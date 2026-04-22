from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
from tensorflow import keras

from ..model import (
    create_ffn,
    MultiHeadGraphAttention,
    LinkQueryAttention,
    ConstrainedLoss,
    MyR2,
)


class GraphTemporalBlock(keras.layers.Layer):
    """
    Alternates temporal mixing and graph attention on a sequence of node states.

    Input / output shape:
      x: [B, W, N, C]
    """

    def __init__(
        self,
        channels,
        edge_list,
        num_heads,
        num_nodes_orig,
        temporal_kernel_size=3,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.dropout_rate = float(dropout)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.graph_attention = MultiHeadGraphAttention(
            units=self.channels // num_heads,
            edge_list=edge_list,
            num_heads=num_heads,
            num_nodes_orig=num_nodes_orig,
        )
        self.temporal_norm = keras.layers.LayerNormalization()
        self.spatial_norm = keras.layers.LayerNormalization()
        self.ffn_norm = keras.layers.LayerNormalization()
        self.temporal_conv = keras.layers.Conv1D(
            filters=self.channels,
            kernel_size=self.temporal_kernel_size,
            padding="causal",
            activation="gelu",
        )
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(self.channels * 2, activation="gelu"),
                keras.layers.Dropout(self.dropout_rate),
                keras.layers.Dense(self.channels),
            ]
        )

    def call(self, x, training=False):
        b = tf.shape(x)[0]
        w = tf.shape(x)[1]
        n = tf.shape(x)[2]
        c = tf.shape(x)[3]

        xt = self.temporal_norm(x)
        xt = tf.transpose(xt, [0, 2, 1, 3])
        xt = tf.reshape(xt, [b * n, w, c])
        xt = self.temporal_conv(xt)
        xt = self.dropout(xt, training=training)
        xt = tf.reshape(xt, [b, n, w, c])
        xt = tf.transpose(xt, [0, 2, 1, 3])
        x = x + xt

        xs = self.spatial_norm(x)
        xs = tf.reshape(xs, [b * w, n, c])
        xs = self.graph_attention(xs)
        xs = self.dropout(xs, training=training)
        xs = tf.reshape(xs, [b, w, n, c])
        x = x + xs

        xf = self.ffn_norm(x)
        xf = self.ffn(xf, training=training)
        xf = self.dropout(xf, training=training)
        return x + xf


class GraphAttentionNetworkTemporalBlocks(keras.Model):
    """
    Sequence-aware GAT model for Architecture F.

    Input shape:
        [B, W, N, F]

    Processing:
        1. Project node features to hidden channels while keeping the time axis.
        2. Add projected node positional encodings at each timestep.
        3. Apply stacked graph-temporal blocks:
           - causal temporal convolution over each node history
           - graph attention over nodes at each timestep
           - residual FFN refinement
        4. Decode line flows from the final timestep node states.
    """

    def __init__(
        self,
        num_nodes_orig,
        num_links,
        node_pe_orig,
        link_pe,
        link_edges,
        edge_list,
        g_max,
        d_max,
        f_max,
        withd_m,
        injec_m,
        pca_obj=None,
        lamb=0.001,
        output_weight=None,
        hidden_units=32,
        num_heads=3,
        temporal_blocks=2,
        temporal_kernel_size=3,
        temporal_dropout=0.05,
        name="gat_powerflow_temporal_blocks",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_nodes_orig = num_nodes_orig
        self.num_links = num_links
        self.node_PE = tf.cast(node_pe_orig, tf.float32)
        self.hidden_channels = hidden_units * num_heads

        self.preprocess = create_ffn(self.hidden_channels, layers=1, drop=False)
        self.pe_project = keras.layers.Dense(self.hidden_channels, use_bias=False)

        self.graph_temporal_blocks = [
            GraphTemporalBlock(
                channels=self.hidden_channels,
                edge_list=edge_list,
                num_heads=num_heads,
                num_nodes_orig=num_nodes_orig,
                temporal_kernel_size=temporal_kernel_size,
                dropout=temporal_dropout,
            )
            for _ in range(temporal_blocks)
        ]

        self.decoder_layer = LinkQueryAttention(
            link_edges=link_edges,
            node_pe=node_pe_orig,
            link_pe=link_pe,
            key_dim=64,
            value_dim=128,
            num_heads=4,
            dropout=0.1,
        )

        self.MLP = create_ffn(hidden_units=32, layers=2, drop=True)
        self.output_layer = keras.layers.Dense(
            1,
            activation=None if pca_obj is not None else keras.activations.tanh,
            dtype="float32",
        )

        pca_flag = pca_obj is not None
        pca_compo = pca_obj.components_ if pca_flag else None
        pca_mean = pca_obj.mean_ if pca_flag else None

        self.loss_func = ConstrainedLoss(
            g_max=g_max,
            d_max=d_max,
            f_max=f_max,
            withd_m=withd_m,
            injec_m=injec_m,
            num_nodes_orig=num_nodes_orig,
            pca=pca_flag,
            pca_compo=pca_compo,
            pca_mean=pca_mean,
            lamb=lamb,
            output_weight=output_weight,
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_tracker1 = keras.metrics.Mean(name="loss1")
        self.loss_tracker2 = keras.metrics.Mean(name="loss2")
        self.metric = MyR2("R2")

    def call(self, inputs, training=False):
        x = self.preprocess(inputs)

        b = tf.shape(inputs)[0]
        w = tf.shape(inputs)[1]
        node_pe = self.pe_project(self.node_PE)
        node_pe = tf.reshape(node_pe, [1, 1, self.num_nodes_orig, self.hidden_channels])
        node_pe = tf.repeat(node_pe, b, axis=0)
        node_pe = tf.repeat(node_pe, w, axis=1)
        x = x + node_pe

        for block in self.graph_temporal_blocks:
            x = block(x, training=training)

        node_states = x[:, -1, :, :]
        context = self.decoder_layer(node_states, training=training)
        output = self.MLP(context)
        return tf.squeeze(self.output_layer(output), axis=-1)

    def train_step(self, data):
        node_states, labels = data
        gene_labels = labels[:, : self.num_nodes_orig]
        flow_labels = labels[:, self.num_nodes_orig :]
        demand = node_states[:, -1, :, -1]

        with tf.GradientTape() as tape:
            output = self(node_states, training=True)
            total_loss, resi_loss_flow, cons_loss = self.loss_func.call(
                gene_labels, flow_labels, output, demand
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(total_loss)
        self.loss_tracker1.update_state(resi_loss_flow)
        self.loss_tracker2.update_state(cons_loss)
        self.metric.update_state(flow_labels, output)

        return {
            "loss": self.loss_tracker.result(),
            "loss1": self.loss_tracker1.result(),
            "loss2": self.loss_tracker2.result(),
            "R2": self.metric.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_tracker1, self.loss_tracker2, self.metric]

    def test_step(self, data):
        node_states, labels = data
        gene_labels = labels[:, : self.num_nodes_orig]
        flow_labels = labels[:, self.num_nodes_orig :]
        demand = node_states[:, -1, :, -1]

        output = self(node_states, training=False)
        total_loss, resi_loss_flow, cons_loss = self.loss_func.call(
            gene_labels, flow_labels, output, demand
        )

        self.loss_tracker.update_state(total_loss)
        self.loss_tracker1.update_state(resi_loss_flow)
        self.loss_tracker2.update_state(cons_loss)
        self.metric.update_state(flow_labels, output)

        return {
            "loss": self.loss_tracker.result(),
            "loss1": self.loss_tracker1.result(),
            "loss2": self.loss_tracker2.result(),
            "R2": self.metric.result(),
        }


def build_temporal_block_gat(meta: Dict[str, Any], lamb: float) -> keras.Model:
    return GraphAttentionNetworkTemporalBlocks(
        num_nodes_orig=meta["num_nodes_orig"],
        num_links=meta["num_links"],
        node_pe_orig=meta["node_pe_orig"],
        link_pe=meta["link_pe"],
        link_edges=meta["link_edges"],
        edge_list=meta["edge_list"],
        g_max=meta["p_nom_bus"],
        d_max=meta["demand_max"],
        f_max=meta["flow_max"],
        withd_m=meta["withd_m"],
        injec_m=meta["injec_m"],
        pca_obj=meta["pca"],
        lamb=lamb,
        output_weight=meta["output_weight"],
        hidden_units=32,
        num_heads=3,
        temporal_blocks=2,
        temporal_kernel_size=3,
        temporal_dropout=0.05,
    )
