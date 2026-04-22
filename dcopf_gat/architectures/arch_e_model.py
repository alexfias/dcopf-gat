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


class GraphAttentionNetworkGRU(keras.Model):
    """
    Sequence-aware GAT model for Architecture E.

    Input shape:
        [B, W, N, F]

    where
        B = batch size
        W = window length
        N = number of nodes
        F = number of node features

    Processing:
        1. For each node independently, encode the W-step feature sequence
           with a GRU.
        2. Reshape back to node embeddings [B, N, H].
        3. Apply the graph-attention backbone.
        4. Decode link flows as before.
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
        hidden_units=64,
        num_heads=3,
        num_layers=3,
        gru_units=64,
        name="gat_powerflow_gru",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_nodes_orig = num_nodes_orig
        self.num_links = num_links
        self.node_PE = tf.cast(node_pe_orig, tf.float32)
        self.gru_units = gru_units

        # Temporal encoder: one shared GRU applied to every node sequence.
        self.temporal_gru = keras.layers.GRU(
            gru_units,
            return_sequences=False,
            dtype="float32",
            name="temporal_gru",
        )

        # Same general backbone structure as the snapshot GAT.
        self.preprocess = create_ffn(hidden_units * num_heads, layers=1, drop=False)

        self.attention_layers = [
            MultiHeadGraphAttention(
                units=hidden_units,
                edge_list=edge_list,
                num_heads=num_heads,
                num_nodes_orig=num_nodes_orig,
            )
            for _ in range(num_layers)
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
        """
        inputs: [B, W, N, F]
        """
        node_states = inputs

        shape = tf.shape(node_states)
        B = shape[0]
        W = shape[1]
        N = shape[2]
        F = shape[3]

        # [B, W, N, F] -> [B, N, W, F]
        x = tf.transpose(node_states, perm=[0, 2, 1, 3])

        # [B, N, W, F] -> [B*N, W, F]
        x = tf.reshape(x, [B * N, W, F])

        # GRU over time for each node
        x = self.temporal_gru(x)  # [B*N, gru_units]

        # [B*N, H] -> [B, N, H]
        x = tf.reshape(x, [B, N, self.gru_units])

        # Add node positional encodings
        node_pe_ext = tf.repeat(tf.expand_dims(self.node_PE, axis=0), B, axis=0)

        # Preprocess + PE
        x = tf.concat([self.preprocess(x), node_pe_ext], axis=-1)

        # GAT backbone
        for att in self.attention_layers:
            x = att(x)

        # Link-query decoder
        context = self.decoder_layer(x, training=training)  # [B, L, value_dim]

        output = self.MLP(context)
        return tf.squeeze(self.output_layer(output), axis=-1)  # [B, L]

    def train_step(self, data):
        node_states, labels = data

        gene_labels = labels[:, : self.num_nodes_orig]
        flow_labels = labels[:, self.num_nodes_orig :]

        # Demand from the last timestep in the window
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

        # Demand from the last timestep in the window
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


def build_gru_gat(meta: Dict[str, Any], lamb: float) -> keras.Model:
    return GraphAttentionNetworkGRU(
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
        hidden_units=64,
        num_heads=3,
        num_layers=3,
        gru_units=64,
    )