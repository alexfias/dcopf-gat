# dcopf_gat/model.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend


def create_ffn(hidden_units, drop=False, dropout_rate=0.1, layers=1, active=True):
    ffn_layers = []
    for _ in range(layers):
        ffn_layers.append(keras.layers.Dense(hidden_units))
        if active is True:
            ffn_layers.append(keras.layers.LeakyReLU())
        elif active is not None:
            ffn_layers.append(active)
        if drop:
            ffn_layers.append(keras.layers.Dropout(dropout_rate))
    return keras.Sequential(ffn_layers)


class MyR2(keras.metrics.Metric):
    def __init__(self, name="R2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def r2(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
        y_mean = tf.reduce_mean(y_true, axis=0)
        ss_tot = tf.reduce_sum(tf.square(y_true - y_mean), axis=0)
        out = 1.0 - ss_res / (ss_tot + backend.epsilon())
        return tf.reduce_mean(out)

    def update_state(self, y_true, y_pred, **kwargs):
        t = self.r2(y_true, y_pred)
        t_sum = tf.reduce_sum(t)
        self.total.assign_add(t_sum)
        num = tf.cast(tf.size(t), tf.float32)
        self.count.assign_add(num)

    def result(self):
        return self.total / self.count


class ConstrainedLoss(keras.losses.Loss):
    def __init__(
        self,
        g_max,
        d_max,
        f_max,
        withd_m,
        injec_m,
        num_nodes_orig,
        pca=False,
        pca_compo=None,
        pca_mean=None,
        lamb=0.001,
        output_weight=None,
        gen_bus_abs=False,
        name="constrained_loss",
    ):
        super().__init__(name=name)
        self.ConstraintLossFunc = keras.losses.MeanAbsoluteError()
        self.gene_max = tf.constant(g_max, dtype=tf.float32)
        self.gene_num = len(self.gene_max)
        self.dmand_max = tf.constant(d_max, dtype=tf.float32)
        self.flow_max = tf.constant(f_max, dtype=tf.float32)
        self.withd_m = tf.constant(withd_m, dtype=tf.float32)
        self.injec_m = tf.constant(injec_m, dtype=tf.float32)
        self.num_nodes = num_nodes_orig

        self.pca = pca
        if pca:
            self.pca_component = tf.constant(pca_compo, dtype=tf.float32)
            self.pca_mean = tf.constant(pca_mean, dtype=tf.float32)
        else:
            self.pca_component = None
            self.pca_mean = None

        self.lamb = lamb
        self.weights = output_weight
        self.gen_bus_abs = gen_bus_abs

    def CalPow(self, data):
        withdraw = tf.matmul(data, self.withd_m)
        injection_neg = tf.matmul(data, self.injec_m)
        return withdraw + injection_neg

    def WeightedLogCosh(self, y_true, y_pred):
        diff = y_pred - y_true
        log_cosh = tf.math.log(tf.math.cosh(diff + 1e-12))
        if self.weights is None:
            return tf.reduce_mean(log_cosh)
        else:
            w = tf.constant(self.weights, dtype=tf.float32)
            return tf.reduce_mean(tf.reduce_sum(log_cosh * w, axis=-1))

    def call(self, gene_true, flow_true, flow_pred, demand):
        # if PCA: flow_pred is in PC space, reconstruct
        if self.pca:
            flow_pred_real = tf.matmul(flow_pred, self.pca_component) + self.pca_mean
        else:
            flow_pred_real = flow_pred

        resi_loss_flow = tf.cast(self.WeightedLogCosh(flow_true, flow_pred), tf.float32)

        power_withdraw_pred = self.CalPow(flow_pred_real * self.flow_max)
        demand_real = tf.reshape(demand, (-1, self.num_nodes)) * self.dmand_max

        if not self.gen_bus_abs:
            gene_true = gene_true * self.gene_max

        cons_loss = self.lamb * self.ConstraintLossFunc(gene_true, power_withdraw_pred + demand_real)
        total_loss = resi_loss_flow + cons_loss
        return total_loss, resi_loss_flow, cons_loss


class GraphAttention(keras.layers.Layer):
    def __init__(
        self,
        units,
        edges,
        num_nodes_orig,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.edges = tf.constant(edges, dtype=tf.int32)
        self.num_nodes = num_nodes_orig
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
            trainable=True,
        )
        self.a = self.add_weight(
            shape=(self.units * 2, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        node_states = inputs  # [B, N, F]
        node_states_transformed = tf.squeeze(
            tf.matmul(tf.expand_dims(node_states, axis=-2), self.W),
            axis=-2,
        )  # [B, N, units]

        node_states_expanded = tf.gather(node_states_transformed, self.edges, axis=1)
        node_states_expanded = tf.reshape(
            node_states_expanded, (-1, tf.shape(self.edges)[0], 2 * self.units)
        )  # [B, E, 2*units]

        num_neighborhood = tf.math.bincount(tf.cast(self.edges[:, 0], "int32"))

        attention_scores = tf.squeeze(
            tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.a)),
            axis=-1,
        )  # [B, E]
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2.0, 2.0))

        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=tf.transpose(attention_scores),
            segment_ids=self.edges[:, 0],
            num_segments=tf.reduce_max(self.edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(attention_scores_sum, num_neighborhood, axis=0)
        attention_scores_sum = tf.transpose(attention_scores_sum)
        attention_scores_norm = attention_scores / attention_scores_sum

        node_states_neighbors = tf.gather(
            node_states_transformed, self.edges[:, 1], axis=1
        )  # [B, E, units]

        out = tf.math.unsorted_segment_sum(
            data=tf.transpose(
                node_states_neighbors * attention_scores_norm[:, :, tf.newaxis],
                perm=[1, 0, 2],
            ),
            segment_ids=self.edges[:, 0],
            num_segments=tf.shape(node_states)[1],
        )
        return tf.transpose(out, perm=[1, 0, 2])  # [B, N, units]


class MultiHeadGraphAttention(keras.layers.Layer):
    def __init__(self, units, edge_list, num_heads, num_nodes_orig, **kwargs):
        super().__init__(**kwargs)
        self.heads = [
            GraphAttention(units=units, edges=edge_list[h], num_nodes_orig=num_nodes_orig)
            for h in range(num_heads)
        ]

    def call(self, inputs):
        head_outputs = [h(inputs) for h in self.heads]
        return tf.concat(head_outputs, axis=-1)


class LinkQueryAttention(keras.layers.Layer):
    """
    Level-B decoder (NLAT-like):
      - Per-link query built from endpoint embeddings (+ optional link PE)
      - Keys/Values are node embeddings (+ optional node PE)
      - Cross-attention yields link-specific context
    """
    def __init__(
        self,
        link_edges,          # [L,2] int indices in nodes_orig ordering
        node_pe=None,        # [N,Cn] or None
        link_pe=None,        # [L,Ce] or None
        key_dim=64,
        value_dim=128,
        num_heads=4,
        dropout=0.1,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.link_edges = tf.constant(link_edges, dtype=tf.int32)
        self.node_pe = None if node_pe is None else tf.cast(node_pe, tf.float32)
        self.link_pe = None if link_pe is None else tf.cast(link_pe, tf.float32)

        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.q_proj = keras.layers.Dense(
            self.num_heads * self.key_dim, use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.k_proj = keras.layers.Dense(
            self.num_heads * self.key_dim, use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.v_proj = keras.layers.Dense(
            self.num_heads * self.value_dim, use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.out_proj = keras.layers.Dense(
            self.value_dim, use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.attn_drop = keras.layers.Dropout(self.dropout)

    def _split_heads(self, x, head_dim):
        # x: [B, T, H*D] -> [B, H, T, D]
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, [B, T, self.num_heads, head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, node_states, training=False):
        """
        node_states: [B, N, F]
        return:      [B, L, value_dim]
        """
        B = tf.shape(node_states)[0]

        # Keys/values come from all nodes (+ node PE)
        kv = node_states
        if self.node_pe is not None:
            node_pe_ext = tf.repeat(tf.expand_dims(self.node_pe, axis=0), B, axis=0)  # [B,N,Cn]
            kv = tf.concat([kv, node_pe_ext], axis=-1)  # [B,N,F+Cn]

        # Queries come from link endpoints (+ link PE)
        src = self.link_edges[:, 0]
        dst = self.link_edges[:, 1]
        h_src = tf.gather(node_states, src, axis=1)  # [B,L,F]
        h_dst = tf.gather(node_states, dst, axis=1)  # [B,L,F]
        q_in = tf.concat([h_src, h_dst], axis=-1)    # [B,L,2F]

        if self.link_pe is not None:
            link_pe_ext = tf.repeat(tf.expand_dims(self.link_pe, axis=0), B, axis=0)  # [B,L,Ce]
            q_in = tf.concat([q_in, link_pe_ext], axis=-1)  # [B,L,2F+Ce]

        Q = self.q_proj(q_in)  # [B,L,H*key_dim]
        K = self.k_proj(kv)    # [B,N,H*key_dim]
        V = self.v_proj(kv)    # [B,N,H*value_dim]

        Qh = self._split_heads(Q, self.key_dim)      # [B,H,L,key_dim]
        Kh = self._split_heads(K, self.key_dim)      # [B,H,N,key_dim]
        Vh = self._split_heads(V, self.value_dim)    # [B,H,N,value_dim]

        scale = tf.cast(self.key_dim, tf.float32) ** -0.5
        scores = tf.matmul(Qh, Kh, transpose_b=True) * scale  # [B,H,L,N]
        attn = tf.nn.softmax(scores, axis=-1)                 # [B,H,L,N]
        attn = self.attn_drop(attn, training=training)

        ctx = tf.matmul(attn, Vh)  # [B,H,L,value_dim]
        ctx = tf.transpose(ctx, [0, 2, 1, 3])  # [B,L,H,value_dim]
        ctx = tf.reshape(ctx, [B, tf.shape(ctx)[1], self.num_heads * self.value_dim])  # [B,L,H*value_dim]

        return self.out_proj(ctx)  # [B,L,value_dim]


class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        num_nodes_orig,
        num_links,
        node_pe_orig,
        link_pe,
        link_edges,   # <-- NEW (required for Level-B decoder)
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
        name="gat_powerflow",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_nodes_orig = num_nodes_orig
        self.num_links = num_links
        self.node_PE = tf.cast(node_pe_orig, tf.float32)

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

        # Level-B NLAT-like link-query attention decoder
        self.decoder_layer = LinkQueryAttention(
            link_edges=link_edges,
            node_pe=node_pe_orig,   # keep PE as optional extra signal
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
        node_states = inputs  # [B, N, F]
        B = tf.shape(node_states)[0]
        node_pe_ext = tf.repeat(tf.expand_dims(self.node_PE, axis=0), B, axis=0)

        # preprocess + add node PE (as before)
        x = tf.concat([self.preprocess(node_states), node_pe_ext], axis=-1)

        # GAT backbone
        for att in self.attention_layers:
            x = att(x)

        # NLAT-like link-query attention decoder
        context = self.decoder_layer(x, training=training)  # [B, L, value_dim]

        output = self.MLP(context)
        return tf.squeeze(self.output_layer(output), axis=-1)  # [B, L]

    def train_step(self, data):
        node_states, labels = data
        gene_labels = labels[:, : self.num_nodes_orig]
        flow_labels = labels[:, self.num_nodes_orig :]
        demand = node_states[:, :, -1]

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
        flow_labels = labels[:, self.num_nodes_orig:]
        demand = node_states[:, :, -1]

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
