# dcopf_gat/data_pipeline.py
import tensorflow as tf


def make_dataset(x, y, batch_size, shuffle=False):
    """
    Create a tf.data.Dataset that works efficiently on CPU and GPU.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 10000))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds