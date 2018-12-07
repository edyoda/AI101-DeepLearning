import tensorflow as tf;
from layers import fully_connected;


def build_model(inp,label):
    y1 = fully_connected(inp, activation=tf.nn.relu,scope="fc-1")
    out = fully_connected(y1, activation=tf.nn.softmax,scope="fc-2")
    loss = tf.losses.softmax_cross_entropy(logits=out,onehot_labels=label)
    return out,loss;

