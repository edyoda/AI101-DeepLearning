from __future__ import absolute_import;
import tensorflow as tf;
import layers;
import param;

p = param.Param()

def build_model(inp,label):
    y1 = layers.fully_connected(inp=inp,nout=p.hidden_units[0], activation=tf.nn.relu,scope="fc-1")
    out = layers.fully_connected(inp=y1,nout=p.num_labels, activation=tf.nn.softmax,scope="fc-2")
    with tf.variable_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(logits=out,onehot_labels=label)
        tf.summary.scalar('loss',loss)
    with tf.variable_scope('accuracy'):        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,1),tf.argmax(out,1)),tf.float64))
        tf.summary.scalar('accuracy',accuracy)

    return out,loss,accuracy;
