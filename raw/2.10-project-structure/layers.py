import tensorflow as tf;

def fully_connected(inp,nout,scope='fc',activation = None):
	with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
		if activation == None:
			activation = lambda x : x;
		nin = inp.shape[-1]
		w = tf.get_variable(name='w',shape=(nin,nout),dtype=tf.float64)
		b = tf.get_variable(name='b',shape=(nout,),dtype=tf.float64)
		out = tf.matmul(inp,w)+b
	return activation(out);


