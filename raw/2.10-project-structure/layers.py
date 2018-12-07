import tensorflow;




def fully_connected(self,inp,nout,scope='fc',activation = None):
	with tf.name_scope(scope):
		if activation = None:
			activation = lambda x : x;
		nin = inp.shape[-1]
		w = tf.get_variable(name='w',shape=(nin,nout))
		b = tf.get_variable(name='b',shape=(nout,))
		out = tf.matmul(inp,w)+b
	return activation(out);


