import tensorflow as tf;
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sklearn
import sklearn.datasets;

import pandas as pd

def get_dataset(epoch,batchsize):
		"""
		Get the (data,label) pair
		"""
		iris_ds = sklearn.datasets.load_iris(return_X_y=False)
		iris_data = pd.DataFrame(data=iris_ds.data,columns=iris_ds.feature_names)
		min_max_scaler = MinMaxScaler()
		scaled_data = min_max_scaler.fit_transform(iris_data)
		encoder = OneHotEncoder(n_values=3)
		label = encoder.fit_transform(iris_ds.target.reshape(-1,1))
		label = label.todense()
		trainx,testx,trainy,testy = train_test_split(scaled_data,label)	
		#Creating Dataset
		train_ds = tf.data.Dataset.from_tensor_slices((trainx,trainy)).shuffle(1000).repeat(epoch).batch(batchsize)
		#Creating Dataset
		test_ds = tf.data.Dataset.from_tensors((testx,testy)).shuffle(1000)
		return train_ds,test_ds;





