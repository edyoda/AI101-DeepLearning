
import keras
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf;
sess = tf.Session()
K.set_session(sess)
save_path = 'model/exp-5.7/model'
from random import shuffle
import os;
from PIL import Image
import numpy as np

from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf;

# In[15]:


from keras.applications import VGG16



def get_dataset(epoch,batchsize, test_batchsize):
    """
    Get the (data,label) pair
    """
    VGG_MEAN = np.array([123.68, 116.78, 103.94])

    test_prefix = '/home/jaley/Downloads/all/test1'
    def my_gen():
        prefix = '/home/jaley/Downloads/all/train'
        filelist = [filename for filename in os.listdir(prefix)]
        shuffle(filelist)
        for filename in filelist:
            filename=str(filename)
            label = np.zeros((2,))
            if filename.startswith('dog'):                    
                path = prefix + '/'+filename
                label[1]=1.0
                img = np.array(Image.open(path).resize((224,224)),dtype='float32')/255
                yield  img,label;
            if filename.startswith('cat'):                    
                path = prefix + '/'+filename
                label[0]=1.0
                img = np.array(Image.open(path).resize((224,224)),dtype='float32')/255
                yield  img,label;

    
    #Creating Dataset from Generators
    
    train_ds = tf.data.Dataset.from_generator(my_gen,output_types=(tf.float32,tf.float32),output_shapes=((224,224,3),(2,)))
    test_ds = tf.data.Dataset.from_generator(my_gen,output_types=(tf.float32,tf.float32),output_shapes=((224,224,3),(2,)))
    train_ds = train_ds.repeat(epoch).batch(batchsize)
    test_ds = test_ds.repeat(epoch).batch(test_batchsize)
    return train_ds,test_ds;


# In[19]:



train_ds,test_ds = get_dataset(1,20,10)  # Generating the dataset here  
morphing_iter = tf.data.Iterator.from_structure(train_ds.output_types,
                                           train_ds.output_shapes)
print (train_ds.output_shapes)
inp, labels = morphing_iter.get_next()
train_init_op = morphing_iter.make_initializer(train_ds)
test_init_op = morphing_iter.make_initializer(test_ds)






#model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', \
#                    input_tensor=X, input_shape=(224, 224, 3))

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', input_shape=(224, 224, 3),input_tensor=inp,is_training=True)


# In[16]:

output_layer = "fc1"

with tf.variable_scope('finetuning'):
    y1 = vgg_conv.get_layer(output_layer).output
    y2 = layers.Dense(256, activation='relu')(y1)
    y3 = layers.Dropout(0.5)(y2)
    ypred = layers.Dense(2, activation='softmax')(y3)


writer = tf.summary.FileWriter('log',tf.get_default_graph())


with tf.variable_scope('finetuning'):
    loss = tf.losses.softmax_cross_entropy(logits=ypred,onehot_labels=labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,1),tf.argmax(ypred,1)),tf.float32))
var_list = []
for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='finetuning'):
    var_list.append(var)
initializer = tf.variables_initializer(var_list)

saver = tf.train.Saver()

with sess.as_default():
    train_init_op.run()
    initializer.run()
    train_loss_lst,train_accuracy_lst=[],[]
    for itr in range(10000):
        _,_loss,_accuracy = sess.run([optimizer,loss,accuracy])
        train_accuracy_lst.append(_accuracy)
        train_loss_lst.append(_loss)
        if itr%1000 == 0:
            saver.save(sess,save_path,global_step=itr)
        if itr%20 == 0:
            print ('train : itr=%d, loss=%2.6f, accuracy=%2.2f'%(itr,np.mean(train_loss_lst),np.mean(train_accuracy_lst)))
            train_loss_lst=[]
            train_accuracy_lst = []
            #test_init_op.run()
            #val_loss_lst,val_accuracy_lst = [],[]
            #for val_itr in range(50):
            #    _loss,_accuracy = sess.run([loss,accuracy])
            #    val_loss_lst.append(_loss)
            #    val_accuracy_lst.append(_accuracy)
            #if itr==0: # For testing the list
            #    print (val_accuracy_lst)
            #print ('val : itr=%d, loss=%2.6f, accuracy=%2.2f'%(itr,np.mean(val_loss_lst),np.mean(val_accuracy_lst)))
            #train_init_op.run()
            #
        
