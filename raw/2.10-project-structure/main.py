from __future__ import absolute_import;
import tensorflow as tf;
import dataset_io as io
import model_v1 as model;
import param

class Trainer:
    def __init__(self,restore=False):
        self.p = param.Param()
        self.train_ds,self.test_ds = io.get_dataset(epoch=self.p.epochs,batchsize=self.p.batchsize)
        self.session = tf.Session()
        self.model_init()
        self.writer = tf.summary.FileWriter(logdir= self.p.log_path, graph = self.session.graph )
        self.saver = tf.train.Saver()
        #Variable Initialization
        with self.session.as_default():
            if(restore==False):            
                 self.initializer.run()
            else:
                self.saver.restore(self.session,save_path=self.p.model_path)


    def model_init(self):
        train_iter = self.train_ds.make_one_shot_iterator()
        self.TRAINX,self.TRAINY = train_iter.get_next()
        y_pred,self.loss,self.train_accuracy = model.build_model(inp=self.TRAINX,label=self.TRAINY)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.p.learning_rate).minimize(self.loss)
        self.merge = tf.summary.merge_all()
        self.initializer = tf.global_variables_initializer()
        
    
    def train(self):
        i=0;
        with self.session.as_default():
            try:
                while(True):
                    self.optimizer.run();
                    self.writer.add_summary(self.merge.eval(),self.p.batchsize*i)
                    print ('loss = %2.6f, accuracy = %2.6f'%(self.loss.eval(),self.train_accuracy.eval()))   
                    i+=1;
                    
            except tf.errors.OutOfRangeError:
                print ('loop ended')          
            self.saver.save(self.session,self.p.model_path)
        

    def predict(self):
        pass

    def score(self):
        pass


if __name__=="__main__":
    trainer = Trainer(restore=False)
    trainer.train()