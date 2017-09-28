import tensorflow as tf
import numpy as np
class model_cnn:
    def __init__(self, vec_len, text_len, w2v):
        self.vec_len = vec_len
        self.text_len = text_len
        self.dropout_rate = 0.5
        self.learning_rate = 1e-4
        self.w2v = w2v
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None
        self.build()
    
    def build(self):
        self.add_placeholder()
        self.add_prediction()
        self.add_loss()
        self.add_optimizer()
    
    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=.5):
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.mask_placeholder: mask_batch,
            #self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict
    
    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=-1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=self.dropout_rate)
        _, loss = sess.run([self.train, self.loss], feed_dict=feed)
        return loss
        
    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.text_len],name='inputs')
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None],name='labels')
        self.mask_placeholder = tf.placeholder(dtype=tf.bool, shape=[None, self.text_len])
        #self.dropout_placeholder = tf.placeholder(dtype=tf.float32)
        
    def add_prediction(self):
        lookup = tf.Variable(self.w2v,dtype=tf.float32,name='w2v') #[len(words) , vec_len]
        embeddings = tf.nn.embedding_lookup(params=lookup,ids=self.input_placeholder,name='embedding') #[None, self.text_len, vec_len, 1]
        #3,4,5x100
        max = [[],[],[]]
        for window in [3,4,5]:
            shape = [window, self.vec_len, 1, 100]
            epsilon = np.sqrt(6/np.sum(shape))
            filter = tf.Variable(tf.random_uniform(shape=shape, minval=-epsilon, maxval=epsilon,dtype=tf.float32),name='filter_%d'%window) #[1, window, vec_len, 100]
            
            conv = tf.nn.conv2d(tf.expand_dims(embeddings,-1),filter,strides=[1,1,1,1],padding='VALID',name='conv_%d'%window) #[None, text_len-window+1, 1, 100]
            bias_conv = tf.Variable(tf.zeros([self.text_len-window+1, 1, 100]),dtype=tf.float32,name='conv_bias_%d'%window)
            features = tf.nn.relu(conv + bias_conv,name='relu_%d'%window)
            
            #valid_features = tf.boolean_mask(mask=self.mask_placeholder,tensor=features)
            max[window-3] = tf.nn.max_pool(features,[1, self.text_len-window+1,1,1],[1,1,1,1],'VALID',name='maxpool_%d'%window) #[None,1,1,100]
            
        max_drop = tf.nn.dropout(tf.squeeze(tf.concat(max,-1),name='features'),keep_prob=1-self.dropout_rate) #[None,100]
        #print(max_drop,tf.squeeze(tf.concat(max,-1)),max[0])
        shape = [300,2]
        epsilon = np.sqrt(6/np.sum(shape))
        W_fc = tf.Variable(tf.random_uniform(shape=shape, minval=-epsilon, maxval=epsilon),dtype=tf.float32,name='fc_weight')
        bias_fc = tf.Variable(tf.zeros([2]),dtype=tf.float32,name='fc_bias')
            
        #self.pred = tf.matmul(max_drop,W_fc) + bias_fc
        self.pred = tf.nn.xw_plus_b(max_drop,W_fc,bias_fc,name='prediction')
        
    def add_loss(self):
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,labels=self.labels_placeholder,name='loss')

       
    def add_optimizer(self):
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

if __name__ == "__main__" :
    sess = tf.InteractiveSession()
    w2v = np.zeros([3,300])
    M = model_cnn(300,10,w2v)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #print(sess.run(tf.report_uninitialized_variables()))
    writer = tf.summary.FileWriter("", sess.graph)
    writer.close()
    for epoch in range(100):
        for batch in range(100):
            inputs_batch = np.zeros([5,10],dtype='int32')
            labels_batch = np.zeros([5],dtype='int32')
            mask_batch = np.zeros([5,10],dtype='bool')
            M.train_on_batch(sess, inputs_batch, labels_batch, mask_batch)
            print(labels_batch,M.predict_on_batch(sess, inputs_batch, mask_batch))