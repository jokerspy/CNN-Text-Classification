import tensorflow as tf
import numpy as np
import sys
import time
import re
from model import *
from word2vec import *
import pandas as pd
class word_datum: #datatype for word
    def __init__(self,ids):
        #self.word = word
        self.id = ids
        #self.vector = vector
        self.count = 0
    def add(self):
        self.count+= 1
class sentence_datum: #datatype for sentence
    def __init__(self,label,sentence,length,cv_split):
        self.label = label
        self.sentence = sentence
        self.length = length
        self.cv_split = cv_split
        self.index = None

def clean_text(text):
    """
    Tokenization/string cleaning for all datasets except for SST.
    From YoonKim's code https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def process_data(sentence,words,cv_split_num):
    file = ['rt-polarity.neg','rt-polarity.pos']
    #load negative/positive samples
    words['PAD_STR'] = word_datum(0)
    max_len = 0
    for label in (0,1):
        with open(file[label],'rb') as f:
            for line in f:
                line_clean = clean_text(line.decode("ISO-8859-1")).split()
                max_len = max(max_len, len(line_clean))
                for word in set(line_clean):
                    if word in words:
                        words[word].add()
                    else:
                        words[word] = word_datum(len(words))
                sentence.append(sentence_datum(label,line_clean,len(line_clean),np.random.randint(0,cv_split_num)))                
                #print(word,vocab[word].id)
                #print('\r processing: %d' % len(sentence),end='')
           
    return max_len
def process_dataset(sentence,words,max_len):
    # transfer word in the sentence into the index in w2v
    #TODO: padding, count length, separate train/test set
    train, test = {'x':[],'y':[]},{'x':[],'y':[]}
    #train, test = [], []
    for sample in sentence:
        sample.index = [words[word].id for word in sample.sentence] # if word in words]
        if len(sample.index)<max_len:
            sample.index+=[0]*(max_len-len(sample.index))
        if sample.cv_split==0:
            #test.append({'input':sample.index,'label':sample.label})
            test['x'].append(sample.index)
            test['y'].append(sample.label)
        else:
            #train.append({'input':sample.index,'label':sample.label})
            train['x'].append(sample.index)
            train['y'].append(sample.label)
    return train, test
def train(w2v='google'):
    vec_len = 300
    cv_split_num = 10 # split number of cross-validation set
    learning_rate = 1e-3
    dropout_rate = 0.5
    sentence = []
    words = {}
    print('Process data from RT dataset')
    max_len = process_data(sentence,words,cv_split_num)
    print('Max sentence length=%d'%max_len)
    W = word2vec()
    if w2v=='google':
        W.load_google_w2v('GoogleNews-vectors-negative300.bin',words)
        print('Total words in w2v=%d'%len(W.w2v_mat))
    else:
        W.load_random_w2v(len(words),vec_len)
    
    train, test = process_dataset(sentence,words,max_len)
    sess = tf.InteractiveSession()
    
    M = model_cnn(vec_len,max_len,W.w2v_mat,learning_rate)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #print(sess.run(tf.report_uninitialized_variables()))
    writer = tf.summary.FileWriter("", sess.graph)
    writer.close()
    for epoch in range(1):
        for batch in range(100):
            inputs_batch = train['x'][0:2]#
            #inputs_batch = list(pd.DataFrame(train[0:2])['input'])
            #print(len(inputs_batch),len(inputs_batch[0]))
            #print(train['x'][0:2])
            labels_batch = train['y'][0:2]#
            #labels_batch = list(pd.DataFrame(train[0:2])['label'])
            M.train_on_batch(sess, inputs_batch, labels_batch, dropout_rate)
            print(labels_batch,M.predict_on_batch(sess, inputs_batch))
    
def test():
    pass
    
if __name__ == "__main__":
    train()