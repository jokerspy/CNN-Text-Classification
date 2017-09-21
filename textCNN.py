import tensorflow as tf
import numpy as np
import sys
import time
class word_datum: #datatype for word
    def __init__(self,word,ids,vector=None):
        self.word = word
        self.id = ids
        self.vector = vector
        self.count = 0
    def add(self):
        self.count+= 1
class sentence_datum: #datatype for sentence
    def __init__(self,label,sentence,length,cv_split):
        self.label = label
        self.sentence = sentence
        self.length = length
        self.cv_split = cv_split
    

def process_data():
    file = ['rt-polarity.neg','rt-polarity.pos']
    sentence = []
    words = {}
    cv_ratio = 10 # percentage of cross-validation set
    #load negative/positive samples
    for label in (0,1):
        with open(file[label],'rb') as f:
            for line in f:
                #TODO: clean the text?
                sentence.append(sentence_datum(label,line,len(line.split()),np.random.randint(0,cv_ratio)))
                for word in line.split():
                    if word in words:
                        words[word].add()
                    else:
                        words[word] = word_datum(word,len(words))
                    #print(word,vocab[word].id)
                #print('\r processing: %d' % len(sentence),end='')
                
def train():
    pass
def test():
    pass
if __name__ == "__main__":