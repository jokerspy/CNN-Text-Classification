import numpy as np
import pickle

class word2vec:
    def __init__(self):
        self.w2v_mat = []
    
    def random_word_vector(self,num_words,vec_len):
        # TODO:what is the correct range?
        return np.random.rand(num_words,vec_len)
        
    def random_init(self,num_words,vec_len):
        # generate random w2v
        # TODO:what is the correct range?
        self.w2v_mat = self.random_word_vector(num_words,vec_len)
        # save the random w2v into rndw2v.bin
        with open('rndw2v_%d_%d.bin' %(num_words,vec_len), 'wb') as f:
            pickle.dump(self.w2v_mat, f)
        return self.w2v_mat
    
    def load_random_w2v(self,num_words,vec_len):
        # load the random w2v from rndw2v.bin
        try:
            with open('rndw2v_%d_%d.bin' %(num_words,vec_len), 'rb') as f:
                self.w2v_mat = pickle.load(f)
                print('Load w2v file for words=%d, vec_len=%d'%(num_words,vec_len))
        except IOError:
            print('Generate random word2vec')
            self.w2v_mat = self.random_init(num_words,vec_len)
        return self.w2v_mat
    
    def load_google_w2v(self,file,words):
        # little modification from YoonKim's code
        try:
            with open('google_w2v_%d.bin' %len(words), 'rb') as f:
                self.w2v_mat = pickle.load(f)
                print('Load google w2v file for words=%d'%len(words))
        except IOError:
            with open(file, "rb") as f:
                header = f.readline()
                words_size, vec_len = map(int, header.split())
                self.w2v_mat = np.zeros(shape = [len(words),vec_len])
                mem_size = np.dtype('float32').itemsize * vec_len
                print('Load google word2vec') 
                for line in range(words_size):
                    word = []
                    while True:
                        ch = f.read(1).decode("ISO-8859-1")
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    if word in words:
                        self.w2v_mat[words[word].id] = np.fromstring(f.read(mem_size), dtype='float32')  
                    else:
                        f.read(mem_size)
                    #print("\rProcessing google w2v: %d / %d" % (line+1, words_size), end='')
                #print("\n")
            #TODO: words that not exist in google w2v
            rnd_vec = self.random_word_vector(len(words),vec_len)
            missing_word_ids = [ids for ids in range(len(words)) if sum(filter(lambda x: x!=0, self.w2v_mat[ids]))==0]
            print('Words not in pre-trained w2v=%d'%len(missing_word_ids))
            self.w2v_mat[missing_word_ids] = rnd_vec[missing_word_ids]
            with open('google_w2v_%d.bin' %len(words), 'wb') as fp:
                pickle.dump(self.w2v_mat, fp)
           
        return self.w2v_mat

    
#W = word2vec()
#print(W.random_init([3,2,3],3))
#print(W.load_random_w2v([3,2,3,4],3))
#print(W.load_google_w2v('GoogleNews-vectors-negative300.bin',{'jokerspt'}))