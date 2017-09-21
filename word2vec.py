import numpy as np
import pickle

class word2vec:
    def __init__(self):
        self.w2v_mat = []
    
    def random_init(self,words,vec_len):
        # generate random w2v
        # TODO:what is the correct range?
        self.w2v_mat = np.random.rand(vec_len,len(words))
        # save the random w2v into rndw2v.bin
        with open('rndw2v_%d_%d.bin' %(len(words),vec_len), 'wb') as f:
            pickle.dump(self.w2v_mat, f)
        return self.w2v_mat
    
    def load_random_w2v(self,words,vec_len):
        # load the random w2v from rndw2v.bin
        try:
            with open('rndw2v_%d_%d.bin' %(len(words),vec_len), 'rb') as f:
                self.w2v_mat = pickle.load(f)
        except IOError:
            
            print("Generating random word2vec")
            self.w2v_mat = self.random_init(words,vec_len)
        return self.w2v_mat
    
    def load_google_w2v(self,file,words):
        # little modification from YoonKim's code
        with open(file, "rb") as f:
            header = f.readline()
            vocab_size, vec_len = map(int, header.split())
            self.w2v_mat = np.zeros(shape = [vec_len,len(words)])
            mem_size = np.dtype('float32').itemsize * vec_len
            for line in range(vocab_size):
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
                print("\rProcessing google w2v: %d / %d" % (line+1, vocab_size), end='')
            print("\n")
        #TODO: words that not exist in google w2v
        return self.w2v_mat

    
W = word2vec()
print(W.random_init([3,2,3],3))
print(W.load_random_w2v([3,2,3,4],3))
