import sys
import numpy as np
import scipy as sp
import pylab
from scipy.special import gammaln
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
from stop_words import get_stop_words
from os import listdir 
from os.path import isfile, join, isdir 
import string
import copy

def sample_index(p):
    """
    Sample from a multinomial distrubtion and return the sample index
    An experiment with one of p possible outcomes, drawing n samples 
    Given the distribution P(z|data), will pull a sample and return the index
    """
    distribution = np.random.multinomial(1,p, size=1);
    return distribution.argmax();

def word_indices(vector):
    """
    We have a vector of size vocab_size, which holds the count of each word
    for a specific documents. We return a sequence with each word index, 
    of the lenght document length
    """
    #nonzero returns a tuple, so we take the 0th element
    for word_index in vector.nonzero()[0]:
        for i in xrange(int(vector[word_index])):
            yield word_index
            #Yeild causes a tempory list to be built; we only need it once 

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class lda_gibbs_sampler(object):

    def __init__(self, n_topics, alpha = 0.0001, beta = 0.000001):
        """
        n_topics: the desired number of topics
        alpha: a scalar
        beta: a scalar
        """
        self.best_like = -10000000;
        self.num_topics = n_topics;
        self.alpha = alpha;
        self.beta = beta;
    
    def _initialize(self, matrix):
        """
        Matrix input
                |
      num docs  |
                |--------
                    vocab size
        """
        num_docs, vocab_size = matrix.shape;

        #Number of times document and topic z co-occur
        self.n_mz = np.zeros((num_docs,self.num_topics));
        self.best_n_mz = np.zeros((num_docs,self.num_topics));
        #Number of times topic z and word w co-occur
        self.n_zw = np.zeros((self.num_topics, vocab_size));
        self.best_n_zw = np.zeros((self.num_topics, vocab_size));

        # Sums
        self.n_m = np.zeros(num_docs);#total_words_in_document 
        self.n_z = np.zeros(self.num_topics);#total_words_in_topic 
        self.topics = {};

        for m_i, cur_doc in enumerate(documents):#for all documents
            # n is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1; the word i_d
            # This returns a vector 3,4,5,5,5,6,8,... of word ids of len doc_len
            for n, w in enumerate(word_indices(matrix[m_i,:])):
                #sample topic index zmn = k ~ Multinomial (1/K)
                z_mn = np.random.randint(self.num_topics);
                #Increment the document-topic count
                self.n_mz[m_i][z_mn] += 1; 
                #Increment the document-topic sum
                self.n_m[m_i] += 1;
                #Increment the topic-term count
                self.n_zw[z_mn][w] += 1;
                #Increment topic-term sum
                self.n_z[z_mn] += 1;
                #Set topic at document, word in document (m,n)
                self.topics[(m_i,n)] = z_mn;

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size num_topics)
        """
        #The size of the vocab is the second dimension of the n_zw matrix
        vocab_size = self.n_zw.shape[1]

        # TODO: check the sum(self.n_zw stuff)
        left = (self.n_zw[:,w] + self.beta) / \
                (self.n_zw[:,w] + self.beta*vocab_size);
        right = (self.n_mz[m,:] + self.alpha) / \
                (self.n_m[m] + self.alpha * self.num_topics);
        p_z = left * right;
        # Normalize
        p_z /= np.sum(p_z);
        return p_z;
    
    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.n_zw.shape[1]
        n_docs = self.n_mz.shape[0]
        lik = 0

        for z in xrange(self.num_topics):
            lik += log_multi_beta(self.n_zw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m_i in xrange(n_docs):
            lik += log_multi_beta(self.n_mz[m_i,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.num_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.n_zw.shape[1]
        num = self.n_zw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix, max_iterations = 50):
        """
        Run the Gibbs sampler
        """

        n_docs, vocab_size = matrix.shape
        self._initialize(matrix);

        for iteration in xrange(max_iterations):
            for m_i in xrange(n_docs):
                for n, w in enumerate(word_indices(matrix[m_i, :])):
                    z = self.topics[(m_i,n)];
                    self.n_mz[m_i][z] -= 1;
                    self.n_m[m_i] -= 1;
                    self.n_zw[z][w] -= 1;
                    self.n_z[z] -= 1; 

                    p_z = self._conditional_distribution(m_i, w);
                    z = sample_index(p_z);

                    self.n_mz[m_i][z] += 1;
                    self.n_m[m_i] += 1;
                    self.n_zw[z][w] += 1;
                    self.n_z[z] += 1;
                    self.topics[(m_i,n)] = z;

            likelihood = sampler.loglikelihood();
            print "Iteration", iteration
            f.write("Iteration {0}".format(iteration));
            print "Likelihood", likelihood
            f.write("Likelihood {0}".format(likelihood));
            if(likelihood > self.best_like):
                self.best_n_mz = copy.copy(self.n_mz)
                self.best_n_zw = copy.copy(self.n_zw)
                self.best_like = likelihood;

   

    def print_data(self):
        num_topics, num_words = self.best_n_zw.shape;
        num_docs, num_topics = self.best_n_mz.shape;

        print "Final likelihood:", self.best_like
        f.write("Final likelihood: {0}".format(self.best_like));
        for topic_index in xrange(num_topics):
            print "====== Topic", topic_index, "======="
            f.write("====== Topic {0} =======".format(topic_index));

            for z in xrange(num_topics):
                if np.sum(self.best_n_zw[z,:]) > 0:
                    self.best_n_zw[z, :] /= np.sum(self.best_n_zw[z, :]); 
            for m in xrange(num_docs):
                if np.sum(self.best_n_mz[m,:]) > 0:
                    self.best_n_mz[m, :] /= np.sum(self.best_n_mz[m, :]);

            word_count_pairs = [(vocabulary[w], self.best_n_zw[topic_index,w]) for w in range(vocabSize)];
            sorted_list = sorted(word_count_pairs, \
                key = lambda tup: tup[1], reverse=True);
  
            loop = 0;
            for index, count in sorted_list:
                if count > 0:
                    print index , '(%.4f),' %count,
                    f.write("{0}, ({1})".format(index, '(%.4f),' %count))
                if loop > 10:
                    break;
                loop += 1;
            print
            f.write('\n');
        """
        for document_index in xrange(num_docs):
            print "====== Doc", document_index, "======="
            f.write("====== Doc", document_index, "=======")
            for t_i, topic in enumerate(self.best_n_mz[document_index,:]):
                print t_i, '(%.4f),' %topic,
                f.write(t_i, '(%.4f),' %topic,)
            print
            f.write('\n');
        print
        f.write('\n');
        """
                
"""
Initialize the document from text file
"""

def readFile(filename):
    global vocabSize, vocabulary
    words = file(filename).read().lower().split();
    words = [w for w in words if w.isalpha() and w not in stopWords];

    tokens = [];
    
    # Create a mapping for words to indicies
    for w in words:
        if w not in word2Index:
            word2Index[w] = vocabSize;
            vocabulary.append(w);
            vocabSize += 1;

        tokens.append(word2Index[w]);

    return tokens

def readSimple(filename):
    global vocabSize, vocabulary

    f = open(filename, 'r');
    for line in f:
        words = re.sub("[^\w]", " ",  line).lower().split();
        words = [w for w in words if w.isalpha() and w not in stopWords];

        tokens = [];
        
        # Create a mapping for words to indicies
        for w in words:
            if w not in word2Index:
                word2Index[w] = vocabSize;
                vocabulary.append(w);
                vocabSize += 1;

            tokens.append(word2Index[w]);

        documents.append(tokens);
    return documents 

word2Index = {};
vocabulary = [];
vocabSize = 0;
N_TOPICS = int(sys.argv[1])
n_iters = int(sys.argv[2])
stopWords = get_stop_words('en');
mode = "COMPLEX";
documents = [];


if __name__ == '__main__':
    if mode == "COMPLEX":
        for dirs in listdir('./bbc'):
            if isdir(join('./bbc',dirs)):
                for files in listdir('./bbc/' + dirs):
                    if(isfile(join('./bbc/'+dirs, files))):
                        filename = './bbc/' + dirs + '/' + files
                        tokens = readFile(filename);
                        documents.append(tokens);

    if mode == "SIMPLE":
        documents = readSimple('corpus.txt');

    """
    Create a matrix such that:
                | entires are counts of each vocabulary word in each document
    documents   |
    M         |
                |______________
                    vocabulary
                        V
    """
    f = open("results.txt", 'wb');
    matrix = np.zeros((len(documents), len(vocabulary)));
    for m_i, doc in enumerate(documents):
        for word in doc:
            matrix[m_i][word] += 1;

    sampler = lda_gibbs_sampler(N_TOPICS);
    sampler.run(matrix, n_iters);

    sampler.print_data();

