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

    def __init__(self, n_topics, alpha = 0.001, beta = 0.000001):
        """
        n_topics: the desired number of topics
        alpha: a scalar
        beta: a scalar
        """
        self.prev_like = -10000000;
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
        #Number of times topic z and word w co-occur
        self.n_zw = np.zeros((self.num_topics, vocab_size));

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
            print "Likelihood", likelihood
            #if(likelihood - self.prev_like < 0):
                #break;
            self.prev_like = likelihood;
   

    def print_data(self):
        num_topics, num_words = self.n_zw.shape;
        num_docs, num_topics = self.n_mz.shape;

        for topic_index in xrange(num_topics):
            print "====== Topic", topic_index, "======="
            word_count_pairs = enumerate(self.n_zw[topic_index,:]);
            sorted_list = sorted(word_count_pairs, \
                key = lambda tup: tup[1], reverse=True);
  
            loop = 0;
            for index, count in sorted_list:
                if count > 0:
                    print I[index], count,
                if loop > 10:
                    break;
                loop += 1;

            print
        """
        for document_index in xrange(num_docs):
            print "====== Doc", document_index, "======="
            for t_i, topic in enumerate(self.n_mz[document_index,:]):
                print topic,
            print
        """
        print
                
"""
Initialize the document from text file
"""


word2Index = {};
vocabulary = [];
vocabSize = 0;
N_TOPICS = int(sys.argv[1])
stopWrods = get_stop_words('en');

"""documents = [];
W_sub = [];
V = {}; I = {};
count = 0;"""

mode = "SIMPLE";

if mode == "COMPLEX":
    dir_list = ['./bbc/sport', './bbc/politics'];
    for dirs in dir_list:
        for files in listdir(dirs):
            if(isfile(join(dirs, files))):
                filename = dirs + '/' + files
                #f = open(filename, 'r');
                words.

                for line in f:
                    wordList = re.sub("[^\w]", " ",  line).split()
                    wordList =  [i for i in wordList if not i in en_stop]
                    for word in wordList:
                        word_lc = string.lower(word);
                        W_sub.append(word_lc);
                        if (word_lc not in V) and (word_lc not in en_stop):
                            V[word_lc] = count; 
                            I[count] = word_lc;
                            count += 1;

                documents.append(W_sub);
                W_sub = [];

"""
#for dirs in listdir('./bbc'):
    #if isdir(join('./bbc',dirs)):
    for files in listdir('./bbc/' + dirs):
        if(isfile(join('./bbc/'+dirs, files))):
            filename = './bbc/' + dirs + '/' + files
            f = open(filename, 'r');
            en_stop = get_stop_words('en');

            for line in f:
                wordList = re.sub("[^\w]", " ",  line).split()
                wordList =  [i for i in wordList if not i in en_stop]
                for word in wordList:
                    W_sub.append(word);
                    if (word not in V) and (word not in en_stop):
                        V[word] = count; 
                        I[count] = word;
                        count += 1;

            documents.append(W_sub);
            W_sub = [];
"""
if mode == "SIMPLE":
    f = open('corpus.txt', 'r');
    en_stop = get_stop_words('en');
    for line in f:
        wordList = re.sub("[^\w]", " ",  line).split()
        wordList =  [i for i in wordList if not i in en_stop]
        for word in wordList:
            word_lc = string.lower(word);
            W_sub.append(word_lc);
            if (word_lc not in V) and (word_lc not in en_stop):
                V[word_lc] = count; 
                I[count] = word_lc;
                count += 1;
        documents.append(W_sub);
        W_sub = [];

"""
Create a matrix such that:
            | entires are counts of each vocabulary word in each document
documents   |
  M         |
            |______________
                vocabulary
                    V
"""
matrix = np.zeros((len(documents), len(V)));
en_stop = get_stop_words('en');
for m_i, doc in enumerate(documents):
    for word in doc:
        word_lc = string.lower(word);
        if word_lc not in en_stop:
            matrix[m_i][V[word_lc]] += 1;

sampler = lda_gibbs_sampler(N_TOPICS);
sampler.run(matrix);

sampler.print_data();

