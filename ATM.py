# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:18:05 2016

two dimensional author topic model (to be renamed) 
  / an extension of Author Topic Model described in 
  / The Author-Topic Model for Authors and Documents
  
@author: Chin-Wei Huang, Pierre-André Brousseau

Adapted from https://gist.github.com/mblondel/542786
  / Cython version to be implemented following
  / https://github.com/fannix/lda-cython/blob/master/lda_gibbs_cython.pyx
"""


import numpy as np
from scipy.special import gammaln
import json

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[1]:
        for i in xrange(int(vec[0,idx])):
            yield idx

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
    
    
class LdaSampler(object):

    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, reviews, item_tags, user_tags):
        n_docs, vocab_size = reviews.shape
        n_itags = item_tags.shape[1]
        n_utags = user_tags.shape[1]

        # number of times itag i, utag u and topic z co-occur
        self.niuz = np.zeros((n_itags, n_utags, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        
        self.niu = np.zeros((n_itags, n_utags))
        self.nz = np.zeros(self.n_topics)
        self.author_topic = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(reviews[m, :])):
                # choose an arbitrary topic as first topic for word i
            
                z = np.random.randint(self.n_topics)
                
                
                xi = np.random.randint(item_tags[m,:].sum())
                xu = np.random.randint(user_tags[m,:].sum())
                
                ii = item_tags[m,:].nonzero()[0][xi]
                uu = user_tags[m,:].nonzero()[0][xu]
                
                self.niuz[ii,uu,z] += 1
                self.niu[ii,uu] += 1
                    
                
                self.nzw[z,w] += 1
                self.nz[z] += 1
                
                self.author_topic[(m,i)] = (ii,uu,z)

    def _conditional_distribution(self, its, uts, w):
        """
        Conditional distribution (matrix of size n_topics x n_authors).
        Sample distribution of latent variable z
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size)
        right = 0
        
        ts = np.zeros((its.shape[0],uts.shape[0],self.n_topics))
        ts_sum = np.zeros((its.shape[0],uts.shape[0]))
        for i,ii in enumerate(its):
            for j,uu in enumerate(uts):
                ts[i,j,:] = self.niuz[ii,uu]
                ts_sum[i,j] = self.niu[ii,uu]
        right = (ts + self.alpha) / \
                (ts_sum[:,:,np.newaxis] + self.alpha * self.n_topics)
                
                
        p_iuz = left[np.newaxis,np.newaxis,:] * right
        # normalize to obtain probabilities
        p_iuz /= np.sum(p_iuz)
        return p_iuz

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for ii in xrange(self.n_itags):
            for uu in xrange(self.n_utags):
                lik += log_multi_beta(self.niuz[ii,uu,:]+self.alpha)
                lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def theta(self):
        """
        Compute theta = p(z|x).
        """
        num = self.niuz + self.alpha
        num /= np.sum(num, axis=2)[:, :, np.newaxis]
        return num

    def fit(self, reviews, item_tags, user_tags, maxiter=30, verbose=False):
        n_docs, vocab_size = reviews.shape
        self.n_itags = n_itags = item_tags.shape[1]
        self.n_utags = n_utags = user_tags.shape[1]
        
        if verbose:
            print("Sampler initializing.\n")
        self._initialize(reviews,item_tags,user_tags)
        ll0 = self.loglikelihood()
        if verbose:
            print("Initial likekihood: " + str(ll0) + "\n")        
        
        liks = [ll0]
        for it in xrange(maxiter):
            if verbose:
                print("Iteration %i" %it)
            
                
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(reviews[m, :])):
                    
                    ii,uu,z = self.author_topic[(m,i)]
                    
                    self.niuz[ii,uu,z] -= 1
                    self.niu[ii,uu] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1
                    
                    its, uts = item_tags[m].nonzero()[0],user_tags[m].nonzero()[0]
                    p_iuz = self._conditional_distribution(its,uts, w)
                    shape = p_iuz.shape
                    sp = np.random.multinomial(1,p_iuz.flatten())
                    sp = sp.reshape(shape)
                    xi,xu,z = np.where(sp==1)
                    xi = xi[0]
                    xu = xu[0]
                    z = z[0]
                    ii = its[xi]
                    uu = uts[xu]
                    self.niuz[ii,uu,z] += 1
                    self.niu[ii,uu] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    
                    self.author_topic[(m,i)] = (ii,uu,z)

            # FIXME: burn-in and lag!
            # yield self.phi()
            lik = self.loglikelihood()
            liks.append(lik)
            if verbose:
                print("\tLog-likelihood: " + str(lik))
                      
        return liks
        
    def predict(self,item_tags, user_tags):
        
        thetas = self.theta()
        phis = self.phi()
        
        msg = "Dimension does not match."
        assert item_tags.shape[0] == user_tags.shape[0], msg
        assert item_tags.shape[1] == thetas.shape[0], msg
        assert user_tags.shape[1] == thetas.shape[1], msg
        
        IT = item_tags.astype(bool)
        UT = user_tags.astype(bool)
        
        for i in range(item_tags.shape[0]):
            tpcs = thetas[IT[i]][:,UT[i]].mean(0).mean(0)
            yield (tpcs[:,np.newaxis] * phis).mean(0)
        
def recoverSampler(filename):
    pack = json.load(open(filename))
    sp = LdaSampler(pack['n_topics'],pack['alpha'],pack['beta'])
    sp.nzw = np.array(pack['nzw'])
    sp.niuz = np.array(pack['niuz'])
    sp.n_utags = pack['n_utags']
    sp.n_itags = pack['n_itags']
    return sp
    
def evaluate(model, reviews, item_tags, user_tags):
    N,V = reviews.shape
    I = item_tags.shape[1]
    U = user_tags.shape[1]
    K = model.n_topics
    score = 0
    for r,proba in enumerate(model.predict(item_tags, user_tags)):
        score += reviews[r].dot( np.log(proba) )
    return {K:score/float(N)}

def highlight(model, dictionary, reviews, item_tags, user_tags):
    pass









