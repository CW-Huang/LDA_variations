
"""

Implementation of Gibbs sampler for Author topic model 
    / by Michal Rosen-Zvi et al.
Adapted from 
    / https://github.com/fannix/lda-cython/blob/master/lda_gibbs_cython.pyx
    

@author: Chin-Wei
"""


import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.int
ctypedef np.int_t DTYPE_t
from scipy.special import gammaln
from scipy.sparse.csr import csr_matrix
import gc



cdef int sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

cdef word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    if not isinstance(vec, csr_matrix):
        vec = csr_matrix(vec)
    li = []

    cdef int i,r
    
    for i in range(vec.indptr[1]):
        for r in range(int(vec.data[i])):
            li.append(vec.indices[i])
    return li
    

cdef log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

                    
                    
class ATMSampler(object):

    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _conditional_distribution(self, int w, 
                                   list its,
                                   list uts,
                                   np.ndarray[DTYPE_t, ndim=2] nzw,
                                   np.ndarray[DTYPE_t, ndim=3] niuz,
                                   np.ndarray[DTYPE_t, ndim=1] nz,
                                   np.ndarray[DTYPE_t, ndim=2] niu):
        """
        Conditional distribution (vector of size n_topics).
                Sample distribution of latent variable z
        """
        cdef int vocab_size = nzw.shape[1]
        cdef int z, n_topics, ii, uu, i, j, k
        n_topics = self.n_topics
        cdef double beta = self.beta
        cdef double alpha = self.alpha
        cdef double left, right, lik
        cdef np.ndarray[np.double_t, ndim=3] p_iuz
        cdef double p_iuz_sum  = 0
        

        #calculate likelihood
        p_iuz = np.zeros((len(its),len(uts),n_topics))
        for k in range(n_topics):
            left = (nzw[k,w] + beta) / \
                   (nz[k] + beta * vocab_size)
            for i,ii in enumerate(its):
                for j,uu in enumerate(uts):
                    right = (niuz[ii,uu,k] + alpha) / \
                            (niu[ii,uu] + alpha * n_topics)
                    lik = left * right
                    p_iuz[i,j,k] = lik
                    p_iuz_sum += lik
               
        #solve conversion rounding error
        cdef double partial_sum = 0
        for i in range(len(its)):
            for j in range(len(uts)):
                for k in range(n_topics):
                    p_iuz[i,j,k] /= p_iuz_sum
                    partial_sum += p_iuz[i,j,k]
        
        partial_sum -= p_iuz[i,j,k]
        p_iuz[i,j,k] = 1.0 - partial_sum
        
        return p_iuz

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        cdef int vocab_size = self.nzw.shape[1]
        cdef double lik = 0
        
        cdef int z, ii, uu
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
        cdef np.ndarray[np.double_t, ndim = 2] num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def theta(self):
        """
        Compute theta = p(z|x).
        """
        cdef np.ndarray[np.double_t, ndim = 3] num = self.niuz + self.alpha
        num /= np.sum(num, axis=2)[:, :, np.newaxis]
        return num
    
    def predict(self,
                np.ndarray[np.int_t, ndim=2] item_tags,
                np.ndarray[np.int_t, ndim=2] user_tags):
        
        cdef np.ndarray[np.double_t, ndim = 3] thetas = self.theta()
        cdef np.ndarray[np.double_t, ndim = 2] phis = self.phi()
        
        cdef str msg = "Dimension does not match."
        assert item_tags.shape[0] == user_tags.shape[0], msg
        assert item_tags.shape[1] == thetas.shape[0], msg
        assert user_tags.shape[1] == thetas.shape[1], msg
        
        cdef int i,n,v
        n = item_tags.shape[0]
        v = phis.shape[1]
        
        IT = item_tags.astype(bool)
        UT = user_tags.astype(bool)
        
        cdef np.ndarray[np.double_t, ndim=2] pred_proba = np.ones((n,v))
        for i in range(item_tags.shape[0]):
            tpcs = thetas[IT[i]][:,UT[i]].mean(0).mean(0)
            pred_proba[i,:] = (tpcs[:,np.newaxis] * phis).mean(0)
            
        return pred_proba
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, reviews,
            list item_tags,
            list user_tags,
            int n_itags,
            int n_utags,
            int maxiter=30,
            int verbose=False):
        """
        Run the Gibbs sampler.
        """
        if not isinstance(reviews, csr_matrix):
            reviews = csr_matrix(reviews)
            
            
        cdef int n_docs, vocab_size
        
        n_docs = reviews.shape[0]
        vocab_size = reviews.shape[1]
        
        self.n_itags = n_itags
        self.n_utags = n_utags
        
        if verbose:
            print("Sampler initializing.\n")
        
        # initialize
        cdef int it, m, i, w, z, ii, uu
        cdef double ll0,lik
        ## number of times tagging i-u pair and topic z co-occur
        cdef np.ndarray[DTYPE_t, ndim=3] niuz = np.zeros((n_itags, n_utags, self.n_topics), dtype=DTYPE)
        ## number of times topic z and word w co-occur
        cdef np.ndarray[DTYPE_t, ndim=2] nzw = np.zeros((self.n_topics, vocab_size), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] niu = np.zeros((n_itags, n_utags), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] nz = np.zeros(self.n_topics, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] its
        cdef np.ndarray[DTYPE_t, ndim=1] uts
        cdef list iis
        cdef list uus
        cdef int itsum, utsum, xi, xu
        
        self.author_topic = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
        
            iis = item_tags[m]      
            uus = user_tags[m]
            
            itsum = len(iis)
            utsum = len(uus)
                
            for i, w in enumerate(word_indices(reviews[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                
                
                xi = np.random.randint(itsum)
                xu = np.random.randint(utsum)
                
                ii = iis[xi]
                uu = uus[xu]
                
                niuz[ii,uu,z] += 1
                niu[ii,uu] += 1
                nzw[z,w] += 1
                nz[z] += 1
                self.author_topic[(m,i)] = (ii,uu,z)
            
            if m % 5000 == 0:
                gc.collect()
        self.nzw = nzw
        self.niuz = niuz
        
        ll0 = self.loglikelihood()
        if verbose:
            print("Initial likekihood: " + str(ll0) + "\n")      
                
        liks = [ll0]
        cdef np.ndarray[np.double_t, ndim=3] p_iuz, sp_back
        cdef int qq,rr,ss, count, length
        cdef np.ndarray[np.double_t, ndim=1] transform
        
        for it in range(maxiter):
            if verbose:
                print("Iteration %i" %it)
                
            for m in range(n_docs):
                iis = item_tags[m]     
                uus = user_tags[m]                    
                for i, w in enumerate(word_indices(reviews[m, :])):
                    ii,uu,z = self.author_topic[(m,i)]
                                        
                    niuz[ii,uu,z] -= 1
                    niu[ii,uu] -= 1
                    nzw[z,w] -= 1
                    nz[z] -= 1
                                       
                    p_iuz = self._conditional_distribution(w,iis,uus,nzw,niuz,nz,niu)

                    length = len(iis) * len(uus) * self.n_topics
                        
                    transform = np.zeros(length)
                    count = 0
                    for qq in range(len(iis)):
                        for rr in range(len(uus)):
                            for ss in range(self.n_topics):
                                transform[count] = p_iuz[qq,rr,ss]
                                count+=1
                    sp = np.random.multinomial(1,transform)                    
                    xi,xu,z = self._get_sample_index(len(iis),
                                                     len(uus),
                                                     self.n_topics,
                                                     sp)
                    ii = iis[xi]
                    uu = uus[xu]
                    
                    niuz[ii,uu,z] += 1
                    niu[ii,uu] += 1
                    nzw[z,w] += 1
                    nz[z] += 1
                    
                    self.author_topic[(m,i)] = (ii,uu,z)
                    
                if m % 5000 == 0:
                    gc.collect()
                    
            self.nzw = nzw
            self.niuz = niuz

            # FIXME: burn-in and lag!
            lik = self.loglikelihood()
            liks.append(lik)
            if verbose:
                print("\tLog-likelihood: " + str(lik))
            
            gc.collect()

        return liks
    
    def _get_sample_index(self,
                          int I,
                          int J,
                          int K,
                          np.ndarray[DTYPE_t, ndim=1] sp):
        cdef int qq,rr,ss, count
        count = 0
        for qq in range(I):
            for rr in range(J):
                for ss in range(K):
                    if sp[count] == 1:
                        return (qq,rr,ss)
                    count+=1
