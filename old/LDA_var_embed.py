import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LDA_gibbs_var:
    def __init__(self, docs, vocab, n_topic, random_state=0):
        self.V = len(vocab)  # size of vocab
        self.k = n_topic  # number of topics
        self.N = np.array([doc.shape[0] for doc in docs])  # num words in each doc
        self.M = len(docs)  # num of docs
        self.docs = docs
        self.vocab = vocab

        np.random.seed(random_state)
        self.alpha = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
        self.eta = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V

        self.n_iw = np.zeros((self.k, self.V), dtype=int)
        self.n_di = np.zeros((self.M, self.k), dtype=int)

        self.assign = None
    #
    def _init_gibbs(self, n_gibbs=2000):
        N_max = max(self.N)
        self.assign = np.zeros((self.M, N_max, n_gibbs + 1), dtype=int)
        
        for d in range(self.M):
            for n in range(self.N[d]):
                w_dn = self.docs[d][n]
                self.assign[d, n, 0] = np.random.randint(self.k)
                i = self.assign[d, n, 0]
                self.n_iw[i, w_dn] += 1
                self.n_di[d, i] += 1
    #
    def _conditional_prob(self, w_dn, d):
        prob = np.empty(self.k)
        for i in range(self.k):
            _1 = (self.n_iw[i, w_dn] + self.eta) / (self.n_iw[i, :].sum() + self.V * self.eta)
            _2 = (self.n_di[d, i] + self.alpha) / (self.n_di[d, :].sum() + self.k * self.alpha)
            prob[i] = _1 * _2
        return prob / prob.sum()
    #
    def _calculate_beta(self):
        beta = np.empty((self.k, self.V))
        for j in range(self.V):
            for i in range(self.k):
                beta[i, j] = (self.n_iw[i, j] + self.eta) / (self.n_iw[i, :].sum() + self.V * self.eta)
        return beta
    #
    def _calculate_theta(self):
        theta = np.empty((self.M, self.k))
        for d in range(self.M):
            for i in range(self.k):
                theta[d, i] = (self.n_di[d, i] + self.alpha) / (self.n_di[d, :].sum() + self.k * self.alpha)
        return theta
    #
    def run_gibbs(self, n_gibbs=2000,burn_in=500, sample_interval=10):
        self._init_gibbs(n_gibbs)
        beta_samples = []
        theta_samples = []

        print(f"V: {self.V}\nk: {self.k}\nN: {self.N[:10]}...\nM: {self.M}")
        print(f"alpha: {self.alpha}\n_eta: {self.eta}")
        print(f"n_iw: dim {self.n_iw.shape}\nn_di: dim {self.n_di.shape}")
        print(f"assign: dim {self.assign.shape}")
        print("\n", "="*10, "START SAMPLER", "="*10)
        for t in range(n_gibbs):
            for d in range(self.M):
                for n in range(self.N[d]):
                    w_dn = self.docs[d][n]
                    i_t = self.assign[d, n, t]
                    self.n_iw[i_t, w_dn] -= 1
                    self.n_di[d, i_t] -= 1
                    prob = self._conditional_prob(w_dn, d)
                    i_tp1 = np.argmax(np.random.multinomial(1, prob))
                    self.n_iw[i_tp1, w_dn] += 1
                    self.n_di[d, i_tp1] += 1
                    self.assign[d, n, t + 1] = i_tp1

            if t > burn_in and (t - burn_in) % sample_interval == 0:
                beta_samples.append(self._calculate_beta())
                theta_samples.append(self._calculate_theta())
            
            if ((t + 1) % 50 == 0):
                print(f"Sampled {t + 1}/{n_gibbs}")
        return beta_samples, theta_samples
    #          
    def sample(self):
        beta = np.empty((self.k,self.V))
        theta = np.empty((self.M, self.k))

        for j in range(self.V):
            for i in range(self.k):
                beta[i, j] = (self.n_iw[i, j] + self.eta) / (self.n_iw[i, :].sum() + self.V*self.eta)

        for d in range(self.M):
            for i in range(self.k):
                theta[d, i] = (self.n_di[d, i] + self.alpha) / (self.n_di[d, :].sum() + self.k*self.alpha)
        return beta
    #
#
class LDA_word_embed_v:
    def __init__(self, docs, vocab, n_topic, word_embeddings, random_state=0):
        self.V = len(vocab)  # size of vocab
        self.k = n_topic  # number of topics
        self.N = np.array([doc.shape[0] for doc in docs])  # num words in each doc
        self.M = len(docs)  # num of docs
        self.docs = docs
        self.vocab = vocab

        np.random.seed(random_state)
        self.alpha = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
        self.eta = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V

        self.n_iw = np.zeros((self.k, self.V), dtype=int)
        self.n_di = np.zeros((self.M, self.k), dtype=int)

        self.assign = None

        # word embeddings
        self.word_embeddings = word_embeddings # array or list of embedding vectors
        self.topic_embeddings = np.zeros((self.k, len(word_embeddings[0])))
    #
    def _update_topic_embeddings(self):
        for i in range(self.k):
            top_words_index = self.n_iw[i,:].argsort()[-10:]
            # each topic has an embedding vector that is the mean of the embeddings of the 
            # top ten words in the topic, updated each iteration of gibbs sampling
            self.topic_embeddings[i] = np.mean(
                [self.word_embeddings[word_idx] for word_idx in top_words_index], axis=0)
    #
    def _init_gibbs(self, n_gibbs=2000):
        N_max = max(self.N)
        self.assign = np.zeros((self.M, N_max, n_gibbs + 1), dtype=int)
        
        for d in range(self.M):
            for n in range(self.N[d]):
                w_dn = self.docs[d][n]
                self.assign[d, n, 0] = np.random.randint(self.k)
                i = self.assign[d, n, 0]
                self.n_iw[i, w_dn] += 1
                self.n_di[d, i] += 1
    #
    def _conditional_prob(self, w_dn, d):
        word_embedding = self.word_embeddings[w_dn]
        prob = np.empty(self.k)
        for i in range(self.k):
            topic_embedding = self.topic_embeddings[i]
            embedding_similarity = max(0, cosine_similarity([word_embedding], [topic_embedding])[0, 0])

            _1 = (self.n_iw[i, w_dn] + self.eta) / (self.n_iw[i, :].sum() + self.V * self.eta)
            _2 = (self.n_di[d, i] + self.alpha) / (self.n_di[d, :].sum() + self.k * self.alpha)
            if embedding_similarity != 0:
                prob[i] = _1 * _2 * embedding_similarity
            else:
                prob[i] = _1 * _2
        return prob / prob.sum()
    #
    def run_gibbs(self, n_gibbs=2000,burn_in=500, sample_interval=10):
        self._init_gibbs(n_gibbs)
        beta_samples = []
        theta_samples = []

        print(f"V: {self.V}\nk: {self.k}\nN: {self.N[:10]}...\nM: {self.M}")
        print(f"alpha: {self.alpha}\n_eta: {self.eta}")
        print(f"n_iw: dim {self.n_iw.shape}\nn_di: dim {self.n_di.shape}")
        print(f"assign: dim {self.assign.shape}")
        print("\n", "="*10, "START SAMPLER", "="*10)
        for t in range(n_gibbs):
            self._update_topic_embeddings()
            for d in range(self.M):
                for n in range(self.N[d]):
                    w_dn = self.docs[d][n]
                    i_t = self.assign[d, n, t]
                    self.n_iw[i_t, w_dn] -= 1
                    self.n_di[d, i_t] -= 1
                    prob = self._conditional_prob(w_dn, d)
                    i_tp1 = np.argmax(np.random.multinomial(1, prob))
                    self.n_iw[i_tp1, w_dn] += 1
                    self.n_di[d, i_tp1] += 1
                    self.assign[d, n, t + 1] = i_tp1
            if t > burn_in and (t - burn_in) % sample_interval == 0:
                beta_samples.append(self._calculate_beta())
                theta_samples.append(self._calculate_theta())
            
            if ((t + 1) % 50 == 0):
                print(f"Sampled {t + 1}/{n_gibbs}")
        return beta_samples, theta_samples
    #          
    def sample(self):
        beta = np.empty((self.k,self.V))
        theta = np.empty((self.M, self.k))

        for j in range(self.V):
            for i in range(self.k):
                beta[i, j] = (self.n_iw[i, j] + self.eta) / (self.n_iw[i, :].sum() + self.V*self.eta)

        for d in range(self.M):
            for i in range(self.k):
                theta[d, i] = (self.n_di[d, i] + self.alpha) / (self.n_di[d, :].sum() + self.k*self.alpha)
        return beta,theta
    #
#