import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self, samples, N=10, k=5):
        ### k is the number of the neighbors
        ### 生成  n倍的数据
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0

    # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N = int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        # print ('neighbors\n',neighbors)
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]
            # print nnarray
            self._populate(N, i, nnarray)
        return self.synthetic

    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap * dif
            self.newindex += 1


