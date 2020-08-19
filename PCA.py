# PCA is a linear dimensionalty resuction technique 
# we are going to implement PCA from scratch here
# other ways for the same are Factor Analysis and Non-negative matrix decompositioin

''' 
PCA can be achieved by finding the Eigenvaectors and Eigen values of the given tranformation
In our case we will be solving the eigenvalue decomposition problem using the QR algorithm
In this algorithm the QR decomposition of the input matrix is found iteratively

'''

from EigenValue_Decomposition_QR_method inport *

class PCA:
	def __init__(self,n_comp=None,whiten=false):

		self.n_comp = n_comp
		self.whiten = whiten

	def fit(self,X):

		n,m=X.shape
		self.mu=X.mean(axis=0)
		X=X-self.mu

		if(self.whiten):
			self.std=X.std(axis=0)
			X=X/self.std

		C=X.T@X/(n-1)

		self.evalues,self.evec=eigen_decomposition(C)

		if self.n_comp is not None:

			self.evalues=self.evalues[0:self.n_comp]
			self.evec=self.evec[:,0:self.n_comp]

		order=np.flip(np.argsort(self.evalues))
		self.evalues=self.evalues[order]
		self.evec=self.evec[:,order]

		return self
		
