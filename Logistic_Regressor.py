

def LogisticClassifier:
	def __init__(self, rate=0.1,tol=1e-4,max_iter=1000):
		self.rate=rate
		self.tol=tol
		self.iter=max_iter

		self.intercept=True
		self.center=True
		self.scale=True

		self.hist=[]

	def matrix_design(self,X):
		if self.center:
			X=X-self.means
		if self.scale:
			X=X/self.standard_error

		if self.intercept:

			X=np.hstack([np.ones((X.shape[0],1)),X])

		return X

	def fit_center_scale(self,X):
		self.means=X.mean(axis=0)
		self.standard_error=np.std(X,axis=0)
	def sigmoid(z):
		return 1/(1+np.exp(-z))


	def fit(self,X,y):
		self.fit_center_scale(X)

		n,k=X.shape

		X=self.matrix_design(X)

		prev_loss=-float('inf')
		self.conv=False
		self.beta=np.zeros(k+(1 if self.add_intercept else 0))

		for i in range(self.iter):

			y_hat=sigmoid(X@self.beta)
			self.loss=np.mean(-y * np.log(y_hat)-(1-y)*np.log(1-y_hat))
			if(abs(prev_loss-self.loss)<self.tol):
				self.conv=True
				break
			else prev_loss=self.loss


			rem=(y_hat-y).reshape((n,1))
			grad=(X*rem).mean(axis=0)
			self.beta-=self.rate*grad

		self.iter=i+1

	def predict_prob(self,X):

		X=self.matrix_design(X)
		return sigmoid(X@self.beta)

	def predict(self,X):

		return (self.predict_prob(X)>0.5).astype(int)

