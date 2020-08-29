
import numpy as np

def sigmoid(z):
	return 1 / ( 1 + np.exp(-z) )

class NeuralNetwork:
	def __init__(self,n_hidden=(100,),learning_rate=1.0,max_iter=10,threshold=0.5):
		self.n_hidden = n_hidden
		self.n_layers = len(n_hidden) + 2
		self.learning_rate = float(learning_rate)
		self.max_iter = int(max_iter)
		self.threshold = float(threshold)

	def _random_initialization(self):

		if not self.n_hidden:
			layer_sizes = [ (self.n_output, self.n_input+1) ]
		else:
			layer_sizes = [ (self.n_hidden[0], self.n_input+1) ]
			previous_size = self.n_hidden[0]

			for size in self.n_hidden[1:]:
				layer_sizes.append( (size, previous_size+1) )
				previous_size = size

			layer_sizes.append( (self.n_output, previous_size+1) )
		
		self.layers = [
			[np.random.normal(0, 0.1, size=layer_size), sigmoid]
			for layer_size in layer_sizes
		]

	def fit(self, X, y):
			self.n_input = X.shape[1]        
			self.n_output = 1
			y = np.atleast_2d(y).T
			self._random_initialization()
			for iteration in range(self.max_iter):
				self.forward_prop(X)
				self.back_prop(y)
	def predict(self, X):
			y_class_probabilities = self.predict_proba(X)
			return np.where(y_class_probabilities[:,0] < self.threshold, 0, 1)
	def predict_proba(self, X):
			self.forward_prop(X)
			return self.act[-1]


	def forward_prop(self, X):
		self.act=[]
		act=X

		for W, act_fun in self.layers:
			bias= np.ones((act.shape[0],1))
			act=np.hstack([bias,act])
			self.act.append(act)
			act=act_fun(act@W.T)


		self.act.append(act)

	def back_prop(self,y):

		N=y.shape[0]

		y_hat=self.act[-1]

		error=y_hat-y


		for layer in range(self.n_layers-2,-1,-1):

			a=self.act[layer]
			delta=(error.T@a)/N

			if layer!=self.n_layers-2:

				delta=delta[1:,:]

			W=self.layers[layer][0]

			if layer>0:

				if layer !=self.n_layers-2:
					error=error[:,1:]

				error=(error@W)*(a*(1-a))


			W-=self.learning_rate*delta

