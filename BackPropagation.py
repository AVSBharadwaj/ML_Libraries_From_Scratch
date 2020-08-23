


def forward_prop(self, X):
	self.act=[]
	act=X

	for W, act_fun in self.layers:
		bias= np.ones((act_fun.shape[0],1))
		act=np.hstack([bias,act])
		self.act.append(act)
		act=act_fun(act@W.T)


	self.act.append(act)

def back_prop(self,y):

	N=y.shape[0]

	y_hat=self.act[-1]

	error=y_hat-y


	for layer in range(self.layers-2,-1,-1):

		a=self.act[layer]
		delta=(error.T@a)/N

		if layer!=self.layers-2:

			delta=delta[1:,:]

		W=self.layers[layer][0]

		if layer>0:

			if layer !=self.layers-2:
				error=error[:,1:]

			error=(error@W)*(a*(1-a))


		W-=self.rate*delta

