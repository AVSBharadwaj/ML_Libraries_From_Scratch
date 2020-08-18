# implement the Linear Regression method 

'''
The most general representation of the linear regression problem is:
		

		Y = M * X

		which can be re written as:

		Y = M * (QR)   // after QR decomposition of X

		Q.T * Y = M * R

		which can be solved easily as R is upper triangular

'''
from Decompositions import *



class LinearRegression:
	def __init__ (self,intercept=True):
		self.intercept = intercept

	def add_intercept(self,X):
		if(self.intercept):
			X=np.hstack([np.ones((X.shape[0],1)),X])
		return X

	def fit(self,X,Y):

		X=self.add_intercept(X)
		Q,R= QR_decomposition(X)

		self.M=solve_equation(R,Q.T@Y)

	def predict(self,X):

		X=self.add_intercept(X)

		return X@self.M



