
import numpy as np

def best_split_point(X,y,column):
	ordering= np.argsort(X[:,column])

	classes= y[ordering]

	class_0_below=(classes==0).cumsum()
	class_1_below=(classes==0).cumsum()

	class_0_above=class_0_below[-1]-class_0_below
	class_1_above=class_1_below[-1]-class_1_below

	below_total=np.arange(1,len(y)+1)
	above_total=np.arange(len(y)-1,-1,-1)

	gini=class_1_below*class_0_below /( below_total**2)+class_1_above*class_0_above /( above_total**2)
	gini[np.isnan(gini)]=1

	best_split_rank=np.argmin(gini)

	best_split_gini=gini[best_split_rank]

	best_split_index=np.argwhere(ordering==best_split_rank).item(0)

	best_split_value=X[best_split_index,column]

	return best_split_gini,best_split_value,column

class Node:
	def __init__(self,X,y):

		self.X=X
		self.y=y

		self.is_leaf=True
		self.column=None
		self.split_point=None

		self.child=None

	def is_pure(self):

		p=self.probabilities()
		if p[0]==1 or p[1]==1:
			return True

		return False

	def split(self,depth=0):

		X,y=self.X,self.y

		if self.is_leaf and not self.is_pure():

			splits=[best_split_point(X,y,column) for column in range(X.shape[1])]
			splits.sort()
			gini, split_point,column=splits[0]
			self.is_leaf =False
			self.column = column
			self.split_point =split_point

			below=X[:,column]<=split_point
			above=X[:,column]>split_point

			self.child=[Node(X[below],y[below]),Node(X[above],y[above])]
			if depth:
				for child in self.child:
					child.split(depth-1)


	def probabilities(self):
		return np.array([np.mean(self.y==0),np.mean(self.y==1),])
	
	def predict_proba(self,row):

		if self.is_leaf:
			return self.probabilities()

		else :
			if row[self.column]<=self.split_point:
				return self.child[0].predict_proba(row)

			else:
				return self.child[1].predict_proba(row)


class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = int(max_depth)
        self.root = None
        
    def fit(self, X, y):
        self.root = Node(X, y)
        self.root.split(self.max_depth)
        
    def predict_proba(self, X):
        results = []
        for row in X:
            p = self.root.predict_proba(row)
            results += [p]
        return np.array(results)
            
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
