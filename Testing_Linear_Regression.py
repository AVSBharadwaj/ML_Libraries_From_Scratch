

# testing out LinearRegression Method
from Linear_Regression import *

from sklearn.datasets import load_boston

import matplotlib
from matplotlib import pyplot as plt

#%matplotlib inline
import numpy as np
from numpy.linalg import det 
from scipy.stats import ortho_group 
from numpy.testing import assert_allclose

boston=load_boston()

X_raw=boston.data
Y_raw=boston.target

shuffle=np.random.permutation(len(Y_raw))
X_full=X_raw[shuffle].copy()
Y_full=Y_raw[shuffle].copy()

split=int(0.8*len(Y_full))
X_train=X_full[:split,:]
Y_train=Y_full[:split]
X_test=X_full[split:,:]
Y_test=Y_full[split:]


model=LinearRegression()
model.fit(X_train,Y_train)



def GOF_report(label,model,X,Y):
	Y_hat=model.predict(X)

	plt.scatter(x=Y,y=Y_hat,label=label,alpha=0.5,color='red')
	plt.title('Predicted vs Actual')
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.legend(loc='upper left')


	mse=np.mean((Y-Y_hat)**2)
	y_bar = np.mean(Y)
	r2 = 1 - np.sum( (Y-Y_hat)**2 ) / np.sum( (Y-y_bar)**2 )
	print("{label: <16} mse={mse:.2f}     r2={r2:.2f}".format(**locals()))

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
GOF_report("Training Set",model,X_train,Y_train)
plt.subplot(1,2,2)
GOF_report("Test Set",model,X_test,Y_test)
plt.show()
