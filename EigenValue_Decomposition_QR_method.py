
'''

The below function calculates and returns the eigenvalues and eigenvectors of the
given matrix
It uses the QR algorithm for the same 
'''



def eigen_decomposition(A,iter=100):
	A_k=A
	Q_k=np.eye(A.shape[1])

	for k in range(iter):

		Q,R=QR_decomposition(A_k)

		Q_k=Q_k@Q
		A_k=R @ Q


	evalues=np.diag(A_k)
	evec=Q_k

	return evalues,evec


