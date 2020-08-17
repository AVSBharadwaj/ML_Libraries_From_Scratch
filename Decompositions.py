
import numpy as np

'''
below is an implmentation of the Householder_reflection tranformation

it takes the input vector a and return the tranformation H that would take a to 
its mirror image point e

'''


def householder_reflection(a,e):
	'''
	return a orthogonal matrix that maps a onto e
	'''
	'''
	u is the vector joining a and its reflection on e
	u=a-||a||e
	'''

	u=a-np.sign(a[0])*np.linalg.norm(a)*e
	

	'''
	v is the normalization of u
	'''

	v=u/np.linalg.norm(u)


	'''
	H is the final householder matrix tranformation 

	Hx=x-2<x,v>v
	  =x-2v<x,v>
	  =x-2v(v.T*x)
	  =x-2(v*v.T)x
	  =[I-2(v*v.T)]x

	hence H=I-2(v*v.T)x
	'''

	H=np.eye(len(a))-2*np.outer(v,v)
	
	return H



'''

Below is the function for QR decomposition 
It takes a matrix A and return Q and R where Q is orthogonal and R is a upper triangular
matrix and
				QR=A

'''

def QR_decomposition(A):

	N,M=A.shape

	'''
	initialise Q as a Identity matrix and R as a upper triangular matrix
	'''

	Q=np.eye(N)
	R=A.copy()

	'''
	iteratively covert R and Q into the desired form while still maintaining
	their original structure
	'''

	for i 	in range(M-int(N==M)):


		r=R[i:,i]

		'''
		if its already in unit form 
		'''
		if np.allclose(r[1:],0):
			continue

		'''
		e is our ouput basis vector of the form (1,0,...,0)
		'''
		e=np.zeros(N-i)
		e[0]=1

		'''
		H is the Householder_reflection matrix we would get from r and e
		'''
		H=np.eye(N)
		H[i:,i:]=householder_reflection(r,e)

		Q=Q @ H.T
		R=H @ R

	return Q,R


# Solve Ax=B using Back-substitution method where A is a upper triangular matrix
# there is also a scipy library function calles scipy.linalg.solve_triangular() which 
# does the same thing bit faster 
def solve_equation(A,b):

	N,M = A.shape

	'''
	last x is just A[n,n]*x=B[n]
	'''
	x=b[(M-1):M] / A[M-1,M-1]

	for i in range(M-2,-1,-1):

		back=np.dot(A[i,(i+1):],x)
		rhs=b[i]-back

		x_i=rhs/A[i,i]
		x=np.insert(x,0,x_i)

	return x




