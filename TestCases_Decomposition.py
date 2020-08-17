
import numpy as np
from Decompositions import *
# testing the solve_equation method

def test_equation():
	A=np.array([[1,1],[0,1]])
	b=np.array([2,3])

	x=solve_equation(A,b)
	
	
	g=np.array([-1.0,3.0])
	
	print(np.allclose(x,g))
		

def test_triangular():
	for N in range(1,20):
		A=np.triu(np.random.normal(size=(N,N)))
		
		x=np.random.normal(size=(N,))
		
		b=A@x
		
		x2=solve_equation(A,b)
		
		print(np.allclose(x,x2,atol=1e-5))


def test_reflection():
	x=np.array([1,1,1])
	e=np.array([1,0,0])
	H=householder_reflection(x,e)

	print(np.allclose(H@x,np.sqrt(3)*e,atol=1e-5))
def test_qr():

	A=np.array([[2,1],[0,3],[4,5],[1,1]])
	Q,R=QR_decomposition(A)

	print(np.allclose(R[1:,0],np.zeros(A.shape[0]-1),atol=1e-5))
	print(np.allclose(R[2:,0],np.zeros(A.shape[0]-2),atol=1e-5))
	print(np.allclose(Q@R,A,atol=1e-5))

if __name__ =="__main__":

	test_equation()
	test_triangular()
	test_reflection()
	test_qr()