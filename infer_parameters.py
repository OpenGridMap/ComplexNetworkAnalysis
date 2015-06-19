import numpy as np

def infer_degree_sequence(n):
	S = degree_sequences_to_matrix(get_example_degree_sequences()) # matrix of size dmax^2
	w = np.sum(S,0) # number of nodes for each example degree sequence
	print(w)
	
	distances = np.matrix([[1,2]])
	print(distances)
	w = np.multiply(distances,w)
	#w = np.transpose(w)*distances
	print(w)
	
	""" needs to be debugged """
	norm = np.sum(w)
	print(norm)
	w /= float(norm)
	print(w)
	
	
	return
	
def get_example_degree_sequences():
	# list of degree sequences
	l = [[0,1,2,6],[3,4]]
	return l

def degree_sequences_to_matrix(l):
	dmax = 0
	for e in l:
		dmax = max(dmax, len(e))
	for e in l:
		e += [0]*(dmax-len(e))
	S = np.transpose(np.matrix(l))
	print(S)
	return S

infer_degree_sequence(0)
	
def infer_assortativity_matrix(n):
	return
	
def get_example_assortativity_matrices():
	return