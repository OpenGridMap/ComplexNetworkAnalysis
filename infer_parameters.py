import numpy as np
import measures_on_simple_data as msd
import networkx as nx
import aux_graphical as aux
from random import choice
from network_utils import degreehisto_to_degreeseq


# list of example feeders
examples_list = [13, 34, 37, 123, 124, 125, 126, 127, 128]
examples_nbnodes_list = [13, 34, 37, 123, 32, 32, 19, 16, 16]


def infer_degree_sequence(n):
	S = degree_distrib_to_max_size(get_example_degree_distribs())  # matrix of size dmax*nb_examples
	nb = np.sum(S, 0)  # number of nodes for each example degree sequence
	
	# If there is an example with same nb of nodes, return its degree histogram
	nblist = np.asarray(np.transpose(nb))
	
	i = 0
	for e in nblist:
		if e == n:
			histo = np.asarray(np.transpose(S[:, i]))[0]
			print(histo)
			return histo
		i += 1
		
	# Elif n greater than max or lower than min of all known examples, return empty
	if n > max(nblist) or n < min(nblist):
		print("Cannot infer degree sequence : " + str(n) + " not in range of known examples.")
		return []
	
	# Else, find two degree sequences examples with nearest number of nodes
	n_upper = n * 10000
	n_lower = 0
	h_upper = []
	h_lower = []
	i = 0
	for e in nblist:
		if (e > n) and (e < n_upper):
			n_upper = e
			histo = np.asarray(np.transpose(S[:, i]))[0]
			h_upper = histo
		elif (e < n) and (e > n_lower):
			n_lower = e
			histo = np.asarray(np.transpose(S[:, i]))[0]
			h_lower = histo
		i += 1
	n_lower = n_lower.item(0)
	n_upper = n_upper.item(0)
	seq_lower = degreehisto_to_degreeseq(h_lower)
	seq_upper = degreehisto_to_degreeseq(h_upper)
	print(n_lower)
	print(seq_lower)
	print(n_upper)
	print(seq_upper)
	
	# prepare my_seq
	my_seq = seq_lower
	my_seq.extend([0] * (n - n_lower))
	my_seq.sort()
	
	### weird!!!! 
# 	print("weird")
# 	print(my_seq)
# 	print(nx.is_graphical(my_seq))
# 	print(aux.is_valid_degree_sequence_erdos_gallai(my_seq))
# 	s1 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
# 	print(s1)
# 	print(nx.is_graphical(s1))
	
	# calculate weighted average number of edges
	w_lower = 1/float(n - n_lower)
	w_upper = 1/float(n_upper - n)
	s = w_lower + w_upper
	w_lower /= s
	w_upper /= s
	double_edges = round(sum(seq_lower)*w_lower + sum(seq_upper)*w_upper)
	# must be even
	double_edges += double_edges%2

	# loop until reaching desired number of edges
	while sum(my_seq) < double_edges:
		# add degrees, keeping the sequence graphical
		i = 0
		while my_seq[i] >= seq_upper[i]:
			i += 1
		my_seq[i] += 1
		j = -1
		while my_seq[j] >= seq_upper[j]:
			j -= 1
		my_seq[j] += 1
		my_seq.sort()
		# my_seq should be graphical at every loop

	print(my_seq)
	print(nx.is_graphical(my_seq))
	print(nx.is_valid_degree_sequence(my_seq))
	print(aux.is_valid_degree_sequence_erdos_gallai(my_seq))
	return my_seq

def infer_degree_sequence_random(n):
	S = degree_distrib_to_max_size(get_example_degree_distribs())  # matrix of size dmax*nb_examples
	nb = np.sum(S, 0)  # number of nodes for each example degree sequence
	
	# If there is an example with same nb of nodes, return its degree histogram
	nblist = np.asarray(np.transpose(nb))
	
	i = 0
	for e in nblist:
		if e == n:
			histo = np.asarray(np.transpose(S[:, i]))[0]
			print(histo)
			return histo
		i += 1
		
	# Elif n greater than max or lower than min of all known examples, return empty
	if n > max(nblist) or n < min(nblist):
		print("Cannot infer degree sequence : " + str(n) + " not in range of known examples.")
		return []
	
	# Else, find two degree sequences examples with nearest number of nodes
	n_upper = n * 10000
	n_lower = 0
	h_upper = []
	h_lower = []
	i = 0
	for e in nblist:
		if (e > n) and (e < n_upper):
			n_upper = e
			histo = np.asarray(np.transpose(S[:, i]))[0]
			h_upper = histo
		elif (e < n) and (e > n_lower):
			n_lower = e
			histo = np.asarray(np.transpose(S[:, i]))[0]
			h_lower = histo
		i += 1
	n_lower = n_lower.item(0)
	n_upper = n_upper.item(0)
	seq_lower = degreehisto_to_degreeseq(h_lower)
	seq_upper = degreehisto_to_degreeseq(h_upper)
	print(n_lower)
	print(seq_lower)
	print(n_upper)
	print(seq_upper)
	
	# prepare my_seq
	my_seq = seq_lower
	my_seq.extend([0] * (n - n_lower))
	my_seq.sort()
	
	### weird!!!! 
# 	print("weird")
# 	print(my_seq)
# 	print(nx.is_graphical(my_seq))
# 	print(aux.is_valid_degree_sequence_erdos_gallai(my_seq))
# 	s1 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
# 	print(s1)
# 	print(nx.is_graphical(s1))
	
	# calculate weighted average number of edges
	w_lower = 1/float(n - n_lower)
	w_upper = 1/float(n_upper - n)
	s = w_lower + w_upper
	w_lower /= s
	w_upper /= s
	double_edges = round(sum(seq_lower)*w_lower + sum(seq_upper)*w_upper)
	# must be even
	double_edges += double_edges%2
	
	# loop until reaching desired number of edges
	while sum(my_seq) < double_edges:
		# add degrees, keeping the sequence graphical
		if 0 in my_seq:
			my_seq[0] = 1
		else:
			k = choice(seq_upper)
			while (k-1) not in my_seq:
				k = choice(seq_upper)
			i = 0
			while my_seq[i] != k-1:
				i += 1
			my_seq[i] = k
		# add an other degree, to keep the sequence graphical
		k = choice(seq_upper)
		while (k-1) not in my_seq:
			k = choice(seq_upper)
		i = 0
		while my_seq[i] != k-1:
			i += 1
		my_seq[i] = k
		
		my_seq.sort()
		# my_seq should be graphical at every loop

	print(my_seq)
	print(nx.is_graphical(my_seq))
	print(nx.is_valid_degree_sequence(my_seq))
	print(aux.is_valid_degree_sequence_erdos_gallai(my_seq))
	return my_seq

		
def get_example_degree_distribs():
	l = []
	for k in examples_list:
		G = msd.readFeederData(k)
		l += [nx.degree_histogram(G)]
	return l
	
def get_example_assortativity_matrices():
	l = []
	for k in examples_list:
		G = msd.readFeederData(k)
		l += [nx.degree_mixing_matrix(G)]
	return l

def get_example_assortativity_coeffs():
	l = []
	for k in examples_list:
		G = msd.readFeederData(k)
		l += [nx.degree_assortativity_coefficient(G)]
	return l

# fill with zeros to have same size in all vectors
def degree_distrib_to_max_size(l):
	dmax = 0
	for e in l:
		dmax = max(dmax, len(e))
	for e in l:
		e += [0] * (dmax - len(e))
	S = np.transpose(np.matrix(l))
	return S

# fill with zeros to have same size in all matrices
def assort_matrices_to_max_size(l):
	dmax = 0
	ll = []
	for e in l:
		dmax = max(dmax, len(e))

	for e in l:
		if dmax - len(e) > 0:
			for k in range(dmax - len(e)):
				e = np.insert(e, len(e[0]), 0, axis=1)
			for k in range(dmax - len(e)):
				e = np.insert(e, len(e), 0, axis=0)
		ll += [e]
	return ll

def infer_assortativity_matrix(n):
	S = assort_matrices_to_max_size(get_example_assortativity_matrices())  # list of matrices of size dmax*dmax
	
	# If there is an example with same nb of nodes, return its assortativity matrix
	i = 0
	for e in examples_nbnodes_list:
		if e == n:
			mat = S[i]
			return mat
		i += 1
		
	# Elif n greater than max or lower than min of all known examples, return empty
	if n > max(examples_nbnodes_list) or n < min(examples_nbnodes_list):
		print("Cannot infer assortativity matrix : " + str(n) + " not in range of known examples.")
		return []
	
	# Else, find two examples with nearest number of nodes
	n_upper = n * 10000
	n_lower = 0
	mat_upper = []
	mat_lower = []
	i = 0
	for e in examples_nbnodes_list:
		if (e > n) and (e < n_upper):
			n_upper = e
			mat_upper = S[i]
		elif (e < n) and (e > n_lower):
			n_lower = e
			mat_lower = S[i]
		i += 1

	# calculate weighted average of upper and lower matrices
	w_lower = 1/float(n - n_lower)
	w_upper = 1/float(n_upper - n)
	s = w_lower + w_upper
	w_lower /= s
	w_upper /= s
	ml = np.multiply(mat_lower,w_lower)
	mu = np.multiply(mat_upper,w_upper)
	my_mat = ml + mu

	return my_mat

def infer_assortativity_coeff(n):
	C = get_example_assortativity_coeffs()  # list of coeffs
	
	# If there is an example with same nb of nodes, return its assortativity matrix
	i = 0
	for e in examples_nbnodes_list:
		if e == n:
			coeff = C[i]
			return coeff
		i += 1
		
	# Elif n greater than max or lower than min of all known examples, return empty
	if n > max(examples_nbnodes_list) or n < min(examples_nbnodes_list):
		print("Cannot infer assortativity coeff : " + str(n) + " not in range of known examples.")
		return []
	
	# Else, find two examples with nearest number of nodes
	n_upper = n * 10000
	n_lower = 0
	C_upper = 10000
	C_lower = -10000
	i = 0
	for e in examples_nbnodes_list:
		if (e > n) and (e < n_upper):
			n_upper = e
			C_upper = C[i]
		elif (e < n) and (e > n_lower):
			n_lower = e
			C_lower = C[i]
		i += 1

	# calculate weighted average of upper and lower matrices
	w_lower = 1/float(n - n_lower)
	w_upper = 1/float(n_upper - n)
	s = w_lower + w_upper
	w_lower /= s
	w_upper /= s
	my_coeff = w_lower*C_lower + w_upper*C_upper

	return my_coeff


