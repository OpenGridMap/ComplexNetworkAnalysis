import networkx as nx

def is_valid_degree_sequence_erdos_gallai(deg_sequence):
	try:
		dmax, dmin, dsum, n, num_degs = _basic_graphical_tests(deg_sequence)
	except nx.NetworkXUnfeasible: 
		return False
	# Accept if sequence has no non-zero degrees or passes the ZZ condition
	if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
		return True

	# Perform the EG checks using the reformulation of Zverovich and Zverovich
	k, sum_deg, sum_nj, sum_jnj = 0, 0, 0, 0
	for dk in range(dmax, dmin - 1, -1):
		if dk < k + 1:  # Check if already past Durfee index
			return True
		if num_degs[dk] > 0:
			run_size = num_degs[dk]  # Process a run of identical-valued degrees
			if dk < k + run_size:  # Check if end of run is past Durfee index
				run_size = dk - k  # Adjust back to Durfee index
			sum_deg += run_size * dk
			for v in range(run_size):
				sum_nj += num_degs[k + v]
				sum_jnj += (k + v) * num_degs[k + v]
			k += run_size
			if sum_deg > k * (n - 1) - k * sum_nj + sum_jnj:
				return False
	return True
	
def _basic_graphical_tests(deg_sequence):
	# Sort and perform some simple tests on the sequence
	p = len(deg_sequence)
	num_degs = [0] * p
	dmax, dmin, dsum, n = 0, p, 0, 0
	for d in deg_sequence:
		# Reject if degree is negative or larger than the sequence length
		if d < 0 or d >= p:
			raise nx.NetworkXUnfeasible
		# Process only the non-zero integers
		elif d > 0:
			dmax, dmin, dsum, n = max(dmax, d), min(dmin, d), dsum + d, n + 1
			num_degs[d] += 1
	# Reject sequence if it has odd sum or is oversaturated
	if dsum % 2 or dsum > n * (n - 1):
		raise nx.NetworkXUnfeasible
	return dmax, dmin, dsum, n, num_degs