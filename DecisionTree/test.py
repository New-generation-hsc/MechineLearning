import dtree
import numpy as np
import scipy.stats as st


def test_entropy():

	data = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
	print(dtree.entropy(data))


def test_count_dict():
	data = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
	print(dtree.get_count_dict(data))
	# {0: 6, 1:9}

def test_info_gain():
	data = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
	attribute_data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
	print(dtree.info_gain(attribute_data, data))
	# 0.0830074998558

def test_mode():
	data = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
	print(st.mode(data)[0])


def test_decision_tree():

	dataset = [[0, 0, 0, 0],
		   [0, 0, 0, 1],
		   [0, 1, 0, 1],
		   [0, 1, 1, 0],
	       [0, 0, 0, 0],
		   [1, 0, 0, 0],
		   [1, 0, 0, 1],
		   [1, 1, 1, 1],
		   [1, 0, 1, 2],
		   [1, 0, 1, 2],
		   [2, 0, 1, 2],
		   [2, 0, 1, 1],
		   [2, 1, 0, 1],
		   [2, 1, 0, 2],
		   [2, 0, 0, 0]]
	labels = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]

	attributes = ['age', 'job', 'house', 'credit']
	tree = dtree.DecisionTree(np.array(dataset), np.array(labels), attributes)
	print(tree.children[1].parent)

def test_lenses():
	fr = open("lenses.txt")
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

	dataset = [lense[:-1] for lense in lenses]
	labels = [lense[-1] for lense in lenses]

	tree = dtree.DecisionTree(np.array(lenses), np.array(labels), lensesLabels)
	print(tree)


if __name__ == '__main__':

	test_lenses()