import numpy as np
import scipy.stats as st


# Attributes: A1, A2, A3, ..., Am (m elements)
# Data row 1: V11, V12, V13, ..., V1m
# Data row 2: V21, V22, V23, ..., V2m
# Data row 3: V31, V32, V33, ..., V3m
# Data row 4: V41, V42, V43, ..., V4m
# Data row 5: V51, V52, V53, ..., V5m
# 			.
# 			.
# 			.
# Data row n: Vn1, Vn2, Vn3, ..., Vnm
# 
# 
# Labels: L1, L2, L3, ..., Ln (n elements)

def entropy(attribute_data):
	""" 
	Calculate Shannon entropy
	we can use np.unique function and set the `return_counts` True
	:param attribute_data: type:[np.array]-> data from a single feature
	"""
	_, frequences = np.unique(attribute_data, return_counts=True)

	# calculate the probility of every unique value
	val_prob = frequences / len(attribute_data)
	return -val_prob.dot(np.log2(val_prob))


def info_gain(attribute_data, labels):
	"""
	Caculate information gain
	:param attribute_data: type:[np.array]-> data from a single feature
	"""
	attr_value_counts = get_count_dict(attribute_data)
	total_count = len(labels)

	EA = 0.0
	for attr_val, attr_value_count in attr_value_counts.items():
		EA += attr_value_count*entropy(labels[attribute_data == attr_val])

	return entropy(labels) - EA / total_count


def get_count_dict(attribute_data):
	"""
	return a dict where the key is unique value, the value is frequency
	:param attribute_data: type:[np.array]-> data from a single feature
	"""
	values, frequences = np.unique(attribute_data, return_counts=True)
	return dict(zip(values, frequences))


class DecisionTree(object):
	"""
	Create a decision tree
	"""
	# prediction at this node
	label = None
	# Split attribute for the children
	attribute = None
	# Attribute value
	attribute_value = None
	# A list of child nodes (DecisionTree)
	children = None
	# the parent node
	parent = None

	def __init__(self, dataset, labels, attributes, parent=None, value=None):

		"""
		A decision tree node
		:param value: Value of the parent's split attribute
		"""
		if value is not None:
			self.attribute_value = value

		if parent is not None:
			self.parent = parent

		if dataset.size == 0 or not attributes:
			try:
				self.label = st.mode(labels)[0][0]
			except:
				self.label = labels[len(labels)-1]
			return

		if np.all(labels[:] == labels[0]):
			print("all")
			self.label = labels[0]
			return

		self.build(dataset, labels, attributes)
		return

	def __repr__(self):

		if self.children is None:
			return "DecisionTree< x[{0}]={1}, label={2} >".format(self.parent.attribute, self.attribute_value, self.label)
		else:
			if self.parent is not None:
				return "DecisionTree< x[{0}]={1} , {2}>".format(self.parent.attribute, self.attribute_value, self.children)
			else:
				return "DecisionTree< {0} , {1}>".format(self.attribute, self.children)

	def build(self, dataset, labels, attributes):
		"""
		build a subtree
		"""
		self.choose_best_attribute(dataset, labels, attributes)

		best_attribute_column = attributes.index(self.attribute)
		attribute_data = dataset[:, best_attribute_column]

		child_attributes = attributes[:]
		child_attributes.remove(self.attribute)

		self.children = []
		for val in np.unique(attribute_data):
			child_data = np.delete(dataset[attribute_data == val, :], best_attribute_column, 1)
			child_labels = labels[attribute_data == val]
			self.children.append(DecisionTree(child_data, child_labels, child_attributes, value=val, parent=self))

	def choose_best_attribute(self, dataset, labels, attributes):
		"""
		Choose the attribute where the information gain is the largest
		"""
		best_gain = float('-inf')
		for attribute in attributes:
			attribute_data = dataset[:, attributes.index(attribute)]
			gain = info_gain(attribute_data, labels)
			if gain > best_gain:
				best_gain = gain
				self.attribute = attribute
		return