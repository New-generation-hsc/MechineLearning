from collections import defaultdict
from collections import Counter
import math


class DecisionTree(object):
	"""A classifier"""
	def __init__(self, dataset, labels, attributes):
		self.dataset = dataset
		self.labels = labels
		self.attributes = attributes

	def compute_prob(self, data_list):
		"""compute the probility
		of each label
		
		[description]
		Arguments:
			labels {[list]} -- [classifier of each recoder]
		return a dict of each lable probility
		"""
		freq_dict = defaultdict(lambda: 0)
		# a default dict recode the frequency of each lable: default value is 0
		for data in data_list:
			freq_dict[data] += 1

		amount = len(data_list)
		return {key: freq_dict[key]/amount for key in freq_dict}

	def compute_entropy(self, labels):
		"""compute the entropy
		by given the probility of each lable"""
		probilities = self.compute_prob(labels)

		entropy = 0.0
		for key in probilities:
			entropy += probilities[key]*math.log2(probilities[key])
		return -entropy

	def compute_cond_entropy(self, dataset, labels, attr):
		"""compute the conditional entropy"""
		index = self.attributes.index(attr)

		data_list = [data[index] for data in dataset]
		prob_dict = self.compute_prob(data_list)
		# this is the probility of attribute value
		
		
		cond_dict = defaultdict(list)
		# this is the dict store the same attribute data in the data
		for i, data in enumerate(dataset):
			cond_dict[data[index]].append(labels[i])

		# compute the conditional entropy
		sub_prob_dict = {}
		for key in cond_dict:
			entropy = self.compute_entropy(cond_dict[key])
			sub_prob_dict[key] = entropy

		cond_entropy = 0.0
		for key in prob_dict:
			cond_entropy += prob_dict[key]*sub_prob_dict[key]
			
		return cond_entropy

	def choose_attribute(self, dataset, labels, attrs):
		"""Choose the max gain information"""
		entropy = self.compute_entropy(labels)
		maximum = 0.0
		max_attr = None
		for attr in attrs:
			gain = entropy - self.compute_cond_entropy(dataset, labels, attr)
			print('attr: {} --------> {}'.format(attr, gain))
			if gain > maximum:
				max_attr = attr
				maximum = gain
		return max_attr

	def get_values(self, dataset, best):
		"""duplicate data list"""
		index = self.attributes.index(best)

		data_list = [record[index] for record in dataset]
		data = list(set(data_list))
		return data

	def get_examples(self, dataset, labels,  best, val):
		"""just return a list of all records in dataset that have the val for
		the best attribute"""
		index = self.attributes.index(best)
		sub_dataset = [data for data in dataset if data[index] == val]
		sub_labels = [labels[i] for i, data in enumerate(dataset) if data[index] == val]
		return sub_dataset, sub_labels


	def create_decision_tree(self, dataset, labels, attributes):

		default = Counter(labels).most_common(1)[0][1]
		# Counter return a list (element, frequency)
		
		"""If the dataset is empty or the attribute list is empty, return
		the default value."""
		if not dataset or len(labels) < 1:
			return default
		elif labels.count(labels[0]) == len(labels):
			return labels[0]
		else:
			# Choose the next best attribute to best classify our data
			best = self.choose_attribute(dataset, labels, attributes)

			# Create a new decision tree/node with the best attribute
			tree = {best:{}}

			# Create a subtree for the current value under the "best" field
			for data in self.get_values(dataset, best):
				sub_dataset, sub_labels = self.get_examples(dataset, labels, best, data)
				subtree = self.create_decision_tree(
								sub_dataset, sub_labels,
								[attr for attr in attributes if attr != best]
								)
				tree[best][data] = subtree
		return tree