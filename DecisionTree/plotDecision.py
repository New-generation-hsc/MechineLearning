import matplotlib.pyplot as plt


DECISION_NODE = dict(boxstyle='sawtooth', fc="0.8")
LEAF_NODE = dict(boxstyle='round4', fc='0.8')
ARROW_ARGS = dict(arrowstyle='<-')


class PlotTree(object):
	"""using the matplotlib module to represent the process of decision tree
	"""
	def __init__(self, tree):
		"""[summary]
		
		the tree is a nested dictionary. like this:
		{'no surfacing': {0: 'no', 1: {'flippers': {0:'no', 1:'yes'}}}}
		"""
		self.tree = tree
		self.width = self.get_leafs(self.tree)
		self.hegiht = self.get_depth(self.tree)

		self.xOff = -0.5 / self.width
		self.yOff = 1.0

		self.axes = self._register_graph()

	@staticmethod
	def _register_graph():
		fig = plt.figure(1, facecolor='white')
		fig.clf() 
		axes = plt.subplot(111, frameon=False) # create a sub graph and clear the axis lines
		return axes

	@classmethod
	def get_leafs(cls, tree):
		"""compute the leafs of the given tree"""
		num_leafs = 0

		node = list(tree.keys())[0]
		branch = tree[node]

		for key in branch.keys():
			if isinstance(branch[key], dict):
				num_leafs += cls.get_leafs(branch[key])
			else:
				num_leafs = 1
				# reach the leaf node
		return num_leafs

	@classmethod
	def get_depth(cls, tree):
		"""compute the depth of the given tree"""
		max_depth = 0

		node = list(tree.keys())[0]
		branch = tree[node]

		for key in branch.keys():
			if isinstance(branch[key], dict):
				this_depth = 1 + cls.get_depth(branch[key])
			else:
				this_depth = 1
			if this_depth > max_depth:
				max_depth = this_depth
		return max_depth

	def plot_node(self, nodeText, curent_pos, parent_pos, nodeType):
		"""
		Arguments:
			nodeText {str} -- [the text in the node]
			curent_pos {tuple} -- [current node position]
			parent_pos {tuple} -- [parent node position]
		"""
		self.axes.annotate(nodeText, xy=parent_pos, xycoords='axes fraction', \
				xytext=curent_pos, textcoords='axes fraction', va='center', \
				ha='center', bbox=nodeType, arrowprops=ARROW_ARGS)

	def plot_branch_text(self, current_pos, parent_pos, text):

		x_mid = (parent_pos[0] - current_pos[0])/2 + current_pos[0]
		y_mid = (parent_pos[1] - current_pos[0])/2 + current_pos[1]
		self.axes.text(x_mid, y_mid, text)

	def plot(self, tree, parent_pos, nodeText):
		"""The main function of this class, plot every node, and branch recursively"""

		num_leafs = self.get_leafs(tree)
		depth = self.get_depth(tree)
		node = list(tree.keys())[0]

		current_pos = (self.xOff + (1.0 + num_leafs)/2.0/self.width, self.yOff)
		# plot the branch text
		self.plot_branch_text(current_pos, parent_pos, nodeText)

		# plot the decision node
		self.plot_node(node, current_pos, parent_pos, DECISION_NODE)

		branch = tree[node]
		self.yOff = self.yOff - 1.0 / self.hegiht
		for key in branch.keys():
			if isinstance(branch[key], dict):
				self.plot(branch[key], current_pos, str(key))
			else:
				self.xOff = self.xOff + 1.0 / self.width
				self.plot_node(branch[key], (self.xOff, self.yOff), current_pos, LEAF_NODE)
				self.plot_branch_text((self.xOff, self.yOff), current_pos, str(key))
		self.yOff = self.yOff + 1.0 / self.hegiht


if __name__ == '__main__':

	plot = PlotTree({'no surfacing': {0: 'no', 1: {'flippers': {0:'no', 1:'yes'}}}})
	plot.plot_node('Hello', (2, 1), (3, 1.5), LEAF_NODE)
	plt.show()
