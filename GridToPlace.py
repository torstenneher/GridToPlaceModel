import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)
mpl.rcParams['backend'] = "Qt4Agg"
print mpl.rcParams['backend']
#mpl.use('Agg')
from sets import Set
import random
import time
import copy
import scipy.stats
import scipy.misc
import scipy.signal
import scipy.ndimage
from matplotlib import cm
#import pprint
import cPickle as pickle
#import pickle as pickle
import matplotlib.image as mpimg
import matplotlib._png as png
import matplotlib.gridspec as gridspec
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1 import ImageGrid
#from mpl.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.path import Path
import matplotlib.patches as patches
#from matplotlib_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from matplotlib_toolkits.axes_grid1.inset_locator import mark_inset
#from cvxopt import matrix, solvers
from sklearn import svm, linear_model , lda, decomposition, preprocessing
#from sklearn import svm


#mpl Figure parameter
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'legend.handlelength': 1.})
mpl.rcParams.update({'legend.labelspacing': 0.1})
#mpl.rcParams['figure.figsize'] = [23.17, 12.39]
#mpl.rcParams['figure.figsize'] = [7.5, 6]
mpl.rcParams['font.family'] = ['sans-serif']
#mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['mathtext.default'] = 'regular'

begin = time.time()
# useable functions#############################################################################################
def save_figures(path = '/home/torsten/Documents/figures/', start = 0, title='', file_type = 'png'):
	figures=[manager.canvas.figure for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]

	for i, figure in enumerate(figures):
		print file_type
		figure.savefig(path+ title +'%d.'% int(i+start) +file_type, dpi = 600)
		#figure.savefig(path+ title +'%d.png' % int(i+start))


def normalize(pattern, one_by_one = True): #normalize pattern over last dimension;
	
	dim =len(pattern.shape)
	
	if dim == 4:
		norms = np.sqrt(np.einsum('enpj,enpj->enp', pattern*1.,pattern*1.)).repeat(pattern.shape[-1]).reshape(pattern.shape)
		norms[norms == 0] =1
		pattern /= norms
	
	if dim == 3:
		norms = np.sqrt(np.einsum('npj,npj->np', pattern*1.,pattern*1.)).repeat(pattern.shape[-1]).reshape(pattern.shape)
		norms[norms == 0] =1
		pattern /= norms
	
	if dim ==2:
		if one_by_one:
			for p in pattern:
				p /= np.sqrt(np.dot(p,p))
		else:
			pattern /= np.sqrt(np.einsum('pj,pj->p', pattern*1.,pattern*1.)).repeat(pattern.shape[1]).reshape(pattern.shape)
	
	if dim ==1:
		pattern/= np.sqrt(np.sum(pattern**2))


def makeFigure2Spatialcells(fig = None, p1=None, p2=None, locations=None, label1 = None, label2 = None, title = None, legend = 0, geometry = [1,1,1], colors = ['b', 'g']):
	
	if fig == None:
		fig = plt.figure()
	ax1 = fig.add_subplot(geometry[0],geometry[1],geometry[2] )
	non_z1 = np.flatnonzero(p1)
	non_z2 = np.flatnonzero(p2)
	ax1.scatter( locations[:,0][non_z1], locations[:,1][non_z1], c= colors[0], ec = 'none', label = label1)
	ax1.scatter( locations[:,0][non_z2], locations[:,1][non_z2], c= colors[1], alpha = 0.5, ec = 'none', label = label2)
	ax1.set_xlim(0,1)
	ax1.set_ylim(0,1)
	if title != None:
		ax1.set_title(title)
	if legend:
		ax1.legend()
	return ax1

def makeInset(fig = None, ax = None, ax_xy = None, ax_width = None, ax_height = None): #make inset to axes ax in figure fig at axcoordinates (bottomleft) ax_xy; return inset, instances of Axes
	
	display_xy = ax.transData.transform(ax_xy)
	display_area = ax.transData.transform((ax_width, ax_height))- ax.transData.transform((0, 0))
	inv = fig.transFigure.inverted()
	fig_xy = inv.transform(display_xy)
	fig_area = inv.transform(display_area)
	return fig.add_axes([fig_xy[0], fig_xy[1], fig_area[0], fig_area[1]])
	
def makeLabel(ax = None, label = None, sci = False): #puts letter at the upper right corner at axes
	
	if sci:
		ax.annotate(xytext=(-0.15, 1.04), xy=(-0.15, 1.04), textcoords = 'axes fraction' ,s =label, size = 'xx-large', weight = 'extra bold', visible = True, annotation_clip=False)
		ax.yaxis.get_major_ticks()[-1].label1.set_visible(False)
	else:
		ax.annotate(xytext=(-0.2, 1.1), xy=(-0.2, 1.1),textcoords = 'axes fraction' ,s =label, size = 'xx-large', weight = 'extra bold', visible = True, annotation_clip=False)


def plotCell(In, patterns = None, env = 0, cell = None, binary = False, fig = None, noise = None, only_stored=True, ax_index = None, ax = None, color = 'r', cb = 0, size =20, zeros = False,vmin = None, vmax = None):
	'''
	abstract
	plots cell firing over space of cell cell. Location of pixels are provided by Input Instance In. Plot is made in ax of figure fig.
	:param In: Input Instance that provides locations of the pixels. Additional if patterns is not given, it uses self.input_stored in In.
	:type In: Input Instance; In.number_patterns must be the same as number of patterns in patterns
	:param patterns: cell population firing over space
	:type patterns: array of dimension 3 (envirionment, pattern, cell fire)
	:param cell: Index of which cell is plotted
	:type cell: int
	:param env: Index of environment
	:type env: int
	:param binary: Whether firing is plotted as 1 or 0
	:type binary: bool
	:param fig: Figure Instance in which the plot is plotted; if None, new Figure is created
	:type fig:Figure Instance 
	:param ax: optional, gives the axes into which plot is plotted. If not given, a new axes instance is created in fig
	:type ax:axes instance 
	:param ax_index: index of new created axes Intance in the figure fig
	:type ax_index lsit of length 3
	:param color: color of pixel, when binary = True
	:type color: mpl color
	:param cb: Whether colorbar is added
	:type cb: bool
	:param size: size of pixel
	:type size: int
	'''



	if fig ==None:
		fig = plt.figure()
	if patterns ==None:
		patterns = In.input_stored
	if ax == None:
		ax = fig.add_subplot(ax_index[0], ax_index[1], ax_index[2])
		#ax.set_title('Firing of cell '+str(cell))
	if only_stored:
		loc = In.locations[In.store_indizes]
	else:
		loc = np.tile(In.locations, (env+1,1,1))
	ax.set_xlim(-0.05,In.cage[0])
	ax.set_ylim(-0.05,1.0, In.cage[1])
	print 'plotcell', patterns.shape, loc.shape
	cb_possible = True
	if zeros == False:
		if noise == None:
			if binary:

				s = ax.scatter(loc[env][:,0][np.flatnonzero(patterns[env][:,cell] != 0)], loc[env][:,1][np.flatnonzero(patterns[env][:,cell] != 0)], c = color, faceted = False, s = size)
				
			else:
				s = ax.scatter(loc[env][:,0][np.flatnonzero(patterns[env][:,cell] != 0)], loc[env][:,1][np.flatnonzero(patterns[env][:,cell] != 0)], c = patterns[env][:,cell][np.flatnonzero(patterns[env][:,cell] != 0)], edgecolor = 'none',cmap=cm.jet, s = size, vmin = vmin, vmax = vmax)
				if np.flatnonzero(patterns[env][:,cell] != 0).shape < 1:
					cb_possible = False
		else:
			if binary:
				s = ax.scatter(loc[env][:,0][np.flatnonzero(patterns[env, noise][:,cell] != 0)], loc[env][:,1][np.flatnonzero(patterns[env,noise][:,cell] != 0)], c = color, faceted = False, s = size)
			else:
				print loc.shape
				print patterns.shape
				s = ax.scatter(loc[env][:,0][np.flatnonzero(patterns[env, noise][:,cell] != 0)], loc[env][:,1][np.flatnonzero(patterns[env,noise][:,cell] != 0)], c = patterns[env,noise][:,cell][np.flatnonzero(patterns[env,noise][:,cell] != 0)], faceted = False,cmap=cm.jet, s = size, vmin = vmin, vmax = vmax)
				if np.flatnonzero(patterns[env,noise][:,cell] != 0).shape < 1:
					cb_possible = False
	if zeros == True:
		if noise == None:
			if binary:
				s = ax.scatter(loc[env][:,0], loc[env][:,1], c = color, faceted = False, s = size)
			else:
				s = ax.scatter(loc[env][:,0], loc[env][:,1], c = patterns[env][:,cell], faceted = False,cmap=cm.jet, s = size, vmin = vmin, vmax = vmax)
		else:
			if binary:
				s = ax.scatter(loc[env][:,0], loc[env][:,1], c = color, faceted = False, s = size)
			else:
				s = ax.scatter(loc[env][:,0], loc[env][:,1], c = patterns[env,noise][:,cell], faceted = False,cmap=cm.jet, s = size, vmin = vmin, vmax = vmax)
	if cb and cb_possible:
		fig.colorbar(s)
	return [ax,s]


# to fasten up the simulations, exp(x) is calculated beforehand for many x values and stored in a list. When exp(x) has to be calculated later it is just looked up in the list
#list of arguments for them np.exp() are calculated
exp_list = np.arange(-50, 30, 0.001)
# list of calculated np.exp() for the arguemnts defined above 
exp = np.exp(exp_list)
def calcExp(x): #faster exp(x) function
	'''
	abstract
	substition for np.exp(x). Searches for x the argument (index) which is most similar as in exp_list and returns np.exp() for this index. This is much faster, since np.exp() is calculated only once for the arguments specified in exp_list
	''' 
	index = np.int32((x-exp_list[0])/0.001)
	index[index < 0] = 0
	index[index >= exp_list.shape[0]] = exp_list.shape[0] -1
	return exp[index]



#####################################################################Networks########################################################################################################
class Network(object): # Generic network class 
	
	'''
	
	
	:param input_cells: number of input cells
	:type input_cells: int
	:param connectivity: proportion of input cells to which one outputcell is connected to
	:type connectivity: float in [0,1]
	:param learnrate: factor used for learning each pattern
	:type learnrate: float
	:param subtract_input_mean: determines whether input mean is subtracted by applying Hebbian learning+
	:param subtract_input_mean: bool
	:param subtract_output_mean: determines whether output mean is subtracted by applying Hebbian learning+
	:param subtract_output_mean: bool
	:param actFunction: How output is computed; e.g. activation Function
	:type actFunction: Network.getOutput method
	:param number_winner: number of firing neurons in the output in one pattern if actFunction is a WTA function
	:type number_winner: int
	:param e_max: Parameter for getOutput function ' getOutputEMax'. Determines activity threshold.
	:type weight_mean: float
	:param active_in_env: number of cells that are allowed to be active in one environment; if None all cells can fire in all environments
	:type active_in_env: int in [0, self.cells]
	:param n_e: Number of environments; only necessary if active_in_environnment != None
	:type n_e: int
	:param initMethod: How the weights are initialized
	:type initMethod: makeWeights method in :class:`Network`
	:param weight_sparsity: parameter needed of weight initMethod 'Network.makeWeightsSparsity'
	:type weight_sparsity: float in [0,1]
	:param weght_mean: Parameter for weight init function ' makeWeightsNormalDistributed'. Determines mean of the normal distribution how weights are initialized
	:type weight_mean: float
	:param weght_sigma: Parameter for weight init function ' makeWeightsNormalDistributed'. Determines sigma of the normal distribution how weights are initialized
	:type weight_mean: float
	'''
	
	
	def __init__(self, input_cells=None, cells=None, connectivity = None, learnrate= None, subtract_input_mean = None, subtract_output_mean = None, actFunction = None, number_winner=None, e_max = 1, active_in_env = None, n_e = 1, initMethod = None, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5):
		 

		self.no_presynapses = int(connectivity*input_cells)
		self.learnrate = learnrate
		self.number_winner = number_winner
		self.sparsity = self.number_winner/(cells*1.)
		self.cells = cells
		self.input_cells = input_cells
		self.subtract_input_mean = subtract_input_mean
		self.subtract_output_mean = subtract_output_mean
		self.getOutput = actFunction
		self.n_e = n_e
		self.e_max = e_max		
		if active_in_env == None:
			active_in_env = self.cells
		self.active_in_env = active_in_env
		
		self.initActiveCells()
		self.initConnection()
		initMethod(self, sparsity=weight_sparsity, mean = weight_mean, sigma = weight_sigma)
		
		self.Cor = {} #Dictionarry of Corelation instances constisting of input and outputstatistics
		self.output_stored = None #stored output pattern
		self.noisy_output = None #reconstructed stored output when noisy input was given 



	####Initialize the connectivity matrix
	def initConnection(self):
		self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
		self.connection[np.mgrid[0:self.cells, 0:self.no_presynapses][0], np.array(map(random.sample, [range(self.input_cells)]*self.cells,  [self.no_presynapses]*self.cells))] = 1


	def initConnectionModular(self, border = None, no_synapses = None):

		self.connection = np.zeros([self.cells, self.input_cells], 'bool') # connection matrix; [i,j] = 1 iff there is a connection from j to i, otherwise 0
		if no_synapses[0]>0:
			print 'synapses in init networks'
			print border
			print no_synapses
			self.connection[np.mgrid[0:self.cells, 0:no_synapses[0]][0], np.array(map(random.sample, [range(border[0])]*self.cells,  [no_synapses[0]]*self.cells))] = 1
		if no_synapses[1]>0:
			self.connection[np.mgrid[0:self.cells, 0:no_synapses[1]][0], np.array(map(random.sample, [range(border[0], border[1])]*self.cells,  [no_synapses[1]]*self.cells))] = 1
		if no_synapses[2]>0:
			self.connection[np.mgrid[0:self.cells, 0:no_synapses[2]][0], np.array(map(random.sample, [range(border[1], self.input_cells)]*self.cells,  [no_synapses[2]]*self.cells))] = 1
		
		
	###Determines which cells can be active in the environments
	def initActiveCells(self):
		self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')
		self.active_cells_vector = np.zeros([self.n_e, self.cells], 'bool')

		for h in range(self.n_e):
				self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
				self.active_cells_vector[h][self.active_cells[h]]=1

	#Methods for initializing the weights##################################################################
	# important for you
		
	def makeWeightsZero(self, **kwargs):
		'''
		
		init weights as zero
		'''
		self.weights = np.zeros(self.connection.shape)
	
	def makeWeightsUniformDistributed(self,**kwargs):
		'''
		
		init weighs created over uniform distribution between [0,1] and normalize them;
		'''
		self.weights = np.random.uniform(0,1, self.connection.shape)
		self.weights*=self.connection
		normalize(self.weights)
		
	def makeWeightsNormalDistributed(self, mean = None, sigma = None, **kwargs):
		'''
		
		init weighs created due to normal distribution with mean = mean and sigma = sigma. Finally, weights are normalized
	
		:param mean: Mean of distribution
		:type mean: float
		:param sigma: sigma of distribution
		:type sigma: float 
		'''
		self.weights = np.random.normal(loc = mean, scale =sigma, size = self.connection.shape)
		self.weights*=self.connection
		normalize(self.weights)
		
	def makeWeightRealisticDistributed(self, **kwargs):
		print 'make weights realistc distributed'
		self.weights = np.zeros(self.connection.shape)
		#self.makeWeightsUniformDistributed()
		s = np.linspace(0,0.2, 1000)
		pdf_s = 100.7*(1-np.exp(-s/0.022))*(np.exp(-s/0.018)+0.02*np.exp(-s/0.15))
		cdf_s = np.zeros(pdf_s.shape)
		rand = np.random.uniform(0, np.sum(pdf_s), size = (self.cells, self.input_cells))
		for i in range(pdf_s.shape[0]):
			cdf_s[i:] += pdf_s[i]
		w_ind = 0
		for w in self.weights:
			for i in range(self.input_cells):
				w[i] = s[np.flatnonzero(rand[w_ind, i]> cdf_s)[-1]]
			w_ind +=1
		self.weights = self.weights/0.2*(self.weights/(self.weights+0.0314)) *self.connection
		#plt.hist(list(self.weights[0:5]), bins = 50, histtype = 'step')
		#plt.hist(list(self.weights[self.weights>0]), bins = 100, histtype = 'step')
		#plt.show()
		print 'make weights realistc distributed -------- Done'
	
	
	
	
	#not important
	def makeWeightsOne(self, **kwargs):
		'''
		
		init weights as one and normalize them
		'''
		self.weights = np.ones(self.connection.shape)
		self.weights *= self.connection
		normalize(self.weights)

	def makeWeightsSparsity(self, sparsity=0.05, **kwargs):
		'''
		
		init weights as 0 or 1 and normalize them
		
		:param sparsity: proportion of ones in one incomming weight vector of one cell. This proportion cannot be higher as self.connectivity
		:type sparsity: float in (0,1]
		'''
		self.weights = np.zeros(self.connection.shape)
		active_units = np.int(sparsity*self.no_presynapses)
		for row in range(self.connection.shape[0]):
			make_active = random.sample(np.nonzero(self.connection[row])[0], active_units) #which existing weights are set to 1
			self.weights[row][make_active] = 1
		self.weights*=self.connection
		normalize(self.weights)


		
		
	##### Learning Methods ##################################################
	# These methods learn patterns by adjusting self.weights 
	def hebbianLearning(self,input_pattern = None, output_pattern = None, learnrate = None):
		'''
		
		adjusts weights according to the standard hebbian rule wij = k*p_i*q_j, k = learnfactor = self.learnrate; if self.subtract_input_mean, input_mean is subtracted from input before learning; similar if self.subtract_output_mean
	
		:param input_pattern: input to associate
		:type input_pattern: array of max 3 dimensions, last one must be self.connection.shape[1]
		:param output_pattern: output to associate
		:type output_pattern: array of max 3 dimensions, last one must be self.connection.shape[0]
		'''
		if learnrate == None:
			learnrate = self.learnrate
		if learnrate != 0:
			input_scaled = np.copy(input_pattern)*1.0
			output_scaled = np.copy(output_pattern)*1.0
	
			if len(input_pattern.shape) == 1:
				self.weights += learnrate * np.einsum('j,i,ij->ij', input_scaled, output_scaled, self.connection)
			if len(input_pattern.shape) == 2: #(pattern,cell)
				if self.subtract_input_mean:
					input_mean =np.einsum('pi->i',input_scaled)/(input_scaled.shape[0]+0.0)
					input_mean2 =np.sum(input_scaled, axis = -2)/(input_scaled.shape[0]+0.0)
					print "newput mean correct?"
					print (input_mean == input_mean2) 
					
					input_scaled -= input_mean
				if self.subtract_output_mean:
					output_mean =np.einsum('pi->i',output_scaled)/(output_scaled.shape[0]+0.0)
					output_scaled -= output_mean
				self.weights += learnrate * np.einsum('pj,pi,ij->ij', input_scaled, output_scaled, self.connection)
				w2 = np.tensordot(output_scaled, input_scaled, (-2,-2)).reshape(self.weights.shape) *self.connection
				print 'w=w2 in hebb learning ? Change it!!!!!!!'
				print self.weights == w2
				
			if len(input_pattern.shape) == 3:#(environment, pattern, cell)
				if self.subtract_input_mean:
					#input_mean =np.einsum('epi->i',input_scaled)/(input_pattern.shape[0]*input_pattern.shape[1]+0.0)
					input_mean = np.sum(np.sum(input_scaled, 0), 0)/(input_pattern.shape[0]*input_pattern.shape[1]+0.0)
					input_scaled -= input_mean
				if self.subtract_output_mean:
					output_mean =np.einsum('epi->i',output_scaled)/(output_pattern.shape[0]*output_pattern.shape[1]+0.0)
					output_scaled -= output_mean
				print "adjust weights"
				if input_scaled.shape[0] >1:
					self.weights += learnrate * np.einsum('epj,epi,ij->ij', input_scaled, output_scaled, self.connection)
				else: #faster
					self.weights = np.tensordot(output_scaled, input_scaled, (-2,-2)).reshape(self.weights.shape) *self.connection
	
	def learnRegression(self,input_pattern = None, output_pattern = None, key = 'StoredStored'):
		
		'''
		
		learns weight by using linear regression between input pattern and outputpattern
	
		:param input_pattern: input
		:type input_pattern: array of max 3 dimensions, last one must be self.connection.shape[1]
		:param output_pattern: output
		:type output_pattern: array of max 3 dimensions, last one must be self.connection.shape[0]
		'''
		
		if (self.connection == 0).any():
			self.weights = self.calcRegressionNoFullConnectivity(input_pattern = input_pattern, output_pattern = output_pattern)
		else:
			self.weights = self.calcRegression(input_pattern = input_pattern, output_pattern = output_pattern)

		self.output_stored = output_pattern
		self.input_stored = input_pattern
		self.Cor[key] = Corelations(patterns_1 = self.output_stored, patterns_2 = np.tile(self.output_stored, (2,1,1)))
	
	def makeWeightsPattern(self, **kwargs):
		
		'''
		
		stores pattern by setting incomming weights to patterns that are going to be stored. If possible each pattern occures number_winner times in the set of weights
		'''
		no_pattern = max(int(self.cells/self.number_winner), self.input_stored.shape[1])# If possible each pattern occures number_winner times in the set of weights
		self.weights = np.zeros(self.weights.shape)
		for i in range(no_pattern):
			self.weights[i*self.number_winner : (i+1)*self.number_winner] = np.tile(self.input_stored[0][i], (self.number_winner, 1))
		self.weights *= self.connection
		normalize(self.weights)
		self.output_stored = self.getOutputWTALinear(input_pattern = self.input_stored)
	
	
	
	
	
	#####################actFunction:############################################
	#what you need at the beginning
	def calcActivity(self,input_pattern=None): 
		'''
		
		calculates the output activity
		
		:param input_pattern: input_pattern
		:type input_pattern: numpy.array of arbritrary dimension
		:param return: Activity in the output
		:type return: numpy array. of same dimension as input_pattern, only last dimension differ if number of input cells is different to number of output cells.
		'''
		activity = np.tensordot(input_pattern, self.weights, (-1,-1))
		return activity
	
	def getOutputWTA(self,input_pattern=None, env = None, **kwargs): #returns the firing of the network given the input patterns; 
		'''
		
		calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire; firing rate is either 1 or 0; Only cells that are allowed to be active in the enviroment are considered.
		
		:param input_pattern: input
		:type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]. Dimesnions are (environments, noise_level, patterns, cells)
		:param env: specifies current enviromnent if input dimension = 1 
		:param return: firing of the outputcells 
		:type return: array
		'''
		size = list(np.shape(input_pattern)) #dimension of input
		size[-1] = self.weights.shape[0] # change to dimension of output
		
		#set activity of those cells to zero, that are not allowed to be active in the environment
		if len(size) == 1:
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]
		if len(size) == 2: #env, pattern--- not possible; use len(size) = 1 instead and specify env.
			print 'len 2 not possible getoutput wta'
		if len(size) == 3:#env, pattern, cell
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
		else: #env, noise, pattern, cell
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)
		

		winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

		fire_rate = np.ones(size, 'bool')
		out_fire = np.zeros(size, 'bool')

			
		if len(size) ==1:#pattern
			out_fire[winner] = fire_rate[winner]
		if len(size) ==2:#env, pattern--- not possible
			out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
		if len(size) ==3: #env, pattern, cells
			indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
			out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
		if len(size) ==4: # env, noise, time, pattern
			indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
			out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
		if len(size) > 4:
				print 'error in input dimension in calckWTAOutput'
		return out_fire
	
	def getOutputWTALinear(self,input_pattern=None, env = 0, **kwargs): #returns the firing of the network given the input patterns; output is the activity of the winners
		'''
		
		Same as getOutputWTA, but now outputfiring is equal to activity of cell
		'''
		
		size = list(np.shape(input_pattern)) #dimension of input
		size[-1] = self.weights.shape[0] # change to dimension of output

		
		if len(size) == 1:#cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]
		if len(size) == 2:
			print 'len 2 not possible getoutput wta'
		if len(size) == 3:#env, pattern, cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
		if len(size) == 4:#env, noise, pattern,cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)
		

		winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

		fire_rate = activity # activities < 0 are set to 0
		fire_rate[fire_rate< 0] = 0
		out_fire = np.zeros(size)

		if len(size) ==1:

			out_fire[winner] = fire_rate[winner]
		if len(size) ==2:
			out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
		if len(size) ==3:
			indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
			out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
		if len(size) ==4: # env, noise, time, pattern
			indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
			out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
		if len(size) > 4:
				print 'error in input dimension in calckWTAOutput'
		#normalize(out_fire)
		return out_fire
	
	
	def getOutputLinthresOptimalErr(self, activation2 = None, output = None):
		

		activity = np.copy(activation2)
		activity[activity < 0] = 0
		activity /= np.tile(np.max(activity, axis = -2), (activity.shape[0], 1))
		err = np.einsum('lc, lc-> c', activity - output, activity - output)

		for k in np.linspace(np.min(activity),np.max(activity), 1000):
			activity_help = np.copy(activity)
			activity_help[activity_help <= k] = 0
			err_new = np.einsum('lc, lc-> c', activity_help - output, activity_help - output)
			if (err_new > err).any():
				print 'break precalc linthreshold output at k = '+str(k)
				break
			else:
				activity = activity_help
				err = np.copy(err_new)


		too_bad_rows = err_new < err
		activity[activity == 0] = 10**4
		min_activity_cell_loc = np.argmin(activity, -2) 
		activity[min_activity_cell_loc[too_bad_rows], np.arange(activity.shape[-1])[too_bad_rows]] = 0
		activity[activity == 10**4] = 0
		err_new = np.einsum('lc, lc-> c', activity - output, activity - output)
	

		optimal_thres = False
		i=0
		while not optimal_thres:
			if (i/100.) == int(i/100):
				print i
			too_bad_rows = err_new < err
			if not too_bad_rows.any():
				optimal_thres = True
				print 'number cells reduced to zero'
				print i
				break
			activity[activity == 0] = 10**4
			min_activity_cell_loc = np.argmin(activity, -2) 
			activity[min_activity_cell_loc[too_bad_rows], np.arange(activity.shape[-1])[too_bad_rows]] = 0
			activity[activity == 10**4] = 0
			err = np.copy(err_new)
			err_new = np.einsum('lc, lc-> c', activity - output, activity - output)
			i+=1
			
		return activity
	
	
	def getOutputLinthresOptimalCor(self, activation = None, output = None): #does not work
		
		

		activity = np.copy(activation)
		cor = Corelations(patterns_1 = activity, patterns_2 = output, env = 1, in_columns =1 ).getCorOrigOrig(at_noise = 0)

		for k in np.linspace(np.min(activity),np.max(activity), 1000):
			activity_help = np.copy(activity)
			activity_help[activity_help <= k] = 0
			cor_new = Corelations(patterns_1 = activity_help, patterns_2 = output, env = 1, in_columns =1 ).getCorOrigOrig(at_noise = 0)
			#cor_new[cor_new <0 ] = 0
			if (cor_new  < cor).any():
				print 'break precalc linthreshold output at k = '+str(k)
				break
			else:
				activity = activity_help
				cor = np.copy(cor_new)



		too_bad_rows = cor_new > cor
		activity[activity == 0] = 10**8
		min_activity_cell_loc = np.argmin(activity, -2) 
		activity[min_activity_cell_loc[too_bad_rows], np.arange(activity.shape[-1])[too_bad_rows]] = 0
		activity[activity == 10**8] = 0
		cor_new = Corelations(patterns_1 = activity, patterns_2 = output, env = 1, in_columns =1 ).getCorOrigOrig(at_noise = 0)

		optimal_thres = False
		i=0
		while not optimal_thres:
			if (i/100.) == int(i/100):
				print i
			too_bad_rows = cor_new > cor
			if not too_bad_rows.any():
				optimal_thres = True
				print 'number cells reduced to zero'
				print i
				break
			activity[activity == 0] = 10**8
			min_activity_cell = np.argmin(activity, -2) 
			activity[min_activity_cell_loc[too_bad_rows], np.arange(activity.shape[-1])[too_bad_rows]] = 0
			activity[activity == 10**8] = 0
			cor = np.copy(cor_new)
			cor_new = Corelations(patterns_1 = activity, patterns_2 = output, env = 1, in_columns =1 ).getCorOrigOrig(at_noise = 0)
			i+=1
		return activity
	
	
	#do not need it right now
	def getOutputWTARolls(self,input_pattern=None, **kwargs): #returns the firing of the network given the input patterns; output is the highest activity of the winners
		# ToDo : active cells in environment is not supported yet!!! 
		'''
		
		calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire; firing neurons have all the same firing rate, which is equal to the maximal activity in that pattern
	
		:param input_pattern: input
		:type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]. Dimesnions are (environments, noise_level, patterns, cells)
		:param return: firing of the outputcells 
		:type return: array
		'''
		activity = self.calcActivity(input_pattern=input_pattern)
		size = list(np.shape(input_pattern)) #dimension of input
		size[-1] = self.weights.shape[0] # change to dimension of output
		winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

		max_activity_pattern = np.max(activity, -1)
		fire_rate = max_activity_pattern.repeat(activity.shape[-1]).reshape(activity.shape)
		fire_rate = np.ones(activity.shape)
		out_fire = np.zeros(size)
			
		if len(size) ==1:
			out_fire[winner] = fire_rate[winner]
		if len(size) ==2:
			out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
		if len(size) ==3:
			indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
			out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
		if len(size) ==4: # env, noise, time, pattern
			indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
			out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
		if len(size) > 4:
				print 'error in input dimension in calckWTAOutput'
		return out_fire

	def getOutputSign(self,input_pattern=None, **kwargs): #returns the firing of the network given the input patterns; output is binary, all cells with activity above 0 fire, the others do not.

		'''
		
		calculates outputfiring  given input_pattern; neuron fires if activation >=0;
	
		:param input_pattern: input
		:param return: firing of the outputcells;
		:type return: array
		'''
		activity = self.calcActivity(input_pattern=input_pattern)
		size = list(np.shape(input_pattern)) #dimension of input
		size[-1] = self.weights.shape[0] # change to dimension of output
		fire = np.zeros(size)
		fire[np.nonzero(activity > 0)] = 1
		return fire
	
	def getOutputId(self, input_pattern=None, **kwargs):
		'''
		
		calculates outputfiring  given input_pattern; Firing rate of all neurons is equal to their activity level
	
		:param input_pattern: input
		:param return: firing of the outputcells;
		:type return: array
		'''
		activity = self.calcActivity(input_pattern = input_pattern)
		return activity
	
	def getOutputEMax(self, input_pattern = None, env = 0, e_max = None,**kwargs):
		
		'''
		
		calculates outputfiring  given input_pattern; EMax activation function is used. Cells that are within (1-emax) times the maximal activation of one cell in that pattern fire, the others not. Firing rate is activation level
	
		:param input_pattern: input
		:param return: firing of the outputcells;
		:type return: array
		'''
		
		if e_max == None:
			e_max = self.e_max
		size = list(np.shape(input_pattern)) #dimension of input

		size[-1] = self.weights.shape[0] # change to dimension of output
		if len(size) == 1:#cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector[env]

		if len(size) == 2:
			print 'len 2 not possible getoutput wta'
		if len(size) == 3:#env, pattern, cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
		if len(size) == 4:#env, noise, pattern,cells
			activity = self.calcActivity(input_pattern=input_pattern) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)
			
		if e_max < 1:
			min_activity = (np.max(activity, -1) * (1-e_max)).repeat(activity.shape[-1]).reshape(activity.shape)#max act in one pattern *emax = min act one cell must have to be active
			activity[activity < min_activity] = 0
		return activity
		
	def getOutputLinearthreshold(self,input_pattern=None, env = None, **kwargs):
		
		'''
		
		calculates outputfiring  given input_pattern; Firing is calcualted accoriding to Rolls 1995 activation function. A threshold is determined for each pattern individually to assure sparsity level.
	
		:param input_pattern: input
		:param return: firing of the outputcells; silent cells are 0 
		:type return: array
		'''
		
		activity = self.calcActivity(input_pattern = input_pattern)		
		activity[activity <= 0] = 0
		for k in np.linspace(np.min(activity),np.min(np.max(activity, -1)), 100):
			if k>=np.max(activity):
				if len(input_pattern.shape) >= 2:
					print 'break precalc since otherwise all 0'
				break
			activity_help = np.copy(activity)
			activity_help[activity_help <= k] = 0
			a = Network.calcFireSparsity(activity_help)
			if (a < self.sparsity).any():
				if len(input_pattern.shape) >= 2:
					print 'break precalc linthreshold output at k = '+str(k)
				break
			else:
				activity = activity_help
		
		
		if len(input_pattern.shape) >= 2:
			
			print 'calc Output Hetero Linthreshold'
			a = self.calcFireSparsity(activity)
			not_sparse = True
			i=0
			while not_sparse:
				if (i/100.) == int(i/100):
					print i
				too_large_a_rows = a > self.sparsity
				if not too_large_a_rows.any():
					not_sparse = False
					print 'number cells reduced to zero'
					print i
					break
				activity[activity == 0] = 10**15
				min_activity_cell = np.argmin(activity, -1) 
				activity[too_large_a_rows, min_activity_cell[too_large_a_rows]] = 0
				activity[activity == 10**15] = 0
				a = self.calcFireSparsity(activity)
				i+=1
			normalize(activity)
				
		if len(input_pattern.shape) == 1:
			activity = self.calcActivity(input_pattern = input_pattern)
			a = self.calcFireSparsity(activity)
			not_sparse = True
			i=0
		
			while a > self.sparsity:
				min_activity = np.min(activity[activity !=0])
				activity[activity == min_activity] = 0
				a = self.calcFireSparsity(activity)
				i+=1
			normalize(activity)
		out_fire = activity
			
		
		return activity
	
	def getOutputLinearthresholdNegative(self,input_pattern=None, **kwargs):
		
		'''
		
		calculates outputfiring  given input_pattern; Firing is calcualted accoriding to Rolls 1995 activation function. A threshold is determined for each pattern individually to assure sparsity level. Difference to getOutputLinearthreshold is that activities below zero are allowed.
	
		:param input_pattern: input
		:param return: firing of the outputcells; silent cells are  0 
		:type return: array
		'''
		
		activity = self.calcActivity(input_pattern = input_pattern)

		for k in np.linspace(np.min(activity),np.min(np.max(activity, -1)), 100):
			if k>=np.max(activity):
				if len(input_pattern.shape) >= 2:
					print 'break precalc since otherwise all 0'
				break
			activity_help = np.copy(activity)
			activity_help[activity_help <= k] = 0
			a = Network.calcFireSparsity(activity_help)
			if (a < self.sparsity).any():
				if len(input_pattern.shape) >= 2:
					print 'break precalc linthreshold output at k = '+str(k)
				break
			else:
				activity = activity_help
		
		
		if len(input_pattern.shape) >= 2:
			
			print 'calc Output Hetero Linthreshold'
			a = self.calcFireSparsity(activity)
			not_sparse = True
			i=0
			while not_sparse:
				if (i/100.) == int(i/100):
					print i
				too_large_a_rows = a > self.sparsity
				if not too_large_a_rows.any():
					not_sparse = False
					print 'number cells reduced to zero'
					print i
					break
				activity[activity == 0] = 10**15
				min_activity_cell = np.argmin(activity, -1) 
				activity[too_large_a_rows, min_activity_cell[too_large_a_rows]] = 0
				activity[activity == 10**15] = 0
				a = self.calcFireSparsity(activity)
				i+=1
			normalize(activity)
				
		if len(input_pattern.shape) == 1:
			activity = self.calcActivity(input_pattern = input_pattern)
			a = self.calcFireSparsity(activity)
			not_sparse = True
			i=0
		
			while a > self.sparsity:
				min_activity = np.min(activity[activity !=0])
				activity[activity == min_activity] = 0
				a = self.calcFireSparsity(activity)
				i+=1
			normalize(activity)
		out_fire = activity
		return activity
	
	
	def getOutputMinRate(self, activity = None, min_rate = 0.15):
		min_fire = np.max(activity, axis = -2) * min_rate
		min_fire = np.tile(min_fire , (activity.shape[-2],1)).reshape(activity.shape)
		activity[activity < min_fire] = 0
		return activity
	
	def getOutputLinearthresholdTreves(self,input_pattern=None, env = None, **kwargs):
		
		'''
		
		calculates outputfiring  given input_pattern; Firing is calcualted accoriding to Rolls 1995 activation function. A threshold is determined for each pattern individually to assure sparsity level.
	
		:param input_pattern: input
		:param return: firing of the outputcells; silent cells are 0 
		:type return: array
		'''
		activity = self.calcActivity(input_pattern = input_pattern)		
		activity[activity <= 0] = 0
		
		if len(input_pattern.shape) >= 2:
			for k in np.linspace(np.min(activity),np.min(np.max(activity, -1)), 100):
				if k>=np.max(activity):
					if len(input_pattern.shape) >= 2:
						print 'break precalc since otherwise all 0'
					break
				activity_help = np.copy(activity)
				activity_help[activity_help <= k] = 0
				a = Network.calcFireSparsity(activity_help)
				if (a < self.sparsity).any():
					if len(input_pattern.shape) >= 2:
						print 'break precalc linthreshold output at k = '+str(k)
					break
				else:
					activity = activity_help
		
			print 'calc Output Hetero Linthreshold'
			a = self.calcFireSparsity(activity)
			not_sparse = True
			i=0
			while not_sparse:
				if (i/100.) == int(i/100):
					print i
				too_large_a_rows = a > self.sparsity
				if not too_large_a_rows.any():
					not_sparse = False
					print 'number cells reduced to zero'
					print i
					break
				activity[activity == 0] = 10**15
				min_activity_cell = np.argmin(activity, -1) 
				activity[too_large_a_rows, min_activity_cell[too_large_a_rows]] = 0
				activity[activity == 10**15] = 0
				a = self.calcFireSparsity(activity)
				i+=1
			gain = self.sparsity*activity.shape[-1]*1./np.sum(activity, axis =-1).repeat(activity.shape[-1]).reshape(activity.shape) # g * mean = a
		
		if len(input_pattern.shape) == 1:
			activity = self.calcActivity(input_pattern = input_pattern)
			a = self.calcFireSparsity(activity)
			not_sparse = True
			i=0
			while a > self.sparsity:
				min_activity = np.min(activity[activity !=0])
				activity[activity == min_activity] = 0
				a = self.calcFireSparsity(activity)
				i+=1
			gain = self.sparsity*activity.shape[0]*1./np.sum(activity) # g * mean = a

		
		return gain * activity
			
	########################### Linear Regression help methods #############################################################################
	def calcRegression(self,input_pattern = None, output_pattern = None): #learn linear map from input pattern to output pattern
		print 'Full used______________________________'
		weights = np.zeros(self.weights.shape)
		#####np.lstsq solves b = ax. In our case input_pattern = output_pattern * weights.T
		if len(input_pattern.shape) <=2:
			weights = np.linalg.lstsq(input_pattern, output_pattern)[0].T
		if len(input_pattern.shape) ==3:
			weights = np.linalg.lstsq(input_pattern.reshape(input_pattern.shape[0]*input_pattern.shape[1], input_pattern.shape[2]), output_pattern.reshape(output_pattern.shape[0]*output_pattern.shape[1],output_pattern.shape[2]))[0].T
		if len(input_pattern.shape) >3:
			print 'too much dim in learn regression'
		return weights
	
	def calcRegressionNoFullConnectivity(self,input_pattern = None, output_pattern = None): #learn linear map from input pattern to output pattern without full connectivity
		weights = np.zeros(self.weights.shape)
		print 'no full used_____________________________________________'
		#####np.lstsq solves b = ax. In our case output_pattern = input_pattern * weights.T
		if len(input_pattern.shape) <=2:
			for i in range(output_pattern.shape[-1]):
				weights[i][np.flatnonzero(self.connection[i]==1)] = np.linalg.lstsq(input_pattern[:, np.flatnonzero(self.connection[:,i]==1)], output_pattern[:,i])[0].T
		if len(input_pattern.shape) ==3:
			input_p = input_pattern
			for i in range(output_pattern.shape[-1]):
				weights[i][np.flatnonzero(self.connection[i]==1)] = np.linalg.lstsq(input_p[0][:, np.flatnonzero(self.connection[i]==1)], output_pattern[0][:,i])[0]
		if len(input_pattern.shape) >3:
			print 'too much dim in learn regression'
		return weights
		
	def calcRegressionOutput(self,input_pattern = None, output_pattern = None, x = None, **kwargs):
		
		'''
		
		calculates outputfiring  given x as input; Output firing is computed after linear regression is applied on input and output. Linear regression is not learned here in the weights
	
		:param input_pattern: input
		:param return: firing of the outputcells;
		:type return: array
		'''
		if (self.connection == 0).any():
			weights = self.calcRegressionNoFullConnectivity(input_pattern = input_pattern, output_pattern = output_pattern)
		else:
			weights = self.calcRegression(input_pattern = input_pattern, output_pattern = output_pattern)
		return np.tensordot(x, weights, (-1,-1))
	
	def calcDifftoLinReg(self): #return reg(mean(p)) - 2*,mean(q)
		input_mean =np.einsum('epi -> i', self.input_stored)/(self.input_stored.shape[0]*self.input_stored.shape[1]+0.0)
		output_mean =np.einsum('epi->i',self.output_stored)/(self.output_stored.shape[0]*self.output_stored.shape[1]+0.0)
		diff = self.calcRegressionOutput(input_pattern = self.input_stored - input_mean, output_pattern = self.output_stored, x = input_mean) #- 2* output_mean
		return diff

	def printExpectedSparsity(self): 

		input_mean =np.sum(np.sum(self.input_stored, 0),0)/(self.input_stored.shape[0]*self.input_stored.shape[1]+0.0)
		diff = self.calcDifftoLinReg()
		q_hat = self.calcRegressionOutput(input_pattern = self.input_stored-input_mean, output_pattern = self.output_stored, x =self.input_stored )
		sparsity = self.calcFireSparsity(patterns = q_hat +diff)

		
		
		
	######################## Recall Function #########################################
	def recall(self, input_pattern = None, key = '', first = None):
		
		'''
		
		calculates output given input cues and creates Correlation Classes comparing stored and recalled patterns and recalled with recalled ones.
	
		:param input_pattern: input
		:type input_pattern: array
		:param first: if first != None, only first 'first' patterns are considered for analysis (if first >0 ). If first <0 only last stored patterns are considered. 
		:type first: integer
		'''
		
		if first == None:
			first = input_pattern.shape[-2]	
			
		if self.output_stored == None: #if nothing is stored
			self.output_stored = np.zeros([1,1,first])
		
		if first >=0 :
			input_pattern = input_pattern[:,:,:first]
			self.noisy_output = self.getOutput(self,input_pattern)
			self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,:first], patterns_2 = self.noisy_output)
			self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)
		
		if first < 0:
			input_pattern = input_pattern[:,:,first:]
			self.noisy_output = self.getOutput(self,input_pattern)
			self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,first:], patterns_2 = self.noisy_output)
			self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)

	@classmethod
	def calcFireSparsity(cls, patterns = None):#patterns 0 (loc, pattern)
		
		'''
		
		Help function to determine sparsity threshold for getOutputLinearthreshold
		'''
		enumerator = (np.sum(patterns*1., axis = -1)/patterns.shape[-1])**2
		denominator = np.sum((patterns*1)**2, -1)/patterns.shape[-1]
		return enumerator/denominator
	
	def getActivationActiveSilentCells(self, input_pattern  = None, at_location = None):
		
		'''
		
		gets the activation of active (silent) cells during storage when the input is presented after learning; if at_location != None only at a specific stored location 
	
		:param input_pattern: input
		:param return: activity levels of active and silent cells
		:type return: list of arrays
		'''
		activity = self.calcActivity(input_pattern = input_pattern)
		if len(activity.shape) == 3:
			activity = activity.reshape(activity.shape[0]*activity.shape[1], activity.shape[2])
		if at_location == None: #all locations
			output_pattern = self.Cor['StoredStored'].patterns_1
			active_cells = np.nonzero(output_pattern)
			silent_cells = np.nonzero(output_pattern==0)
		else:
			output_pattern = self.Cor['StoredStored'].patterns_1[at_location]
			active_cells = np.nonzero(output_pattern)
			silent_cells = np.nonzero(output_pattern==0)
		return [activity[active_cells], activity[silent_cells]]
		
	def getSilentCellWithMaxActivity(self, input_pattern  = None, at_location = None):
		
		'''
		
		return cell index of cell that was silent duirng storage at location at_lolcation and has now highest activity given the input input_pattern
		'''
		
		activity = self.getActivationActiveSilentCells(input_pattern  = input_pattern, at_location = at_location)[1]
		arg_max = np.argmax(activity, axis = -1)
		return arg_max
		
	def getActiveCellWithMaxActivity(self, input_pattern  = None, at_location = None):
		'''
		
		return cell index of cell that was active duirng storage at location at_lolcation and has now highest activity given the input input_pattern
		'''
		activity = self.getActivationActiveSilentCells(input_pattern  = input_pattern, at_location = at_location)[0]
		arg_max = np.argmax(activity, axis = -1)
		return arg_max
		
	def getActiveCellWithMinActivity(self, input_pattern  = None, at_location = None):
		'''
		
		return cell index of cell that was active duirng storage at location at_lolcation and has now lowest activity given the input input_pattern
		'''
		activity = self.getActivationActiveSilentCells(input_pattern  = input_pattern, at_location = at_location)[0]
		arg_min = np.argmin(activity, axis = -1)
		return arg_min
		
class HeteroAssociation(Network):
	
	def learnAssociation(self,input_pattern = None, output_pattern = None, key = 'StoredStored', first = None): #Association of input pattern and output pattern
		'''
		
		hebbian association input with output; self.input_stored becomes input_pattern; self.output_stored becomes outputpattern. Creates Correlation Class self.Cor[key] that analsizes the stored output
		
		:param input_pattern: input to associate
		:type input_pattern: array of max 4 dimensions, last one must be self.connection.shape[1]
		:param output_pattern: output to associate
		:type output_pattern: array of max 4 dimensions, last one must be self.connection.shape[0]
		:param first: if first != None, only the first 'first' stored pattern are considered for analysis. If negative, only the last ones are considered. However all patterns are stored.
		:type first: integer
		'''

		self.hebbianLearning(input_pattern = input_pattern, output_pattern = output_pattern)
		self.output_stored = output_pattern
		self.input_stored = input_pattern
		if first == None:
			first = input_pattern.shape[-2]
		if first >= 0:
			self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,:first])
		else:

			self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,first:])
	

class AutoAssociation(HeteroAssociation):
	
	
	'''
	
	Network with recurrent dynamics. getOutputfunctions as in :class: 'Network' but now activation cycles are possible with external input clamped on.
	
	:param cycles: Number of activation cycles. In one cycle all neurons are updated synchronously.
	:type cycles: int
	:param external_force: Determines influence of external input during dynamics; if 0 no clapmed external input is considered.
	:type external_force: int
	:param internal_force: Determines influence of recurrent input during dynamics;
	:type internal_force: int
	:param external_weights: Weight matrix that connect external input to the network. Necessary when external_force != 0
	:type external weights: np.array of dimenstion (input_cells, cells)
	
	'''	

	def __init__(self, input_cells=None, cells=None, number_winner=None, connectivity = None, learnrate= None, subtract_input_mean = None, subtract_output_mean = None, initMethod = None, weight_sparsity = None, actFunction = None, weight_mean = None, weight_sigma = None, cycles = None, external_force = None, internal_force = None, external_weights = None, active_in_env = None, n_e = None):
		

		self.cycles = cycles
		self.external_force = external_force
		self.internal_force = internal_force
		self.external_weights = external_weights
		super(AutoAssociation, self).__init__(input_cells=input_cells, cells=cells, number_winner=number_winner, connectivity = connectivity, learnrate= learnrate, subtract_input_mean = subtract_input_mean, subtract_output_mean = subtract_output_mean, initMethod = initMethod, weight_sparsity = weight_sparsity, actFunction = actFunction, active_in_env = active_in_env, n_e = n_e, weight_mean = weight_mean, weight_sigma = weight_sigma)
	
	
	
	
	### Helper Function; returns activity that arrives externally
	def calcExternalActivity(self, external_pattern = None):
		activity = np.tensordot(external_pattern, self.external_weights, (-1,-1))
		return activity

	##### Output Function ################
	#As in Network class, but now implemented with recurrent dynamics
	def getOutputWTARolls(self,input_pattern=None, external_activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
		'''
		
		calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
	
		:param input_pattern: input
		:type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: firing of the outputcells 
		:type return:  array possible dimensions (env, noise, pattern, cellfire)
		
		'''		
		
		def calcOutput(internal_pattern = None, external_activity = None):
			
			internal_activity = self.calcActivity(input_pattern=internal_pattern)
			normalize(internal_activity)
		
			activity = internal_activity*self.internal_force  + external_activity*self.external_force
			
			size = list(np.shape(input_pattern)) #dimension of input
			size[-1] = self.weights.shape[0] # change to dimension of output
			winner = np.argsort(activity)[...,-self.number_winner:size[-1]]
	
			max_activity_pattern = np.max(activity, -1)
			fire_rate = max_activity_pattern.repeat(activity.shape[-1]).reshape(activity.shape)
			#fire_rate = np.ones(size, 'bool')
			out_fire = np.zeros(size)

				
			if len(size) ==1:
				out_fire[winner] = fire_rate[winner]
			if len(size) ==2:
				out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
			if len(size) ==3:
				indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
				out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
			if len(size) ==4: # env, noise, time, pattern
				indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
				out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
			if len(size) > 4:
					print 'error in input dimension in calckWTAOutput'
			

			return out_fire

		if external_activity != None: 
			normalize(external_activity)
			
			out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
				if c == self.cycles -1:
					print 'all ' +str(c) +' cycles used'
			return out_fire
			
		else: #rec dynamics without consistent input form the outside
			out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, discrete = discrete)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, discrete = discrete)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
			return out_fire
	
	def getOutputWTA(self,input_pattern=None, external_activity = None, env = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
		'''
		
		calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
	
		:param input_pattern: input
		:type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: firing of the outputcells 
		:type return:  array possible dimensions (env, noise, pattern, cellfire)
		'''		
		
		def calcOutput(internal_pattern = None, external_activity = None, env = None):

			internal_activity = self.calcActivity(input_pattern=internal_pattern)
			normalize(internal_activity)

			size = list(np.shape(input_pattern)) #dimension of input
			size[-1] = self.weights.shape[0] # change to dimension of output
			if len(size) == 1:
				activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector[env]
			if len(size) == 2:
				print 'len 2 not possible getoutput wta'
			if len(size) == 3:
				activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
			else:
				activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)
		

			winner = np.argsort(activity)[...,-self.number_winner:size[-1]]
	
			fire_rate = np.ones(size, 'bool')
			out_fire = np.zeros(size, 'bool')
				
			if len(size) ==1:
				out_fire[winner] = fire_rate[winner]
			if len(size) ==2:
				out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
			if len(size) ==3:
				indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
				out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
			if len(size) ==4: # env, noise, time, pattern
				indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
				out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
			if len(size) > 4:
					print 'error in input dimension in calckWTAOutput'
			return out_fire


		if external_activity != None: 
			normalize(external_activity)
			out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity, env = env)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity, env = env)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
				if c == self.cycles -1:
					print 'all ' +str(c) +' cycles used'
			
		else: #rec dynamics without consistent input form the outside
			out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, env = env)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, env = env)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
		normalize(out_fire)
		return out_fire
	
	def getOutputWTALinear(self,input_pattern=None, external_activity = None, env = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
		'''
		
		calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
	
		:param input_pattern: input
		:type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: firing of the outputcells 
		:type return: array possible dimensions (env, noise, pattern, cellfire)
		'''		
		
		def calcOutput(internal_pattern = None, external_activity = None, env = None):
			
			internal_activity = self.calcActivity(input_pattern=internal_pattern)
			normalize(internal_activity)
		
			#activity = (internal_activity*self.internal_force  + external_activity*self.external_force)* self.active_cells_vector
			
			size = list(np.shape(input_pattern)) #dimension of input
			size[-1] = self.weights.shape[0] # change to dimension of output
			if len(size) == 1:
				activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector[env]
			if len(size) == 2:
				print 'len 2 not possible getoutput wta'
			if len(size) == 3:
				activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1], axis = 0).reshape(size)
			else:
				activity = (internal_activity*self.internal_force  + external_activity*self.external_force) * self.active_cells_vector.repeat(size[1]*size[2], axis = 0).reshape(size)
		

			winner = np.argsort(activity)[...,-self.number_winner:size[-1]]
	
			activity[activity<0] = 0
			fire_rate = activity
			out_fire = np.zeros(size, 'bool')

				
			if len(size) ==1:
				out_fire[winner] = fire_rate[winner]
			if len(size) ==2:
				out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
			if len(size) ==3:
				indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
				out_fire[indices[0], indices[1], winner] = fire_rate[indices[0], indices[1], winner]
			if len(size) ==4: # env, noise, time, pattern
				indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
				out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
			if len(size) > 4:
					print 'error in input dimension in calckWTAOutput'
			return out_fire

		

		if external_activity != None: 
			normalize(external_activity)

			out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity, env = env)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity, env = env)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
				if c == self.cycles -1:
					print 'all ' +str(c) +' cycles used'
			return out_fire
			
		else: #rec dynamics without consistent input form the outside
			out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, discrete = discrete, env = env)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, discrete = discrete, env = env)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
			return out_fire
	
	def getOutputLinearthreshold(self,input_pattern=None, external_activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
		'''
		
		calculates outputfiring  given input_pattern; the highest self.number_winner activated neurons fire;
	
		:param input_pattern: input
		:type input_pattern: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: firing of the outputcells 
		:type return: array possible dimensions (env, noise, pattern, cellfire)
		'''		
		
		def calcOutput(internal_pattern = None):
			
			internal_activity = self.calcActivity(input_pattern=internal_pattern)
			normalize(internal_activity)
				
			activity = self.internal_force * internal_activity + self.external_force * external_activity
		
			if len(internal_pattern.shape) >= 2:
				a = Network.calcFireSparsity(activity)
				not_sparse = True
				i=0
		
				while not_sparse:

					too_large_a_rows = a > self.sparsity
					if not too_large_a_rows.any():
						not_sparse = False


						break

					activity[activity == 0] = 10**5
					min_activity_cell = np.argmin(activity, -1) 
					activity[too_large_a_rows, min_activity_cell[too_large_a_rows]] = 0
					activity[activity == 10**5] = 0

					a = Network.calcFireSparsity(activity)
					i+=1
					#time.sleep(5)


				i=0

				normalize(activity)
				
			if len(internal_pattern.shape) == 1:
				a = Network.calcFireSparsity(activity)
				not_sparse = True
				i=0
		
				while a > self.sparsity:
					min_activity = np.min(activity[activity !=0])
					activity[activity == min_activity] = 0
					a = Network.calcFireSparsity(activity)
					i+=1

				normalize(activity)

			return activity

		

		if external_activity != None: 
			normalize(external_activity)
			
			out_fire = calcOutput(internal_pattern = input_pattern)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = calcOutput(internal_pattern =out_fire)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
				if c == self.cycles -1:
					print 'all ' +str(c) +' cycles used'
			return out_fire
			
		else: #rec dynamics without consistent input form the outside
			out_fire = super(AutoAssociation, self).getOutputLinearthreshold(input_pattern=input_pattern, discrete = discrete)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = super(AutoAssociation, self).getOutputLinearthreshold(input_pattern=out_fire, discrete = discrete)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
			return out_fire
	
	def getOutputId(self,  external_activity = None, input_pattern=None):
		
		
		def calcOutput(internal_pattern = None, external_activity = None):
			
			internal_activity = self.calcActivity(input_pattern=internal_pattern)
			normalize(internal_activity)
		
			activity = internal_activity*self.internal_force  + external_activity*self.external_force

			return activity

		

		if external_activity != None: 
			normalize(external_activity)

			out_fire = calcOutput(internal_pattern = input_pattern, external_activity = external_activity)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = calcOutput(internal_pattern =out_fire, external_activity = external_activity)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
				if c == self.cycles -1:
					print 'all ' +str(c) +' cycles used'
			return out_fire
			
		else: #rec dynamics without consistent input form the outside
			out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=input_pattern, discrete = discrete)
			for c in range(self.cycles):
				out_old = np.copy(out_fire)
				out_fire = super(AutoAssociation, self).getOutputWTA(input_pattern=out_fire, discrete = discrete)
				if (out_old == out_fire).all():
					print 'stop after ' + str(c) +' cycles'
					break
			return out_fire
	
		
		return activity
		
	
	#########Recall Function###########
	#As in Network; Now with additional external activity possible 
	def recall(self, input_pattern = None, external_activity = None, key = '', first = None):
		if first == None:
			first = input_pattern.shape[-2]
		if first >= 0:
			input_pattern = input_pattern[:,:,:first]
			self.noisy_output = self.getOutput(self,input_pattern = input_pattern, external_activity = external_activity[:,:,:first])
			self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,:first], patterns_2 = self.noisy_output)
			self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)
		else:
			input_pattern = input_pattern[:,:,first:]
			self.noisy_output = self.getOutput(self,input_pattern = input_pattern, external_activity = external_activity[:,:,first:])
			self.Cor['StoredRecalled'+key] = Corelations(patterns_1 = self.output_stored[:,first:], patterns_2 = self.noisy_output)
			self.Cor['RecalledRecalled'+key] = Corelations(patterns_1 =self.noisy_output[:,0], patterns_2 = self.noisy_output)
		
class OneShoot(Network):

	'''
	
	Network where input first triggers output with existing weights and then weights are adjusted. Different to Association networks, where a pair of patterns is associated. 
	'''


	def __init__(self, **kwargs):
		

		super(OneShoot, self).__init__(**kwargs)
		self.co_factor = 1.0/self.cells # needed for neural gas; it decrases learning amount through time
	#one shoot learning methods
	
	def learnFiringDependent(self,input_pattern = None, output_pattern = None, cofactor_increase = None): # one shoot learning of patterns; hebbian association with input and outputfiring; outputpattern is opiional
		if output_pattern ==None:
			output_pattern = self.getOutput(self,input_pattern = input_pattern)
		self.weights += self.learnrate * np.einsum('j,i,ij->ij', input_pattern*1, output_pattern*1, self.connection)
		normalize(self.weights)
		
	def learnActivityDependent(self,input_pattern =None, output_pattern = None, cofactor_increase = 0): # one shoot version, here learning rate is dependent on output activity not firing; output_pattern is not needed
		activity = self.calcActivity(input_pattern = input_pattern)
		activity[np.argsort(activity)[:-self.number_winner]]=0
		self.hebbianLearning(input_pattern = input_pattern, output_pattern = activity)
		normalize(self.weights)
		
	def learnSiTreves(self,input_pattern =None, output_pattern = None, cofactor_increase = None): 

		#input_mean = np.sum(input_pattern)*1./input_pattern.shape[0]
		
	
		input_mean = (np.dot(self.connection, input_pattern)*1./np.sum(self.connection, axis = -1)).repeat(input_pattern.shape[0]).reshape(self.connection.shape[0], input_pattern.shape[0])
		input_p = np.tile(input_pattern, (self.connection.shape[0],1)) -input_mean
		self.weights += self.learnrate * np.einsum('ij,i,ij->ij', input_p, output_pattern*1, self.connection)
		self.weights[self.weights< 0] = 0
		normalize(self.weights)
		
		
	def learnOneShootAllPattern(self,input_pattern = None, method = None, key = 'StoredStored', first = None, store_output = 1): # one shoot learning of all patterns using giving method; output activity during learning is stored
		
		print 'one shoot learning all patterns'
		normalize(self.weights)
		
		if store_output:
			output = np.zeros([input_pattern.shape[0],input_pattern.shape[1], self.cells])
			if self.learnrate != 0:
				for env in range(input_pattern.shape[0]):
					for p in range(input_pattern.shape[1]):
						output[env,p] = self.getOutput(self,input_pattern = input_pattern[env, p], env = env)
						method(self,input_pattern = input_pattern[env,p], output_pattern = output[env,p])
			else:
				print 'learn rate one shoot = 0'
				output = self.getOutput(self,input_pattern = input_pattern)
			self.output_stored = output
			self.input_stored = input_pattern
			if first == None:
				first = input_pattern.shape[-2]
			if first >= 0:
				self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,:first])
			else:
				self.Cor[key] = Corelations(patterns_1 = self.output_stored[:,first:])
				
		else:
			for env in range(input_pattern.shape[0]):
				for p in range(input_pattern.shape[1]):
					method(self,input_pattern = input_pattern[env,p], output_pattern = self.getOutput(self,input_pattern = input_pattern[env, p], env = env))

	def learnAccordingToRank(self, input_pattern = None, output_pattern = None, cofactor_increase = 0): # outputpattern not needed
		
		activity = self.calcActivity(input_pattern = input_pattern)
		activity_sort = np.argsort(activity)
		learnrate = calcExp(-self.co_factor*(np.arange(self.cells)))[activity_sort]
		self.weights += np.outer(learnrate, input_pattern) *self.connection
		normalize(self.weights)
		self.co_factor += cofactor_increase
				

class Incremental(OneShoot):

	
	'''
	
	OneShoot Network that learns input over several epochs 
	'''
	
	def __init__(self, input_cells=None, cells=None, number_winner=None, connectivity = None, learnrate= None, subtract_input_mean = None, subtract_output_mean = None,  initMethod = None, sparsity = None, storage_mode=None, actFunction = None, active_in_env = None, n_e = None,weight_mean = None, weight_sigma = None):
		

		OneShoot.__init__(self, input_cells = input_cells, cells = cells, number_winner = number_winner, connectivity = connectivity, learnrate = learnrate, subtract_input_mean = subtract_input_mean, subtract_output_mean = subtract_output_mean, initMethod = initMethod, sparsity = sparsity, actFunction = actFunction,active_in_env = active_in_env, n_e = n_e,weight_mean = weight_mean, weight_sigma = weight_sigma)
		self.storage_mode = storage_mode
		normalize(self.weights)
		
	
	def learnIncremental(self, input_pattern= None, input_pattern_to_store=None, no_times = None, number_to_store = None, method = None, cofactor_increase = 0, env = 0 ): # learns enviroment and determines the pattern to store in the input and its output firing either during learning (online) or after learning (offline)
			
			self.input_stored= np.zeros([self.n_e, number_to_store, self.input_cells])
			if input_pattern_to_store == None:
				self.input_stored[env] = input_pattern[env][np.array(random.sample(range(input_pattern.shape[1]), number_to_store))] # the inputs that are considered as stored
			else:
				self.input_stored[env] = input_pattern_to_store[env]
			self.output_stored = np.zeros([self.n_e,number_to_store, self.weights.shape[0]]) #the stored outputs; i.e. the activation of the stored inputs, at the time when the input is stored
			if self.storage_mode == 'online':
				print 'online'
				time_to_store = np.sort(random.sample(range(no_times), int((0.0+self.number_to_store))))# when pattern is stored; here randomly during learning phase
			else:
				time_to_store = [no_times+1] #output to store is computed after learning

			stored =0 #number of pattern stored yet
			for t in range(no_times):
				if t != time_to_store[stored]:
					method(self, input_pattern = input_pattern[env][np.random.randint(input_pattern.shape[1])], cofactor_increase = cofactor_increase) # learns randomly choosen input_pattern according to method
				else:
					self.output_stored[env][stored] = self.getOutput(self,input_pattern = self.input_stored[env][stored], env = env)
					method(self, input_pattern = self.input_stored[env][stored], output_pattern = self.output_stored[env][stored], cofactor_increase = cofactor_increase) # learns input_pattern to store according to method
					stored +=1
			if self.storage_mode == 'offline':
				self.output_stored[env] = self.getOutput(self,input_pattern = self.input_stored, env = env)
							
##################Paramter####################################################################
class Parameter():
	

	'''
	
	Parameter Class. If some paramters are not given in the Simulation, usually paramters of this class are used instead of raising an error.
	'''
	
	#Parameter
	no_pattern = 400
	number_to_store = 252
	n_e=1
	first  = None
	cells = dict(Ec = 1100, Dg = 12000, Ca3 =2500, Ca1 = 4200)#cell numbers of each region
	sparsity = dict(Ec = 0.35, Dg = 0.005, Ca3 = 0.032, Ca1 = 0.09)#activity level of each region (if WTA network)
	number_winner = dict(Ec = int(cells['Ec']*sparsity['Ec']), Dg = int(cells['Dg']*sparsity['Dg']), Ca3 = int(cells['Ca3']*sparsity['Ca3']), Ca1 = int(cells['Ca1']*sparsity['Ca1']) ) 
	connectivity = dict(Ec_Dg = 0.32, Dg_Ca3 = 0.0006, Ca3_Ec = 0.32, Ec_Ca3 =0.32, Ca3_Ca3 = 0.24, Ca3_Ca1 = 0.32, Ca1_Ec = 0.32, Ec_Ca1 = 0.32, Ca1_Sub = 0.32, Sub_Ec = 0.32, Ec_Sub = 0.32)# probability given cell is connected to given input cell
	learnrate = dict(Ec_Dg = 0.5, Dg_Ca3 = None, Ca3_Ec = 1, Ec_Ca3 =1, Ca3_Ca3=1, Ca3_Ca1 = 0.5, Ec_Ca1 = 1, Ca1_Ec = 1, Ca1_Sub = 1, Sub_Ec = 1, Ec_Sub = 0)
	initMethod = dict(Ec_Dg = Network.makeWeightsUniformDistributed, Dg_Ca3 = Network.makeWeightsUniformDistributed, Ec_Ca3 =Network.makeWeightsZero, Ca3_Ec =Network.makeWeightsZero, Ca3_Ca3 = Network.makeWeightsZero, Ec_Ca1 = Network.makeWeightsNormalDistributed, Ca3_Ca1 = Network.makeWeightsZero, Ca1_Ec =Network.makeWeightsZero)
	
	#initMethod = dict(Ec_Dg = Network.makeWeightsNormalDistributed, Dg_Ca3 = Network.makeWeightsNormalDistributed, Ec_Ca3 =Network.makeWeightsZero, Ca3_Ec =Network.makeWeightsZero, Ca3_Ca3 = Network.makeWeightsZero, Ec_Ca1 = Network.makeWeightsNormalDistributed, Ca3_Ca1 = Network.makeWeightsZero, Ca1_Ec =Network.makeWeightsZero)
	
	actFunctionsRegions = dict(Ec_Dg = Network.getOutputWTALinear, Dg_Ca3 = Network.getOutputWTA, Ca3_Ec = Network.getOutputWTALinear, Ec_Ca3 = Network.getOutputWTA, Ca3_Ca3 = AutoAssociation.getOutputWTA, Ca3_Ca1= Network.getOutputWTALinear, Ca1_Ec = Network.getOutputWTALinear, Ec_Ca1 = Network.getOutputWTALinear)
	
	rolls = 0
	if rolls:
		cells = dict(Ec = 600, Dg = 1000, Ca3 =1000, Ca1 = 1000)#cell numbers of each region
		sparsity = dict(Ec = 0.05, Dg = 0.05, Ca3 = 0.05, Ca1 = 0.01, Sub = 0.097)#activity level of each region (if WTA network)
		number_winner = dict(Ec = int(cells['Ec']*sparsity['Ec']), Dg = int(cells['Dg']*sparsity['Dg']), Ca3 = int(cells['Ca3']*sparsity['Ca3']), Ca1 = int(cells['Ca1']*sparsity['Ca1']) )
		connectivity = dict(Ec_Dg = 60./600, Dg_Ca3 = 4./1000, Ca3_Ec = 60./1000, Ec_Ca3 =120./600., Ca3_Ca3 = 200./1000, Ca3_Ca1 = 200./1000., Ca1_Ec = 1000./1000, Ec_Ca1 = 1, Ca1_Sub = 0.32, Sub_Ec = 0.32, Ec_Sub = 0.32)
		learnrate = dict(Ec_Dg = 0.5, Dg_Ca3 = None, Ca3_Ec = 1, Ec_Ca3 =1, Ca3_Ca3=1, Ca3_Ca1 = 10, Ec_Ca1 = 1, Ca1_Ec = 1, Ca1_Sub = 1, Sub_Ec = 1, Ec_Sub = 0)
		initMethod = dict(Ec_Dg = Network.makeWeightsUniformDistributed, Dg_Ca3 = Network.makeWeightsUniformDistributed, Ec_Ca3 =Network.makeWeightsZero, Ca3_Ec =Network.makeWeightsZero, Ca3_Ca3 = Network.makeWeightsZero, Ec_Ca1 = Network.makeWeightsNormalDistributed, Ca3_Ca1 = Network.makeWeightsZero, Ca1_Ec =Network.makeWeightsZero)
		actFunctionsRegions = dict(Ec_Dg = Network.getOutputLinearthreshold, Dg_Ca3 = Network.getOutputWTA, Ec_Ca3 = Network.getOutputWTA, Ca3_Ca3 = AutoAssociation.getOutputWTA, Ca3_Ca1= Network.getOutputLinearthreshold, Ca1_Ec = Network.getOutputWTA, Ec_Ca1 = Network.getOutputWTALinear, Ca3_Ec = None )
	
	
	
	active_in_env = dict(Ec = int(cells['Ec']), Dg =  int( cells['Dg']), Ca3 =int(cells['Ca3']), Ca1 = int(cells['Ca1']))		
	active_in_env = dict(Ec = None, Dg =  None, Ca3 =None, Ca1 = None)	
	radius_ca3_field = 0.2111
	radius_ca1_field = 0.2523
	
	
	active_env = 0
	if active_env: #only proportion of cells active in each env
		active_in_env = dict(Ec = cells['Ec'], Dg =  int(0.035 * cells['Dg']), Ca3 =int(0.2325 * cells['Ca3']), Ca1 = int(0.4625 * cells['Ca1'])) #number cells active in each
		sparsity = dict(Ec = 0.35, Dg = 0.0064/active_in_env['Dg']*cells['Dg'], Ca3 = 0.03255/active_in_env['Ca3']*cells['Ca3'], Ca1 = 0.0925/active_in_env['Ca1']*cells['Ca1'])#activity level given cell is active in env


	
	
	incrementalLearnMethod = OneShoot.learnFiringDependent#learns either activtiy dpendent, or firing dependent
	incremental_storage_mode = None # learns pattern either 'online', i.e. during learning statisitcs or 'offline', i.e. after learning input statistics
	no_incremental_times = None
	
	#rec dynamics
	external_force = 1
	internal_force = 3
	cycles = 15
	
	noise_levels = np.arange(0,cells['Ec']+1, int(cells['Ec']/15))
	noise_levels_ca3 =[0]
	noise_levels_ca1 =[0]

	subtract_input_mean = 0
	subtract_output_mean = 1 #=1 for all Ca3_Ca3 Autoassociations

	def __init__():
		pass
################################################################################################		



################################################ Analysis Classes #######################################################
class Corelations(object):
	
	'''
	
	computes pearson correlations <a,b>/ab; where a is element of patterns_1 and b of patterns_2: Copmutes the correlation pairwise for patterns in each noise level. Different Environments are lumped together and treated as one. 
	
	:param patterns_1: original stored pattern
	:type patterns_1: array of dimension 2 (pattern, cellfire) or 3 (env, pattern, cellfire)
	:param patterns_2:  noisy_original pattern; if not given autocorrelation with patterns_1 is computed
	:type patterns_2: array of dimension 3 (noise, pattern, cellfire) or 4 (env, noise, pattern, cellfire)
	:param in_columns: If True, transpoes data given as (fire, pattern) or (noise, fire, pattern) into right dimension order.
	:type in_colums: Bool
		 
	 '''
		
		
	
	def __init__(self, patterns_1=None, patterns_2=None, in_columns = False, env = 1):
		
		
		if patterns_2 != None:
			print 'patterns_2.shape Correlations init'
			print patterns_2.shape
		self.orig_vs_orig = None #
		self.orig_vs_other = None #average corelation of original pattern and noisy version of the original; array has length len(noise_levels)
		self.over_orig_vs_orig = None#overlaps
		self.over_orig_vs_other = None
		if len(patterns_1.shape) == 3:
			self.patterns_1= patterns_1.reshape(patterns_1.shape[0]*patterns_1.shape[1], patterns_1.shape[-1]) # env, pat, cell into env*pat,cell
		else:
			self.patterns_1= patterns_1
		if len(patterns_1.shape) == 1: #pattern
			self.patterns_1 = patterns_1.reshape(patterns_1.shape[0], 1) #pattern, cell

		if patterns_2 == None:
			self.patterns_2 = np.tile(self.patterns_1, (2,1,1))
			self.one_patterns = True
		else:
			self.one_patterns = False
			if len(patterns_2.shape) == 4:
				self.patterns_2= np.swapaxes(patterns_2, 0,1).reshape(patterns_2.shape[1], patterns_2.shape[0]*patterns_2.shape[2], patterns_2.shape[-1])
			if len(patterns_2.shape) == 3:#env, pattern, cell
				if env:#env, pattern, cell
					self.patterns_2 = np.tile(patterns_2.reshape(patterns_2.shape[0]*patterns_2.shape[1], patterns_2.shape[-1]), (1,1,1)) #2, env*patt, cell
				else:#noise, pattern, cell
					self.patterns_2 = patterns_2
			if len(patterns_2.shape) == 2:#noise, pattern
				self.patterns_2 = patterns_2.reshape(patterns_2.shape[0],patterns_2.shape[1],1)
	

		if in_columns: #if data is given as columns, transpose them into rows !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! not tested with env framework
			self.patterns_1 = np.transpose(self.patterns_1)
			self.patterns_2 = np.transpose(self.patterns_2, (0,2,1))
			
			


		
		self.distance_matrix = None # matrix that has at (i,j) the euclidean distance between pattern i and j as entry
		self.corelations = None # Corelation Matrix; (abc) is corelation of pattern b and c with noise a
		self.calculated_cor = False #indicates if self.corelation is yet computed, this way double computation is avoided
		self.overlaps = None# Matrix of overlaps;(abc) is overlap of pattern b and c with noise a
		self.calculated_over = False #indicates if self.overlaps is yet computed, this way double computation is avoided
		
		self.covariance = None
		self.covariance_ev = None
		self.covariance_ew = None
		self.number_relevant = None
		self.projection_matrix = None
		
		
		self.distances = None
		self.all_distances = None
		self.p_d =None
		self.p_fire_given_distance = None
		
		self.fire_fire_distances = []
		self.fire_distances = []
		self.silent_distances = []
		self.fire_silent_distances = []
		self.p_d_fire= []
		self.p_d_silent = []
		self.p_fire_given_fire_and_distance = []
		self.p_fire_given_silent_and_distance = []
		self.p_fire_given_fire_and_distance_weighted_distance = []
		self.p_fire_given_silent_and_distance_weighted_distance = []
		for i in range(self.patterns_1.shape[-1]):
			self.fire_fire_distances +=[None]
			self.fire_distances +=[None]
			self.silent_distances +=[None]
			self.fire_silent_distances +=[None]
			self.p_d_fire +=[None]
			self.p_d_silent +=[None]
			self.p_fire_given_fire_and_distance +=[None]
			self.p_fire_given_silent_and_distance +=[None]
			self.p_fire_given_fire_and_distance_weighted_distance +=[None]
			self.p_fire_given_silent_and_distance_weighted_distance +=[None]
		print 'self.patterns_2.shape Correlations init'
		print self.patterns_2.shape
		
	######### calc Methods ####################
	def calcCor(self, noise_level = None, in_steps = False):
		'''
		
		computes the pearson corelation matrix self.corelations; self.corelation[abc] = corleation of pattern b and  pattern c at noise_level a =  <b - <b>, c - <c>)/||b||*||c|
		
		:param noise_level: noise_level at which the matrix is computed; if None then it is computed for all levels
		:type noise_level: int in the intervall (0, len(noise_levels))
		:param in_steps: If True and noise_level != None, corrleations are calculated individually for each pair. This is necessary only for huge data sets, when memory demands are too high.
		:type in_steps: Bool
		'''
		if noise_level == None:
			if not self.calculated_cor:
				if self.patterns_1.shape[-1] != 1:
					mean_subtracted_1 = self.patterns_1 - (np.einsum('pi->p', self.patterns_1*1)/(self.patterns_1.shape[-1]+0.0)).repeat(self.patterns_1.shape[-1]).reshape(self.patterns_1.shape)
					mean_subtracted_2 = self.patterns_2 - np.einsum('npi->np', self.patterns_2*1).repeat(self.patterns_2.shape[-1]).reshape(self.patterns_2.shape)/(self.patterns_2.shape[-1]+0.0)
					self.p1_norm_inverse = 1./np.sqrt(np.einsum('...ai, ...ai ->...a', mean_subtracted_1, mean_subtracted_1)) #input_norm_inverse[b]= 1/norm(b)
					self.p2_norm_inverse = 1./np.sqrt(np.einsum('...bi, ...bi ->...b', mean_subtracted_2, mean_subtracted_2))#p2_norm_inverse[a,b]= 1/norm(b(a)) ; b at noise level a
					self.corelations = np.einsum('pi, nqi, p, nq-> npq', mean_subtracted_1, mean_subtracted_2, self.p1_norm_inverse, self.p2_norm_inverse )#cor[abc] = at noise a over of patt b (noise =0) and c (noise = a)= <b - <b>, c - <c>)/||b||*||c||
				else:
					mean_subtracted_1 = self.patterns_1  - (np.einsum('pi->i', self.patterns_1*1)/(self.patterns_1.shape[-2]+0.0)).repeat(self.patterns_1.shape[-2]).reshape(self.patterns_1.shape)
					mean_subtracted_2 = self.patterns_2 - np.einsum('npi->ni', self.patterns_2*1).repeat(self.patterns_2.shape[-2]).reshape(self.patterns_2.shape)/(self.patterns_2.shape[-2]+0.0)
					self.p1_norm_inverse = 1./np.sqrt(np.einsum('ai, ai -> ', mean_subtracted_1, mean_subtracted_1)) #input_norm_inverse[b]= 1/norm(b)
					self.p2_norm_inverse = 1./np.sqrt(np.einsum('nbi, nbi -> n', mean_subtracted_2, mean_subtracted_2))#p2_norm_inverse[a,b]= 1/norm(b(a)) ; b at noise level a
					self.corelations = np.einsum('pi, nqi, n-> n', mean_subtracted_1, mean_subtracted_2, self.p2_norm_inverse ).reshape(1, self.patterns_2.shape[0]) * self.p1_norm_inverse #cor[abc] = at noise a over of patt b (noise =0) and c (noise = a)= <b - <b>, c - <c>)/||b||*||c||
				

				self.calculated_cor = True
		else:
			if self.corelations == None:
				self.corelations = np.zeros([self.patterns_2.shape[-3],self.patterns_2.shape[-2], self.patterns_2.shape[-2]])
				self.p2_norm_inverse = np.zeros([self.patterns_2.shape[-3], self.patterns_2.shape[-2]])
			if (self.corelations[noise_level] == 0).all():
				mean_subtracted_1 = self.patterns_1 - (np.einsum('pi->p', self.patterns_1*1)/(self.patterns_1.shape[-1]+0.0)).repeat(self.patterns_1.shape[-1]).reshape(self.patterns_1.shape)
				mean_subtracted_2 = self.patterns_2[noise_level] - np.einsum('pi->p', self.patterns_2[noise_level]*1).repeat(self.patterns_2.shape[-1]).reshape(self.patterns_2[0].shape)/(self.patterns_2.shape[-1]+0.0)
				self.p1_norm_inverse = 1./np.sqrt(np.einsum('...ai, ...ai ->...a', mean_subtracted_1, mean_subtracted_1)) #input_norm_inverse[b]= 1/norm(b)
				self.p2_norm_inverse[noise_level] = 1./np.sqrt(np.einsum('bi, bi ->b', mean_subtracted_2, mean_subtracted_2))#p2_norm_inverse[a,b]= 1/norm(b(a)) ; b at noise level a
				if in_steps:
					for i in range(mean_subtracted_1.shape[0]):
						for j in range(mean_subtracted_2.shape[0]):
							self.corelations[noise_level][i][j] = np.dot(mean_subtracted_1[i],mean_subtracted_2[j])* self.p1_norm_inverse[i]* self.p2_norm_inverse[noise_level][j]

				else:
					print self.patterns_2.shape
					print self.patterns_1.shape
					print self.corelations[noise_level].shape
					print mean_subtracted_1.shape
					print mean_subtracted_2.shape
					print self.p1_norm_inverse.shape
					print self.p2_norm_inverse.shape
					self.corelations[noise_level] = np.einsum('pi, qi, p, q-> pq', mean_subtracted_1, mean_subtracted_2, self.p1_norm_inverse, self.p2_norm_inverse[noise_level] )# cor[abc] = at noise a over of patt b (noise =0) and c (noise = a)= <b(0), 
	
	def calcOverlaps(self, noise_level = None):
		'''
		
		computes the overlap matrix self.overlaps at noise_level; overlap(p,q) = p^T q (inner product); self.overlaps[abc] = overlap of pattern b and  pattern c at noise_level a
	
		:param noise_level: noise_level at which the matrix is computed; if None then it is computd for all levels
		:type noise_level: int in the intervall (0, len(noise_levels))
		'''
		
		if noise_level == None:
			if not self.calculated_over:
				self.overlaps = np.einsum('bi, aci->abc', self.patterns_1*1, self.patterns_2*1)#noise a, stored_pattern b, recalled c
				self.calculated_over = True
		else:
			if self.overlaps == None:
				self.overlaps = np.zeros([self.patterns_2.shape[-3],self.patterns_2.shape[-2], self.patterns_2.shape[-2]])
			if (self.overlaps[noise_level] == 0).all():
				self.overlaps[noise_level] = np.einsum('bi, ci->bc', self.patterns_1*1, self.patterns_2[noise_level]*1)
	
	def calcOverlapsNoMean(self, noise_level = None):
		'''
		
		computes the overlap matrix self.overlapsNoMean at noise_level; overlapNoMean(p,q) = (p)^T * (q-q_0) where q_0 is the mean of q
	
		:param noise_level: noise_level at which the matrix is computed; if None then it is computd for all levels
		:type noise_level: int in the intervall (0, len(noise_levels))
		'''
		if noise_level == None:
			#mean_subtracted_2 = self.patterns_2 - np.einsum('npi->np', self.patterns_2*1).repeat(self.patterns_2.shape[-1]).reshape(self.patterns_2.shape)/(self.patterns_2.shape[-1]+0.0)
			mean_subtracted_2 = self.patterns_2 - np.einsum('npi->ni', self.patterns_2*1).repeat(self.patterns_2.shape[-2]).reshape(self.patterns_2.shape)/(self.patterns_2.shape[-2]+0.0)
			self.overlapsNoMean = np.einsum('bi, aci->abc', self.patterns_1*1, mean_subtracted_2*1)#noise a, stored_pattern b, recalled c
		else:
			self.overlapsNoMean = np.zeros([self.patterns_2.shape[-3],self.patterns_2.shape[-2], self.patterns_2.shape[-2]])
			#mean_subtracted_2 = self.patterns_2[noise_level] - np.einsum('pi->p', self.patterns_2[noise_level]*1).repeat(self.patterns_2.shape[-1]).reshape(self.patterns_2[0].shape)/(self.patterns_2.shape[-1]+0.0)
			mean_subtracted_2 = self.patterns_2[noise_level] - np.einsum('pi->i', self.patterns_2[noise_level]*1).repeat(self.patterns_2.shape[-2]).reshape(self.patterns_2[0].shape)/(self.patterns_2.shape[-2]+0.0)
			self.overlapsNoMean[noise_level] = np.einsum('bi, ci->bc', self.patterns_1*1, mean_subtracted_2*1)
	
	def calcDistanceMatrix(self, locations, round_it = 'fix', bins = None):
		
		'''
		
		calculates Distance Matrix given locations. Entry (i,j) is the eucledean distance between location i and j. 
		
		:param round_it:I f round_it != 0, distances are rounded. if round_it is an number, this number indicates the number of digits it is rounded to. If round it = 'fix', distances are rounded to next bin, specified by bins.
		:type round_it: int or 'fix'
		:param: bins: if round_it equals 'fix', distances are rounded up to their next bin in bins. If none, bins = np.linspace(0,1.5, 20)
		:type bins: np.array
		'''
		

		self.distance_matrix = np.sqrt(np.sum((np.tile(locations, (1,self.patterns_1.shape[0])).reshape(self.patterns_1.shape[0], self.patterns_1.shape[0],2) - np.tile(locations, (self.patterns_1.shape[0],1)).reshape(self.patterns_1.shape[0], self.patterns_1.shape[0],2))**2, axis = -1)) # (i,j) = eucl distance loc_i loc_j 
		if round_it != 0:
			print 'calc Dis matrix Cor with rounding'
			if round_it =='fix':
				print 'use fix bins for distances'
				if bins == None:
					print 'make fix bins'
					bins = np.linspace(0,1.5, 20)
				for i in range(1,bins.shape[0]): # distance are divided into evenly distributed 50 bins; first bin only contains 0 distance 
						self.distance_matrix[(self.distance_matrix <= bins[i]) - (self.distance_matrix <= bins[i-1])] = bins[i] #every entry that is <= bins[i] but without beeing <= bins[i-1] becomes bins[i]

			else:
				self.distance_matrix = np.round(self.distance_matrix, round_it)
			
		
	#########################gets with Corelations######################
	def getCor(self, at_noise=None): #get correlations at_noise
		'''
		
		get correlations at noise level at_noise
		'''
		self.calcCor(noise_level = at_noise)
		return self.corelations[at_noise]
		
	def getCorOrigOrig(self, at_noise=None):
		'''
		
		get diagonal of self.corelation; These are the correlations of original patterns with reconstructed version; if at_noise = None, gets all noise_levels and gets array of dimension two (noise, diagonal)
		'''
		self.calcCor()
		if at_noise == None:
			corelations = np.copy(self.corelations)
		else:
			corelations = np.copy(self.corelations[at_noise])
		return np.diagonal(corelations, 0, -1,-2)
		
	def getCorOrigOther(self, at_noise=None, subtract_orig_orig = 1, pattern = None, column = False): #
		
		'''
		
		get correlations entries of self.corealtions at_noise_level; if at_noise = None, gets all noise_levels and gets array of dimension two (noise, entries). If self.one_pattern, only entries (i,j) with i<=j are returned, since the entry (i,j) is equal to the entry (j,i) here.
	
		:param pattern: if pattern != None, it gets only the correation of this pattern with all others.
		:type pattern: integer indicating index of pattern
		:param column: if pattern != None, it determines whether self.correlations[at_noise, pattern] (row) or self.correlations[at_noise, :, pattern] (column) is returned
		:type column: bool
		:param subtract_orig_orig: If True, only correlation(p_j, p_i) with i!= j are returned; i.e. not the entries at the diagonal
		:type subtract_orig_orig: Bool
		 orig_vs_other as array; if noise = None, we have additional noise dimension in return, if one_patterns this is triangle entries of self.corelations, else it is all entries; subtract_orig_orig determines, if diagonal is included (orig_orig)
		'''
		self.calcCor(noise_level = at_noise)
		if pattern == None: # all patterns
			if at_noise == None: #
				corelations = np.copy(self.corelations)
				if self.one_patterns:#triangle is sufficent since cor matrix is symetric, here we use lower triangle
					if subtract_orig_orig:#without diag
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						return_value = corelations[ind0, tril0, tril1]
					else: #with diagonal
						tril0 = np.tril_indices(corelations.shape[-1], 0)[0]
						tril1 = np.tril_indices(corelations.shape[-1], 0)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
					
						return_value = corelations[ind0, tril0, tril1]
				else: #lower and upper differ and must be used both
					if subtract_orig_orig:
						#lower triangel
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_lower = corelations[ind0, tril0, tril1]
					
						#upper triangel
						tril0 = np.triu_indices(corelations.shape[-1], 1)[0]
						tril1 = np.triu_indices(corelations.shape[-1], 1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_upper = corelations[ind0, tril0, tril1]
					
						return_value = np.zeros([corelations.shape[0], corelations.shape[-1]**2 - corelations.shape[-1]])
						for i in range(corelations.shape[0]):
							return_value[i,:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower[i]
							return_value[i,(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper[i]	
					else:
						for i in range(corelations.shape[0]):
							return_value[i] = np.ravel(corelations[i])
			else:
				corelations = np.copy(self.corelations[at_noise])
				if self.one_patterns:
					if subtract_orig_orig:
						return_value = corelations[np.tril_indices(corelations.shape[-1], -1)]
					else:
						return_value = corelations[np.tril_indices(corelations.shape[-1], 0)]
				else:
					if subtract_orig_orig:
						cor_lower = corelations[np.tril_indices(corelations.shape[-1], -1)]
						cor_upper = corelations[np.triu_indices(corelations.shape[-1], 1)]
						return_value = np.zeros(corelations.shape[-1]**2 -corelations.shape[-1])
						return_value[:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower
						return_value[(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper
					else:
						return_value = np.ravel(corelations)
		else: #one pattern
			if at_noise != None:
				if column:
					return_value = self.corelations[at_noise][:,pattern]# column pattern, i.e. StoredRecalled, REcalled pattern used
				else:
					return_value = self.corelations[at_noise][pattern]# Stored pattern used
			else:
				return_value = np.zeros([self.corelations.shape[0],self.corelations.shape[1]])
				for i in range(self.corelations.shape[0] ):
					if column:
						return_value[i] = np.ravel(self.corelations[i,:,pattern])# column pattern, i.e. StoredRecalled, REcalled pattern used
					else:
						return_value[i] = np.ravel(self.corelations[i,pattern])# Stored pattern used	
		return return_value
	
	
	def getOrigOtherDistances(self, locations = None, subtract_orig_orig = 1, pattern = None): #
		
		'''
		
		get distances of corelations ,ie. the return array of CorOrigOther()
		'''
		distances = np.copy(self.getDistanceMatrix(round_it = 0, locations = locations))
		if pattern == None: # all patterns
			if self.one_patterns:
				if subtract_orig_orig:
					return_value = distances[np.tril_indices(distances.shape[-1], -1)]
				else:
					return_value = distances[np.tril_indices(distances.shape[-1], 0)]
			else:
				if subtract_orig_orig:
					cor_lower = distances[np.tril_indices(distances.shape[-1], -1)]
					cor_upper = distances[np.triu_indices(distances.shape[-1], 1)]
					return_value = np.zeros(distances.shape[-1]**2 -distances.shape[-1])
					return_value[:(distances.shape[-1]**2 -distances.shape[-1])/2] = cor_lower
					return_value[(distances.shape[-1]**2 -distances.shape[-1])/2:] = cor_upper
				else:
					return_value = np.ravel(distances)
		else: #one pattern
			return_value = distances[pattern]# Stored pattern used
		#self.distance_matrix = None
		return return_value
	
	
	
	
	def getOrigVsOrig(self):
		
		'''
		
		returns the average Corelation of the diagonal in self.correlations. If at_noise = None, at all noise_levels. Then return has dimension (noise_levels)
		'''
		if self.orig_vs_orig == None:
			if self.patterns_1.shape[-1] == 1:
				self.calcCor()
				self.orig_vs_orig = self.corelations[0]
			else:
				self.orig_vs_orig = np.sum(self.getCorOrigOrig(), axis = -1)/(self.patterns_1.shape[0]+0.0)
		return self.orig_vs_orig
		
	def getOrigVsOther(self, at_noise = None):  #returns average Corelation over all patterns, if at_noise = None, we have additional dimension over noise in Return
		'''
		
		returns the average Corelation of the entries away from the diagonal in self.correlations. If at_noise = None, at all noise_levels. Then return has dimension (noise_levels)
		'''
		if at_noise == None:
			if self.orig_vs_other == None:
				divide = self.patterns_1.shape[0]**2 - self.patterns_1.shape[0] +0.0
				if self.one_patterns:
					divide /= 2
				self.orig_vs_other = np.sum(self.getCorOrigOther(subtract_orig_orig = 1, at_noise = at_noise), axis = -1)/divide
			return self.orig_vs_other
			
		else:
			divide = self.patterns_1.shape[0]**2 - self.patterns_1.shape[0] +0.0
			if self.one_patterns:
				divide /= 2
			return np.sum(self.getCorOrigOther(subtract_orig_orig = 1, at_noise = at_noise), axis = -1)/divide

	
	###########gets with Overlaps###############
	#Same as gets with Correlation but uses now self.overlaps instead of self.correlation
	def getOverOrigOrig(self, at_noise=None):
		
		self.calcOverlaps(noise_level = at_noise)
		if at_noise == None:
			overlaps = np.copy(self.overlaps)
		else:
			overlaps = np.copy(self.overlaps[at_noise])
		return np.diagonal(overlaps, 0, -1,-2)
	
	def getOverOrigOther(self, at_noise=None, subtract_orig_orig = 1, pattern = None, column = False):
		self.calcOverlaps(noise_level = at_noise)
		if pattern == None: # all patterns
			if at_noise == None: #
				corelations = np.copy(self.overlaps)
				if self.one_patterns:#triangle is sufficent since cor matrix is symetric, here we use lower triangle
					if subtract_orig_orig:#without diag
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						return_value = corelations[ind0, tril0, tril1]
					else: #with diagonal
						tril0 = np.tril_indices(corelations.shape[-1], 0)[0]
						tril1 = np.tril_indices(corelations.shape[-1], 0)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
					
						return_value = corelations[ind0, tril0, tril1]
				else: #lower and upper differ and must be used both
					if subtract_orig_orig:
						#lower triangel
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_lower = corelations[ind0, tril0, tril1]
					
						#upper triangel
						tril0 = np.triu_indices(corelations.shape[-1], 1)[0]
						tril1 = np.triu_indices(corelations.shape[-1], 1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_upper = corelations[ind0, tril0, tril1]
					
						return_value = np.zeros([corelations.shape[0], corelations.shape[-1]**2 - corelations.shape[-1]])
						for i in range(corelations.shape[0]):
							return_value[i,:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower[i]
							return_value[i,(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper[i]	
					else:
						for i in range(corelations.shape[0]):
							return_value[i] = np.ravel(corelations[i])
			else:
				corelations = np.copy(self.overlaps[at_noise])
				if self.one_patterns:
					if subtract_orig_orig:
						return_value = corelations[np.tril_indices(corelations.shape[-1], -1)]
					else:
						return_value = corelations[np.tril_indices(corelations.shape[-1], 0)]
				else:
					if subtract_orig_orig:
						cor_lower = corelations[np.tril_indices(corelations.shape[-1], -1)]
						cor_upper = corelations[np.triu_indices(corelations.shape[-1], 1)]
						return_value = np.zeros(corelations.shape[-1]**2 -corelations.shape[-1])
						return_value[:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower
						return_value[(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper
					else:
						return_value = np.ravel(corelations)
		else: #one pattern
			if at_noise != None:
				if column:
					return_value = self.overlaps[at_noise][:,pattern]# column pattern, i.e. StoredRecalled, REcalled pattern used
				else:
					return_value = self.overlaps[at_noise][pattern]# Stored pattern used
			else:
				return_value = np.zeros([self.overlaps.shape[0],self.overlaps.shape[1]])
				for i in range(self.overlaps.shape[0] ):
					if column:
						return_value[i] = np.ravel(self.overlaps[i,:,pattern])# column pattern, i.e. StoredRecalled, REcalled pattern used
					else:
						return_value[i] = np.ravel(self.overlaps[i,pattern])# Stored pattern used	
		return return_value
	
	def getOverOrigVsOrig(self, normed = 1): #returns average Corelation over all patterns, dimension over noise
		
		'''
		
		As getOrigVsOrig, nut now with self.overlaps instead of self.correlation. if normed, then retun is divided by no winner to get value between 1 and 0
		'''
		
		if self.over_orig_vs_orig == None:
			self.over_orig_vs_orig = np.sum(self.getOverOrigOrig(), axis = -1)/(self.patterns_1.shape[0]+0.0)
			if normed == 1:
				self.over_orig_vs_orig /= np.sum(self.patterns_1[0]) *1. # divide by no winner to get value between 1 and 0
		return self.over_orig_vs_orig

	def getOverOrigVsOther(self, normed = 1):  #returns average Corelation over all patterns, dimension over noise
		
		'''
		
		As getOrigVsOther, nut now with self.overlaps instead of self.correlation. if normed, then retun is divided by no winner to get value between 1 and 0
		'''
		
		if self.over_orig_vs_other == None:
			divide = self.patterns_1.shape[0]**2 - self.patterns_1.shape[0] +0.0
			if self.one_patterns:
				divide /= 2
			self.over_orig_vs_other = np.sum(self.getOverOrigOther(subtract_orig_orig = 1), axis = -1)/divide
			if normed == 1:
				self.over_orig_vs_other /= np.sum(self.patterns_1[0]) *1. # divide by no winner to get value between 1 and 0
		return self.over_orig_vs_other


	###########gets with OverlapsNoMean###############
	#Same as gets with Correlation but uses now self.overlapsNoMean instead of self.correlation
	def getOverOrigOrigNoMean(self, at_noise=None):
		
		self.calcOverlapsNoMean(noise_level = at_noise)
		if at_noise == None:
			overlaps = np.copy(self.overlapsNoMean)
		else:
			overlaps = np.copy(self.overlapsNoMean[at_noise])
		return np.diagonal(overlaps, 0, -1,-2)
	
	def getOverOrigOtherNoMean(self, at_noise=None, subtract_orig_orig = 1, pattern = None, column = False):
		self.calcOverlapsNoMean(noise_level = at_noise)
		if pattern == None: # all patterns
			if at_noise == None: #
				corelations = np.copy(self.overlapsNoMean)
				if self.one_patterns:#triangle is sufficent since cor matrix is symetric, here we use lower triangle
					if subtract_orig_orig:#without diag
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						return_value = corelations[ind0, tril0, tril1]
					else: #with diagonal
						tril0 = np.tril_indices(corelations.shape[-1], 0)[0]
						tril1 = np.tril_indices(corelations.shape[-1], 0)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
					
						return_value = corelations[ind0, tril0, tril1]
				else: #lower and upper differ and must be used both
					if subtract_orig_orig:
						#lower triangel
						tril0 = np.tril_indices(corelations.shape[-1], -1)[0]
						tril1 = np.tril_indices(corelations.shape[-1], -1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_lower = corelations[ind0, tril0, tril1]
					
						#upper triangel
						tril0 = np.triu_indices(corelations.shape[-1], 1)[0]
						tril1 = np.triu_indices(corelations.shape[-1], 1)[1]
						ind0 = np.arange(corelations.shape[0]).repeat(tril0.shape[0]).reshape(corelations.shape[0], tril0.shape[0])
						cor_upper = corelations[ind0, tril0, tril1]
					
						return_value = np.zeros([corelations.shape[0], corelations.shape[-1]**2 - corelations.shape[-1]])
						for i in range(corelations.shape[0]):
							return_value[i,:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower[i]
							return_value[i,(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper[i]	
					else:
						for i in range(corelations.shape[0]):
							return_value[i] = np.ravel(corelations[i])
			else:
				corelations = np.copy(self.overlapsNoMean[at_noise])
				if self.one_patterns:
					if subtract_orig_orig:
						return_value = corelations[np.tril_indices(corelations.shape[-1], -1)]
					else:
						return_value = corelations[np.tril_indices(corelations.shape[-1], 0)]
				else:
					if subtract_orig_orig:
						cor_lower = corelations[np.tril_indices(corelations.shape[-1], -1)]
						cor_upper = corelations[np.triu_indices(corelations.shape[-1], 1)]
						return_value = np.zeros(corelations.shape[-1]**2 -corelations.shape[-1])
						return_value[:(corelations.shape[-1]**2 -corelations.shape[-1])/2] = cor_lower
						return_value[(corelations.shape[-1]**2 -corelations.shape[-1])/2:] = cor_upper
					else:
						return_value = np.ravel(corelations)
		else: #one pattern
			if at_noise != None:
				if column:
					return_value = self.overlapsNoMean[at_noise][:,pattern]# column pattern, i.e. StoredRecalled, REcalled pattern used
				else:
					return_value = self.overlapsNoMean[at_noise][pattern]# Stored pattern used
			else:
				return_value = np.zeros([self.overlapsNoMean.shape[0],self.overlapsNoMean.shape[1]])
				for i in range(self.overlapsNoMean.shape[0] ):
					if column:
						return_value[i] = np.ravel(self.overlapsNoMean[i,:,pattern])# column pattern, i.e. StoredRecalled, REcalled pattern used
					else:
						return_value[i] = np.ravel(self.overlapsNoMean[i,pattern])# Stored pattern used	
		return return_value
	
	def getOverOrigVsOrigNoMean(self, at_noise = None, normed = 0): #returns average Corelation over all patterns, dimension over noise
		
		'''
		
		As getOrigVsOrig, nut now with self.overlapsNoMean instead of self.correlation. if normed, then retun is divided by no winner to get value between 1 and 0
		'''
		
		over = np.sum(self.getOverOrigOrigNoMean(), axis = -1)/(self.patterns_1.shape[0]+0.0)
		if normed == 1:
			over /= np.sum(self.patterns_1[0]) *1. # divide by no winner to get value between 1 and 0
		if at_noise != None:
			over = over[at_noise]
		return over

	def getOverOrigVsOtherNoMean(self, normed = 0):  #returns average Corelation over all patterns, dimension over noise
		
		'''
		
		As getOrigVsOther, nut now with self.overlapsNoMean instead of self.correlation. if normed, then retun is divided by no winner to get value between 1 and 0
		'''
		
		if self.over_orig_vs_other == None:
			divide = self.patterns_1.shape[0]**2 - self.patterns_1.shape[0] +0.0
			if self.one_patterns:
				divide /= 2
			self.over_orig_vs_other = np.sum(self.getOverOrigOther(subtract_orig_orig = 1), axis = -1)/divide
			if normed == 1:
				self.over_orig_vs_other /= np.sum(self.patterns_1[0]) *1. # divide by no winner to get value between 1 and 0
		return self.over_orig_vs_other


	### Analysis of Distances and Overlaps
	def getOverNoMeanDistances(self, locations= None, bins = None, location = None, at_noise =0):
		
		'''
		
		gets the entries of self.overlapsNoMean sorted over distances of the corresponding locations. bins specifies the bins in which the overlaps are sorted. Does only consider entries (i,j) in self.overlaps where i>j. Also returns used_bins, that are the bins of bins that are non_empty.
		'''
		self.calcOverlapsNoMean(noise_level = at_noise)
		overlaps = np.copy(self.overlapsNoMean[at_noise])

		overlaps[np.tril_indices(overlaps.shape[-1],-1)] = -10**5 # only i>= j
		#overlaps[np.diag_indices(overlaps.shape[-1])] = -10**5 # only i !=j
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		return_value = []
		used_bins = []
		if location != None:
			overlaps = overlaps[location]
			dis = dis[location]
		
		for i in range(bins.shape[0]):
			append = overlaps[dis == bins[i]][overlaps[dis == bins[i]]>-10**5]
			if append.shape[0] != 0:
				return_value.append(np.copy(append))
				used_bins.append(bins[i])
		return [return_value, used_bins]
		
	def getAverageOverlapOverNoMeanDistance(self, locations = None, bins = None, location = None):
		'''
		
		gets the average overlap at each distance bin
		'''
		
		over, used_bins = self.getOverNoMeanDistances(locations, bins, location)
		av = np.zeros(len(used_bins))
		for i in range(len(over)):
			av[i] = np.sum(over[i])/(over[i].shape[0]*1.)
			i+=1
		return [av, used_bins]
		
		
		
	def getOverDistances(self, locations= None, bins = None, location = None, at_noise =0):
		'''
		
		gets the entries of self.overlaps sorted over distances of the corresponding locations. bins specifies the bins in which the overlaps are sorted. Does only consider entries (i,j) in self.overlaps where i=>j. Also returns used_bins, that are the bins of bins that are non_empty. If location != None, only this specific location (pattern) is considered for index i
		'''
		self.calcOverlaps(noise_level = at_noise)
		overlaps = np.copy(self.overlaps[at_noise])

		overlaps[np.tril_indices(overlaps.shape[-1],-1)] = -10**5 # only i>= j
		#overlaps[np.diag_indices(overlaps.shape[-1])] = -10**5 # only i !=j
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		return_value = []
		used_bins = []
		if location != None:
			overlaps = overlaps[location]
			dis = dis[location]
		
		for i in range(bins.shape[0]):
			append = overlaps[dis == bins[i]][overlaps[dis == bins[i]]>-10**5]
			if append.shape[0] != 0:
				return_value.append(np.copy(append))
				used_bins.append(bins[i])
		return [return_value, used_bins]

	def getAverageOverlapOverDistance(self, locations = None, bins = None, location = None):
		'''
		
		gets the average overlap at each distance bin
		'''
		
		over, used_bins = self.getOverDistances(locations, bins, location)
		av = np.zeros(len(used_bins))
		for i in range(len(over)):
			av[i] = np.sum(over[i])/(over[i].shape[0]*1.)
			i+=1
		return [av, used_bins]
		
		
	def getCorDistances(self, locations= None, bins = None, location = None, at_noise =0):
		'''
		
		gets the entries of self.overlaps sorted over distances of the corresponding locations. bins specifies the bins in which the overlaps are sorted. Does only consider entries (i,j) in self.overlaps where i=>j. Also returns used_bins, that are the bins of bins that are non_empty. If location != None, only this specific location (pattern) is considered for index i
		'''
		self.calcCor(noise_level = at_noise)
		overlaps = np.copy(self.corelations[at_noise])

		overlaps[np.tril_indices(overlaps.shape[-1],-1)] = -10**5 # only i>= j
		#overlaps[np.diag_indices(overlaps.shape[-1])] = -10**5 # only i !=j
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		return_value = []
		used_bins = []
		if location != None:
			overlaps = overlaps[location]
			dis = dis[location]
		
		for i in range(bins.shape[0]):
			append = overlaps[dis == bins[i]][overlaps[dis == bins[i]]>-10**5]
			if append.shape[0] != 0:
				return_value.append(np.copy(append))
				used_bins.append(bins[i])
		return [return_value, used_bins]
		
	def getAverageCorOverDistance(self, locations = None, bins = None, location = None):
		'''
		
		gets the average overlap at each distance bin
		'''
		
		over, used_bins = self.getCorDistances(locations, bins, location)
		av = np.zeros(len(used_bins))
		for i in range(len(over)):
			av[i] = np.sum(over[i])/(over[i].shape[0]*1.)
			i+=1
		return [av, used_bins]
		
		

	#### Methods used for findPlaceWeight Analysis
	def getLocationsWithinDistance(self, distance = None, locations = None, location = None, bins = None):
		'''
		
		gets locations within the radius distance and outside the radius around location. Distances are rounded according to bins
		'''
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		distances = dis[location]
		inside = np.flatnonzero(distances <= distance)
		outside = np.flatnonzero(distances > distance)
		return [inside, outside]
		
	def getLocationsAtDistance(self, distance = None, locations = None, location = None, bins = None):
		'''
		
		gets locations at the exact distance to location. Distances are rounded according to bins
		'''
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		distances = dis[location]
		at_distance = np.flatnonzero(distances == distance)
		return at_distance
		
	def getPopulationWithinDistance(self, distance = None, locations = None, location = None, bins = None, with_noise = 0):
		
		'''
		
		gets patterns within the radius distance and outside the radius around location. Distances are rounded according to bins; if with_noise, then all noisy versions of the patterns are returned too.
		'''
		
		[inside, outside] = self.getLocationsWithinDistance(distance = distance, locations = locations, location = location, bins = bins)
		inside_pop = self.patterns_2[0][inside]
		outside_pop =self.patterns_2[0][outside]
		if with_noise:
			for i in range(self.patterns_2.shape[0]-1):
				inside_pop = np.concatenate((inside_pop, self.patterns_2[i+1][inside]))
				outside_pop = np.concatenate((outside_pop, self.patterns_2[i+1][outside]))
		return [inside_pop, outside_pop]
		
	def findPlaceWeight(self, distance = None, locations = None, location = None, bins = None, with_noise = 0, max_weight = -1, boarder = 0, influence = 0, non_negative = 1, alpha = 1):
		
		'''
		
		gets solution of ....
		'''
		
		
		[inside, outside] = self.getPopulationWithinDistance(distance = distance, locations = locations, location = location, bins = bins, with_noise = with_noise)
		
		G = np.concatenate((outside, - inside), axis = 0)
		h = [1.]* outside.shape[0] + [-1.] *inside.shape[0]
		
		if non_negative:
			G = np.concatenate((G, - np.eye(inside.shape[1])*1.), axis = 0)
			h = h + [0] *inside.shape[1]
			
			
		if max_weight != -1:
			#each weight less equal 0.3
			print 'each weight less equal ' +str(max_weight)
			G = np.concatenate((G, np.eye(inside.shape[1])*1.), axis = 0)
			h = h + [max_weight] *inside.shape[1]
			if not non_negative:# w >= - max
				G = np.concatenate((G, -np.eye(inside.shape[1])*1.), axis = 0)
				h = h + [max_weight] *inside.shape[1]
			
		if boarder:
			at_distance = self.patterns_1[self.getLocationsAtDistance(distance = distance, locations = locations, location = location, bins = bins)]
			print 'boarder findplace weight'

			G = np.concatenate((G, at_distance*1. - np.tile(self.patterns_1[location], (at_distance.shape[0], 1))), axis = 0)
			h = h + [0] * at_distance.shape[0]
			
		if influence:
			print 'influence findplace weight'
			ones = np.ones([1,inside.shape[-1]])
			ones[0][:self.modules[1]] = -1 *1./self.modules[1] # 1/n * sum module 4 - 1/m * sum module 1 <= 0
			ones[0][self.modules[1]:self.modules[3]] = 0 # mod 2 and 3 not considered
			ones[0][self.modules[3]:] = 1./(self.modules[4]-self.modules[3])
			
			G = np.concatenate((G, ones*1.), axis = 0)
			h = h + [0]
			
			#same for module 2:  1/n * sum module 4 - 1/m * sum module 2 <= 0
			ones = np.ones([1,inside.shape[-1]])

			ones[0][:self.modules[1]] = 0 
			ones[0][self.modules[1]:self.modules[2]] = -1 *1./(self.modules[2]-self.modules[1])
			ones[0][self.modules[2]:self.modules[3]] = 0
 			ones[0][self.modules[3]:] = 1./(self.modules[4]-self.modules[3])
			
			G = np.concatenate((G, ones*1.), axis = 0)
			h = h + [0]
			
			
			
			
		G = matrix(G)
		h = matrix(h)
			
		#c = matrix([1.] * inside.shape[1]) #minimize sum over components x_i
		
		# maximize differences px - qx for 100 random chosen inside p and outside q - sum x_i as penalty for big x.
		if inside.shape[0] < 10:
			rand_in = np.tile(self.patterns_1[location], (1000,1))
		else:
			if inside.shape[0] < 50:
				rand_in =  100* np.array(random.sample(inside, 10))
			else:
				if inside.shape[0] < 100:
					rand_in =  20* np.array(random.sample(inside, 50))
				else:
					if inside.shape[0] < 500:
						rand_in =  10 * np.array(random.sample(inside, 100))
					else:
						if inside.shape[0] < 1000:
							rand_in =  2 * np.array(random.sample(inside, 500))
						else:
							if inside.shape[0] >= 1000:
								rand_in =  np.array(random.sample(inside, 1000))
		
		if outside.shape[0] >= 1000:
			rand_out = np.array(random.sample(outside, 1000))
		else: 
			if outside.shape[0] >= 100:
				rand_out = 10 *np.array(random.sample(outside, 100))
			else:
				rand_out = 100 *np.array(random.sample(outside, 10))
		c = np.sum(rand_out, axis = -2) - np.sum(rand_in, axis = -2) + np.ones(inside.shape[1], 'float')
		
		
		c = matrix(c)
		
		##old linear 
		#if non_negative:
			#solve = solvers.lp(c,G,h)['x']
		#else:
		##new quadratic. necessary since min x_i^2 and not sim x_i
		p = matrix(alpha * np.eye(inside.shape[1])*1.) # xPx + qx
		c = 1/1000. * (matrix(np.sum(rand_out, axis = -2) - np.sum(rand_in, axis = -2)))# q=c, now without np.ones(inside.shape[1], 'float') term
		
		## min (outside . center) * w
		#c = 1./outside.shape[0] * np.sum(inside, axis = -2) -self.patterns_1[location]
		#c = matrix(c)
		
		solve = solvers.qp(p,c,G,h)['x'] #min sum_i x_i^2 
		return solve
		
	def getNumberWrongPixels(self, activation_map = None, location = None, locations = None, distance = None, bins=None):#return proportion inside wornd and outside wrong
		'''
		
		determines proportion pixel that fire inside the field wrong and outside.
		'''
		[inside, outside] = self.getLocationsWithinDistance(distance = distance, locations = locations, location = location, bins = bins)
		wrong_inside = np.flatnonzero(activation_map[inside] < 1).shape[0]
		wrong_outside = np.flatnonzero(activation_map[outside] > 1).shape[0]
		return [wrong_inside/(inside.shape[0]+0.0), wrong_outside/(outside.shape[0]+0.0)]
		
		
	def findPlaceWeightSVC(self, distance = None, locations = None, location = None, bins = None, with_noise = None, **kwargs):
	
		[inside, outside] = self.getPopulationWithinDistance(distance = distance, locations = locations, location = location, bins = bins, with_noise = with_noise)
		X = np.concatenate((inside, outside))
		Y = np.concatenate((np.ones(inside.shape[0]), np.ones(outside.shape[0])*-1))

		SVC = svm.LinearSVC()
		SVC.fit(X,Y)
		w = SVC.coef_

		return w




	def getOverNoMeanActiveCellsDistances(self, output_patterns, locations = None, bins = None):#get <p_1, p_2 - bar p> for all p such that q_i^1 = q^2 = 1; sorts it into bins, acroding to in which distance 1,2 have
		
		self.calcOverlapsNoMean(noise_level = 0)
		over = np.copy(self.overlapsNoMean[0])
		over[np.diag_indices_from(over)] = -10**10 #to exclude overlap of pattern with itself later
		p_1 = np.array([], 'int') #inidzes pattern_1
		p_2 = np.array([], 'int')#inidzes pattern_2
		cells = np.array([], 'int')
		for cell in range(output_patterns.shape[-1]):
			
			non_z_cell = np.flatnonzero(output_patterns[:,cell]) #cells that fire together with cell
			size = non_z_cell.shape[0]
			p_1 = np.concatenate((non_z_cell.repeat(size, axis = 0),p_1))
			p_2 = np.concatenate((np.tile(non_z_cell, size), p_2))
			cells = np.concatenate((np.ones(size*size, 'int')*cell, cells))#cell index repeated as often as cell is part of one overlap pair

		
		self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		return_value = []
		used_bins = []
		for i in range(bins.shape[0]):
			append = over[p_1,p_2][dis[p_1,p_2] == bins[i]] * output_patterns[p_2, cells][dis[p_1,p_2] == bins[i]]
			append = append[append> -10**4]
			if append.shape[0] != 0:
				return_value.append(append)
				used_bins.append(bins[i])

		return [return_value, used_bins]
		
	def getOverNoMeanSilentCellsDistances(self, output_patterns, locations = None, bins = None):#get <p_1, p_2 - bar p> for all p such that q_i^1 = 0, q_i^2 = 1
		
		self.calcOverlapsNoMean(noise_level = 0)
		over = np.copy(self.overlapsNoMean[0])
		over[np.diag_indices_from(over)] = -10**10 #to exclude overlap of pattern with itself later
		p_1 = np.array([], 'int') #inidzes pattern_1
		p_2 = np.array([], 'int')#inidzes pattern_2
		cells = np.array([], 'int')
		for cell in range(output_patterns.shape[-1]):
			
			output_patterns[:,cell]
			non_z_cell = np.flatnonzero(output_patterns[:,cell]) #cells that fire together with cell
			z_cell = np.flatnonzero(output_patterns[:,cell] == 0)
			size_non = non_z_cell.shape[0]
			size_z= z_cell.shape[0]
			p_1 = np.concatenate((z_cell.repeat(size_non, axis = 0),p_1))
			p_2 = np.concatenate((np.tile(non_z_cell, size_z), p_2))
			cells = np.concatenate((np.ones(size_z*size_non, 'int')*cell, cells))#cell index repeated as often as cell is part of one overlap pair

		
		self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		return_value = []
		used_bins = []
		for i in range(bins.shape[0]):
			append = over[p_1,p_2][dis[p_1,p_2] == bins[i]] * output_patterns[p_2, cells][dis[p_1,p_2] == bins[i]]
			append = append[append> -10**4]
			if append.shape[0] != 0:
				return_value.append(append)
				used_bins.append(bins[i])

		return [return_value, used_bins]

	def getMeanActivityActiveCells(self): #gets the average firing of an active cell
		active_cells = np.nonzero(self.patterns_1)
		return np.sum(self.patterns_1[active_cells], -1)/(np.flatnonzero(self.patterns_1).shape[0]*1.)
	
	def getOverLocationWithAllOthers(self, locations = None, location = None):
		overlaps = np.dot(self.patterns_1, self.patterns_1[location])
		return overlaps
		
	def getOverLocationWithAllOthersNoAverage(self, locations = None, location = None):
		overlaps = np.dot(self.patterns_1, self.patterns_1[location])
		av = np.sum(overlaps)/(self.patterns_1.shape[0]*1.)

		return overlaps -av
	
	#Misc
	def getDistanceMatrix(self, locations, round_it = 'fix', bins = None): #returns matrix (loc, loc) with the distance of the two as entry 
		if self.distance_matrix == None:
			self.calcDistanceMatrix(locations, round_it = round_it, bins = bins)

		return self.distance_matrix
	
	def getOccurencesDistances(self, locations, bins = None): #all distances (multiple times) as an array 
		if self.distances == None:
			self.distances= np.ravel(self.getDistanceMatrix(locations, bins = bins))

		return self.distances
		
	def calcOccurencesDistancesFireFire(self,locations, cells= None, bins = None):#returns all distances (multple times) of two locations, whenever cell fires at both; Oc( q=1, q=1, D=d). If cell = None, all cells are considered
		
		distance_matrix = self.getDistanceMatrix(locations = locations, bins = bins)
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		
		for cell in cells:
			if self.fire_fire_distances[cell] == None:
				self.fire_fire_distances[cell] = np.zeros(self.getAllDifferentDistances(locations, bins = bins ).shape[0])
				fire_locs = np.nonzero(self.patterns_1[:,cell])[0]
				distances = distance_matrix[fire_locs.repeat(fire_locs.shape[0]), np.tile(fire_locs, (1,fire_locs.shape[0]))]
				i=0
				for d in self.getAllDifferentDistances(locations, bins = bins ):
					self.fire_fire_distances[cell][i] = np.nonzero(distances == d)[0].shape[0]
					i+=1
					
	def calcOccurencesFireDistances(self,locations, bins = None, cells= None):#returns all distances (multple times) of a location, where a cell fires and all other locations; OC(q = 1, D=d); should be proportional to getOccurencesDistances (since firing and location is independent). In Fact OC(q = 1 , D=d) = oc(D=d) * oc(q=1) = getOccurencesDistances * cell_number (=1 if cells != None) * oc(q=1); if cells = None all cells are considered
		
		distance_matrix = self.getDistanceMatrix(locations = locations, bins = bins)
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]

		for cell in cells:
			if self.fire_distances[cell] == None:
				self.fire_distances[cell] = np.zeros(self.getAllDifferentDistances(locations, bins = bins ).shape[0])
				fire_locs = np.nonzero(self.patterns_1[:,cell])[0]
				distances =np.ravel(distance_matrix[fire_locs])
				i=0
				for d in self.getAllDifferentDistances(locations, bins = bins ):
					self.fire_distances[cell][i] = np.nonzero(distances == d)[0].shape[0]
					i+=1

		
	def calcOccurencesSilentDistances(self,locations, bins = None, cells= None):#returns all distances (multple times) of a location, where cell is silent and all other locations; OC(q = 0, D=d); should be proportional to getOccurencesDistances (since independent). In Fact OC(q = 0 , D=d) = oc(D=d) * P(q=0) = getOccurencesDistances * cell_number * P(q=0)
		distance_matrix = self.getDistanceMatrix(locations = locations, bins = bins )
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.silent_distances[cell] == None:
				self.silent_distances[cell] = np.zeros(self.getAllDifferentDistances(locations, bins = bins ).shape[0])
				silent_locs = np.nonzero(self.patterns_1[:,cell]-1)[0]
				distances = np.ravel(distance_matrix[silent_locs])
				i=0
				for d in self.getAllDifferentDistances(locations, bins = bins ):
					self.silent_distances[cell][i] = np.nonzero(distances == d)[0].shape[0]
					i+=1

	def calcOccurencesDistancesFireSilent(self,locations, bins = None, cells= None):#returns all distances (multple times) of a location, where cell fires and one where not; OC(q=0, q=1, D=d)
		
		distance_matrix = self.getDistanceMatrix(locations = locations, bins = bins )
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.fire_silent_distances[cell] == None:
				self.fire_silent_distances[cell] = np.zeros(self.getAllDifferentDistances(locations, bins = bins ).shape[0])
				fire_locs = np.nonzero(self.patterns_1[:,cell])[0]
				silent_locs = np.nonzero(self.patterns_1[:,cell]-1)[0]
				distances = distance_matrix[fire_locs.repeat(silent_locs.shape[0]), np.tile(silent_locs, (1,fire_locs.shape[0]))]
				i=0
				for d in self.getAllDifferentDistances(locations, bins = bins ):
					self.fire_silent_distances[cell][i] = np.nonzero(distances == d)[0].shape[0]
					i+=1

	def getAllDifferentDistances(self, locations, bins = None): #returns all ocurring distances one time (deleting all multple ones) in increasing size
		if self.all_distances == None:
			self.all_distances = np.sort(list(set(self.getOccurencesDistances(locations = locations, bins = bins))))
		return self.all_distances
		
	def getPDistance(self, locations,bins = None): #returns probablity of a distance; P(D=d) = OC(d)/OC(all d)
		if self.p_d ==None:
			all_distances = self.getAllDifferentDistances(locations, bins = bins )
			self.p_d = np.zeros(all_distances.shape[0])
			i=0
			for d in all_distances:
				self.p_d[i] = np.nonzero(self.getOccurencesDistances(locations, bins = bins ) == d)[0].shape[0]
				i+=1
			self.p_d /= (np.sum(self.p_d)*1.)

		return self.p_d
		
	def calcPDistanceFireLocations(self,locations, bins = None, cells= None): #returns probablity of a distance; P(D=d) = OC(q =1, d)/OC(all q = 1)
		
		self.calcOccurencesFireDistances(locations, bins = bins , cells = cells)
		all_distances = self.getAllDifferentDistances(locations, bins = bins )
		
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.p_d_fire[cell]== None:
				self.p_d_fire[cell] = self.fire_distances[cell]
				if np.sum(self.fire_distances[cell])*1. != 0:
					self.p_d_fire[cell] /=(np.sum(self.fire_distances[cell])*1.)
		
	def calcPDistanceSilentLocations(self,locations, bins = None, cells = None): #returns probablity of a distance; P(D=d) = OC(q =0, d)/OC(all q = 0)
		
		self.calcOccurencesSilentDistances(locations, bins = bins , cells = cells) 
		all_distances = self.getAllDifferentDistances(locations, bins = bins )
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.p_d_silent[cell] == None:
				self.p_d_silent[cell] = self.silent_distances[cell]/(np.sum(self.silent_distances[cell])*1.)
		
	def getPFireGivenDistance(self,locations, bins = None, cells = None):#returns probablity P(q=1 | D=d) = P(q=1,D=d)/P(D=d) = OC(q=1, D=d)/OC(D=d); OC(D=d) must be same counting as OC(q=1, D=d)
		
		all_distances = self.getAllDifferentDistances(locations, bins = bins )
		i=0
		occurences_fire_distances = self.getOccurencesFireDistances(locations, bins = bins , cells = cells)
		occurences_distances = self.getOccurencesDistances(locations, bins = bins )
		for d in all_distances:
			self.p_fire_given_distance[i] = np.nonzero(occurences_fire_distances == d)[0].shape[0]
			self.p_fire_given_distance[i] /= np.nonzero(occurences_distances == d)[0].shape[0]
			i+=1
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		self.p_fire_given_distance/= (len(cells)*1.0) #Divide by cell numbers, since OC(D=d) is counted for each cell once and self.getOccurencesDistances(locations) just one time
		print 'getPFireGivenDistance'
		print 'should be not used!!!!!!!!!!1'
		return self.p_fire_given_distance
		
	def calcPFireGivenFireAndDistance(self, locations, cells = None, bins = None): #returns probablity P(q=1 | q(t) = 1, D=d) = P(q=1 + q(t)=1 + D=d)/P(D=d + q(t)=1) = OC(q=1, q=1, D=d)/OC(q=1, D=d)
		
		all_distances = self.getAllDifferentDistances(locations, bins = bins)
		
		self.calcOccurencesFireDistances(locations, cells = cells, bins = bins)
		self.calcOccurencesDistancesFireFire(locations, cells = cells, bins = bins)
		
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.p_fire_given_fire_and_distance[cell] == None:
				self.p_fire_given_fire_and_distance[cell] = self.fire_fire_distances[cell]
				self.p_fire_given_fire_and_distance[cell][self.fire_distances[cell]!=0] /= (self.fire_distances[cell][self.fire_distances[cell]!=0]*1.)

		
	def calcPFireGivenSilentAndDistance(self, locations, bins = None,  cells = None): #returns probablity P(q=1 | q(t) = 0, D=d) = P(q=1 , q(t)=0 + D=d)/P(D=d + q(t)=0) = OC(q=1, q=0, D=d)/OC(q=0, D=d)
		
		all_distances = self.getAllDifferentDistances(locations, bins = bins )
		self.calcOccurencesSilentDistances(locations, bins = bins , cells = cells) 
		self.calcOccurencesDistancesFireSilent(locations, bins = bins , cells = cells)
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.p_fire_given_silent_and_distance[cell] == None:
				self.p_fire_given_silent_and_distance[cell] = self.fire_silent_distances[cell]
				self.p_fire_given_silent_and_distance[cell][self.silent_distances[cell] != 0] /= (self.silent_distances[cell][self.silent_distances[cell] != 0]*1.)

	def calcPFireGivenFireAndDistanceWeightedDistances(self, locations, bins = None,  cells = None): #returns P(q =1 | q_t = 1, D=d) * P(D=d|q = 1) = Oc(1,1,d)/oc(1,d) * oc(1,d)/oc(1) = oc(1,1,d)/oc(1)
		
		self.calcPFireGivenFireAndDistance(locations, cells, bins = bins)
		self.calcPDistanceFireLocations(locations, bins = bins )
		
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.p_fire_given_fire_and_distance_weighted_distance[cell] == None:
				self.p_fire_given_fire_and_distance_weighted_distance[cell] = self.p_fire_given_fire_and_distance[cell] * self.p_d_fire[cell]

	def calcPFireGivenSilentAndDistanceWeightedDistances(self, locations, bins = None,  cells = None):#returns P(q =1 | q_t = 0, D=d) * P(D=d) = P(q=1, D=d | q_t =0)

				
		self.calcPFireGivenSilentAndDistance(locations, cells, bins = bins)
		self.calcPDistanceSilentLocations(locations, bins = bins )
		
		if cells == None:
			cells = range(self.patterns_1.shape[1])
		else:
			cells = [cells]
		for cell in cells:
			if self.p_fire_given_silent_and_distance_weighted_distance[cell] == None:
				self.p_fire_given_silent_and_distance_weighted_distance[cell] = self.p_fire_given_silent_and_distance[cell] * self.p_d_silent[cell]
	
	def getPFireGivenFireAndDistanceAll(self, locations, bins = None):
		self.calcPFireGivenFireAndDistance(locations, cells = None, bins = bins)
		return np.sum(self.p_fire_given_fire_and_distance, axis = 0)/(self.patterns_1.shape[-1]*1.)
	
	def getPFireGivenSilentAndDistanceAll(self, locations, bins = None):
		self.calcPFireGivenSilentAndDistance(locations, bins = bins ,  cells = None)
		return np.sum(self.p_fire_given_silent_and_distance, axis = 0)/(self.patterns_1.shape[-1]*1.)
	
	def getPFireGivenFireAndDistanceWeightedDistancesAll(self, locations, bins = None):
		self.calcPFireGivenFireAndDistanceWeightedDistances(locations, bins = bins , cells = None)
		return np.sum(self.p_fire_given_fire_and_distance_weighted_distance, axis = 0)/(self.patterns_1.shape[-1]*1.)
	
	def getPFireGivenSilentAndDistanceWeightedDistancesAll(self, locations, bins = None):
		self.calcPFireGivenSilentAndDistanceWeightedDistances(locations, bins = bins , cells = None)
		return np.sum(self.p_fire_given_silent_and_distance_weighted_distance, axis = 0)/(self.patterns_1.shape[-1]*1.)
	

	#def calcEigenValuesCovariance(self): #calc ev and ew form covariance matrix of patterns_1, where each column is one observation (data point); e-values and e-vectors are ordered, may first
		
		#cov = np.cov(self.patterns_1.T)#patterns_1 is (observation, variable), thus transpose it to (variabele, observation) so that column become data point
		#print 'calc covariance'

		#w ,v = np.linalg.eig(cov) #w[i] = eigenvalue i and v[:,i] eigenvector of i, thus the columns of v are the eigenvectors
		#arg = np.argsort(w)[::-1]
		#self.covariance_ew = w[arg]
		#self.covariance_ev = v[:, arg]


	#def getEigenvaluesCovariance(self, number_relevant= None): #returns the eigenvalues of the covariance of patterns; each colum of pattern is one datapoint
		
		#if number_relevant == None:
			#number_relevant = self.patterns_1.shape[1]
		#if self.covariance_ew == None:
			#self.calcEigenValuesCovariance()
		#return self.covariance_ew[:number_relevant]

	#def getEigenvectorsCovariance(self, number_relevant = None):#patterns = (variable ,observation)
		
		#if number_relevant == None:
			#number_relevant = self.patterns_1.shape[1]
		#if self.covariance_ew == None:
			#self.calcEigenValuesCovariance()
		#return self.covariance_ev[:,:number_relevant]
		
	#def getProjectionMatrix(self, number_relevant = None): # returns  ON Matrix that projects data into space spanend by highest evectors, number e-v = number_relevant
		
		#if number_relevant == None:
			#print 'no dimension reduced during projection'
			#number_relevant = self.patterns_1.shape[1]
		#if number_relevant == self.number_relevant: #if computed before 
			#pass
		#else:
			#w = self.getEigenvaluesCovariance(number_relevant)
			#v = self.getEigenvectorsCovariance(number_relevant)
			#q,r = np.linalg.qr(v)# q is ON Basis of column space of relevant ev v
			#self.projection_matrix = np.dot(q, q.T)#Projection matrix
			#print '-----getporjectmatrix done'
			
			#self.number_relevant = number_relevant
		#return self.projection_matrix
		
	#def getProjectedData(self, to_project_data = None, number_relevant = None):# project given data (given as colums) by the projection matrix, retun data is given in colums too
		
		#p = self.getProjectionMatrix(number_relevant)
		#projected = np.einsum('...ij,...jk->...ik',p, to_project_data*1.)
		#return projected
	
	#def getCorrelationInstanceProjectedData(self, patterns1 = None, patterns2 = None, number_relevant = None, binary =1):#returns Correlation instance of projected data given by p1 and p2 (as columns), if binary k-wta is applied on porjected data first. 
		
		#number_winner = int(np.sum(patterns1[:,0]))
		#p1 = self.getProjectedData(to_project_data = patterns1, number_relevant = number_relevant).T #now dat ain rows again
		#p2 = np.transpose(self.getProjectedData(to_project_data = patterns2, number_relevant = number_relevant), (0,2,1))
		#if binary:
			#size = list(p1.shape)
			#winner = np.argsort(p1)[...,-number_winner:size[-1]]
			#p1 = np.zeros(size)
			#p1[np.mgrid[0:size[0], 0:number_winner][0], winner] = 1
			
			#size = list(p2.shape)
			#winner = np.argsort(p2)[...,-number_winner:size[-1]]
			#p2 = np.zeros(size)
			#indices = np.mgrid[0:size[0],0:size[1],0:number_winner]
			#p2[indices[0], indices[1], winner] =1
		#return Corelations(patterns_1 = p1, patterns_2 = p2, in_columns =0)
		
	
	def calcEigenValuesCovariance(self): #calc ev and ew form covariance matrix of patterns_1, where each column is one observation (data point); e-values and e-vectors are ordered, may first
		
		cov = np.cov(self.patterns_1.T)#patterns_1 is (observation, variable), thus transpose it to (variabele, observation) so that column become data point
		print 'calc covariance'

		w ,v = np.linalg.eig(cov) #w[i] = eigenvalue i and v[:,i] eigenvector of i, thus the columns of v are the eigenvectors
		arg = np.argsort(w)[::-1]
		self.covariance_ew = w[arg]
		self.covariance_ev = v[:, arg]


	def getEigenvaluesCovariance(self, number_relevant= None): #returns the eigenvalues of the covariance of patterns; each colum of pattern is one datapoint
		
		if number_relevant == None:
			number_relevant = self.patterns_1.shape[1]
		if self.covariance_ew == None:
			self.calcEigenValuesCovariance()
		return self.covariance_ew[:number_relevant]

	def getEigenvectorsCovariance(self, number_relevant = None):#patterns = (variable ,observation)
		
		if number_relevant == None:
			number_relevant = self.patterns_1.shape[1]
		if self.covariance_ew == None:
			self.calcEigenValuesCovariance()
		return self.covariance_ev[:,:number_relevant]
		
		
	def calcPCA(self, number_relevant = None):
		self.PCA = decomposition.PCA(n_components = number_relevant)
		self.PCA.fit(self.patterns_1)
		
	def getProjectedCor(self, number_relevant = None, patterns_to_project_1 = None, patterns_to_project_2 = None, project_just_1 = 0):
		self.calcPCA(number_relevant)
		projected_patterns_1 = self.PCA.transform(patterns_to_project_1)
		if not project_just_1:
			projected_patterns_2 = np.zeros([patterns_to_project_2.shape[0],patterns_to_project_2.shape[1],number_relevant] )
			for i in range(patterns_to_project_2.shape[0]):
				projected_patterns_2[i] = self.PCA.transform(patterns_to_project_2[i])
		else:
			projected_patterns_2 = patterns_to_project_2
		return Corelations(patterns_1 = projected_patterns_1, patterns_2 = projected_patterns_2, env = 0)
		
	
	
	
	
	def getEucledeanDifferenceAfterProjection(self,patterns=None, number_relevant = None): #return av euclidean norm of normalized patterns (data as colums) and projected patterns
		
		p = self.getProjectionMatrix(number_relevant)
		normalize(patterns.T)
		projected = np.dot(p, patterns*1.)
		diff = patterns - projected
		diff_norm_sq = np.dot(diff.T, diff)
		av_norm = np.sum(np.sqrt(np.diagonal(diff_norm_sq)))/(diff.shape[0]*1.)
		return av_norm
		
	def getCorrelationAfterProjection(self,patterns=None, number_relevant = None): #return correlation of patterns (colums) and projected patterns
		
		p = self.getProjectionMatrix(number_relevant)
		projected = np.dot(p, patterns*1.)
		av_norm = Corelations(patterns_1 = patterns, patterns_2 = projected, in_columns = True).getOrigVsOrig()
		return av_norm
		
	def getWhiteData(self):
		if self.covariance_ew == None:
			self.calcEigenValuesCovariance()
		

		ew = np.sqrt(np.copy(self.covariance_ew))
		ew[ew == 0] = np.min(ew[ew != 0])
		d = np.diag(1./ew)
		return np.dot(d, np.dot(self.covariance_ev.T, self.patterns_1.T))
		
		
		
	def getFireTimes(self): #returns array that indicate number of times each cell fires in all pattern
		binary_pattern = np.copy(self.patterns_1)
		binary_pattern[binary_pattern !=0] = 1
		return np.sum(binary_pattern, axis = 0)
		
	def getCellIndizesMaxFire(self, number_cells = 1): #return number_of_cell cells that fire most often during all pattern; 
		fire_times = self.getFireTimes()
		return np.argsort(fire_times)[-number_cells:]
		
	def getCellIndizesMinFire(self, number_cells = 1): #return number_of_cell cells that fire at fewest during all pattern
		fire_times = self.getFireTimes()
		return np.argsort(fire_times)[:number_cells]
	
	def getSpatialStored(self,cell = None, recall = False, at_noise = None): # return firing of cell over enviroment
	
		if recall:
			fire = self.patterns2[at_noise,:,cell]
		else:
			fire = self.patterns_1[:,cell]
		return fire

	def getMeanOver(self, at_noise=None):
		
		self.calcOverlaps(noise_level = at_noise)
		overlaps = np.copy(self.overlaps[at_noise])
		
		if self.one_patterns:
			overlaps[np.tril_indices(overlaps.shape[-1])] =0
			mean = np.sum(overlaps)/((overlaps.shape[-1]**2- overlaps.shape[-1]+0.0)/2.)
		else:
			overlaps[np.diag_indices(overlaps.shape[-1])] = 0
			mean = np.sum(overlaps)/(overlaps.shape[-1]**2- overlaps.shape[-1]+0.0)
		return mean

	def getMeanOverAllNoise(self):
		self.calcOverlaps()
		return np.array(map(self.getMeanOver, range(self.overlaps.shape[0])))

	def getMaxIndRow(self, at_noise = 0): #returns column index from each row of self.corelations where cor is max 
		
		self.calcCor(noise_level = at_noise)
		over = np.round(self.corelations[at_noise],2)
		max_ind_row = np.argsort(over)[:,-1]
		return max_ind_row
		
	def getMaxIndCol(self, at_noise = 0): #returns row index from each column of self.corelations where cor is max 
		
		self.calcCor(noise_level = at_noise)
		over = np.round(self.corelations[at_noise],2).T
		max_ind_col = np.argsort(over)[:,-1]
		return max_ind_col
	
	def getNotMaxIndRow(self, at_noise = 0): #returns the row indizes of self.corelations in which the maximum is not on the diagonal; thus, the rows, in which the corrlation between orig orig is smaller than at least one corelation orig other 
		
		max_ind_in_row = self.getMaxIndRow(at_noise) #column index of each row with maximal argument; dimension = row
		r = np.arange(self.corelations.shape[-1])
		not_max_row = r[max_ind_in_row != r]# each row where the max ind is not on the diagonal
		return not_max_row
		
	def getNotMaxIndCol(self, at_noise = 0): #returns the col indizes of self.corelations in which the maximum is not on the diagonal; thus, the cols, in which the corrlation between orig orig is smaller than at least one corelation orig other 
		
		max_ind_in_col = self.getMaxIndCol(at_noise) #column index of each row with maximal argument; dimension = row
		r = np.arange(self.corelations.shape[-1])
		not_max_col = r[max_ind_in_col != r]# each row where the max ind is not on the diagonal
		return not_max_col
	
	def getNotMaxCorrelationsRow(self, at_noise = 0): #returns the corelation <p,~p>, whenver there is a q!= p such that <p,~q> > <p,~p>
		cor = self.getCor(at_noise = at_noise)
		not_max_row = self.getNotMaxIndRow(at_noise)
		return cor[not_max_row, not_max_row]
		
	
	def getNotMaxCorrelations(self, at_noise = 0): #returns the corelation <p,~p>, whenver there is a q!= p such that <q,~p> > <p,~p>
		cor = self.getCor(at_noise = at_noise)
		not_max_col = self.getNotMaxIndCol(at_noise)
		return cor[not_max_col, not_max_col]

	def getProportionCompletedToWrong(self, at_noise = None):
		if at_noise == None:
			return_value = np.zeros(self.patterns_2.shape[0])
			noise = 0
			for noise in range(self.patterns_2.shape[0]):
				return_value[noise] = np.round(self.getNotMaxIndCol(noise).shape[0]*1./self.patterns_1.shape[0], 3) *100

		else:
			return_value = np.round(self.getNotMaxIndCol(at_noise).shape[0]*1./self.patterns_1.shape[0], 3) *100
		return return_value
	
	
	def plotCorrelationsHeatMap(self,at_noise = 0, pattern = None, In = None, fig = None, ax_index = None, title = None):
		
		ax = fig.add_subplot(ax_index[0], ax_index[1],ax_index[2], aspect = 'equal')
		loc = In.locations[In.store_indizes]		
		cors = self.getCor(at_noise=at_noise)[:,pattern]
		ax.set_title(title)
		ax.set_ylim(-0.02,0.962)
		ax.set_xlim(-0.02,0.962)
		ax.scatter(loc[0][:,0], loc[0][:,1], c = cors, marker = 's', s = 150, faceted = False, cmap=cm.jet)
		#fig.colorbar()
		#mpl.colorbar.ColorbarBase(mpl.colorbar.make_axes(ax)[0])
		ax.scatter(loc[0][pattern,0], loc[0][pattern,1], c = 'k', s= 600, alpha = 1, facecolors = 'None', edgecolor = 'k')
	
	
	def plotCorrelationsHeatMapRow(self,at_noise = 0, pattern = None, In = None, fig = None, ax_index = None, title = None):
		
		ax = fig.add_subplot(ax_index, aspect = 'equal')
		loc = In.locations[In.store_indizes]		
		cors = self.getCor(at_noise=at_noise)[pattern]
		ax.set_title(title)
		ax.set_ylim(0,1)
		ax.set_xlim(0,1)
		ax.scatter(loc[0][:,0], loc[0][:,1], c = cors, marker = 's', s = 140, faceted = False, cmap=cm.jet)
		#fig.colorbar()
		mpl.colorbar.ColorbarBase(mpl.colorbar.make_axes(ax)[0])
		ax.scatter(loc[0][pattern,0], loc[0][pattern,1], c = 'k', s= 600, alpha = 1, facecolors = 'None', edgecolor = 'k')

	def getSimilarPatterns(self, correlation =0.3):
		cor = self.getCor(at_noise = 0)
		similar = Set([])
		for i in range(cor.shape[0]):
			if i not in similar: # if pattern i not have been removed yet
				similar = similar.union(np.flatnonzero(cor[i,i+1:]>= correlation) + i+1) # indices that are similar to pattern i, but have not been considered at previous i. We add i to get original indices.
		return similar
		
	def getSimilarPatternsOverlapNoMean(self, overlap = 100):
		over = self.getOverOrigOtherNoMean(at_noise = 0)
		similar = Set([])
		for i in range(cor.shape[0]):
			if i not in similar: # if pattern i not have been removed yet
				similar = similar.union(np.flatnonzero(over[i,i+1:]>= overlap) + i+1) # indices that are similar to pattern i, but have not been considered at previous i. We add i to get original indices.
		return similar
		
	def getDifferentPatterns(self, correlation = 0.3):
		similar = self.getSimilarPatterns(correlation = correlation)
		different = Set(range(self.corelations.shape[-1])) - similar
		return list(different)
		
	def getCorSpecialPatterns(self, indices = None):
		Cor = Corelations(patterns_1 = self.patterns_1[indices], patterns_2 = np.tile(self.patterns_2[:,indices], (1,1,1,1)))
		return Cor
		
	def getCorDifferentPatterns(self, correlation = 0.3):
		similar = self.getSimilarPatterns(correlation = correlation)
		different = Set(range(self.corelations.shape[-1])) - similar
		Cor = Corelations(patterns_1 = self.patterns_1[list(different)])
		return Cor
			
	def getOverNoMeanActiveCells(self, output_patterns):#get <p_1, p_2 - bar p> for all p such that q_i^1 = q^2 = 1
		
		self.calcOverlapsNoMean(noise_level = 0)
		over = np.copy(self.overlapsNoMean[0])
		over[np.diag_indices_from(over)] = -10**10 #to exclude overlap of pattern with itself later
		p_1 = np.array([], 'int') #inidzes pattern_1
		p_2 = np.array([], 'int')#inidzes pattern_2
		cells = np.array([], 'int')#cell index that fire

		for cell in range(output_patterns.shape[-1]):
			
			non_z_cell = np.flatnonzero(output_patterns[:,cell]) #cells that fire together with cell
			size = non_z_cell.shape[0]
			p_1 = np.concatenate((non_z_cell.repeat(size, axis = 0),p_1))# index of pattern 1, where cell fires in. It is repeated, since it is used for all other indices in non_z_cell
			p_2 = np.concatenate((np.tile(non_z_cell, size), p_2))# index of pattern 2, where cell fires in
			cells = np.concatenate((np.ones(size*size, 'int')*cell, cells))#cell index repeated as often as cell is part of one overlap pair
		r_value = over[p_1,p_2] * output_patterns[p_2, cells]
		return r_value[r_value> -10**3]
		
	def getOverNoMeanSilentCells(self, output_patterns): #get <p_1, p_2 - bar p> for all p such that q_i^1 = q^2 = 1
		
		self.calcOverlapsNoMean(noise_level = 0)
		over = np.copy(self.overlapsNoMean[0])
		over[np.diag_indices_from(over)] = -10**10 #to exclude overlap of pattern with itself later
		p_1 = np.array([], 'int') #inidzes pattern_1
		p_2 = np.array([], 'int')#inidzes pattern_2
		cells = np.array([], 'int')#cell index that fire
		for cell in range(output_patterns.shape[-1]):
			
			#output_patterns[:,cell]
			non_z_cell = np.flatnonzero(output_patterns[:,cell]) #cells that fire together with cell
			z_cell = np.flatnonzero(output_patterns[:,cell] == 0)
			size_non = non_z_cell.shape[0]
			size_z= z_cell.shape[0]
			p_1 = np.concatenate((z_cell.repeat(size_non, axis = 0),p_1))#index where cell of patterns in which it is silent. It is repeated to compare it with all the indices in non_z_cell.
			p_2 = np.concatenate((np.tile(non_z_cell, size_z), p_2))#index where cell of patterns in which it fires. It is repeated to compare it with all the indices in z_cell.
			cells = np.concatenate((np.ones(size_non*size_z, 'int')*cell, cells))#cell index repeated as often as cell is part of one overlap pair

		r_value = over[p_1, p_2] * output_patterns[p_2, cells]#overlaps of paris, weighted by the outputfiring
		return r_value[r_value> -10**3]

	def getCorsWithin(self,size = None, at_noise = 0, n_e = None, subtract_orig_orig = 1):
		self.calcCor(noise_level = at_noise)
		for h in range(n_e):
			ind = np.mgrid[h*size:(h+1)*size, (h+1)*size:self.patterns_1.shape[-2]]
			self.corelations[at_noise][ind[0], ind[1]] = -100
			self.corelations[at_noise][ind[1], ind[0]] = -100
			ind = np.mgrid[h*size:(h+1)*size, 0:h*size]
			self.corelations[at_noise][ind[0], ind[1]] = -100
			self.corelations[at_noise][ind[1], ind[0]] = -100
		cors = self.getCorOrigOther(at_noise=at_noise, subtract_orig_orig = subtract_orig_orig)
		self.calculated_cor = False #make sure self.cor is calculated later again
		self.corelations[at_noise] = 0
		return cors[cors!= -100]
		
	def getDistancesWithin(self,locations = None, subtract_orig_orig = 1):
		distances = np.copy(self.getDistanceMatrix(locations = locations, round_it = 0))
		n_e, size = np.shape(locations)[:-1]
		for h in range(n_e):
			ind = np.mgrid[h*size:(h+1)*size, (h+1)*size:self.patterns_1.shape[-2]]
			distances[ind[0], ind[1]] = -100
			distances[ind[1], ind[0]] = -100
			ind = np.mgrid[h*size:(h+1)*size, 0:h*size]
			distances[ind[0], ind[1]] = -100
			distances[ind[1], ind[0]] = -100		
		if self.one_patterns:
			if subtract_orig_orig:
				return_value = distances[np.tril_indices(distances.shape[-1], -1)]
			else:
				return_value = distances[np.tril_indices(distances.shape[-1], 0)]
		else:
			if subtract_orig_orig:
				cor_lower = distances[np.tril_indices(distances.shape[-1], -1)]
				cor_upper = distances[np.triu_indices(distances.shape[-1], 1)]
				return_value = np.zeros(distances.shape[-1]**2 -distances.shape[-1])
				return_value[:(distances.shape[-1]**2 -distances.shape[-1])/2] = cor_lower
				return_value[(distances.shape[-1]**2 -distances.shape[-1])/2:] = cor_upper
			else:
				return_value = np.ravel(distances)
		self.distance_matrix = None
		return return_value[return_value!= -100]
	


	def getCorsAcross(self,size = None, at_noise = 0, n_e = None, subtract_orig_orig = 1):
		self.calcCor(noise_level = at_noise)
		for h in range(n_e):
			ind = np.mgrid[h*size:(h+1)*size, h*size:(h+1)*size]
			self.corelations[at_noise][ind[0], ind[1]] = -100
		cors = self.getCorOrigOther(at_noise=at_noise, subtract_orig_orig = subtract_orig_orig)
		self.calculated_cor = False #make sure self.cor is calculated later again
		self.corelations[at_noise] = 0
		return cors[cors!= -100]
		
	def getDistancesAcross(self,locations = None, subtract_orig_orig = 1):
		#self.distance_matrix = None
		distances = np.copy(self.getDistanceMatrix(locations = locations, round_it = 0))
		n_e, size = np.shape(locations)[:-1]
		for h in range(n_e):
			ind = np.mgrid[h*size:(h+1)*size, h*size:(h+1)*size]
			distances[ind[0], ind[1]] = -100
		if self.one_patterns:
			if subtract_orig_orig:
				return_value = distances[np.tril_indices(distances.shape[-1], -1)]
			else:
				return_value = distances[np.tril_indices(distances.shape[-1], 0)]
		else:
			if subtract_orig_orig:
				cor_lower = distances[np.tril_indices(distances.shape[-1], -1)]
				cor_upper = distances[np.triu_indices(distances.shape[-1], 1)]
				return_value = np.zeros(distances.shape[-1]**2 -distances.shape[-1])
				return_value[:(distances.shape[-1]**2 -distances.shape[-1])/2] = cor_lower
				return_value[(distances.shape[-1]**2 -distances.shape[-1])/2:] = cor_upper
			else:
				return_value = np.ravel(distances)
		self.distance_matrix = None
		return return_value[return_value!= -100]

class Spatial():
	
	'''
	
	Spatial firing statistics of cells.
	It finds place fields and calculates their sizes etc. 
	
	:param patterns: population activites that the class shall analysis, should be an array of dimension 2 (pattern, cell)
	:param In: If In != None, it analyses the pattern of that Input Class
	:param  min_rate: It is the minimal activity of the cell that is considered as an active location. It will be given as the fraction of the peak activity of the cell.
	:param min_size: minimum size of a field such that it is considered as place field. Given in cm
	:param max_size: maximum size of a field such that it is considered as place field. Given in cm
	:param si_criterion: criterion whether an field as an place field by si and Treves 2009
	'''
	
	def __init__(self, In = None, patterns = None, cage = [1,1], min_rate = None, centers = np.array([]),min_size = None, max_size = None, si_criterion = False, **kwargs):
		
		if In != None:
			patterns = In.getInput()
			centers = In.centers
			cage = In.cage
		
		self.patterns_1 = patterns #dim 2 (location, cell)
		if len(centers.shape) == 3:
			self.centers = centers.reshape(centers.shape[0]*centers.shape[1], centers.shape[2])#center of the place fields; if determined by hand
		else:
			 self.centers = centers
		self.cage = cage
		self.cluster_size = np.zeros([patterns.shape[-1], patterns.shape[0]], dtype = 'int') # (cell, cluster) = cluster_size in cm^2; since number of clusters is yet unkown, last dimension is number patterns.
		self.clusters_colored = np.zeros(list(np.shape(self.patterns_1)[::-1]))# locations that are considered as firing fields; (cell, location); different fields have differnt numeric value
		self.number_fields = np.zeros(self.patterns_1.shape[1], 'int')# no of fields per cell
		self.noise = np.zeros(self.patterns_1.shape[1], 'int') #no of firing locations of a cell that do not belong to a firing place field
		self.makeLocations(self.patterns_1.shape[0])
		#no pixel in x,y direction
		self.x_length = self.cage_proportion*self.space_res
		self.y_length = self.space_res
		
		self.pixel_to_cm2 = cage[0]*cage[1]*10000./self.patterns_1.shape[0]
		self.min_size = None
		self.max_size = None
		self.si_criterion = si_criterion
		if si_criterion:
			print 'si criterion'
		if min_size != None:
			self.min_size = min_size * 1./self.pixel_to_cm2 #min_size in pixel
			print 'min size to be pf', self.min_size, ' pixel and ' , min_size, ' cm^2'
		if max_size != None: #not implemented
			self.max_size = max_size * 1./self.pixel_to_cm2 #min_size in pixel
			print 'max size to be pf', self.max_size, ' pixel and ' , max_size, ' cm^2'
		if min_rate != None:
			print 'min rate ' , min_rate
			min_fire = np.max(self.patterns_1, axis = -2) * min_rate
			min_fire = np.tile(min_fire , (self.patterns_1.shape[0],1))

			self.patterns_1[self.patterns_1 < min_fire] = 0
		
		#self.x_length = np.sqrt(self.patterns_1.shape[0])# number of pixel for one horizontol row in the enviroment
		#self.locations = np.ravel((np.mgrid[0:self.x_length, 0:self.x_length] + 0.0)/self.x_length, order = 'F').reshape(self.patterns_1.shape[0], 2)# all locations in the enviroment
		self.patterns_2d = self.patterns_1.reshape(self.y_length, self.x_length, self.patterns_1.shape[1])
		self.patterns_white = preprocessing.scale(self.patterns_1)
	
	def makeLocations(self, number_patterns): #help function
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.sqrt(number_patterns*1./self.cage_proportion)

		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
			#self.locations = self.locations.reshape(number_patterns, 2)
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]
				#self.locations = self.locations.reshape(number_patterns, 2)
	
	def getSpatial(self, cell = None):#help function
		'''
		
		returns spatial firing of given cell
		'''
		return np.copy(self.patterns_1[:,cell])
	
	def calcClusters(self, cells=None):#help function
		
		print 'start calculating number fields'
		begin = time.time()
		fields = []
		in_cluster = set([])
		max_size = 1 #side length of quader, two firing locations in that quader are considered connected.
		#!
		#if cells == None:
		cells = range(self.patterns_1.shape[1])
		for cell in cells:
			cell_fire = self.getSpatial(cell = cell)
			all_cluster_cell = [] #all clusters the cell have
			loc_fire = set(np.flatnonzero(cell_fire))
			not_visited = set(np.flatnonzero(cell_fire))
			while len(not_visited) >0:
				quader = set([])
				loc = not_visited.pop() # random fire loc, is called and removed

				ball_ind_x =np.arange(-max_size, max_size+1) + loc #x coordinates of quader
				y_row = int(loc/self.x_length) # y coordinated (row) of loc


				ball_ind_x = ball_ind_x[ball_ind_x - y_row*self.x_length >= 0] #cut ball if it goes over one row on the left

				ball_ind_x = ball_ind_x[ball_ind_x - y_row*self.x_length < self.x_length]#cut if it goes over the row on the right

				ball =np.tile(ball_ind_x , 2*max_size+1) + (np.arange(-max_size,max_size+1)*self.x_length).repeat(len(ball_ind_x)) #expand it in y direction
				
				ball = ball[ball >=0] # cut it if it goes below the first row
				ball= ball[ball < cell_fire.shape[0]]#cut it if it goes beyond the last one

				quader = set(ball).intersection(loc_fire)# the firing locations in the quader

				
				if len(quader) >=2: # if loc is not isolated
					ind_overlap = np.flatnonzero(np.array(map(quader.isdisjoint, all_cluster_cell))-1)
					if len(ind_overlap)== 0: #if no connection to any cluster
						all_cluster_cell.append(quader)
					else: #if quader has overlap with some other cluster, union those two
						all_cluster_cell[ind_overlap[0]]=all_cluster_cell[ind_overlap[0]].union(quader)
						for i in ind_overlap[1:][::-1]: # and if new quader connects two cluster that where disjoint before, union them as one
							all_cluster_cell[ind_overlap[0]]=all_cluster_cell[ind_overlap[0]].union(all_cluster_cell[i])
							all_cluster_cell.pop(i)
						#all_cluster_cell = list(np.delete(all_cluster_cell, ind_overlap[1:]))
					#not_visited -= quader
				else:
					self.noise[cell]+= len(quader)
			
			if self.si_criterion:
				max_fire = np.max(self.patterns_1)
				for i in np.arange(len(all_cluster_cell))[::-1]:
					av_fire_field = np.sum(self.patterns_1[:,cell][list(all_cluster_cell[i])])*1./(len(all_cluster_cell[i]))
					max_fire_field = np.max(self.patterns_1[:,cell][list(all_cluster_cell[i])])

					if (av_fire_field < 0.1 * max_fire) or max_fire_field < 0.15 * max_fire:
						print 'field removed'
						print av_fire_field
						print max_fire_field
						self.noise[cell]+= len(all_cluster_cell.pop(i))
			if self.min_size != None:
				for i in np.arange(len(all_cluster_cell))[::-1]:
					if len(all_cluster_cell[i]) < self.min_size:
						#print 'pf size < min size!'
						#print len(all_cluster_cell)
						#print all_cluster_cell[i]
						#print len(all_cluster_cell[i])
						self.noise[cell]+= len(all_cluster_cell.pop(i)) #deletes all_cluster_cell[i] and returns it to noise
						#print len(all_cluster_cell)
					else:
						if self.max_size!= None and len(all_cluster_cell[i]) > self.max_size:
							self.noise[cell]+= len(all_cluster_cell.pop(i)) #deletes all_cluster_cell[i] and returns it to noise
						

			for i in np.arange(len(all_cluster_cell))[::-1]:
				self.clusters_colored[np.array([cell]*len(all_cluster_cell[i]), 'int'),np.array(list(all_cluster_cell[i]), 'int')] = 1*(1+10*(i+1))
				self.cluster_size[cell][i] = len(all_cluster_cell[i])*self.pixel_to_cm2
				#print self.cluster_size[cell][i]
			self.number_fields[cell] = copy.copy(len(all_cluster_cell))
		tim = time.time() - begin
		print 'finished in '+str(int(tim/3600)) + 'h '+str(int((tim-int(tim/3600)*3600)/60)) + 'min ' +str(int(tim - int(tim/3600)*3600- int((tim-int(tim/3600)*3600)/60)*60)) + 'sec'
	
	def getDistanceOfMin(self, activity_map = None, ref_point = None):#help function
		#activity_map = self.getSpatial
		loc_min = self.location[np.argmin(activity_map)]
		min_dis = self.getDistanceMatrix(locations = self.locations)[loc_min, ref_point]
		return min_dis
	
	
	################################## average number of fields in certain populations #############################################
	def getAverageFieldNumber(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sum(self.number_fields)*1./(self.patterns_1.shape[1])
		
	def getAverageFieldNumberActiveCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sum(self.number_fields)*1./len(np.flatnonzero(self.number_fields+self.noise))
	
	def getAverageFieldNumberActiveCellsWithField(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.sum(self.number_fields)*1./len(np.flatnonzero(self.number_fields))
	
	def getAverageFieldNumberActiveCellsWithFieldStd(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.std(self.number_fields)
	
		
	################################## average sizes of fields in certain populations #############################################
	def getAverageFieldSize(self): # returns average field size of all fields;
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.sum(self.cluster_size)*1./np.sum(self.number_fields)
		
	def getAverageFieldSizeStd(self): # returns average field size of all fields;
		if (self.cluster_size == 0).all():
			self.calcClusters()

		return np.std(self.cluster_size)
		
	def getAverageFieldSizeCell(self): #returns averge field size of each cell (cell); a cell without have field has av size 0
		if (self.cluster_size == 0).all():
			self.calcClusters()
		av_size = np.zeros(self.patterns_1.shape[1])#(cells)
		non_z = np.nonzero(self.number_fields)# cells that have fields
		av_size[non_z] = np.sum(self.cluster_size[non_z], axis = -1)/(self.number_fields[non_z]+0.0)
		return np.round(av_size,3)
		
	def getAverageCoverActiveCellsWithField(self): #average cover of pc, (sum pf coverage of all fields the cell has)
		if (self.cluster_size == 0).all():
			self.calcClusters()
		av_size = np.zeros(self.patterns_1.shape[1])#(cells)
		non_z = np.nonzero(self.number_fields)# cells that have fields
		av_size[non_z] = np.sum(self.cluster_size[non_z], axis = -1)
		non_z = np.flatnonzero(av_size).shape[0]*1.
		return np.sum(av_size)/non_z
		
	def getAverageCoverActiveCells(self): #average coverage of a active cell
		if (self.cluster_size == 0).all():
			self.calcClusters()
		av_size = np.zeros(self.patterns_1.shape[1])#(cells)
		non_z = np.nonzero(self.number_fields+self.noise)# cells that have fields
		av_size[non_z] = np.sum(self.cluster_size[non_z], axis = -1) + np.sum(self.cluster_size[non_z], axis = -1)
		non_z = np.flatnonzero(av_size).shape[0]*1.
		return np.sum(av_size)/non_z
		
		
	################################## other helpful statistics #############################################		
	def getAverageNoiseActiceCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sum(self.noise)/(len(np.flatnonzero(self.number_fields+self.noise))+0.0)
		
	def getNumberActiveCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return len(np.flatnonzero(self.number_fields+self.noise))+0.0
		
	def getNumberCellsWithField(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return len(np.flatnonzero(self.number_fields))+0.0
		
	def getActiveCell(self, no_cells = 1): #return indizes of active cells
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.flatnonzero(self.number_fields+self.noise)[:no_cells]
	
	def getPlaceCell(self, no_cells = 1): #return indizes of place cells
		if (self.cluster_size == 0).all():
			self.calcClusters()
		cells = np.flatnonzero(self.number_fields)
		np.random.shuffle(cells)
		return cells[:no_cells]
		
	def getProportionActiveCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return self.getNumberActiveCells()/self.patterns_1.shape[-1]
		
	def getProportionCellsWithField(self):
		return self.getNumberCellsWithField()/self.patterns_1.shape[-1]
		
	def getSizesOverNumberFields(self):#returns field sizes dependent of number fields the cell has, (number_fields, sizes)
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_fields = np.max(self.number_fields)

		sizes = []
		for i in range(1,max_fields+1):
			#sizes1 = np.ravel(self.cluster_size[self.number_fields == i])
			#if size1s.shape[0]>=0:
			sizes.append(np.ravel(self.cluster_size[self.number_fields == i][self.cluster_size[self.number_fields == i]>0]))
		return sizes
		
	def getCellsMaxFieldSize(self, cells = 1):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_fields = np.max(self.cluster_size, axis = -1)
		return np.argsort(max_fields)[-cells:]
		
	def getfieldSizes(self):#returns the field sizes considered as fields
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return self.cluster_size[self.cluster_size!= 0]
		
	def getfieldSizesSorted(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.sort(self.cluster_size[self.cluster_size!= 0])
		
	def getCellsWithFieldSize(self, size, cells =1):# return cell indizes (cells many) that have place field size just above size; if there are not enough many of that kind, returns indizes with max size
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_size_cell = np.max(self.cluster_size, axis = -1)
		argsort = np.argsort(max_size_cell)
		return_cell = argsort[max_size_cell[argsort]>=size]
		if return_cell.shape[0] < cells:
			return_cell = self.getCellsMaxFieldSize(cells)
		else:
			return_cell = return_cell[:cells]
		return return_cell
		
	def getMaxFieldSizeCells(self):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return np.max(self.cluster_size, axis = -1)
	
	def getAverageInsideOutsideWrong(self, noisy_patterns):
		
		fire_cells = np.flatnonzero(self.number_fields)
		fire_locs = np.copy(self.clusters_colored[fire_cells])
		noisy_fire_locs = np.copy(noisy_patterns).T[fire_cells] #cells, locations
		fire_locs[fire_locs!=0] = 1

		noisy_fire_locs[noisy_fire_locs!=0] = 1
		wrong = fire_locs- noisy_fire_locs
		
		inside_wrong = np.copy(wrong)
		inside_wrong[inside_wrong == -1] = 0
		inside_wrong = np.sum(inside_wrong, axis =-1)*1./np.sum(fire_locs, axis = -1)

		
		
		outside_wrong = np.copy(wrong)*-1
		outside_wrong[outside_wrong == -1] = 0
		outside_wrong = np.sum(outside_wrong, axis =-1)*1./(self.patterns_1.shape[0]-np.sum(fire_locs, axis = -1))

		return [inside_wrong,outside_wrong]
		
	def getAverageProportionWrong(self, noisy_patterns):
		if (self.cluster_size == 0).all():
			self.calcClusters()
		wrong = np.zeros([noisy_patterns.shape[0], len(np.flatnonzero(self.number_fields))]) #noise; cells with field
		for noise in range(noisy_patterns.shape[0]):
			inside, outside = self.getAverageInsideOutsideWrong(noisy_patterns[noise])
			wrong[noise] = (inside +outside)/2.
		return np.mean(wrong, axis = -1)
		
	def getAverageRadiusCell(self): #the average radius of one cells place fields
		return np.sqrt(self.getAverageFieldSizeCell()/(np.pi*self.patterns_1.shape[0])) # no_pixel = space_res**2 * pi * r**2
	
	def getAverageRadius(self): #the average radius of all place fields in the population
		return np.sqrt(self.getAverageFieldSize()/(np.pi*self.patterns_1.shape[0])) # no_pixel = space_res**2 * pi * r**2
	
	def getSparsityCell(self): #the sparsity level of each cell
		if (self.cluster_size == 0).all():
			self.calcClusters()
		return (np.sum(self.cluster_size, axis = -1) + self.noise)/(self.patterns_1.shape[0]*1.)
		
	def getSparsityLocation(self): #the sparsity level of each location
		return np.sum(self.patterns_1, -1)*1./self.patterns_1.shape[-1] 
		
	def getAverageSparsityLocation(self): #the average sparsity level over locations
		sp = self.getSparsityLocation() #the sparsity level of each location
		return np.sum(sp)/sp.shape[0]
	def plotBump(self, pattern = None):
		

		fig = plt.Figure()
		plt.jet()
		plt.scatter(self.centers[:,0], self.centers[:,1], c= 'w', faceted = 1)
		plt.scatter(self.centers[:,0][np.nonzero(self.patterns_1[pattern])], self.centers[:,1][np.nonzero(self.patterns_1[pattern])], c= 'r', ec = 'none')
		plt.scatter(self.locations[:,0][pattern], self.locations[:,1][pattern], c= 'b', ec = 'none', s = 60)
		#plt.colorbar()
		return fig
	
	def plotCellFiringClusters(self, ax = None, cell = None): #plot the cells spatial firing, each detected place field has different color
		
		if (self.cluster_size == 0).all():
			self.calcClusters()
		
		ax.set_xlim(0,self.cage[0])
		ax.set_ylim(0,self.cage[1])
		#plt.scatter(self.locations[:,0][np.flatnonzero(self.clusters_colored[cell])], self.locations[:,1][np.flatnonzero(self.clusters_colored[cell])], c= 'w', faceted = 1)
		#plt.scatter(self.locations[:,0][np.flatnonzero(self.clusters_colored[cell])], self.locations[:,1][np.flatnonzero(self.clusters_colored[cell])], c= 'g', alpha = 0.5, faceted = False)
		ax.scatter(self.locations[:,0][np.flatnonzero(self.clusters_colored[cell])] ,self.locations[:,1][np.flatnonzero(self.clusters_colored[cell])], c= self.clusters_colored[cell][np.flatnonzero(self.clusters_colored[cell])], alpha = 0.5, faceted = False)
		ax.text(0,1, ('no_field = '+str(self.number_fields[cell]) +'\ns '+ str(self.cluster_size[cell][self.cluster_size[cell]!=0])))
	
	def plotCellFiring(self, cell = None):#just plots the spatial firing

		plt.figure()
		plt.jet()
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.scatter(self.locations[:,0][np.flatnonzero(self.patterns_1[:,cell])], self.locations[:,1][np.flatnonzero(self.patterns_1[:,cell])], c= self.patterns_1[:,cell][np.flatnonzero(self.patterns_1[:,cell])], ec = 'none')
		plt.colorbar()
	
	def plotCellFiring2D(self, cell = None):#just plots the spatial firing
		
		patterns_2d = self.patterns_1.reshape(self.x_length, self.y_length, self.patterns_1.shape[1])[:,:,cell]
		plt.figure()
		plt.jet()
		plt.xlim(0,self.cage[0])
		plt.ylim(0,self.cage[1])
		plt.scatter(np.nonzero(patterns_2d)[0]/(self.x_length*1.), np.nonzero(patterns_2d)[1]/(self.y_length*1.), c= patterns_2d[np.nonzero(patterns_2d)], ec = 'none')
	
	def calcSpatialAutocorrelation2(self, cell = None, mode = 'full'): #calc spatial autocorrelation (Pearson Coefficient); border effects are corrected to 1, -1
		mode = 'full'
		self.number_evaluations = scipy.signal.correlate(np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[1]]), np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[1]]), mode)
		#r = np.zeros(self.number_evaluations.shape)
		xy = scipy.signal.correlate(self.patterns_2d[:,:,cell]*1, self.patterns_2d[:,:,cell]*1, mode)/(self.number_evaluations*1.)
		
		mean = np.sum(self.patterns_1[:,cell])*1./self.patterns_1.shape[0] #mean of firing map
		std = np.sqrt(mean - mean**2)#std of firing map
		
		mean_shifted_map = scipy.signal.correlate(np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[1]]), self.patterns_2d[:,:,cell]*1, mode)/(self.number_evaluations*1.) #means of the shifted maps; differs form mean, since not all locations are considered if map is shifted and zero padded (with zeros extended at the border)
		std_shifted_map = np.sqrt(mean_shifted_map - mean_shifted_map**2)
		
		#r[std_shifted_map!=0] = (xy[std_shifted_map!=0] - mean * mean_shifted_map[std_shifted_map!=0])/(std * std_shifted_map[std_shifted_map!=0])
		r = (xy - mean * mean_shifted_map)/(std * std_shifted_map) # pearson coefficeint
		
		#map corrected outsiders
		#plt.figure()
		#if mode == 'full':
			#grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)

		#if mode == 'same':
			#grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		#plt.scatter(grid[0][r**2>1],grid[1][r**2>1], r[r**2>1])
		#plt.jet()
		#plt.colorbar()
		r[r>1] = 1 # correct outsiders
		r[r<-1] = -1
		return r.T
		
	def calcSpatialAutocorrelation(self, cell = None, mode = 'full'): #calc spatial autocorrelation (Pearson Coefficient); border effects are corrected to 1, -1

		
		cell_firing = self.patterns_white[:,cell].reshape(self.y_length, self.x_length)
		xy = scipy.signal.correlate2d(cell_firing,cell_firing, mode = mode)
		return xy		
		
	def getSpatialAutocorrelation(self, cell = None, mode = 'full'):#plot spatial autocorrelation of cell, or if cell = None, the average of spatial corrleations of all cells
		
		if cell ==None:
			auto_corr = self.calcSpatialAutocorrelation(cell = 0, mode = mode)
			for cell in range(1,self.patterns_1.shape[1]):
				auto_corr += self.calcSpatialAutocorrelation(cell = cell, mode = mode)
			auto_corr /= self.patterns_1.shape[1]*1.* self.patterns_1.shape[0]

		
		else:
			auto_corr = self.calcSpatialAutocorrelation(cell = cell, mode = mode)/(1.* self.patterns_1.shape[0])
		
		#if mode == 'full':
			#grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)

		#if mode == 'same':
			#grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		return auto_corr
	
	def plotSpatialAutocorrelation(self, cell = None, mode = 'full'):#plot spatial autocorrelation of cell, or if cell = None, the average of spatial corrleations of all cells
		
		if cell ==None:
			auto_corr = self.calcSpatialAutocorrelation(cell = 0, mode = mode)
			for cell in range(1,self.patterns_1.shape[1]):
				auto_corr += self.calcSpatialAutocorrelation(cell = cell, mode = mode)
			auto_corr /= self.patterns_1.shape[1]*1.

		else:
			auto_corr = self.calcSpatialAutocorrelation(cell = cell, mode = mode)
		
		plt.figure()
		plt.jet()
		if mode == 'full':
			grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[1]+1:self.patterns_2d.shape[1]]/(self.x_length+0.0)

		if mode == 'same':
			grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[1]/2:self.patterns_2d.shape[1]/2]/(self.x_length+0.0)
		plt.scatter(grid[0],grid[1],c =auto_corr)
		plt.colorbar()
		
	def getSpatialAutoOverlap(self, cell = None, mode = 'full'):#calc overlap with original and shifted map; normalized; border effects are corrected
		mode = 'full'
		
		def calc():
			no_firing_times = np.sum(self.patterns_1[:,cell])
			if no_firing_times == 0:
				r = np.ones(np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]][0].shape)			
			else:
				self.number_evaluations = scipy.signal.correlate(np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[0]]), np.ones([self.patterns_2d.shape[0], self.patterns_2d.shape[0]]), mode)#number of locations that are compared (less shift many locations are compared, great shift, just little many)
				r = scipy.signal.correlate(self.patterns_2d[:,:,cell]*1, self.patterns_2d[:,:,cell]*1, mode)*self.patterns_1.shape[0]/(self.number_evaluations*1.*no_firing_times)#normalized by number of locations considered and 1/a
				r[r>1] = 1 # correct outsiders
			return r.T
			
			
		if cell == None:
			cell = 0
			r = calc()
			for i in range(1,self.patterns_1.shape[1]):
				cell +=1
				r += calc()
			r /= self.patterns_1.shape[1]*1.
		else:
			r = calc()
		return r
		
	def plotSpatialAutoOverlap(self, cell = None, mode = 'full'): #plots  spatial overlap of cell, or average of all cells
		
		
		if cell ==None:
			auto_corr = self.calcSpatialAutoOverlap(cell = 0, mode = mode)
			for cell in range(1,self.patterns_1.shape[1]):
				auto_corr += self.calcSpatialAutoOverlap(cell = cell, mode = mode)
			auto_corr /= self.patterns_1.shape[1]*1.
		else:
			auto_corr = self.calcSpatialAutoOverlap(cell = cell, mode = mode)
		
		plt.figure()
		plt.jet()
		if mode == 'full':
			grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)
		if mode == 'same':
			grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		plt.scatter(grid[0],grid[1],c =auto_corr)
		plt.xlim([-1,1])
		plt.ylim([-1,1])
		plt.colorbar()
		
	def getAutoCorrelationPoints(self, mode = 'full'): #returns points to plot autocorrelation scatterplot

		if mode == 'full':
			grid = np.mgrid[-self.patterns_2d.shape[0]+1: self.patterns_2d.shape[0], -self.patterns_2d.shape[0]+1:self.patterns_2d.shape[0]]/(self.x_length+0.0)
		if mode == 'same':
			grid = np.mgrid[-self.patterns_2d.shape[0]/2: self.patterns_2d.shape[0]/2, -self.patterns_2d.shape[0]/2:self.patterns_2d.shape[0]/2]/(self.x_length+0.0)
		return grid

	def getSpatialInformation(self):
		mean_cell = np.sum(self.patterns_1, -2)
		non_zero_cells = np.flatnonzero(mean_cell != 0)
		mean_cell = np.tile(mean_cell, (self.patterns_1.shape[0],1))/(self.patterns_1.shape[0]*1.)
		mean_cell[mean_cell==0] = 1

		terms = self.patterns_1*1./mean_cell
		terms[terms == 0.0] = 10**-10
		r = np.zeros(self.patterns_1.shape[-1])

		try: 
			r[non_zero_cells] =np.sum(terms[:,non_zero_cells] * np.log2(terms[:,non_zero_cells]), -2) *1./self.patterns_1.shape[0]
		except RuntimeWarning:
			print 'mins', np.min(terms),'\t',np.min(self.patterns_1),'\t',np.min(mean_cell)
		#print 'mins', np.min(terms),'\t',np.min(self.patterns_1),'\t',np.min(mean_cell)
		return r



################################################Input Classes ######################################################
class Input(Corelations):
	
	'''
	
	Class that creates pattern and builts noisy versions of them.
	
	:param number_cells: number of cells
	:type number_cells: int
	:param n_e: Number of environments
	:type n_e: int
	:param number_patterns: number of total pattern this instance creates
	:type number_patterns: int
	:param store_indizes: indices of patterns that are considered for storing
	:type store_indizes: array of dimension 2 (environment, indices)
	:param number_to_store: number of patterns that are considered for storing
	:type number_to_store: int
	:param inputMethod: How patterns are created
	:type inputMethod: Input.makeInput method
	:param noiseMethod: How noise is created
	:type noiseMethod: Input.makeNoise method
	:param actFunction: How activation is transformed into firing
	:type actFunction: Input.getOutput method
	:param sparsity: Proportion of cells beeing active in each pattern; for getOutputWTA necessary
	:type sparsity: float
	:param noise_levels: levels of noise beeing applied on the pattern; mostly number of cells firing wrongly
	:type noise_levels: list or array of noise_levels
	:param normed: Whether patterns are normalized
	:type normed: bbol
	:param In: When given it uses its paramters
	:type In: Instance of Input
	:param patterns: When given, it uses these patterns as self.patterns
	:type patterns: array of dimension 3 (environment, pattern, cell fire)
	'''
	def __init__(self, number_cells = Parameter.cells['Ec'], n_e = Parameter.n_e,  number_patterns= Parameter.no_pattern, store_indizes = None, number_to_store = None, inputMethod = None, noiseMethod = None, actFunction = None, sparsity = Parameter.sparsity['Ec'], noise_levels = Parameter.noise_levels, normed = 0, In = None, patterns = None, lec_cells = 0, **kwargs):


		self.n_e = n_e
		self.number_patterns = number_patterns #per env
		self.number_to_store = number_to_store #per env
		self.inputMethod = inputMethod
		self.noiseMethod = noiseMethod
		self.actFunction = actFunction
		self.store_indizes = store_indizes

		self.cells = number_cells
		self.noise_levels = np.array(noise_levels)
		self.sparsity = sparsity
		self.n_lec = lec_cells
		
		self.number_winner = int(sparsity*self.cells)


		if number_to_store ==  None:
			self.number_to_store = self.number_patterns
		if In != None: #uses Parameter of given Input Instance
			self.n_e = In.n_e
			self.number_patterns = In.number_patterns #per env
			self.number_to_store = In.number_to_store #per env
			self.inputMethod = In.inputMethod
			self.noiseMethod = In.noiseMethod
			self.actFunction = In.actFunction
		if patterns == None: # uses given patterns as self.patterns
			self.patterns = np.zeros([self.n_e,self.number_patterns, self.cells])
			self.patterns_given = False
		else:
			self.patterns = patterns
			self.patterns_given = True
		
		self.input_stored = np.zeros([self.n_e, self.number_to_store, self.cells])#patterns that are considered for storing
		self.noisy_input_stored = np.zeros([self.n_e,self.noise_levels.shape[0], self.number_to_store, self.cells])#noisy versions of self.input_stored

		#creates self.patterns

		self.makeInput(**kwargs)
		if normed:
			normalize(self.patterns)
		#create self.input_stored and self.noisy_imput_stored
		self.choosePatternToStore(store_indizes = store_indizes)
	
		#Input Instance is a Corelation Instance at the same time.
		print '_____Input_____'
		print self.input_stored.shape
		print self.noisy_input_stored.shape
		super(Input, self).__init__(patterns_1 = self.input_stored, patterns_2= self.noisy_input_stored)
		
	def makeInput(self, **kwargs):
		'''
		
		creates the input; all input patterns are self.patterns. It has dimensions (envionment, pattern, cell firing)
		'''
		if not self.patterns_given:
			if self.inputMethod == None:
				print 'noinput Method gven'
				self.makeInputSparsityLevel()
			else:
				self.inputMethod(self)

		else:
			pass
		
	def makeNoise(self, env = None):
		'''
		
		creates the noisy version of the input pattern self.patterns.
	
		:param env: Environment for which noise is created
		'''
		if self.noiseMethod == None:
			noise =self.makeNoiseRandomFire(pattern = self.patterns[env], noise_levels= self.noise_levels)
		else:
			noise = self.noiseMethod(self, pattern = self.patterns[env], noise_levels = self.noise_levels)
			
		return noise
	
	def choosePatternToStore(self, number_to_store = None, store_indizes = None):
		'''
		
		sets the input patterns in self.patterns that are going to be stored. These patterns are then self.input_stored and their noisy version are in self.noisy_input_stored. 
	
		:param number_to_sore: How many patterns are stored; if None, self.number_to_store is used
		:type number_to_store: int
		:param store_indizes: Indices of patterns that are stored. If None, store_indizes are created by self.makeStoreIndizes(store_indizes)
		:type store_indizes: array of dimension two; (environment, indizes)
		'''
		if number_to_store != None:
			self.number_to_store = number_to_store
		self.makeStoreIndizes(store_indizes)
		self.input_stored = np.zeros([self.n_e, self.number_to_store, self.cells])
		self.noisy_input_stored = np.zeros([self.n_e,self.noise_levels.shape[0], self.number_to_store, self.cells])
		for h in range(self.n_e):
			self.input_stored[h] = self.patterns[h][self.store_indizes[h]] #the pattern that are actually given as inputs; note that self.input_stored[i] corresponds to self.location[self.store_indized][i]
			self.noisy_input_stored[h] = self.makeNoise(h)[:,self.store_indizes[h]]
		print 'self nois input sortede'
		print self.noisy_input_stored[0,-1,0]

	def makeStoreIndizes(self, store_indizes=None):
		'''
		
		creates store indizes randomly if store_indizes = None. If store_inidizes != None, these indizes are used.
		
		:param store_indizes: Indices of patterns that are stored. If None, store_indizes are created by self.makeStoreIndizes(store_indizes)
		:type store_indizes: array of dimension two; (environment, indizes)
		'''
		if self.number_to_store > self.number_patterns:
			print 'not enpugh input patterns'
		if store_indizes == None:
			print 'make store indizes'
			self.store_indizes = np.array(map(random.sample, [range(self.patterns.shape[1])]*self.n_e, [self.number_to_store]*self.n_e))
		else:
			self.store_indizes = store_indizes
			print 'store indizes given'
		
	def makeNewNoise(self, method = None):
		'''
		
		creates noisy patterns, when Input Instance was already created. Old noise is overwritten
		
		:param method: Which method is used for creation of noise
		:type method: Input.makeNoise method.
		'''
		self.noiseMethod = method
		self.noisy_input_stored = np.zeros([self.n_e,self.noise_levels.shape[0], self.number_to_store, self.cells])
		for h in range(self.n_e):
			self.noisy_input_stored[h] = self.makeNoise(h)[:,self.store_indizes[h]]
		super(Input, self).__init__(patterns_1 = self.input_stored, patterns_2= self.noisy_input_stored)
		
	
	#################inputMethods#########################
	##creates the input patterns self.patterns
	def makeInputNormalDistributed(self):
		'''
		
		creates patterns, each cell activity is a sample of a normal distribution
		'''
		activity = np.random.normal(loc = 1, scale = 1, size = self.patterns.shape)
		self.patterns = self.actFunction(self, activity = activity)
		
	def makeInputSparsityLevel(self):
		'''
		
		creates patterns, the proportion of cell that fire in each pattern is self.sparsity. These cells have value 1, the others 0
		'''
		active_units = np.int(self.sparsity*self.patterns.shape[-1])
		for h in range(self.n_e):
			for p in self.patterns[h]:
				p[np.array(random.sample(range(self.patterns.shape[-1]), active_units))] = 1
	
	def makeInputExponential(self):
		'''
		
		creates patterns; cells fire according to an exponetial distribution. The proportion of cells firing is self.sparsity 
		'''
		silent_p = 1-2*self.sparsity
		uniform = np.random.uniform(0,1, (self.patterns.shape))
		silent_cells = uniform <= silent_p
		activity = scipy.stats.expon.rvs(size = self.patterns.shape)*2
		activity[silent_cells] = 0
		a = Network.calcFireSparsity(activity)
		self.patterns = activity
	
	
	#####################noiseMethods##########################################
	# These methods are used to create noisy version of given patterns
	def makeNoiseAccordingSparsity(self,pattern=None, noise_levels=None):
		'''
		
		Each noise cell fires with P(fire) = self.sparsity. A cell that fires have valeu 1 others 0
	
		:param pattern: Patterns which are made noisy. 
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of cells that fire wrongly
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		j=1
		for i in noise_levels[1:]:
			wrong = np.array(map(random.sample, [range(pattern.shape[-1])]*pattern.shape[-2], [i]*pattern.shape[-2]))
			noise[j][np.mgrid[0:pattern.shape[0], 0:i][0], wrong] = np.random.uniform(0,1, size =(wrong.shape)) <= self.sparsity
			j +=1
		return noise

	def makeNoiseRandomFire(self,pattern=None, noise_levels=None): 
		'''
		
		Each noisy cell fires accroding to the rate of an arbritrayly chosen cell in that pattern
	
		:param pattern: Patterns which are made noisy. 
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of cells that fire wrongly
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		j=1
		for i in noise_levels[1:]:
			wrong = np.array(map(random.sample, [range(pattern.shape[-1])]*pattern.shape[0], [i]*pattern.shape[0])) # for each pattern a random set of cells is chosen to fire wrongly 
			noise[j][np.mgrid[0:pattern.shape[0], 0:i][0], wrong] = np.array(random.sample(np.ravel(pattern), i*pattern.shape[0])).reshape(wrong.shape)
			j +=1
		return noise

	def makeNoiseZero(self,pattern=None, noise_levels=None): 
		'''
		
		Each noisy cell becomes silent. 
	
		:param pattern: Patterns which are made noisy. 
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here the noise_levels increase linearly from 0 to number_winner in noise_levels.shape[0] steps. Each level determines, how many cells that fired before are silent in the noisy pattern
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		max_noise = pattern.shape[-1]
		for p in pattern:
			max_noise = min(np.flatnonzero(p).shape[0], max_noise)
		self.noise_levels = np.array(np.linspace(0, max_noise-1, noise_levels.shape[0]), dtype = 'int')
		noise = np.tile(pattern, (len(noise_levels),1,1))
		level=1

		for i in self.noise_levels[1:]:
			for j in range(pattern.shape[0]):
				wrong = random.sample(np.flatnonzero(pattern[j]), i)
				noise[level,j][wrong] = 0
			level +=1

		return noise
	
	def makeNoiseDoNothing(self,pattern=None, noise_levels=None):
		return self.patterns[0].reshape([self.noise_levels.shape[0], self.number_patterns, self.cells])
	
	#later important
	def makeNoiseVector(self,pattern=None, noise_levels=None):
		'''
		
		For each noise_level a noise vector is added to the pattern. The noise vector is a random vector created by a normal distribution.
	
		:param pattern: Patterns which are made noisy. 
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of noise vectors added.
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		rand_vectors = np.random.normal(size = (len(noise_levels), self.cells))
		normalize(rand_vectors)
		rand_vectors *= 10
		noise_vector = rand_vectors[0]
		for i in range(len(noise_levels)-1):
			noise[i+1] += noise_vector
			noise_vector += rand_vectors[i+1]
		return noise
	
	def makeNoiseCovariance(self,pattern=None, noise_levels=None):
		'''
		
		For each noise_level a noise vector is added to the pattern. The first noise vector is the eigenvector with the largest eigenvalue of the covariance matrix of self.patterns. The second is the sum of the first two and so on.  
		
		:param pattern: Patterns which are made noisy. 
		:type pattern: array of dimension 2 (pattern, cell fire)
		:param noise_levels: levels of how much amount of noise is given to pattern. Here it is the number of noise vectors added.
		:type noise_levels: array of dimension 1
		:param return: noisy version of patterns
		:type return: array of dimension 3 (noise_level, pattern, cell fire)
		'''
		noise = np.tile(pattern, (len(noise_levels),1,1))
		Cor = Corelations(patterns_1 = self.patterns)
		ev = Cor.getEigenvectorsCovariance().T
		normalize(ev)
		ev *= 10
		noise_vector = ev[0]
		for i in range(len(noise_levels)-1):
			noise[i+1] += noise_vector
			noise_vector += ev[i+1]
		return noise
		
	###################### getOutputFunctions #################
	### Similar as in class Network
	#important for you
	def getOutputWTA(self, activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
		'''
		
		calculates outputfiring  given activity; the highest self.number_winner activated neurons fire;
	
		:param activtiy: activity
		:type activity: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: array
		'''
		size = activity.shape
		winner = np.argsort(activity)[...,-self.number_winner:size[-1]]
		fire_rate = np.ones(size, 'bool')
		out_fire = np.zeros(size, 'bool')
			
		if len(size) ==1:
			out_fire[winner] = fire_rate[winner]
		if len(size) ==2:
			out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
		if len(size) ==3:
			indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
			out_fire[indices[0], indices[1], winner] =1#fire_rate[indices[0], indices[1], winner]
		if len(size) ==4: # env, noise, time, pattern
			indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
			out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
		if len(size) > 4:
				print 'error in input dimension in calckWTAOutput'
		return out_fire
	
	def getOutputWTALinear(self, activity = None): #returns the winner of the network given the input patterns; if not discrete, output is the membrane potential of the winners
		'''
		
		calculates outputfiring  given activity; the highest self.number_winner activated neurons fire;
		
		:param activtiy: activity
		:type activtiy: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: array
		'''
		size = activity.shape
		winner = np.argsort(activity)[...,-self.number_winner:size[-1]]


		fire_rate = activity
		print fire_rate.shape
		#for f in fire_rate:
			#for e in f:
				#print e
			#print '----------------------------------------'
		fire_rate[fire_rate<=0] = 0
		out_fire = np.zeros(size)
			
		if len(size) ==1:
			out_fire[winner] = fire_rate[winner]
		if len(size) ==2:

			out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
		if len(size) ==3:
			indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
			out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
		if len(size) ==4: # env, noise, time, pattern
			indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
			out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
		if len(size) > 4:
				print 'error in input dimension in calckWTAOutput'
		return out_fire
	
	#not important
	def getOutputWTARolls(self, activity = None):
		'''
		
		calculates outputfiring  given activity; the highest self.number_winner activated neurons fire;
	
		:param activtiy: activity
		:type activtiy: array of max 4 dimension, last one must be self.connection.shape[1]
		:param return: array
		'''
		size = activity.shape
		winner = np.argsort(activity)[...,-self.number_winner:size[-1]]

		max_activity_pattern = np.max(activity, -1)
		fire_rate = max_activity_pattern.repeat(activity.shape[-1]).reshape(activity.shape)
		out_fire = np.zeros(size)
			
		if len(size) ==1:
			out_fire[winner] = fire_rate[winner]
		if len(size) ==2:
			out_fire[np.mgrid[0:size[0], 0:self.number_winner][0], winner] = fire_rate[np.mgrid[0:size[0], 0:self.number_winner][0], winner]
		if len(size) ==3:
			indices = np.mgrid[0:size[0],0:size[1],0:self.number_winner]
			out_fire[indices[0], indices[1], winner] =fire_rate[indices[0], indices[1], winner]
		if len(size) ==4: # env, noise, time, pattern
			indices = np.mgrid[0:size[0],0:size[1],0:size[2],0:self.number_winner]
			out_fire[indices[0], indices[1], indices[2], winner] = fire_rate[indices[0], indices[1], indices[2], winner]
		if len(size) > 4:
				print 'error in input dimension in calckWTAOutput'
		return out_fire

	def getOutputLinearthreshold(self, activity = None):
		
		activity[activity <0] = 0
		for k in np.linspace(np.min(activity),np.min(np.max(activity, -1)), 20):
			
			if k>=np.max(activity):
				print 'break precalc since otherwise all 0'
				break
			activity_help = np.copy(activity)
			activity_help[activity_help <= k] = 0
			a = Network.calcFireSparsity(activity_help)
			if (a < self.sparsity).any():
				print 'break precalc linthreshold output at k = '+str(k)
				break
			else:
				activity = activity_help
		
		if len(self.patterns.shape) >= 2:
			a = Network.calcFireSparsity(activity)
			not_sparse = True
			i=0
			while not_sparse:
				too_large_a_rows = a > self.sparsity
				if not too_large_a_rows.any():
					not_sparse = False
					break
				activity[activity == 0] = 10**10
				min_activity_cell = np.argmin(activity, -1) 
				activity[too_large_a_rows, min_activity_cell[too_large_a_rows]] = 0
				activity[activity == 10**10] = 0
				a = Network.calcFireSparsity(activity)
				i+=1	
		if len(self.patterns.shape) == 1:
			a = Network.calcFireSparsity(activity)
			not_sparse = True
			i=0
		
			while a > self.sparsity:
				min_activity = np.min(activity[activity !=0])
				activity[activity == min_activity] = 0
				a = Network.calcFireSparsity(activity)
				i+=1
			normalize(activity)
		return activity

	def getOutputId(self, activity = None):
		return activity

	def getOutputMinRate(self, activity = None, min_rate = 0):
		min_fire = np.max(activity, axis = -2) * min_rate
		min_fire = np.tile(min_fire , (activity.shape[-2],1)).reshape(activity.shape)
		activity[activity < min_fire] = 0
		return activity

	def getNoiseInd(self, at_corelation = None): # 
		'''
		
		return best index i where Input has at noise_level[i] correlation most similar to at_corlation
		'''
		noise_ind = np.argmin(np.abs(self.getOrigVsOrig() - at_corelation))
		return noise_ind

	def getInput(self):
			return self.patterns
	
	def getNoise(self):
		return self.noise
		
class Grid(Input, Spatial):
	'''
	
	Input Class that creates Grid pattern.
	
	:param grid_mode: how grid parameter are set up. If 'linear' spacings increase linearly from 30-50cm, orientations are random. If 'modules' it has 4 Modules with similar spacings and orientations. Phase is in both cases randomly chosen
	:type grid_mode: 'linear' or 'modules'
	:param rat: When grid_mode is 'modules' it determines how modules are split up. rat = 0 it has only modules 1 and 2, rat=1 it has all 4, rat=2 it has only modules 3 and 4.
	:type rat: 0,1 or 2
	:param space: If grid_mode = 'linear' it uses space as spacings, rather than 30-50cm
	:type space: array with lenght equal to number_cells
	'''

	def __init__(self,  grid_mode = 'modules', rat =1, spacings = None, peak = None, theta = None, phase = None, cage = [1,1], r_to_s = 0.25,**kwargs):
		


		#Grid Attributes
		self.spacing = spacings
		self.peak_firing_rate = peak
		self.theta = theta
		self.phase = phase

		
		self.theta_given = 0
		self.spacing_given = 0
		self.peak_given = 0
		self.phase_given = 0
		if spacings != None:
			self.spacing_given = True
		if peak != None:
			self.peak_given = True
		if theta != None:
			self.theta_given = True
		if phase != None:
			self.phase_given = True


		self.grid_mode = grid_mode
		self.rat = rat
		self.cage = cage
		self.r_to_s =r_to_s # radius of field / spacing of cell; Defining field as everthing above 0.2 peak rate, then in Knierim r_to_s = .17. Using Hafting r_to_s is in [0.3, 0.32]; Supposing that at 0.21 * spacing fire rate is 0.5 * peak (as in deAlmeida), r_to_s = 0.32.
		self.k = self.r_to_s**2/np.log(5) #see calcActivity
		
		self.locations = None #x and y coordinates of all pixel 
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		Input.__init__(self, **kwargs)
		#print 'grid init', self.patterns_1.shape
		Spatial.__init__(self, patterns = self.patterns[0], cage = cage, **kwargs)

		
	def makeLocations(self, number_patterns):
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.int(np.sqrt(number_patterns/self.cage_proportion))

		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
				self.x_length= self.cage_proportion*self.space_res
				self.y_length = self.space_res
			#self.locations = self.locations.reshape(number_patterns, 2)
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)

				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]
				self.y_length= self.cage_proportion*self.space_res
				self.x_length = self.space_res
				#self.locations = self.locations.reshape(number_patterns, 2)

		
	

	
	def makeInput(self,**kwargs):
		print 'kwargs in makeinput Grid', kwargs
		if self.n_lec >0:
			self.Lec = Lec( number_cells = self.n_lec, n_e = self.n_e,  number_patterns= self.number_patterns, store_indizes = self.store_indizes, number_to_store = self.number_to_store, inputMethod = self.inputMethod, noiseMethod = Lec.makeNoiseDoNothing, actFunction = Input.getOutputWTALinear, sparsity = self.sparsity, noise_levels = [0], cage = self.cage, **kwargs)
		self.makeGrid()
		self.patterns = self.calcGridFiring()

	def makeGrid(self): # makes grid; set ups the parameter
		
		self.n_grid = self.cells - self.n_lec
		ec_firing = np.zeros([self.n_e, self.space_res**2, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent, self.cells = number grid cells
		if not self.spacing_given:
			self.spacing = np.zeros([self.n_e,self.n_grid]) +0.0	# space between two peaks in one grid cell
		if not self.theta_given:
			self.theta = np.zeros([self.n_e,self.n_grid])		# Orientation of the grid			
		if not self.peak_given:
			self.peak_firing_rate = np.ones([self.n_e,self.n_grid, 50, 50])# Peak firing rate
		if not self.phase_given:
			self.centers = np.zeros([self.n_e, self.n_grid, 2])	# Origin of the grid
		else:
			self.centers = np.ones([self.n_e, self.n_grid, 2]) * self.phase	# Origin of the grid
		self.field_size = np.zeros([self.n_e,self.n_grid])	# Field size in one grid cell
		self.theta_arc =np.zeros([self.n_e,self.n_grid])	#Orientation of the grid presented in Pi coordinates
		self.rotation = np.zeros([self.n_e,self.n_grid,2,2]) #Rotation Matrices; We first bulit always a grid with 0 orientation and then rotate the whole thing to the true orientation via these matrices


		# for each enviromnent define grid parameters similar found in literature; linear
		if self.grid_mode == 'linear':
			self.gridLinear()
		if self.grid_mode == 'modules':
			self.gridModules()
		
		# parameters needed for calculating the rate:
		self.cos_spacing = self.spacing*np.cos(np.pi/3) # how far i go to the right-left direction, when I move one grid point upwards
		self.sin_spacing = self.spacing*np.sin(np.pi/3) # how far i go to the up-down direction, when I move one grid point to the right
		#self.std = self.field_size**2/np.log(5) #modulates the Gauss according to place field size; 
		
	
	def gridLinear(self):
		'''
		
		Grid is made according to Hafting 2005:  with random origin (phase), random orientation, spacing increasing from 30-50cm.
		'''
		self.modules = np.array(np.linspace(0, self.n_grid, 5), 'int')
		for h in range(self.n_e):
			theta_change = np.random.uniform(0,60)
			origin_change = np.random.uniform(-.25, 0.25, size = 2)
			for i in range(self.n_grid):
				noise_spacing = 0#random.gauss(0, 0.01) #noise mean 0, varianz=1cm
				noise_field_size = 0#random.gauss(0, 0.01)#noise varianz=1cm
				if not self.peak_given:
					self.peak_firing_rate[h][i] = 1
				if h == 0:
					if not self.spacing_given:
						self.spacing[h][i] = 0.3 + (0.2/self.n_grid)*i + noise_spacing 	#Generate a baseline spacing, starting at 0.30m and increasing linearly to0.50m
					#if self.fields == None:
						#self.field_size[h][i] = 0.0977 + 0.0523*(1.0/self.n_grid)*i+ noise_field_size 	#Generate a baseline field size, starting at 9,77cm and increasing linearly 15cm; field size in radius; (area from 300-700cm**2)
					#else:
						#self.field_size[h][i] = self.fields[i]
					if not self.theta_given:
						self.theta[h][i] = random.uniform(0, 360)
					if not self.phase_given:
						self.centers[h][i] = np.array([random.uniform(0, 1),random.uniform(0, 1)])
				else:
					self.spacing[h][i] = self.spacing[0][i] + noise_spacing
					self.field_size[h][i] = self.field_size[0][i] + noise_field_size
					self.theta[h][i] = self.theta[0][i] + theta_change #+ random.gauss(0,2)
					self.centers[h][i] = self.centers[0][i] + origin_change
				self.theta_arc[h][i] = (2*np.pi*self.theta[h][i])/360.
				self.rotation[h][i] =[[np.cos(self.theta_arc[h][i]), np.sin(self.theta_arc[h][i])], [-np.sin(self.theta_arc[h][i]), np.cos(self.theta_arc[h][i])]]


	def gridModules(self):
		'''
		
		Grid is made according to Solstad 2012: It consists of 4 Modules, each having similar orientation and spacing. Phase is random.
		'''
		
		if self.rat ==0: #m1 and m2 only
			m1 = 19 + 2.5
			m2 = 30+2.5
			m3 = 0
			m4 = 0
			thres = np.array([m1, m1+m2, m1+m2+m3, 54.0001])/54.
			spacing_mean = [38.8, 48.4, 65., 98.4]
			spacing_var = [8,8,6,16]
			orient_mean = [-5, -5, 5, -5]
			orient_var = [3,3, 6, 3]
		
		
		if self.rat ==1:
			m1 = 19 + 2.5 #34.7%
			m2 = 30+2.5 #52.4%
			m3 = 5 #8%
			m4 = 3 #4.8%
			thres = np.array([m1, m1+m2, m1+m2+m3, 62.0001])/62.
			spacing_mean = [38.8, 48.4, 65., 98.4]
			#spacing_var = [8,8,6,16]
			spacing_var = [8,8,8,8]
			#orient_mean = [-5, -5, 5, -5]
			#orient_mean = np.random.uniform(0,60, size = 4)
			orient_mean = np.array([15,30,45,60])
			#orient_var = [3,3, 6, 3]
			orient_var = [3,3, 3, 3]
			
		if self.rat ==2:#only m3 and m4
			m1=0
			m2=0
			m3 = 5
			m4 = 3
			thres = np.array([m1, m1+m2, m1+m2+m3, 8.0001])/8.
			spacing_mean = [38.8, 48.4, 65., 98.4]
			spacing_var = [8,8,6,16]
			orient_mean = [-5, -5, 5, -5]
			orient_var = [3,3, 6, 3]
			#!!!!
			#orient_mean = [-5, -5, -5, -5]
			#orient_var = [3,3, 3, 3]
			
		self.modules = np.zeros(5, 'int')	
		choice = np.random.uniform(low = 0, high =1, size = self.n_grid) + 0.000001
		for i in range(1,self.modules.shape[0]):
			self.modules[i] = np.flatnonzero(choice <= thres[i-1]).shape[0]
			self.spacing[0][self.modules[i-1]:self.modules[i]] = np.random.normal(loc = spacing_mean[i-1], scale = np.sqrt(spacing_var[i-1]), size = self.modules[i]- self.modules[i-1])/100.
			self.theta[0][self.modules[i-1]:self.modules[i]] = np.random.normal(loc = orient_mean[i-1], scale = np.sqrt(orient_var[i-1]), size = self.modules[i]- self.modules[i-1])
		self.centers[0] = np.random.uniform(low =0, high = 1, size=(self.n_grid, 2))
		self.theta_arc[0] = (2*np.pi*self.theta[0])/360.
		for i in range(self.n_grid):
			self.rotation[0][i] =[[np.cos(self.theta_arc[0][i]), np.sin(self.theta_arc[0][i])], [-np.sin(self.theta_arc[0][i]), np.cos(self.theta_arc[0][i])]]
		
		if self.n_e >1:
			for h in range(1, self.n_e):
				theta_change = np.random.uniform(0,60, size = 4)
				origin_change = np.random.uniform(-.25, 0.25, size = (4,2))
				noise_spacing = 0#random.gauss(0, 0.01) #noise mean 0, varianz=1cm
				noise_field_size = 0#random.gauss(0, 0.01)#noise varianz=1cm
				for m in range(1,5):
					for i in range(self.modules[m-1],self.modules[m]):
						if not self.peak_given:
							self.peak_firing_rate[h][i] = 1
						self.spacing[h][i] = self.spacing[0][i] + noise_spacing
						self.field_size[h][i] = self.field_size[0][i] + noise_field_size
						self.theta[h][i] = self.theta[0][i] + theta_change[m-1]
						self.centers[h][i] = self.centers[0][i] + origin_change[m-1]
						self.theta_arc[h][i] = (2*np.pi*self.theta[h][i])/360.
						self.rotation[h][i] =[[np.cos(self.theta_arc[h][i]), np.sin(self.theta_arc[h][i])], [-np.sin(self.theta_arc[h][i]), np.cos(self.theta_arc[h][i])]]



	def calcActivity(self, h, location):
		'''
		 calculates and returns the activity in environment h at location location
		'''
		
		t_loc = np.einsum('ijk,ik->ij' ,self.rotation[h], (location - self.centers[h])) # shift and rotate location into coordinates, where the grid has origin at 0 and rotation 0
		k= np.floor(t_loc[:,1]/(self.sin_spacing[h])) #nearest vertex in up-down (y)  direction is the kth or k+1th one
		y1 = (t_loc[:,1] - k*self.sin_spacing[h]) # assume it is the kth one
		y2 = (t_loc[:,1] - (k+1)*self.sin_spacing[h]) # assume it is the k +1th one
		kx1 = np.round((t_loc[:,0]- k*self.cos_spacing[h])/(self.spacing[h])) 
		kx2 = np.round((t_loc[:,0]- (k+1)*self.cos_spacing[h])/(self.spacing[h]))
		x1 = (t_loc[:,0]-k*self.cos_spacing[h]- kx1*self.spacing[h])# nearest vertex in x direction for y1 (if it is the kth one)
		x2 = (t_loc[:,0]-(k+1)*self.cos_spacing[h]- kx2*self.spacing[h])# nearest vertex in x direction for y2
		#dis = np.minimum((x1*x1 + y1*y1), x2*x2+y2*y2) # = euclidean distance**2 to nearest peak
		
		arg = np.argmin(np.array([x1*x1 + y1*y1, x2*x2+y2*y2]), axis = -2)
		vertX = np.array([kx1, kx2], 'int')[arg, np.arange(arg.shape[0])]
		vertY = np.array([k, k+1], 'int')[arg, np.arange(arg.shape[0])]
		dis = np.array([[x1*x1 + y1*y1], [x2*x2+y2*y2]])[arg, np.zeros(arg.shape[0], 'int'), np.arange(arg.shape[0])]

		#activity= np.exp(-dis/self.std[h])*self.peak_firing_rate[h] # rate is calcualted acording to the distance
		activity = np.exp(-dis/(self.spacing[h]**2 * self.k))*self.peak_firing_rate[h][np.arange(vertY.shape[0]), vertX, vertY] # a = exp(-d**2/(s**2*k) as in Knierim, where k = r_to_s**2/log(5) defining everythin above 0.2 is in field


		return activity

		
	def calcGridFiring(self):
		'''
		
		returns population activity of grid over all environments and locations
		'''
		ec_firing = np.zeros([self.n_e, self.number_patterns, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent, self.cells = number grid cells
		for h in range(self.n_e): # calc firing rate at each location in each enviroment
			activity = np.zeros([self.number_patterns, self.cells])
			activity[:, :self.n_grid] = np.array(map(self.calcActivity, [h]*self.number_patterns, self.locations))
			if self.n_lec >0:
				activity[:,self.n_grid:] = self.Lec.patterns[h]
			ec_firing[h] = self.actFunction(self, activity = activity)
		return ec_firing
	
	def getTrueRadius(self): #to avoid the smaller sizes aat the boarder, we compute the radius only of the largest field assuming all other have the same size
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_field_size_cell = np.max(self.cluster_size, axis = -1)
		return np.sqrt(max_field_size_cell/(self.x_length**2*np.pi))
		
	def getTrueRadiusAverage(self):
		return np.sum(self.getTrueRadius())/(self.cells+0.0)
		
	def getStoredlocations(self):
		'''
		
		returns the locations in which pattern are stored
		'''

		return self.locations[self.store_indizes]
		
class PlaceFields(Input, Spatial):
	
	
	'''
	
	Input Class that creates pattern with place cells.

	:param no_fields: Number place fields per cell
	:type no_fields: int
	:param field_size: size of each field; Note if size of fields depend on self.sparsity for certain actFunction methods!
	:type field_size: int
	:param ordered: determines whether field centers of the cells are ordered or randomly distributed.
	:type ordered: bool
	:param active_in_env: Determines how many cells are allowed to be active in one environment
	:type active_in_env: int		
	'''
	
	def __init__(self, no_fields = None,field_size = None, centers = None, ordered = 0, active_in_env = None, peak = None, cage = [1,1], center_cage = None, **kwargs):

		#Place Fields Attributes
		self.ordered = ordered #whether centers of place fields are evenly distrivuted or random
		self.centers =centers # place field center
		self.no_fields = no_fields #no fields per cell
		self.field_size = field_size # radius of one field; if actFunction is for example getOutputWTA size is only determined by number winner; otherwise, activity is set to 0 outside of the field, inside it has a guassian fire rate disribution
		self.n_e = kwargs['n_e']
		self.cells = kwargs['number_cells']
		self.peak_firing_rate = peak

		if no_fields == None:
			no_fields = np.ones([self.n_e, self.cells], 'int')*1
		if peak == None:
			self.peak_firing_rate = np.ones([self.n_e, self.cells, np.max(no_fields)])
		self.active_in_env = active_in_env
		if self.active_in_env == None:
			self.active_in_env = self.cells
		self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')#which cells are active in which envrionment
		self.no_fields = np.zeros([self.n_e, self.cells],'int')
		for h in range(self.n_e):
				self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
				self.no_fields[h][self.active_cells[h]] = no_fields[h][self.active_cells] #no fields per cell
		self.max_fields = np.max(self.no_fields, axis = 1) #max field number in each env.
		if self.field_size == None:
			field_size = np.random.normal(loc = .2500, scale = .01, size = (self.n_e, self.cells, max(self.max_fields)))
			self.field_size = np.sqrt(field_size/np.pi)



		self.locations = None #x and y coordinates of all pixel 
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.cage = cage
		if center_cage == None: # locations of possible field centers; can be bigger or smaller than cage
			self.center_cage = np.array(self.cage)
		else:
			self.center_cage = np.array(center_cage)
		#self.number_patterns= kwargs['number_patterns']
		#if kwargs['sparsity'] == None:
			#sparsity = 
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		Input.__init__(self, **kwargs)
		Spatial.__init__(self, patterns = self.input_stored[0], cage = cage, **kwargs)


	def makeInput(self,**kwargs):
		self.number_winner = min(self.number_winner, self.cells)
		self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')#which cells are active in which envrionment
		if self.active_in_env != self.cells:
			print 'not all cells active in envoronment!!!!'
			for h in range(self.n_e):
					self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
		else:
			for h in range(self.n_e):
				self.active_cells[h] = np.arange(self.cells)
		self.makeFields()
		self.patterns = self.calcFiring()

	def makeFields(self):
		if self.centers == None:
			self.centers = np.zeros([self.n_e, self.cells, np.max(self.no_fields), 2])
			if self.ordered:
				for h in range(self.n_e):
					self.centers[h][:int(np.sqrt(self.active_in_env))] = np.ravel((np.mgrid[0:int(np.sqrt(self.active_in_env)), 0: int(np.sqrt(self.active_in_env))] + 0.0)/np.sqrt(self.cells), order = 'F').reshape(self.cells,1, 2)
					self.centers[h][int(np.sqrt(self.active_in_env)):] = np.random.sample(size = [self.active_in_env -int(np.sqrt(self.active_in_env)) ,1, 2])
			else:
				for h in range(self.n_e):
					self.centers[h] = np.random.sample(size = [self.cells, np.max(self.no_fields), 2]) * np.array(self.cage)*self.center_cage - (0.5*(self.center_cage-1)) * np.array(self.cage)
					for cell in range(self.cells):
						self.centers[h, cell, self.no_fields[h, cell]:] = -10000 #center of fields that exceed number of fields of a cell are put far away

		self.std = self.field_size**2/np.log(5) #modulates the Gauss according to place field size; 
	
	def calcActivity(self, env ,location, field_number): #calc rate map for all cells at location for field_number


		#print loc_diff.shape
		loc_diff = self.centers[env][:, field_number] - location# diff loation to all fields (cells, (x,y))
		dis = np.sum(loc_diff * loc_diff, axis = -1)# eucledan distance squared to each field; (cell, x**2 +y**2)
		activity= np.exp(-dis/self.std[env, :, field_number])*self.peak_firing_rate[env,:,field_number] # rate is calcualted acording to the distance; activity = 0.2* peak rate if dis = size
		activity[activity < 0.2] = 0
		return activity #env ,pattern, cell_activity (with field)

	def calcFiring(self):
		#firing = np.zeros([self.n_e,self.number_patterns, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent, self.cells = number grid cells
		activity = np.zeros([self.n_e, self.number_patterns, self.cells])
		for h in range(self.n_e):
			for i in range(self.max_fields[h]): # calc rate map for each field and sum up
				activity[h] += np.array(map(self.calcActivity, [h]*self.number_patterns, self.locations, [i]*self.number_patterns))
		indices = np.mgrid[0:self.n_e,0:self.number_patterns,0:self.active_in_env]
		#act_cells = self.active_cells.repeat(self.number_patterns, axis = 0).reshape(self.n_e, self.number_patterns, self.active_in_env)
		#firing[indices[0], indices[1], act_cells] = self.actFunction(self,activity = activity)
		firing = self.actFunction(self, activity = activity)
		return firing
		#return activity

	def getTrueRadius(self): #in a constructed PlaceField Code with known number of fields, we can calculate the radius in antoherway. To avoid smaller fieled sizes at border,we look at the cell with the largest Place fields and calculate the radius.
		if (self.cluster_size == 0).all():
			self.calcClusters()
		max_field_size_cell = np.max(self.getAverageFieldSizeCell())/(self.no_fields+0.0)
		return np.sqrt(max_field_size_cell/(self.x_length**2*np.pi))
		
	def getStoredlocations(self):
		return self.locations[self.store_indizes]

	def makeLocations(self, number_patterns):
		#if number_patterns == None:
			#number_patterns = self.number
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.sqrt(number_patterns*1./self.cage_proportion)

		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
			#self.locations = self.locations.reshape(number_patterns, 2)
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]

class Lec(Input, Spatial): #Weakly spatially modullated cells
	
	'''
	Class for Weakly spatially modullated cells. It creates wsm cells acoring to a specified method. Possible methods are  Lec.makeActiveFilter, Lec.makeActiveRegionsPlaceFields and Lec.makeActiveRegionsSquares
	
	:param size_kernel: size of the gaussian kernel in cm that smoothes the randomly created rate map to produce the cells final rate map (only if actFunction = Lec.makeActiveFilter)
	:param place_cells: only neccessary when actFunction = makeActiveRegionsPlaceFields
	'''
	
	
	def __init__(self, size_kernel = None, cage = [1,1],place_fields = None, **kwargs):
		
		self.locations = None #x and y coordinates of all pixel 
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.cage = cage
		self.inputMethod = kwargs['inputMethod']
		self.place_fields = place_fields
		
		self.size = size_kernel #kernel size in cm
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		Input.__init__(self, **kwargs)
		#Spatial.__init__(self, patterns = self.input_stored[0], cage = cage, **kwargs)
		print 'actfunction LEC',self.actFunction, self.sparsity, self.place_fields, self.size 
		
	
	def makeLocations(self, number_patterns): #help function
		if self.cage[0]> self.cage[1]:
			self.transposed_cage = True
			self.cage_proportion = self.cage[0]/self.cage[1]
		else:
			self.transposed_cage = False
			self.cage_proportion = self.cage[1]/self.cage[0]
			if int(self.cage_proportion)*self.cage[0] != self.cage[1]:
				print 'cage adjusted to', self.cage
		self.space_res = np.sqrt(number_patterns*1./self.cage_proportion)

		if self.transposed_cage:
			self.x_length = self.cage_proportion * self.space_res 
			self.y_length = self.space_res
		else:
			self.y_length = self.cage_proportion * self.space_res
			self.x_length = self.space_res
		
		
		if int(self.space_res * self.cage_proportion*self.space_res) - self.space_res * self.cage_proportion*self.space_res != 0:
			print 'number_patterns not suitable'
			self.locations = None
		else:
			if self.transposed_cage:
				self.locations = np.ravel((np.mgrid[0:self.cage_proportion*self.space_res, 0:self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.cage_proportion*self.space_res/self.cage[0]
				self.locations[:,1] /= self.space_res/self.cage[1]
			#self.locations = self.locations.reshape(number_patterns, 2)
			else:
				self.locations = np.ravel((np.mgrid[0:self.space_res, 0:self.cage_proportion*self.space_res] + 0.0), order = 'F').reshape(number_patterns, 2)
				self.locations[:,0] /= self.space_res/self.cage[0]
				self.locations[:,1] /= self.cage_proportion*self.space_res/self.cage[1]
				

		
		
	def makeActiveRegionsSquares(self): # method to create wsm cells
		### fileds shaped as cage
		if self.transposed_cage:
			self.x_length = self.cage_proportion * self.space_res 
			self.y_length = self.space_res
			x_pixel = int(self.space_res*self.cage_proportion/5)
			y_pixel = int(self.space_res/5)
		else:
			self.y_length = self.cage_proportion * self.space_res
			self.x_length = self.space_res
			x_pixel = int(self.space_res/5)
			y_pixel = int(self.space_res*self.cage_proportion/5)
			
		self.space_res = int(self.space_res)
		if self.space_res/5 - self.space_res/5. != 0:
			print 'number_patterns not suitbael for LEC'
		region_ind = []
		region = []
		
		for i in range(int(5)):
			for j in range(int(5)):
				for y in range(y_pixel):
					#region_ind.append(range(i*x_pixel + j* self.space_res + y*self.space_res , i*x_pixel + j* self.space_res + y*self.space_res+x_pixel))
					if self.transposed_cage:
						region_ind.append(range(i*self.cage_proportion * self.space_res*y_pixel + j*x_pixel + y*self.cage_proportion * self.space_res, i*self.cage_proportion * self.space_res*y_pixel + j*x_pixel + y*self.cage_proportion * self.space_res + x_pixel))
					else:
						region_ind.append(range(i* self.space_res*y_pixel + j*x_pixel + y * self.space_res, i* self.space_res*y_pixel + j*x_pixel + y * self.space_res + x_pixel))
				region.append(np.ravel(region_ind))
				region_ind = []
		
		
		
		#### quadratic fields
		#if self.transposed_cage:
			#self.x_length = self.cage_proportion * self.space_res
			#self.y_length = self.space_res
			#x_pixel = int(self.space_res/5)
			#y_pixel = int(self.space_res/5)
		#else:
			#self.y_length = self.cage_proportion * self.space_res
			#self.x_length = self.space_res
			#x_pixel = int(self.space_res/5)
			#y_pixel = int(self.space_res/5)
			
		#self.space_res = int(self.space_res)
		#if self.space_res/5 - self.space_res/5. != 0:

		#region_ind = []
		#region = []
		
		#for i in range(int(5)):
			#for j in range(int(5*self.cage_proportion)):
				#for y in range(y_pixel):
					##region_ind.append(range(i*x_pixel + j* self.space_res + y*self.space_res , i*x_pixel + j* self.space_res + y*self.space_res+x_pixel))
					#if self.transposed_cage:
						#region_ind.append(range(i*self.cage_proportion * self.space_res*y_pixel + j*x_pixel + y*self.cage_proportion * self.space_res, i*self.cage_proportion * self.space_res*y_pixel + j*x_pixel + y*self.cage_proportion * self.space_res + x_pixel))
					#else:
						#region_ind.append(range(i* self.space_res*y_pixel + j*x_pixel + y * self.space_res, i* self.space_res*y_pixel + j*x_pixel + y * self.space_res + x_pixel))
				#region.append(np.ravel(region_ind))
				#region_ind = []
		no_active = np.random.normal(loc = int(len(region)/2.), scale = 1.5, size = (self.patterns.shape[0],self.patterns.shape[-1]) )
		for env in range(self.patterns.shape[0]):
			for i in range(self.patterns.shape[-1]):

				active = np.ravel(random.sample(region, int(no_active[env][i])))
				self.patterns[env,:,i][active] = 1
				
		self.calcFireUniform()
		for env in range(self.patterns.shape[0]):
			for i in range(self.patterns.shape[-1]):
				self.patterns[env,:,i] = self.blur_image(self.patterns[env,:,i].reshape(self.y_length, self.x_length)[::-1]).reshape(self.patterns[env,:,i].shape)
	
	
	def makeActiveRegionsPlaceFields(self): #method to create wsm cells. Here a rate map is the normailized summation of many place cells
		print '---------------- place fields LEC --------------------------------------------------', self.place_fields
		no_fields = np.random.randint(30,150, (self.n_e, self.cells))
		no_fields = 2 * np.random.randint(30,150, (self.n_e, self.cells))
		#no_fields = np.ones([self.n_e, self.cells], 'int') * 100
		max_fields = np.max(no_fields)
		#field_sizes = np.random.normal(loc = 1100, scale = 300, size = self.cells*max_fields*self.n_e)
		if self.place_fields == 'small':
			field_sizes = np.random.normal(loc = 667, scale = 167, size = self.cells*max_fields*self.n_e)
		if self.place_fields == None or self.place_fields == 'large':
			field_sizes = np.random.normal(loc = 6000, scale = 1500, size = self.cells*max_fields*self.n_e)
		if self.place_fields == 'x-large':
			field_sizes = np.random.normal(loc = 10000, scale = 2500, size = self.cells*max_fields*self.n_e)
		if self.place_fields == 'medium':
			print 'medium'
			field_sizes = np.random.normal(loc = 2000, scale = 500, size = self.cells*max_fields*self.n_e)
		
		field_rate = np.ones([self.n_e, self.cells, max_fields])
		#field_rate = np.random.normal(loc = 1, scale = 0.3, size = (self.n_e, self.cells, max_fields))
		#field_rate[field_rate < 0] *= -1
		#field_rate = np.random.uniform(0,1, (self.n_e, self.cells, max_fields))
	
		field_sizes[field_sizes < 0] *= -1
		field_sizes = field_sizes.reshape(self.n_e, self.cells , max_fields)
		field_in_r = np.sort(np.sqrt(field_sizes/np.pi)/100.)

		HelpPlaceField = PlaceFields(number_cells = self.cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = self.number_patterns ,n_e =self.n_e,noise_levels = [0], normed = 0, store_indizes = None, cage = self.cage, center_cage = 2., field_size = field_in_r, no_fields = no_fields, peak = field_rate, active_in_env = self.cells, sparsity = 0.1)
		self.patterns = np.copy(HelpPlaceField.patterns)

	
	def makeInput(self,**kwargs): #help function
		self.inputMethod(self)
		max_fire = np.tile(np.max(self.patterns, -2), (self.number_patterns)).reshape(self.patterns.shape)*1.
		print 'min max fire'
		print np.min(max_fire)
		self.patterns = self.patterns/max_fire


		
	def calcFireUniform(self):	 #help function	
		self.patterns[self.patterns == 1 ] = np.random.uniform(0.3, 1, size = (self.patterns[self.patterns == 1].shape))
		self.patterns[self.patterns == 0 ] = np.random.uniform(0, 0.7, size = (self.patterns[self.patterns == 0].shape))

	def makeActiveFilter(self):#method to create wsm cells. Here a rate map is created as described in the paper 'From grid cell to place cells with realistic field sizes'

		patterns = np.random.uniform(0,1000, size = (self.n_e, self.cells, self.x_length, self.y_length))
		#patterns = self.actFunction(self, patterns).reshape(self.n_e, self.cells, self.x_length, self.y_length)
		#patterns = np.random.randint(2, size = (self.n_e, self.cells, self.number_patterns))*1.
		#size = np.random.normal(loc = self.size, scale = self.size/5., size = (self.n_e, self.cells))
		#size[size<5] = 5
		size = np.ones([self.n_e, self.cells])*self.size
		for env in range(self.n_e):
			for cell in range(self.cells):
				#self.patterns[env, :,cell] = self.blur_image(im = patterns[env, cell], n = size[env, cell]).reshape(self.number_patterns, order = 'F')
				#self.patterns[env, :,cell] = scipy.ndimage.filters.gaussian_filter(patterns[env,cell], sigma = size[env,cell], truncate = 6).reshape(self.number_patterns, order = 'F')
				self.patterns[env, :,cell] = scipy.ndimage.filters.gaussian_filter(patterns[env,cell], sigma = size[env,cell]).reshape(self.number_patterns, order = 'F')
		min_fire = np.tile(np.min(self.patterns, -2), (self.number_patterns)).reshape(self.patterns.shape)*1.
		self.patterns -= min_fire - 0.0001
		max_fire = np.tile(np.max(self.patterns, -2), (self.number_patterns)).reshape(self.patterns.shape)*1.
		self.patterns = self.patterns/max_fire
		self.patterns = self.actFunction(self, self.patterns)
		print 'done filter'
		

		
	def gauss_kern(self, size, sizey=None): #help function	
		""" Returns a normalized 2D gauss kernel array for convolutions """
		size = int(size)    
		if sizey == None:
			sizey = size
		else:
			sizey = int(sizey)               
  
		x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
		g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
		g / g.sum()		
		#print 'g shape', g.shape
		#g = np.tile(g, (self.n_e, self.cells, 1,1))
		#print 'g shape', g.shape
		return g
	
	def blur_image(self, im, n = None, ny=None): #help function	
		""" blurs the image by convolving with a gaussian kernel of typical
		size n. The optional keyword argument ny allows for a different
		size in the y direction.
		"""
		if n == None:
			if self.size == None:
				self.size = 17 * self.space_res/(np.min(self.cage)*100) # size = 17cm as in da costa 2005
				#size in pixel
			n = self.size* self.space_res/(np.min(self.cage)*100)
		else:
			n *= self.space_res/(np.min(self.cage)*100.)
		if n>=1:
			g = self.gauss_kern(n, sizey=ny)
			#improc = scipy.signal.convolve2d(im,g, mode='same', boundary = 'fill', fillvalue = self.sparsity)
			improc = scipy.ndimage.filters.convolve(im,g, mode='reflect')
		else:
			improc = im
		return(improc)    
	
	def example(self): #help function	
		xmin, xmax, ymin, ymax = -70, 70, -70, 70
		extent = xmin, xmax, ymin, ymax
		
		X, Y = np.mgrid[-70:70, -70:70]
		Z = np.cos((X**2+Y**2)/200.)+ np.random.normal(size=X.shape)    
		#Z = cos((X**2+Y**2)/200.)
		
		fig1 = plt.figure(1)
		fig1.clf()
		ax1a = fig1.add_subplot(131)
		ax1a.imshow(np.abs(Z), cmap=cm.jet, alpha=.9, interpolation='bilinear', extent=extent)
		ax1a.set_title(r'Noisey')
		
		P = self.gauss_kern(3)
		
		ax1b = fig1.add_subplot(132)
		ax1b.imshow(np.abs(P), cmap=cm.jet, alpha=.9, interpolation='bilinear', extent=extent)
		ax1b.set_title(r'Convolving Gaussian')
		
		U = self.blur_image(Z, 3)
		
		ax1c = fig1.add_subplot(133)
		ax1c.imshow(np.abs(U), cmap=cm.jet, alpha=.9, interpolation='bilinear', extent=extent)
		ax1c.set_title(r'Cleaned')

class Border(Input,Spatial): # Border cells
	
	def __init__(self, no_fields = None, field_size = None, centers = None, ordered = 0, active_in_env = None, cage = [1,1], center_cage = None, **kwargs):

		#Place Fields Attributes

		self.active_in_env = active_in_env
		if self.active_in_env == None:
			self.active_in_env = kwargs['number_cells']
		self.locations = None #x and y coordinates of all pixel 
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.cage = cage
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		Input.__init__(self, **kwargs)




	def makeInput(self,**kwargs):
		self.number_winner = min(self.number_winner, self.cells)
		self.active_cells = np.zeros([self.n_e, self.active_in_env], 'int')#which cells are active in which envrionment
		if self.active_in_env != self.cells:
			print 'not all cells active in envoronment!!!!'
			for h in range(self.n_e):
					self.active_cells[h] = np.array(random.sample(range(self.cells), self.active_in_env))
		else:
			for h in range(self.n_e):
				self.active_cells[h] = np.arange(self.cells)
		self.makeFields()
		self.patterns = self.calcFiring()

	def makeFields(self):
		self.field_size = np.abs(np.random.normal(loc = 20, scale = 5, size = self.cells))
		self.borders = np.random.randint(0,4, size = self.cells)
		self.distance_to_border = np.abs(np.random.normal(loc = 0, scale = 15, size = self.cells))
		self.std = self.field_size**2/np.log(5) #modulates the Gauss according to place field size; 
		self.peak_firing_rate = 1#np.ones(self.centers.shape[0])
	
	def calcActivity(self, location): #calc rate map for all cells at location for field_number
		
		activity = np.zeros([self.cells])
		for i in range(4):
			cells = np.flatnonzero(self.borders == i)
			if i == 0:
				dis = location[0]*100 - self.distance_to_border[cells]
			if i == 1:
				dis = location[1]*100 - self.distance_to_border[cells]
			if i == 2:
				dis = self.cage[0]*100 - self.distance_to_border[cells] - location[0]*100
			if i == 3:
				dis = self.cage[1]*100 - self.distance_to_border[cells] - location[1]*100
			activity[cells] = np.exp(-dis**2/self.std[cells])*self.peak_firing_rate # rate is calcualted acording to the distance; activity = 0.2* peak rate if dis = size
		return activity #env ,pattern, cell_activity (with field)

	def calcFiring(self):
		firing = np.zeros([self.n_e,self.number_patterns, self.cells]) # self.n_e = number enviromnents, self.space_res**2 = number of different firing rate in each enviromnent
		for h in range(self.n_e):
			firing[h]  = np.array(map(self.calcActivity, self.locations))
		return firing
		#return activity
		
class JointInput(Lec): # class to combine different cell types (grid, lec, border, place). If you want to use only grid and wsm cells, you can also use the Grid class and specifiy the desired lec_cells during initialization.
	
	def __init__(self, grid_cells = 0, border_cells = 0, lec_cells = 0, place_cells = 0, **kwargs):
		
		self.grid_cells = grid_cells
		self.border_cells = border_cells
		self.lec_cells = lec_cells
		self.place_cells = place_cells
		self.cells = grid_cells + border_cells + lec_cells +place_cells
		self.cage = kwargs['cage']
		self.locations = None #x and y coordinates of all pixel 
		self.distance_matrix = None # matrix where entry (i,j) is the eucledean distance between location i and j
		self.makeLocations(number_patterns = kwargs['number_patterns'])
		kwargs['number_cells'] = self.cells
		Input.__init__(self, **kwargs)
		Spatial.__init__(self, patterns = self.input_stored[0], **kwargs)	
	def makeInput(self, **kwargs):
		if self.grid_cells !=0:
			self.Grid = Grid(number_cells = self.grid_cells, n_e = self.n_e, number_patterns= self.number_patterns, store_indizes = self.store_indizes, number_to_store = self.number_to_store, noiseMethod = Input.makeNoiseDoNothing, actFunction = self.actFunction, sparsity = self.sparsity, noise_levels = [0], normed = 0,**kwargs)
			self.patterns[:,:, :self.grid_cells] = self.Grid.patterns
		if self.border_cells !=0:
			self.Border = Border(number_cells = self.border_cells, n_e = self.n_e, number_patterns= self.number_patterns, store_indizes = self.store_indizes, number_to_store = self.number_to_store, noiseMethod = Input.makeNoiseDoNothing, actFunction = self.actFunction, sparsity = self.sparsity, noise_levels = self.noise_levels, normed = 0,**kwargs)
			self.patterns[:,:, self.grid_cells: self.border_cells + self.grid_cells] = self.Border.patterns
		if self.lec_cells !=0:
			self.Lec = Lec(inputMethod = self.inputMethod,number_cells = self.lec_cells, n_e = self.n_e, number_patterns= self.number_patterns, store_indizes = self.store_indizes, number_to_store = self.number_to_store, noiseMethod =  Input.makeNoiseDoNothing, actFunction = self.actFunction, sparsity = self.sparsity, noise_levels = [0], normed = 0,**kwargs)
			self.patterns[:,:, self.grid_cells+self.border_cells :self.grid_cells+self.border_cells +self.lec_cells] = self.Lec.patterns
		if self.place_cells !=0:
			self.PlaceCells = PlaceFields(number_cells = self.place_cells, n_e = self.n_e, number_patterns= self.number_patterns, store_indizes = self.store_indizes, number_to_store = self.number_to_store, noiseMethod =  Input.makeNoiseDoNothing, actFunction = self.actFunction, sparsity = self.sparsity, noise_levels = self.noise_levels, normed = 0,**kwargs)
			self.patterns[:,:, -self.place_cells:] = self.PlaceCells.patterns
	
	def makeNoiseLEC(self, pattern=None, noise_levels=None): #makes onl ynoise in the lec cells (for stabilit analysis)
		mec_border = int(self.cells/2)
		noise = np.tile(pattern, (len(noise_levels),1,1))
		LecHelp = Lec(inputMethod = self.Lec.inputMethod, number_cells = self.cells-mec_border, n_e = self.n_e, number_patterns= self.number_patterns, store_indizes = self.store_indizes, number_to_store = self.number_to_store, noiseMethod =  Input.makeNoiseDoNothing, actFunction = self.actFunction, sparsity = self.sparsity, noise_levels = [0], normed = 0, cage = self.Lec.cage, size_kernel = self.Lec.size,place_fields =self.Lec.place_fields)
		for n in range(len(noise_levels)):
		#noise[:,:, self.grid_cells+self.border_cells :self.grid_cells+self.border_cells +self.lec_cells] = self.Lec.makeNoiseRandomFire(pattern = self.Lec.patterns[0], noise_levels = noise_levels)
			#print LecHelp.patterns.shape
			#print noise.shape
			#print LecHelp.patterns[0].shape
			#print noise[n, self.grid_cells+self.border_cells :self.grid_cells+self.border_cells +self.lec_cells].shape
			noise[n, :,mec_border:] += noise_levels[n]  * LecHelp.patterns[0]
			
			max_fire = np.tile(np.max(noise[n, :,mec_border:], -2), (self.number_patterns)).reshape(noise[n, :,mec_border:].shape)*1.
			noise[n, :,mec_border:] = noise[n, :,mec_border:]/max_fire
		
		
		print 'noise makenoiseLec'
		print noise[-1,0]
		return noise
	
	
	
	def getStoredlocations(self):
		'''
		
		returns the locations in which pattern are stored
		'''

		return self.locations[self.store_indizes]
	


#############################################Hippocampal Networks######################################################################################
class Hippocampus():
	
	
		
	'''
	
	Base Class for all network architectures. Possible Networks are Ec_Dg, Dg_Ca3, Ec_Ca3, Ca3_Ca3, Ca3_Ca1, Ca1_Ec, Ca3_Ec, Ec_Ca1. It has all the arguments as the class 'Network' and its childs has. It takes them as an dictionary. For example cells = dict('Ec_Ca3' = 1000, 'Ec_Ca1 = 2000'). If a Parameter is not given. Standard Paramter is taken from class 'Parameter'. Furthermore, whole Network instances can be passed as arguments,too. In this case, this network is no loner initilized but uses the passed network as a network. This makes sense, if you have a trained network already and you want to use it again. Additional parameters are described below.
	
	:param just_ca3: Whether the whole architecture is simulated or only till Ca3
	:type just_ca3: bool 
	:param rec: Whether the CA3 network is modelled as an autoassociation memory
	:type rec:  bool
	:param In: Input Instance that gives the patterns that are going to be stored in the architecture.
	:type In: Input Instance, must have same cell number as self.cells['Ec']
	:param InCa1: Input Instance that give input to Ec_CA1 via the weights Ec_Ca1
	:type InCa1: Input Instance, must have same cell number as self.cells['Ec']
	:param Ca3Activation: If CA3Activation != None, then this Input Instance sets up the patterns in CA3.
	:type Ca3Activation: Input Instance, must have same cell number as self.cells['CA3']
	:param Ca1Activation: If CA1Activation != None, then this Input Instance sets up the patterns in CA1. If it is None, InCa1 = In
	:type Ca1ctivation: Input Instance, must have same cell number as self.cells['CA1']
	'''
	
	def __init__(self, just_ca3 = 0,  rec = 1, In = None, InCa1 = None, Ca3Activation = None, Ca1Activation = None, Ec_Dg = None, Dg_Ca3 = None, Ec_Ca3 = None, Ec_Ca1 = None, Ca3_Ca3 = None, Ca3_Ca1 = None, Ca1_Ec = None, Ca3_Ec = None,cells = Parameter.cells, number_winner = Parameter.number_winner, connectivity = Parameter.connectivity, learnrate = Parameter.learnrate, subtract_input_mean = Parameter.subtract_input_mean, subtract_output_mean = Parameter.subtract_output_mean, actFunctionsRegions = Parameter.actFunctionsRegions, initMethod = Parameter.initMethod, cycles  = Parameter.cycles, incremental_storage_mode=Parameter.incremental_storage_mode, no_incremental_times = Parameter.no_incremental_times, number_to_store = Parameter.number_to_store, incrementalLearnMethod = Parameter.incrementalLearnMethod,external_force = Parameter.external_force, internal_force = Parameter.internal_force, first = Parameter.first, active_in_env = Parameter.active_in_env):


		self.In = In
		self.Ec_Dg = Ec_Dg
		self.Dg_Ca3 = Dg_Ca3
		self.Ec_Ca3 = Ec_Ca3
		self.Ec_Ca1 = Ec_Ca1
		self.Ca3_Ca3 = Ca3_Ca3
		self.Ca3_Ca1 = Ca3_Ca1
		self.Ca3_Ec = Ca3_Ec
		self.Ca1_Ec = Ca1_Ec
	
		self.rec = rec
		self.just_ca3 = just_ca3
		self.InCa1 = InCa1
		self.Ca3Activation = Ca3Activation
		self.Ca1Activation = Ca1Activation
	
		self.cells = copy.copy(cells)
		self.number_winner = copy.copy(number_winner)
		self.connectivity = copy.copy(connectivity)
		self.learnrate = copy.copy(learnrate)
		self.subtract_input_mean = copy.copy(subtract_input_mean)
		self.subtract_output_mean = copy.copy(subtract_output_mean)
		self.actFunctionsRegions = copy.copy(actFunctionsRegions)
		self.initMethod = copy.copy(initMethod)
		self.cycles = copy.copy(cycles)
		self.incremental_storage_mode = copy.copy(incremental_storage_mode)
		self.no_incremental_times = copy.copy(no_incremental_times)
		self.number_to_store = copy.copy(number_to_store)
		self.incrementalLearnMethod = incrementalLearnMethod
		self.external_force = copy.copy(external_force)
		self.internal_force = copy.copy(internal_force)
		self.active_in_env = active_in_env
		self.n_e = self.In.n_e
		self.first = first
		if self.first == None:
			self.first = self.In.number_to_store
		
		self.store_ind = np.mgrid[0:self.In.n_e, 0:self.In.number_to_store]#helping attribut, it is needed if CA3Activation is given
	
	
	##########methods for creating the Network instances##############################
	### In General, each network is an HeteroAssocaition Instance
	### Exceptions are Ec_Dg, which is a OneShoot Instance and Ca3_Ca3, which is an AutoAssociation instance
	def makeEc_Dg(self):
		if self.Ec_Dg == None:
			self.Ec_Dg = OneShoot(input_cells=self.cells['Ec'], cells=self.cells['Dg'], number_winner=self.number_winner['Dg'], connectivity = self.connectivity['Ec_Dg'], learnrate= self.learnrate['Ec_Dg'], subtract_input_mean = 0, subtract_output_mean = 0, initMethod = self.initMethod['Ec_Dg'], actFunction = self.actFunctionsRegions['Ec_Dg'],weight_mean = 1, weight_sigma = 0.01, active_in_env = self.active_in_env['Dg'], n_e = self.n_e) 
			
	def makeDg_Ca3(self):
		if self.Dg_Ca3 == None:
			self.Dg_Ca3 = HeteroAssociation(input_cells = self.cells['Dg'], cells = self.cells['Ca3'], number_winner = self.number_winner['Ca3'], connectivity = self.connectivity['Dg_Ca3'], learnrate = self.learnrate['Dg_Ca3'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = self.subtract_output_mean, initMethod = self.initMethod['Dg_Ca3'],actFunction = self.actFunctionsRegions['Dg_Ca3'], active_in_env = self.active_in_env['Ca3'], n_e = self.n_e) 
		
	def makeEc_Ca3(self):
		if self.Ec_Ca3 == None:
			self.ec_ca3_given = False
			self.Ec_Ca3 =  HeteroAssociation(input_cells = self.cells['Ec'], cells = self.cells['Ca3'], number_winner = self.number_winner['Ca3'], connectivity = self.connectivity['Ec_Ca3'], learnrate = self.learnrate['Ec_Ca3'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = self.subtract_output_mean, initMethod = self.initMethod['Ec_Ca3'],actFunction = self.actFunctionsRegions['Ec_Ca3'], active_in_env = self.active_in_env['Ca3'], n_e = self.n_e) 
		else:
			self.ec_ca3_given = True
		
	def makeCa3_Ca3(self):
		if self.Ca3_Ca3 == None:
			self.Ca3_Ca3 = AutoAssociation(input_cells = self.cells['Ca3'], cells = self.cells['Ca3'], number_winner = self.number_winner['Ca3'], connectivity = self.connectivity['Ca3_Ca3'], learnrate = self.learnrate['Ca3_Ca3'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = 1, initMethod = self.initMethod['Ca3_Ca3'], cycles = self.cycles,actFunction = self.actFunctionsRegions['Ca3_Ca3'], external_force = self.external_force, internal_force = self.internal_force, active_in_env = self.active_in_env['Ca3'], n_e = self.n_e) 
			self.Ca3_Ca3.active_cells = self.Ec_Ca3.active_cells
			self.Ca3_Ca3.active_cells_vector = self.Ec_Ca3.active_cells_vector
			self.ca3_ca3_given = False
		else:
			self.ca3_ca3_given = True

	def makeCa3_Ca1(self):
		if self.Ca3_Ca1 == None:
			self.Ca3_Ca1 = HeteroAssociation(input_cells = self.cells['Ca3'], cells = self.cells['Ca1'], number_winner = self.number_winner['Ca1'], connectivity = self.connectivity['Ca3_Ca1'], learnrate = self.learnrate['Ca3_Ca1'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = self.subtract_output_mean, initMethod = self.initMethod['Ca3_Ca1'],actFunction = self.actFunctionsRegions['Ca3_Ca1'], active_in_env = self.active_in_env['Ca1'], n_e = self.n_e) 
			self.ca3_ca1_given = False
		else:
			self.ca3_ca1_given = True
		if self.ca3_ca1_given:
			self.ca3_ca1_given = False
			print '!!!!!!!!!!!!!!!!!!!!! make new CA3_CA1!!!!!!'

	def makeEc_Ca1(self):
		if self.Ec_Ca1 == None:
			self.Ec_Ca1 =  HeteroAssociation(input_cells = self.cells['Ec'], cells = self.cells['Ca1'], number_winner =self.number_winner['Ca1'], connectivity = self.connectivity['Ec_Ca1'], learnrate = self.learnrate['Ec_Ca1'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = self.subtract_output_mean, initMethod = self.initMethod['Ec_Ca1'],actFunction = self.actFunctionsRegions['Ec_Ca1'], active_in_env = self.active_in_env['Ca1'], n_e = self.n_e) 
			self.ec_ca1_given = False
			if self.Ca3_Ca1 != None:
				self.Ec_Ca1.active_cells = self.Ca3_Ca1.active_cells
				self.Ec_Ca1.active_cells_vector = self.Ca3_Ca1.active_cells_vector
		else:
			self.ec_ca1_given = True
	
	def makeCa3_Ec(self):
		if self.Ca3_Ec == None:
			self.Ca3_Ec = HeteroAssociation(input_cells = self.cells['Ca3'], cells = self.cells['Ec'], number_winner = self.number_winner['Ec'], connectivity = self.connectivity['Ca3_Ec'], learnrate = self.learnrate['Ca3_Ec'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = self.subtract_output_mean, initMethod = self.initMethod['Ca3_Ec'],actFunction = self.actFunctionsRegions['Ca3_Ec'])
			self.ca3_ec_given = False
		else:
			self.ca3_ec_given = True
	
	def makeCa1_Ec(self):
		if self.Ca1_Ec == None:
			self.Ca1_Ec =  HeteroAssociation(input_cells = self.cells['Ca1'], cells = self.cells['Ec'], number_winner = self.number_winner['Ec'], connectivity = self.connectivity['Ca1_Ec'], learnrate = self.learnrate['Ca1_Ec'], subtract_input_mean = self.subtract_input_mean, subtract_output_mean = self.subtract_output_mean, initMethod = self.initMethod['Ca1_Ec'],actFunction = self.actFunctionsRegions['Ca1_Ec'], n_e = self.n_e) 
			self.ca1_ec_given = False
		else:
			self.ca1_ec_given = True


	### store method defines how patterns are stored in the network.
	def store(self):
		pass
	
	### recall method defines how patterns are recalled in the network.
	def recall(self):
		pass
	
	### sets up the Activity when Ca3Activation is given
	def setUpCa3Activity(self):
		if self.Ca3Activation == None:
			self.makeEc_Dg()
			self.makeDg_Ca3()
		else:
			self.Dg_Ca3 = Network(input_cells = 1, cells = self.cells['Ca3'], connectivity =1, initMethod = Network.makeWeightsZero, number_winner = 0, active_in_env = self.active_in_env['Ca3'], n_e = self.In.n_e) # dummy network
			self.Dg_Ca3.output_stored = self.Ca3Activation.getInput()[self.store_ind[0],self.In.store_indizes]
	
	### sets up the Activity when Ca1Activation is given
	def setUpCa1Activity(self):
		if self.Ca1Activation == None:
			if self.InCa1 == None:
				self.InCa1 = self.In
			self.makeEc_Ca1()
		else:
			self.Ec_Ca1 = Network(input_cells = 1, cells = self.cells['Ca1'], connectivity =1, initMethod = Network.makeWeightsZero, number_winner = 0,active_in_env = self.active_in_env['Ca1'], n_e = self.In.n_e) # dummy network
			self.Ec_Ca1.output_stored = self.Ca1Activation.getInput()[self.store_ind[0],self.In.store_indizes]

class HippocampusFull(Hippocampus):
	
	'''
	
	The full hippocampal loop. If CA3Activation is not given, input patterns trigger activation in Ec_Dg, which adjust its weights in a competitive net during storage. Activity in Ec_Dg triggers one in Dg_CA3. This activity is autoassociated in Ca3_CA3 and hetero assocatiated in Ec_Ca3. If CA1Activation is not given, InCa1 sets activitys in Ec_Ca1, which is hetero associated with the CA3 patterns in Ca3_Ca1. Finally Ec_CA1 patterns are hetero associated with the original patterns in Ca1_Ec. 
	'''
	
	def __init__(self, **kwargs):
		
		Hippocampus.__init__(self, **kwargs)

		self.setUpCa3Activity()
		self.makeEc_Ca3()
		self.makeCa3_Ca1()
		self.makeCa1_Ec()
		if self.rec:
			self.makeCa3_Ca3()
		if not self.just_ca3:
			self.setUpCa1Activity()
		self.store()
		self.recall()
		
	def store(self):
		if not self.ec_ca3_given:
			if self.Ca3Activation ==None:
				if self.learnrate['Ec_Dg'] !=0:
					print 'learn Ec_Dg'
					self.Ec_Dg.learnOneShootAllPattern(input_pattern = self.In.input_stored, method = self.incrementalLearnMethod, first = self.first)
				else:
					print 'DG does not learn'
					self.Ec_Dg.learnOneShootAllPattern(input_pattern = self.In.input_stored, method = self.incrementalLearnMethod, first = self.first)
				self.Dg_Ca3.output_stored = self.Dg_Ca3.getOutput(self.Dg_Ca3, input_pattern = self.Ec_Dg.output_stored)
			print 'learn Ec_Ca3'
			self.Ec_Ca3.learnAssociation(input_pattern = self.In.input_stored, output_pattern = self.Dg_Ca3.output_stored, first = self.first)
		else:
			self.Dg_Ca3.output_stored = self.Ec_Ca3.output_stored
		if self.rec:
			if not self.ca3_ca3_given:
				print 'learn ca3_ca3'
				self.Ca3_Ca3.learnAssociation(input_pattern = self.Dg_Ca3.output_stored, output_pattern = self.Dg_Ca3.output_stored,first = self.first)
			
		if not self.just_ca3:
			if self.Ca1Activation == None:
				if not self.ec_ca1_given:
					print 'calc Ec_Ca1 output'
					self.Ec_Ca1.output_stored = self.Ec_Ca1.getOutput(self.Ec_Ca1,input_pattern = self.InCa1.input_stored)
			if not self.ca3_ca1_given:
				print 'learn ca3_ca1'
				self.Ca3_Ca1.learnAssociation(input_pattern = self.Ec_Ca3.output_stored, output_pattern = self.Ec_Ca1.output_stored,first = self.first)
			if not self.ca1_ec_given:
				print 'learn ca1_ec'
				self.Ca1_Ec.learnAssociation(input_pattern = self.Ec_Ca1.output_stored, output_pattern = self.In.input_stored,first = self.first)

	def recall(self):
		print 'recall'
		if not self.ec_ca3_given:
			print 'calc Ec_Ca3 output'
			self.Ec_Ca3.recall(input_pattern = self.In.noisy_input_stored, first = self.first)
		if self.rec:
			if not self.ca3_ca3_given:
				print 'calc Ca3 rec output'
				self.Ca3_Ca3.recall(input_pattern = self.Ec_Ca3.noisy_output, external_activity = self.Ec_Ca3.calcActivity(input_pattern = self.In.noisy_input_stored), first = self.first)

		if not self.just_ca3:
			if self.rec:
				if not self.ca3_ca1_given:
					print 'ca3_ca1 rec'
					self.Ca3_Ca1.recall(input_pattern = self.Ca3_Ca3.noisy_output, key = 'Rec', first = self.first)
				if not self.ca1_ec_given:
					print 'ca1_ec_rec'
					self.Ca1_Ec.recall(input_pattern = self.Ca3_Ca1.noisy_output, key = 'Rec', first = self.first)
			if not self.ca3_ca1_given:		
				print 'ca3_ca1 no rec'
				self.Ca3_Ca1.recall(input_pattern = self.Ec_Ca3.noisy_output, key = 'NoRec', first = self.first)
			if not self.ca1_ec_given:
				print 'ca1_ec_norec'
				self.Ca1_Ec.recall(input_pattern = self.Ca3_Ca1.noisy_output, key = 'NoRec', first = self.first)

class Solution():
	
	def __init__(self, In = None, distance = None, location = None, max_weight = None, non_negative = None, Method = None, influence = None, influence_degree = 0, weights_given =None):
		self.In = In
		self.locations = In.getStoredlocations()
		
		
		print 'init Sol locations'

		
		self.location = location
		self.distance = distance
		self.max_weight = max_weight
		self.non_negative = non_negative
		self.bins = np.array([self.distance])
		self.t = np.zeros(len(In.noise_levels))
		self.actFunction = False
		self.w = weights_given
		self.norms = np.zeros(4)
		self.influence = influence
		if self.influence != None:
			print 'non negative = True!!!'
			self.non_negative = True
		self.influence_degree = influence_degree
		self.activation = None
		
		self.Method = Method
		Method(self)
		self.calcActivationMap()
	
	
	#def calcNorms(self):
		#for i in range(4)
		#self.norms[i] = np.dot(self.w[self.In.modules(i):self.In.modules(i+1),0])
		
	
	def getLocationsWithinDistance(self, distance = None, locations = None, location = None, bins = None):
		'''
		
		gets locations within the radius distance and outside the radius around location. Distances are rounded according to bins
		'''
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.In.getDistanceMatrix(locations, bins = bins)
		distances = dis[location]
		inside = np.flatnonzero(distances <= distance)
		outside = np.flatnonzero(distances > distance)
		return [inside, outside]
		
	def getLocationsAtDistance(self, distance = None, locations = None, location = None, bins = None):
		'''
		
		gets locations at the exact distance to location. Distances are rounded according to bins
		'''
		#self.calcDistanceMatrix(locations, bins = bins)
		dis = self.getDistanceMatrix(locations, bins = bins)
		distances = dis[location]
		at_distance = np.flatnonzero(distances == distance)
		return at_distance
		
	def getPopulationWithinDistance(self, distance = None, locations = None, location = None, bins = None, with_noise = 0):
		
		'''
		
		gets patterns within the radius distance and outside the radius around location. Distances are rounded according to bins; if with_noise, then all noisy versions of the patterns are returned too.
		'''
		
		[inside, outside] = self.getLocationsWithinDistance(distance = distance, locations = locations, location = location, bins = bins)
		inside_pop = self.In.input_stored[0][inside]
		outside_pop =self.In.input_stored[0][outside]
		if with_noise:
			for i in range(self.In.noisy_input_stored.shape[1]-1):
				inside_pop = np.concatenate((inside_pop, self.In.noisy_input_stored[0][i+1][inside]))
				outside_pop = np.concatenate((outside_pop, self.In.noisy_input_stored[0][i+1][outside]))
		return [inside_pop, outside_pop]
		
	def findPlaceWeight(self, alpha =1):
		
		'''
		
		gets solution of ....
		'''
		
		
		[inside, outside] = self.getPopulationWithinDistance(distance = self.distance, locations = self.locations, location = self.location, bins = self.bins, with_noise = 0)
		
		G = np.concatenate((outside, - inside), axis = 0)
		h = [1.]* outside.shape[0] + [-1.] *inside.shape[0]
		
		if self.non_negative:
			G = np.concatenate((G, - np.eye(inside.shape[1])*1.), axis = 0)
			h = h + [0] *inside.shape[1]
			
			
		if self.max_weight != -1:
			#each weight less equal 0.3
			print 'each weight less equal ' +str(self.max_weight)
			G = np.concatenate((G, np.eye(inside.shape[1])*1.), axis = 0)
			h = h + [self.max_weight] *inside.shape[1]
			if not self.non_negative:# w >= - max
				G = np.concatenate((G, -np.eye(inside.shape[1])*1.), axis = 0)
				h = h + [self.max_weight] *inside.shape[1]
			
		#if boarder:
			#at_distance = self.patterns_1[self.getLocationsAtDistance(distance = distance, locations = locations, location = location, bins = bins)]


			#G = np.concatenate((G, at_distance*1. - np.tile(self.patterns_1[location], (at_distance.shape[0], 1))), axis = 0)
			#h = h + [0] * at_distance.shape[0]
			
		if self.influence == 'M1+M2':
			print 'influence findplace weight M1 and M2'
									
			
			ones = np.zeros([1,inside.shape[-1]])
			ones[0][:self.In.modules[2]] = -1. *self.influence_degree[0]/(self.In.modules[2]*1.) #  influence[1] * av module 4 - influence[0] *av module 1and2 <= 0
			ones[0][self.In.modules[3]:] =  1.*self.influence_degree[1]/(self.In.modules[4]-self.In.modules[3])
			G = np.concatenate((G, ones*1.), axis = 0)
			h = h + [0]
			
			ones2 = np.zeros([1,inside.shape[-1]])
			ones2[0][:self.In.modules[2]] = -1. *self.influence_degree[0]/(self.In.modules[2]*1.) #  influence[1] * av module 3 - influence[0] *av module 1and2 <= 0
			ones2[0][self.In.modules[2]:self.In.modules[3]] =  1.*self.influence_degree[1]/(self.In.modules[3]-self.In.modules[2])
			G = np.concatenate((G, ones2*1.), axis = 0)
			h = h + [0]
			
			
		if self.influence == 'M1':
			print 'influence findplace weight M1'
						
			ones = np.zeros([1,inside.shape[-1]])
			ones[0][:self.In.modules[1]] = -1.* self.influence_degree[0]/(self.In.modules[1]*1.)#  influence[1] *av module 4 - influence[0] * av module 1 <= 0
			ones[0][self.In.modules[2]:self.In.modules[3]] = 1.*self.influence_degree[1]/(self.In.modules[4]-self.In.modules[3])
			G = np.concatenate((G, ones*1.), axis = 0)
			h = h + [0]
			
			ones2 = np.zeros([1,inside.shape[-1]])
			ones2[0][:self.In.modules[1]] = -1.* self.influence_degree[0]/(self.In.modules[1]*1.)#  influence[1] *av module 3 - influence[0] * av module 1 <= 0
			ones2[0][self.In.modules[2]:self.In.modules[3]] =  1.*self.influence_degree[1]/(self.In.modules[3]-self.In.modules[2])
			G = np.concatenate((G, ones2*1.), axis = 0)
			h = h + [0]
			
			#ones = np.ones([1,inside.shape[-1]])
			#ones[0][:self.modules[1]] = -1 *1./self.modules[1] # 1/n * sum module 4 - 1/m * sum module 1 <= 0
			#ones[0][self.modules[1]:self.modules[3]] = 0 # mod 2 and 3 not considered
			#ones[0][self.modules[3]:] = 1./(self.modules[4]-self.modules[3])
			
			#G = np.concatenate((G, ones*1.), axis = 0)
			#h = h + [0]
			
			##same for module 2:  1/n * sum module 4 - 1/m * sum module 2 <= 0
			#ones = np.ones([1,inside.shape[-1]])
			#ones[0][:self.modules[1]] = 0 
			#ones[0][self.modules[1]:self.modules[2]] = -1 *1./(self.modules[2]-self.modules[1])
			#ones[0][self.modules[2]:self.modules[3]] = 0
 			#ones[0][self.modules[3]:] = 1./(self.modules[4]-self.modules[3])
			
			#G = np.concatenate((G, ones*1.), axis = 0)
			#h = h + [0]
			
			
			
			
		G = matrix(G)
		h = matrix(h)
			
		#c = matrix([1.] * inside.shape[1]) #minimize sum over components x_i
		
		# maximize differences px - qx for 100 random chosen inside p and outside q - sum x_i as penalty for big x.
		if inside.shape[0] < 10:
			rand_in = np.tile(self.In.input_stored[0][self.location], (1000,1))
		else:
			if inside.shape[0] < 50:
				rand_in =  100* np.array(random.sample(inside, 10))
			else:
				if inside.shape[0] < 100:
					rand_in =  20* np.array(random.sample(inside, 50))
				else:
					if inside.shape[0] < 500:
						rand_in =  10 * np.array(random.sample(inside, 100))
					else:
						if inside.shape[0] < 1000:
							rand_in =  2 * np.array(random.sample(inside, 500))
						else:
							if inside.shape[0] >= 1000:
								rand_in =  np.array(random.sample(inside, 1000))
		
		if outside.shape[0] >= 1000:
			rand_out = np.array(random.sample(outside, 1000))
		else: 
			if outside.shape[0] >= 100:
				rand_out = 10 *np.array(random.sample(outside, 100))
			else:
				rand_out = 100 *np.array(random.sample(outside, 10))
		c = np.sum(rand_out, axis = -2) - np.sum(rand_in, axis = -2) + np.ones(inside.shape[1], 'float')
		
		
		c = matrix(c)
		
		##old linear 
		#if non_negative:
			#solve = solvers.lp(c,G,h)['x']
		#else:
		##new quadratic. necessary since min x_i^2 and not sim x_i
		p = matrix(alpha * np.eye(inside.shape[1])*1.) # xPx + qx
		c = 1/1000. * (matrix(np.sum(rand_out, axis = -2) - np.sum(rand_in, axis = -2)))# q=c, now without np.ones(inside.shape[1], 'float') term
		
		## min (outside . center) * w
		#c = 1./outside.shape[0] * np.sum(inside, axis = -2) -self.patterns_1[location]
		#c = matrix(c)
		
		self.w = np.copy(solvers.qp(p,c,G,h)['x']) #min sum_i x_i^2 
		
	def doNothing(self):
		pass
		
	def calcNumberWrongPixels(self, activation = None, noise = None, optimal_t = True):#return proportion inside wornd and outside wrong
		'''
		
		determines proportion pixel that fire inside the field wrong and outside.
		'''
		if activation ==None:
			activation = self.activation
		[inside, outside] = self.getLocationsWithinDistance(distance = self.distance, locations = self.locations, location = self.location, bins = self.bins)
		if self.Method == Solution.weightHebbian or Solution.takeWeightDistributed:
			self.t[noise]=0
			if self.Method == Solution.takeWeightDistributed:
				optimal_t = False
		else:
			self.t[noise]=1
		if optimal_t:
			print 'calc optimal t for noise level' + str(noise)
			wrong = 0.5
			#if self.Method == Solution.weightHebbian:
			for k in np.linspace(np.min(activation[noise]), np.max(activation[noise]), 1000):
				wrong_inside = np.flatnonzero(activation[noise][inside] <= k).shape[0]
				wrong_outside = np.flatnonzero(activation[noise][outside] > k).shape[0]
				wrong_new = wrong_inside/(inside.shape[0]+0.0)/2. + wrong_outside/(outside.shape[0]+0.0)/2.
				if wrong_new < wrong:
					wrong = wrong_new
					self.t[noise] = k
			print self.w.shape
			print self.w[:,0][self.w[:,0]!=0]
			print self.t
			print wrong
			#else:
				#for k in np.linspace(-10, 10, 1000):
					#wrong_inside = np.flatnonzero(activation[noise][inside] < k).shape[0]
					#wrong_outside = np.flatnonzero(activation[noise][outside] > k).shape[0]
					#wrong_new = wrong_inside/(inside.shape[0]+0.0)/2. + wrong_outside/(outside.shape[0]+0.0)/2.
					#if wrong_new < wrong:
						#wrong = wrong_new
						#t = k
		wrong_inside = np.flatnonzero(activation[noise][inside] <= self.t[noise]).shape[0]
		wrong_outside = np.flatnonzero(activation[noise][outside] > self.t[noise]).shape[0]
		return [wrong_inside/(inside.shape[0]+0.0), wrong_outside/(outside.shape[0]+0.0)]
		
	def calcNumberWrongPixelsTotal(self, activation = None, noise = None, optimal_t = True):#return proportion inside wornd and outside wrong
		'''
		
		determines proportion pixel that fire inside the field wrong and outside.
		'''
		if activation ==None:
			activation = self.activation
		[inside, outside] = self.getLocationsWithinDistance(distance = self.distance, locations = self.locations, location = self.location, bins = self.bins)
		if self.Method == Solution.weightHebbian or Solution.takeWeightDistributed:
			self.t[noise]=0
			if self.Method == Solution.takeWeightDistributed:
				optimal = False
		else:
			self.t[noise]=1
		if optimal_t:
			wrong = 0.5
			#if self.Method == Solution.weightHebbian:
			for k in np.linspace(np.min(activation[noise]), np.max(activation[noise]), 1000):
				wrong_inside = np.flatnonzero(activation[noise][inside] <= k).shape[0]
				wrong_outside = np.flatnonzero(activation[noise][outside] > k).shape[0]
				wrong_new = wrong_inside/(inside.shape[0]+0.0)/2. + wrong_outside/(outside.shape[0]+0.0)/2.
				if wrong_new < wrong:
					wrong = wrong_new
					self.t[noise] = k
			#else:
				#for k in np.linspace(-10, 10, 1000):
					#wrong_inside = np.flatnonzero(activation[noise][inside] < k).shape[0]
					#wrong_outside = np.flatnonzero(activation[noise][outside] > k).shape[0]
					#wrong_new = wrong_inside/(inside.shape[0]+0.0)/2. + wrong_outside/(outside.shape[0]+0.0)/2.
					#if wrong_new < wrong:
						#wrong = wrong_new
						#t = k
		wrong_inside = (np.flatnonzero(activation[noise][inside] <= self.t[noise]).shape[0])*1./activation.shape[-2]
		wrong_outside = np.flatnonzero(activation[noise][outside] > self.t[noise]).shape[0]*1./activation.shape[-2]
		return [wrong_inside, wrong_outside]

	def getNumberWrongPixels(self, activation = None, noise = None, sumed = False, both_sides = False):
		

		#activation_thres = np.copy(activation)
		if sumed:
			if noise == None:
				wrong = np.zeros(len(self.In.noise_levels))
				for i in range(len(self.In.noise_levels)):
					wrong[i] = np.sum(self.calcNumberWrongPixelsTotal(noise = i, activation = activation))
			else:
				wrong = np.sum(self.calcNumberWrongPixelsTotal(noise = noise, activation = activation))
		else:
			if noise == None:
				wrong = np.zeros(len(self.In.noise_levels))
				for i in range(len(self.In.noise_levels)):
					wrong[i] = np.sum(self.calcNumberWrongPixels(noise = i, activation = activation))/2.
			else:

				wrong = np.sum(self.calcNumberWrongPixels(noise = noise, activation = activation))/2.
		
		if both_sides:
			t_pos = np.copy(self.t)
			if sumed:
				if noise == None:
					wrong2 = np.zeros(len(self.In.noise_levels))
					for i in range(len(self.In.noise_levels)):
						wrong2[i] = np.sum(self.calcNumberWrongPixelsTotal(noise = i, activation = -activation))
				else:
					wrong2 = np.sum(self.calcNumberWrongPixelsTotal(noise = noise, activation = -activation))
			else:
				if noise == None:
					wrong2 = np.zeros(len(self.In.noise_levels))
					for i in range(len(self.In.noise_levels)):
						wrong2[i] = np.sum(self.calcNumberWrongPixels(noise = i, activation = -activation))/2.
					
				else:
					wrong2 = np.sum(self.calcNumberWrongPixels(noise = noise, activation = -activation))/2.
			if noise == None:
				wrong1 = np.copy(wrong)
				wrong[wrong1 > wrong2] = wrong2[wrong1 > wrong2]
				wrong[wrong1 <= wrong2] = wrong1[wrong1 <= wrong2]
				self.t[wrong1 <= wrong2] = t_pos[wrong1 <= wrong2]
			else:
				wrong = min(wrong, wrong2)
				if wrong != wrong2:
					self.t[noise] = t_pos[noise]
					self.negative = False
				else:
					self.negative = True
		return wrong
		
	def getCorr(self):
		corr = np.zeros(len(self.In.noise_levels))
		for i in range(len(self.In.noise_levels)):
			corr[i] = scipy.stats.pearsonr(self.activation[0],self.activation[i])[0]
		return corr
		
	def getCorrWithout(self):
		corr = np.zeros(len(self.In.noise_levels))
		self.calcActivationMapWithout()
		for i in range(len(self.In.noise_levels)):
			corr[i] = scipy.stats.pearsonr(self.activation[0],self.activation_without[i])[0]
		return corr
		
		
	def findPlaceWeightSVC(self):
	
		[inside, outside] = self.getPopulationWithinDistance(distance = self.distance, locations = self.locations, location = self.location, bins = self.bins, with_noise = 0)
		X = np.concatenate((inside, outside))
		Y = np.concatenate((np.ones(inside.shape[0]), np.ones(outside.shape[0])*-1))
		SVC = svm.LinearSVC(loss='l2', penalty='l1', dual=False, C = 1)
		#SVC = svm.LinearSVC()
		SVC.fit(X,Y)
		self.w = np.copy(SVC.coef_).T

		
	def weightLogReg(self):
	
		[inside, outside] = self.getPopulationWithinDistance(distance = self.distance, locations = self.locations, location = self.location, bins = self.bins, with_noise = 0)
		X = np.concatenate((inside, outside))
		Y = np.concatenate((np.ones(inside.shape[0]), np.zeros(outside.shape[0])*-1))
		self.LogReg = linear_model.LogisticRegression()
		self.LogReg.fit(X,Y)
		self.w = np.copy(self.LogReg.coef_).T

		

	def weightHebbian(self):
		Place = PlaceFields(number_cells = 1,  actFunction = Input.getOutputId, number_patterns = self.In.number_patterns, number_to_store =self.In.number_patterns, store_indizes = np.tile(np.arange(self.In.number_patterns), (1,1)), cage = self.In.cage, field_size = self.distance, centers = [np.array([self.In.getStoredlocations()[0,self.location]])], active_in_env = 1, noise_levels = [0])

	
		Ec_Ca3 = HeteroAssociation(input_cells=self.In.cells, cells=1, connectivity = 1, learnrate= 1, subtract_input_mean = 1, subtract_output_mean = 0, actFunction = Network.getOutputId, number_winner=1, e_max = 0.1, active_in_env = 1, n_e = 1, initMethod = Network.makeWeightsZero, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
		Ec_Ca3.learnAssociation(input_pattern = self.In.input_stored, output_pattern = Place.input_stored)
		self.w = np.copy(Ec_Ca3.weights.T)

	def takeActivityAsWeight(self, subtract_mean = 1):
		mean = np.einsum('pi -> i', self.In.input_stored[0])/(self.In.input_stored.shape[1]*1.)
		if subtract_mean:
			self.w = (self.In.input_stored[0, self.location] - mean).reshape(self.In.input_stored.shape[-1], 1)
		else:
			self.w = self.In.input_stored[0, self.location].reshape(self.In.input_stored.shape[-1], 1)

	def takeWeightRandom(self):
		self.w = np.random.normal(loc = 0.5, scale = 0.5, size = (self.In.input_stored.shape[-1],1))
		#normalize(self.w)
		
	def takeWeightDistributed(self):
		
		
		#emax =1- np.concatenate((np.linspace(0.01, 0.1, 5), np.linspace(0.12, 0.34, 6)))
		emax = np.arange(1,21)/40.
		#emax = np.array([0.2])
		self.Ec_Ca3 = Network(input_cells=self.In.cells, cells=3500, connectivity = 1, learnrate= 1, subtract_input_mean = 1, subtract_output_mean = 0, actFunction = Network.getOutputEMax, number_winner=1, e_max = emax[0], active_in_env = 3500, n_e = 1, initMethod = Network.makeWeightRealisticDistributed, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
		self.actFunction = True
		#self.Ec_Ca3.weights = self.In.input_stored[0]
		
		wrong = self.In.input_stored.shape[1]
		changed = True
		for e in emax:
			if changed:
				changed = False
				self.Ec_Ca3.e_max = e
				activation = self.Ec_Ca3.getOutput(self.Ec_Ca3, input_pattern = self.In.input_stored)
				for cell in range(self.Ec_Ca3.cells):
					wrong_new = self.getNumberWrongPixels(activation = activation[:,:,cell], noise = 0, sumed = 0)
					if wrong_new < wrong:
						changed = True
						wrong = wrong_new
						self.used_cell = cell
						self.used_emax = e
						used_t = self.t[0]
						
		self.w = np.copy(self.Ec_Ca3.weights[self.used_cell]).reshape(self.In.cells, 1)
		self.Ec_Ca3.e_max = self.used_emax
		self.t[0] = used_t

	
	def weightLDA(self): #does not work well
		[inside, outside] = self.getPopulationWithinDistance(distance = self.distance, locations = self.locations, location = self.location, bins = self.bins, with_noise = 0)
		X = np.concatenate((inside, outside))
		Y = np.concatenate((np.ones(inside.shape[0]), np.zeros(outside.shape[0])))
		self.LDA = lda.LDA()
		self.LDA.fit(X,Y,store_covariance=True)

		cov_inv = np.linalg.inv(self.LDA.covariance_)

		[m_0, m_1] = self.LDA.means_

		self.w = np.dot(cov_inv, (m_1-m_0)).reshape(m_0.shape[0], 1)


	def weightICA(self):
		

		self.Ica = decomposition.FastICA()
		self.Ica.fit(np.copy(self.In.input_stored[0]))


		Y = self.Ica.transform(np.copy(self.In.input_stored[0]))

		wrong = 0.5
		for cell in range(Y.shape[1]):
			wrong_new = self.getNumberWrongPixels(activation = Y[:,cell].reshape(1, Y[:,cell].shape[0]), noise = 0, both_sides = True)
			if wrong_new < wrong:
				self.used_cell = cell
		

		unmix = np.linalg.inv(self.Ica.get_mixing_matrix())

		
		#self.w = self.Ica.components_[self.used_cell].reshape(self.Ica.components_[self.used_cell].shape[0], 1)
		self.w = unmix[self.used_cell].reshape(unmix[self.used_cell].shape[0], 1)
		if self.negative:
			self.w*=-1

		
	def weightLinReg(self):
		Place = PlaceFields(number_cells = 1,  actFunction = Input.getOutputId, number_patterns = self.In.number_patterns, number_to_store =self.In.number_patterns, store_indizes = np.tile(np.arange(self.In.number_patterns), (1,1)), cage = self.In.cage, field_size = np.ones([1,1,1])*self.distance, centers = np.array([self.In.getStoredlocations()[0,self.location]]).reshape(1,1,1,2), active_in_env = 1, noise_levels = [0],n_e=1)
		Ec_Ca3 = Network(input_cells=self.In.cells, cells=1, connectivity = 1, learnrate= 1, subtract_input_mean = 1, subtract_output_mean = 0, actFunction = Network.getOutputId, number_winner=1, e_max = 0.1, active_in_env = 1, n_e = 1, initMethod = Network.makeWeightsZero, weight_sparsity = None, weight_mean = 1, weight_sigma = 0.5)
		Ec_Ca3.learnRegression(input_pattern = self.In.input_stored, output_pattern = Place.input_stored)
		self.w = np.copy(Ec_Ca3.weights.T)
		
	def calcActivationMap(self, w =None):
		
		if w ==None:
			w = self.w
			if not self.actFunction:
				activation = np.dot(self.In.noisy_input_stored[0],w)
			else:
				activation = self.Ec_Ca3.getOutput(self.Ec_Ca3, input_pattern = self.In.noisy_input_stored)[0,:, :,self.used_cell].reshape(self.In.noisy_input_stored.shape[1],self.In.noisy_input_stored.shape[2],1 )

			self.activation = np.copy(activation)
		else:
			if not self.actFunction:
				activation = np.dot(self.In.noisy_input_stored[0],w)
			else:
				activation = self.Ec_Ca3.getOutput(self.Ec_Ca3, input_pattern = self.In.noisy_input_stored)[0,:, :,self.used_cell].reshape(self.In.noisy_input_stored.shape[1],self.In.noisy_input_stored.shape[2],1 )
		return activation
		
	def calcActivationMapWithout(self):
		w = np.copy(self.w)
		w[:self.In.modules[1]] = 0
		self.activation_without = self.calcActivationMap(w =w)

	def getActivationMapThres(self, activation = None, noise = 0, both_sides = False, sumed = False):
		if activation ==None:
			activation = self.activation
		activation = np.copy(activation)
		self.getNumberWrongPixels(activation = activation, noise = noise, both_sides = False, sumed = sumed)
		activation[noise][activation[noise] <= self.t[noise]] = 0
		return activation
		
class Model():
	
	def __init__(self):
		pass
		
	def getAverageFieldSize(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getAverageFieldSize()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getAverageFieldSize()
		return av
		
	def getAverageFieldNumber(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getAverageFieldNumber()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getAverageFieldNumber()
		return av
		
	def getAverageFieldNumberActiveCells(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getAverageFieldNumberActiveCells()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getAverageFieldNumberActiveCells()
		return av
		

	def getAverageFieldNumberActiveCellsWithField(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getAverageFieldNumberActiveCellsWithField()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getAverageFieldNumberActiveCellsWithField()
		return av

	def getProportionActiveCells(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getProportionActiveCells()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getProportionActiveCells()
		return av
	
	def getProportionCellsWithField(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getProportionCellsWithField()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getProportionCellsWithField()
		return av
		
	def getfieldSizes(self, var = None):
		if var == None:
			av = []
			for i in range(self.var.shape[0]):
				av.append(self.Spatials[i].getfieldSizes())
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getfieldSizes()
		return av
		
	def getAverageCoverActiveCellsWithField(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getAverageCoverActiveCellsWithField()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getAverageCoverActiveCellsWithField()
		return av

	def getAverageCoverActiveCells(self, var = None):
		if var == None:
			av = np.zeros(self.var.shape[0])
			for i in range(self.var.shape[0]):
				av[i] = self.Spatials[i].getAverageCoverActiveCells()
		else:
			e_max_ind = np.nonzero(var == self.var)
			av = self.Spatials[e_max_ind].getAverageCoverActiveCells()
		return av
		
class Hebb(Model):
	
	def __init__(self,In = None, cells = None, OutputGiven = None, connectivity = None, min_size = None, si_criterion = False, min_rate = None, no_synapses = None, non_zero_activation =False,sub_mean =False, sparsity = None):
		
		self.In = In
		self.weights = []
		self.var_string = np.array(['Given', 'Recalled'])
		self.var = np.array([0,1])
		
		self.output = np.zeros([2, In.input_stored.shape[1], cells])
		self.Spatials = []
		Ec_Ca1 = HeteroAssociation(input_cells=In.cells, cells=cells, connectivity = connectivity, actFunction = Network.getOutputWTALinear, active_in_env = cells, initMethod = Network.makeWeightsZero, number_winner =int(sparsity*cells), learnrate = 1, subtract_input_mean = 1)


		
		#if no_synapses[1]>0 and no_synapses[0]>0:
		print 'synapses in Hebb', no_synapses
		Ec_Ca1.initConnectionModular(border = [In.grid_cells + In.border_cells, In.grid_cells+In.border_cells+In.lec_cells], no_synapses = no_synapses)
		
		if sub_mean:
			mean = np.mean(OutputGiven.input_stored, axis = -2)
			mean = np.tile(mean, (1, OutputGiven.input_stored.shape[-2],1))
			OutputGiven.input_stored -=mean
			
		
		print 'learn asso hebb model'		
		Ec_Ca1.learnAssociation(input_pattern = In.input_stored, output_pattern = OutputGiven.input_stored)
		print "finish learn"		
		self.weights.append(Ec_Ca1.weights)

		
		self.output[0] = OutputGiven.input_stored[0]
		print "calc Spatials"		
		self.Spatials.append(Spatial(patterns = np.copy(self.output[0]), cage = In.cage, min_size = min_size,min_rate = min_rate,si_criterion = si_criterion))
		print "calc act"		
		self.activation = Ec_Ca1.getOutputId(input_pattern = In.input_stored)[0]
		if non_zero_activation:
			min_cell_fire = np.min(self.activation, axis = -2)
			min_cell_fire = np.tile(min_cell_fire, (self.activation.shape[-2],1))
			self.activation -= min_cell_fire
		
		#if sub_mean:
			#mean = np.mean(self.activation, axis = -2)
			#mean = np.tile(mean, (self.activation.shape[-2],1))
			#self.activation -=mean
		
		print "calc output"		
		self.output[1] = Ec_Ca1.getOutput(Ec_Ca1,input_pattern = In.input_stored)[0]
		#self.output[1] = np.copy(self.activation)
		self.Spatials.append(Spatial(patterns = np.copy(self.output[1]), cage = In.cage, min_size = min_size, min_rate = min_rate, si_criterion = si_criterion))
		
		

def MethodsPaper():
	
	grid_cells = 1100
	lec_cells =1100
	res =1600 * 2
	input_cells = grid_cells + lec_cells
	actFunction = Input.getOutputId
	kernel_sizes = np.array([1, 4, 6, 8, 12])
	fig = plt.figure(figsize = [7.5, 8])
	fig.subplots_adjust(left = .12, bottom = .08, right = .9, top = .9, wspace=0.2, hspace=0.5)
	ax_ind = [3,2]
	noise_levels = [0]
	real_info = [0.1] *45 + [0.3]*15 +[0.5]*4 +[0.7]*1 +[.9]#hargreaves

	peak = np.random.normal(loc = 1, scale = .1, size = (1,  grid_cells, 50,50))
	In = Grid(number_cells = grid_cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1] ,peak = peak)
	
	
	## grid cell examples
	grid = ImageGrid(fig, [ax_ind[0], ax_ind[1],1], nrows_ncols = (2,2),axes_pad=0.05, aspect = 1 ,share_all= 1, cbar_mode = 'single', cbar_pad = 0.05)
	for i in range(4):
		grid[i].xaxis.set_visible(False)
		grid[i].yaxis.set_visible(False)
	max_fire = np.max(np.concatenate((In.patterns[0,:, 0], In.patterns[0,:, In.modules[1]+1], In.patterns[0,:, In.modules[2]+1], In.patterns[0,:, In.modules[3]+1])))
	min_fire = np.min(np.concatenate((In.patterns[0,:, 0], In.patterns[0,:, In.modules[1]+1], In.patterns[0,:, In.modules[2]+1], In.patterns[0, :,In.modules[3]+1])))
	s = grid[0].imshow(In.patterns_2d[:,:,0], interpolation = 'none', origin = 'lower', vmin = min_fire, vmax = max_fire)
	grid[1].imshow(In.patterns_2d[:,:,In.modules[1]+1], interpolation = 'none', origin = 'lower', vmin = min_fire, vmax = max_fire)
	grid[2].imshow(In.patterns_2d[:,:,In.modules[2]+1], interpolation = 'none', origin = 'lower', vmin = min_fire, vmax = max_fire)
	grid[3].imshow(In.patterns_2d[:,:,In.modules[3]+1], interpolation = 'none', origin = 'lower', vmin = min_fire, vmax = max_fire)
	fig.colorbar(s, cax = grid.cbar_axes[0], ticks = [0,1])
	makeLabel(ax = grid[0], label = 'A', sci = 0 )
	
	ax = fig.add_subplot(ax_ind[0], ax_ind[1],2)
	ax.set_xlabel('Grid spacing (m)')
	ax.hist([In.spacing[0][:In.modules[1]], In.spacing[0][In.modules[1]: In.modules[2]],In.spacing[0][In.modules[2]: In.modules[3]],In.spacing[0][In.modules[3]: In.modules[4]]], label = ['m1', 'm2', 'm3', 'm4'], bins = 20, histtype = 'barstacked')
	ax.legend(loc='best', prop={'size':12})
	for tick in ax.xaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	for tick in ax.yaxis.get_major_ticks()[1:-1]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax, label = 'B', sci = 0 )
	
	ax = fig.add_subplot(ax_ind[0], ax_ind[1],3)
	ax.set_xlabel('Grid orientation (degree)')
	ax.hist([In.theta[0][:In.modules[1]], In.theta[0][In.modules[1]: In.modules[2]],In.theta[0][In.modules[2]: In.modules[3]],In.theta[0][In.modules[3]: In.modules[4]]], label = ['m1', 'm2', 'm3', 'm4'], bins = 20, histtype = 'barstacked')
	#ax.legend(loc='best', prop={'size':12})
	for tick in ax.xaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	for tick in ax.yaxis.get_major_ticks()[1:-1]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax, label = 'C', sci = 0 )
	
	
	infos = []
	k=0
	grid = ImageGrid(fig, [ax_ind[0], ax_ind[1],4], nrows_ncols = (3,2),axes_pad=0.3, aspect = 1 ,share_all= 1, cbar_mode = 'single',cbar_pad = 0.05)
	for size in kernel_sizes:
		In = JointInput(grid_cells = 0, lec_cells = lec_cells, border_cells = 0, inputMethod = Lec.makeActiveFilter, noiseMethod = Input.makeNoiseRandomFire, actFunction = actFunction, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = [0], normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = .35, size_kernel = size)
		info_sim = In.getSpatialInformation()
		infos.append(np.copy(info_sim))
		if size in [1,6,12]:
			cells_to_plot = np.array([0,1,2])
			for j in range(2):
				grid[k].set_title(str(np.round(info_sim[cells_to_plot[j]], 3)))
				[a,s] = plotCell(In = In, env =0, fig = fig, ax = grid[k], cb = 0, cell = cells_to_plot[j], zeros = 1)
				grid[k].set_xlim(0, In.cage[0])
				grid[k].set_ylim(0, In.cage[1])	
				if j > 0:
					grid[k].yaxis.set_visible(False)
					grid[k].xaxis.set_visible(False)
				else:
					for tick in grid[k].xaxis.get_major_ticks():
						tick.label1.set_visible(False)
					for tick in grid[k].yaxis.get_major_ticks():
						tick.label1.set_visible(False)
					grid[k].set_ylabel(r'$\sigma_N = $' + str(size))
				k+=1
			if np.flatnonzero(In.patterns[0,:,cells_to_plot[0]]).shape[0]>2:
				grid.cbar_axes[0].colorbar(s)
	makeLabel(ax = grid[0], label = 'D', sci = 0 )

	ax4 = fig.add_subplot(ax_ind[0], ax_ind[1], 5)
	n,bins,patches = ax4.hist(infos, label = map(str, kernel_sizes), histtype = 'step',bins = 30, normed =1, cumulative = 1)
	for p in patches:
		p[0].set_xy(p[0].get_xy()[:-1])	
	#n = ax4.hist(real_info, label =['Hargreaves et al.'], bins = [0,0.2,0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2. ,2.2, 2.4, 2.6, 2.8, 3], normed =1, cumulative = 1,histtype = 'step', ec = 'k')[0]
	n,bins,patches = ax4.hist(real_info, label =['Hargreaves\n et al.'], bins = [0,.2, .4, .6, .8, 1], normed =1, cumulative = 1,histtype = 'step', ec = 'k')
	#ax.set_ylim(0, int(np.max(np.concatenate((n[0], n[1])))*1.2))
	#ax4.hist(sizes[-1], label = 'Grid', histtype = 'step', bins = 60, color = 'g')
	#ax45 = ax4.twinx()
	#ax45.hist(sizes[0], label = 'LEC', histtype = 'step', bins = 60, color = 'k')
#	ax4.xaxis.set_ticks([0, 2000, 4000, 6000, 8000])
	ax4.set_ylim(0, 1.01)
	#ax4.set_xlim(0, 1.2)
	for tick in ax4.xaxis.get_major_ticks()[1:-1:2]:
		tick.set_visible(False)
	ax4.set_xlabel('spatial information (bit)')
	ax4.legend(loc = 'lower right',prop={'size':10})
	ax4.set_ylabel('cdf')
	patches[0].set_xy(patches[0].get_xy()[:-1])	
	makeLabel(ax = ax4, label = 'E', sci = 0 )


def problematic(location = 'middle'):
	
	cells = 1100
	res = 1600 *2
	n_lec = 0
	a = res


	noise_levels = [0]
	#bins_av = np.round(np.arange(0,0.7, 0.1), 3)
	ax_ind2 = [2,2]
	#fig = plt.figure()
	fig2 = plt.figure()
	In = Grid(number_cells = cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1])
	
	InNormed = Grid(number_cells = cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 1, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1])
	
	#InLec = JointInput(grid_cells = cells/2, lec_cells = n_lec, border_cells = 0, inputMethod = Lec.makeActiveRegionsPlaceFields, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = [0], normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = 0.75)
	
	a = res/In.cage[0]
	if location == 'middle':
		x = np.sqrt(res/2)
		y = np.sqrt(res/2)*2
		location = 0.5*y *x + 0.5*x 
	else:
		location = np.sqrt(a)/2 *np.sqrt(a)  + np.sqrt(a)/2
	
	x = np.sqrt(res/2)
	y = np.sqrt(res/2)*2
	location = 0.5*y *x + 0.5*y 
	y_length = np.sqrt(a)
	

	
	ax = fig2.add_subplot(ax_ind2[0], ax_ind2[1],2)
	corr = In.getSpatialAutocorrelation(mode = 'same')
	s = ax.imshow(corr, interpolation = 'none', origin = 'lower')
	#for distance in [0.14]:
		#circle = mpl.patches.Ellipse(xy = (In.x_length/2-1,In.y_length/2-1), width = 2*distance*In.y_length, height = 2*distance*In.y_length, angle=0.0, fc = 'none', ec = 'k', lw = 5)
		#ax.add_patch(circle)
	locator = mpl.ticker.FixedLocator([0, np.sqrt(res/2) -1 , 2 *np.sqrt(res/2)-1])
	formater = mpl.ticker.FixedFormatter([str(-np.round(1./np.sqrt(res/2)*np.sqrt(res/2),3)), '0',str(np.round(1./np.sqrt(res/2)*np.sqrt(res/2),3))])
	ax.xaxis.set_major_locator(locator)
	ax.xaxis.set_major_formatter(formater)
	
	locator2 = mpl.ticker.FixedLocator([0, .5*np.sqrt(res/2)-1, np.sqrt(res/2)-1])
	formater2 = mpl.ticker.FixedFormatter([str(-np.round(1./np.sqrt(res/2)*.5 *np.sqrt(res/2),3)), '0',str(np.round(1./np.sqrt(res/2)*.5*np.sqrt(res/2),3))])
	ax.yaxis.set_major_locator(locator2)
	ax.yaxis.set_major_formatter(formater2)
	col = fig2.colorbar(s, ticks = np.round([np.min(corr)+0.04, np.max(corr)],1))
	makeLabel(ax = ax, label = 'B', sci = 0 )
	
	

	
	ax = fig2.add_subplot(ax_ind2[0], ax_ind2[1],3)
	bins = np.linspace(0,np.sqrt(3), 50)
	av = InNormed.getAverageOverlapOverDistance(locations = InNormed.getStoredlocations(), bins = bins)
	av_cor = In.getAverageCorOverDistance(locations = In.getStoredlocations(), bins = bins)
	#ax.plot(av_cor[1], av_cor[0], label = r'$Corr(\mathbf{p}_{t_0}, \mathbf{p}_t)$')
	#ax2 = ax.twinx()
	ax.plot(av[1] ,av[0])
	#ax.plot(av[1][6] ,av[0][6], marker = 'd', color = 'r')
	#ax.plot([0, av[1][14]] ,[av[0][14] ,av[0][14]])
	#ax.yaxis.set_visible(False)
	ax.set_ylabel(r'$cos\left[ \mathbf{p}(\mathbf{r}_i), \mathbf{p}(\mathbf{r}_j)\right] $')
	ax.set_xlabel(r'$||\mathbf{r}_i - \mathbf{r}_j||$')
	for tick in ax.xaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	for tick in ax.yaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	ax.legend(loc = 'upper right', prop={'size':10})	
	makeLabel(ax = ax, label = 'C', sci = 0 )
	
	
	
	
	Sol = Solution(In = In, Method = Solution.doNothing, weights_given = In.input_stored[0][location].reshape(cells,1), distance = .5, location = location)
	ax = fig2.add_subplot(ax_ind2[0], ax_ind2[1],4)
	corr = Sol.activation[0].reshape(In.y_length, In.x_length)
	#In.getSpatialAutocorrelation(mode = 'same')
	s = ax.imshow(corr, interpolation = 'none', origin = 'lower')
	#ax, s = plotCell(In = In, env =0, patterns = np.tile(Sol.activation[0], (1,1,1)), fig = fig2, ax_index = [ax_ind2[0], ax_ind2[1],4], cb = 1, cell = 0)
	for distance in [0.1]:
		circle = mpl.patches.Ellipse(xy = (In.x_length/2-1,In.y_length/2), width = 2*distance*In.y_length, height = 2*distance*In.y_length, angle=0.0, fc = 'none', ec = 'k', lw = 2, ls = 'dashed')
		#circle = mpl.patches.Ellipse(xy = (In.getStoredlocations()[0, location, 0], In.getStoredlocations()[0, location, 1]), width = 2*distance, height = 2*distance, angle=0.0, fc = 'none', ec = 'k',lw = 5, ls = 'dashed')
		ax.add_patch(circle)
	locator = mpl.ticker.FixedLocator([0,2 *np.sqrt(res/2)-1])
	formater = mpl.ticker.FixedFormatter(['0','2'])
	ax.xaxis.set_major_locator(locator)
	ax.xaxis.set_major_formatter(formater)
	
	locator2 = mpl.ticker.FixedLocator([0, np.sqrt(res/2)-1])
	formater2 = mpl.ticker.FixedFormatter(['0', '1'])
	ax.yaxis.set_major_locator(locator2)
	ax.yaxis.set_major_formatter(formater2)
	col = fig2.colorbar(s)
	col.set_ticks([float(col.ax.yaxis.get_majorticklabels()[0].get_text()), float(col.ax.yaxis.get_majorticklabels()[-1].get_text())])
	makeLabel(ax = ax, label = 'D', sci = 0)


def SVCPaper(location = 'middle', distances = [0.10, 0.25, 0.35]):
	cells = 1100
	g_cells = cells
	connections = None
	res = 1600*2
	#a = res
	a = res/2
	location = 'not'
	sumed = 0
	if location == 'middle':
		y = np.sqrt(res/2)
		x = np.sqrt(res/2)*2
		location = np.int(0.5*y *x + 0.5*x)

	else:
		#location = np.int(np.sqrt(a)/2 *np.sqrt(a)  + np.sqrt(a)/2)
		x_pixel = np.sqrt(a)*2
		y_pixel =np.sqrt(a)
		location = int(int(1/3. * y_pixel) *x_pixel + 1/3. * x_pixel)
		
	noise_points = 15.
	noise_levels = np.arange(0,cells+1, int(cells/(noise_points-1)))
	#noise_levels = [0, cells/10]
	#noise_levels = [0]
	#solvers.options['show_progress'] = False
	#bins_av = np.round(np.arange(0,0.7, 0.1), 3)
	ax_ind = [3,3]

	#InBase = Grid(number_cells = cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = 4, number_to_store =4 ,n_e =1,noise_levels = [0], normed = 0, store_indizes = np.tile(np.arange(4), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1]))
	#peak = np.random.uniform(.5, 1.5, size = (1,  g_cells, 50,50))
	#peak = np.random.uniform(.8, 1.2, size = (1,  g_cells, 50,50))
	peak = np.random.normal(loc = 1, scale = .05, size = (1,  g_cells, 50,50))
	#peak = None
	In = Grid(number_cells = cells, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1], sparsity = 1, peak = peak, r_to_s = 0.32)
	print In.noise_levels
	InBig = Grid(number_cells = cells, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=2, cage =[2,1], sparsity = 1, peak = peak, r_to_s = 0.32)


	fig = plt.figure()
	#fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace = 0.24)
	d_ind =0
	letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
	for d in distances:
		bins_av = np.array([d])
		bins= bins_av
		labels = ['M1+M2','all','M3+M4']
		colors =['b', 'g', 'r']
		distance = bins_av[0]
		
		SVCAll = Solution(In = In, Method = Solution.findPlaceWeightSVC, distance = distance, location = location, max_weight = -1, non_negative = 0)
		SVCBig = Solution(In = InBig, Method = Solution.findPlaceWeightSVC, distance = distance, location = location, max_weight = -1, non_negative = 0)

		
	
	
		grid = ImageGrid(fig, [ax_ind[0], ax_ind[1],1+ d_ind*3], nrows_ncols = (3,1),axes_pad=0.1, aspect = 1 ,share_all= 1, cbar_mode = 'none')
		for i in range(3):
			circle = mpl.patches.Ellipse(xy = (In.getStoredlocations()[0, location, 0], In.getStoredlocations()[0, location, 1]), width = 2*distance, height = 2*distance, angle=0.0, fc = 'none', ec = 'k', lw = 2, ls = 'dashed')
			grid[i].add_patch(circle)
			grid[i].set_xlim(0, In.cage[0])
			grid[i].set_ylim(0, In.cage[1])
			grid[i].xaxis.set_visible(False)
			for tick in grid[i].yaxis.get_major_ticks():
				tick.label1.set_visible(False)
		max_fire = np.max(np.concatenate( (SVCAll.activation[1], SVCAll.activation[0])))
		min_fire = np.min(np.concatenate( (SVCAll.activation[1], SVCAll.activation[0])))
		im ,s = plotCell(In = In, env =0, patterns = np.tile(SVCAll.activation[0], (1,1,1)), fig = fig, ax = grid[0], cb = 0, cell = 0, vmin = min_fire, vmax = max_fire )
		#plotCell(In = In, env =0, patterns = np.tile(SVCAll.activation[1], (1,1,1)), fig = fig, ax = grid[1], cb = 0, cell = 0,vmin = min_fire, vmax = max_fire)
		plotCell(In = In, env =0, patterns = np.tile(SVCAll.getActivationMapThres(sumed = sumed)[0], (1,1,1)), fig = fig, ax = grid[1], cb = 0, cell = 0,vmin = min_fire, vmax = max_fire)
		plotCell(In = In, env =0, patterns = np.tile(SVCAll.getActivationMapThres(noise = 1, sumed = sumed )[1], (1,1,1)), fig = fig, ax = grid[2], cb = 0, cell = 0,vmin = min_fire, vmax = max_fire)
		#fig.colorbar(s, cax = grid.cbar_axes[0])
		for t in grid.cbar_axes[0].yaxis.get_major_ticks()[1:-1]:
			t.set_visible(False)
		#grid[0].set_ylabel(r"$\mathbf{w}^T \mathbf{p}$")
		#grid[1].set_ylabel(r"$\mathbf{w}^T \mathbf{p} > c$")
		grid[1].set_ylabel(str(int(np.round(np.pi * d**2 *10000))) + r'$cm^2$')
		#grid[2].set_ylabel(r"$\mathbf{w}^T \tilde{\mathbf{p}} > c$")
		makeLabel(ax = grid[0], label = letters[d_ind * 3], sci = 0 )
		#grid.set_tilte(str(int(np.round(np.pi * d**2 *10000))) + r'$cm^2$')
		
		
		
		
		
		#ax = fig.add_subplot(ax_ind[0], ax_ind[1],1 + d_ind*4)
		#ax.set_title('Correlation')
		#ax.plot(1-In.getOrigVsOrig(), SVCAll.getCorr(), label = 'SVC',c='b', marker = 'd')
		##ax.plot(In.noise_levels, SVCAll.getCorrWithout(), linestyle = ':',c='b')
		#ax.set_ylim(0,1)
		##ax.legend(loc = 'lower left',prop={'size':10})
		
		ax = fig.add_subplot(ax_ind[0], ax_ind[1],2+ d_ind*3)
		#ax.set_title(str(np.round(np.pi * d**2 *10000)) + ' cm2')
		#ax.set_ylabel('proportion wrong')
		#ax.set_xlabel('proportion wrong in EC')
		#ax.plot(np.array(In.noise_levels) * 1./cells ,SVCAll.getNumberWrongPixels(sumed = sumed), label = 'Modules 1-4',c='b',marker = 'd')
		#ax.plot(np.array(In.noise_levels) *1./cells,SVCBig.getNumberWrongPixels(sumed = sumed), label = 'Modules 3+4',c='b',marker = 'd', linestyle = '--')
		#ax.plot(np.array(In.noise_levels)[1] * 1./cells ,SVCAll.getNumberWrongPixels(sumed = sumed)[1],c='r',marker = 'd')
	
		ax.set_ylabel(r'$\epsilon$')
		ax.set_xlabel('no. lesioned cells')
		ax.plot(np.array(In.noise_levels) * 1 ,SVCAll.getNumberWrongPixels(sumed = sumed), label = 'Mod. 1-4',c='b')
		ax.plot(np.array(In.noise_levels) *1,SVCBig.getNumberWrongPixels(sumed = sumed), label = 'Mod. 3+4',c='g')
		ax.plot(np.array(In.noise_levels)[1] * 1 ,SVCAll.getNumberWrongPixels(sumed = sumed)[1],c='r',marker = 'd', markersize =12)
		ax.plot([0,cells], [0, .5], linestyle = '--', c = 'k')
		ax.set_ylim(0, .5)
		ax.legend(loc = 'upper left',prop={'size':10})
		for tick in ax.yaxis.get_major_ticks()[::2]:
			tick.label1.set_visible(False)
		for tick in ax.xaxis.get_major_ticks()[::2]:
			tick.label1.set_visible(False)
		makeLabel(ax = ax, label = letters[d_ind * 3+1], sci = 0 )	
		
	
	
		ax = fig.add_subplot(ax_ind[0], ax_ind[1],3+ d_ind*3)
		#ax.set_title(str(int(np.round(np.pi * d**2 *10000))) + r'$cm^2$')
		ax.set_ylabel('|w|')
		w = np.abs(SVCAll.w[:,0])
		#w = SVCAll.w[:,0]
		ax.set_xlabel('Module')
		#per =  np.percentile(w[:In.modules[1]], [95, 75])
		for i in range(4):
		#ax.boxplot([w[:In.modules[1]], w[In.modules[1]: In.modules[2]],w[In.modules[2]: In.modules[3]],w[In.modules[3]: In.modules[4]]], whis = [.05, .95], flierprops = dict(marker='o'))
			ax.scatter([i+1] * w[In.modules[i]:In.modules[i+1]].shape[0], w[In.modules[i]:In.modules[i+1]])
		#y_lim = ax.get_ylim()
		#ax.set_ylim(-1, y_lim[1]+1)
		#ax.set_ylim(np.min(w)*1.5, np.max(w)*1.1)
		for tick in ax.yaxis.get_major_ticks()[::2]:
			tick.label1.set_visible(False)
		locator = mpl.ticker.FixedLocator([1,2,3,4,5])
		formater = mpl.ticker.FixedFormatter(['1', '2', '3', '4', 'LEC'])
		ax.xaxis.set_major_locator(locator)
		ax.xaxis.set_major_formatter(formater)
		makeLabel(ax = ax, label = letters[d_ind * 3+2], sci = 0 )
		d_ind +=1
	fig.subplots_adjust(hspace=0.5, wspace=0.5, left = .05, bottom = .1, right = .95)


def SVCPaperOneMod(location = 'middle', distances = [0.35]):
	cells = 1100
	g_cells = cells
	connections = None
	res = 1600*2
	#a = res
	a = res/2
	location = 'not'
	sumed = 0
	if location == 'middle':
		y = np.sqrt(res/2)
		x = np.sqrt(res/2)*2
		location = np.int(0.5*y *x + 0.5*x)

	else:
		#location = np.int(np.sqrt(a)/2 *np.sqrt(a)  + np.sqrt(a)/2)
		x_pixel = np.sqrt(a)*2
		y_pixel =np.sqrt(a)
		location = int(int(1/3. * y_pixel) *x_pixel + 1/3. * x_pixel)
		
	noise_points = 15.
	#noise_levels = np.arange(0,cells+1, int(cells/(noise_points-1)))
	noise_levels = np.round(np.linspace(0, cells-1, noise_points))
	#noise_levels = [0, cells/10]
	#noise_levels = [0]
	#solvers.options['show_progress'] = False
	#bins_av = np.round(np.arange(0,0.7, 0.1), 3)
	ax_ind = [max(len(distances),2),3]

	#InBase = Grid(number_cells = cells, noiseMethod = Input.makeNoiseRandomFire, actFunction = Input.getOutputId, number_patterns = 4, number_to_store =4 ,n_e =1,noise_levels = [0], normed = 0, store_indizes = np.tile(np.arange(4), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1]))
	#peak = np.random.uniform(.5, 1.5, size = (1,  g_cells, 50,50))
	#peak = np.random.uniform(.8, 1.2, size = (1,  g_cells, 50,50))
	peak = np.random.normal(loc = 1, scale = .05, size = (1,  g_cells, 50,50))
	#peak = None

	#fig6 = plt.figure()
	#fig5 = plt.figure()
	fig = plt.figure()
	#fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace = 0.24)
	d_ind =0
	letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
	grid_spacing = [.7,1.,1.3,1.6,2., -1]
	#grid_spacing = [ -1, -2]
	number_sims =10
	for d in distances:
		data = np.zeros([len(grid_spacing), noise_points])
		for iteration in range(number_sims):
			bins_av = np.array([d])
			bins= bins_av
			labels = ['M1+M2','all','M3+M4']
			colors =['b', 'g', 'r']
			distance = bins_av[0]
			
			for spacing in grid_spacing:
				spacings =  np.ones([1,cells]) * spacing
				orientations = np.zeros([1,g_cells])
				if spacing == -1 or spacing == -2: #wsm cells
					if spacing == -1:
						grid_cells = int(cells/6.)
						print '-------------------------------------', grid_cells
						In = JointInput(grid_cells = grid_cells, lec_cells = cells - grid_cells, border_cells = 0, inputMethod = Lec.makeActiveFilter, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = 1, size_kernel = 6, peak = None)
						print noise_levels
						print In.noise_levels
						for k in range(4):
	
							cells_to_plot = [0,400, 800, -1]
							#[a,s] = plotCell(In = In, patterns = In.noisy_input_stored, env =0, only_stored = False, noise = 0, fig = fig5, ax_index = [3,4,d_ind*4 +(k+1)], cb = 0, cell = cells_to_plot[k], zeros = 1)
							#a.yaxis.set_visible(False)
							#a.xaxis.set_visible(False)
					else:
						In = JointInput(grid_cells = 0, lec_cells = cells, border_cells = 0, inputMethod = Lec.makeActiveFilter, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'linear', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = 1, size_kernel = 6, peak = None)
	
				else:
					In = Grid(number_cells = cells, noiseMethod = Input.makeNoiseZero, actFunction = Input.getOutputId, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'linear', rat=1, cage =[2,1], sparsity = 1, peak = peak, spacings = spacings, theta = orientations, r_to_s = 0.32 )
					if spacing == 1.3:
						for k in range(4):
		
							cells_to_plot = [0,100, 105, -1]
							#[a,s] = plotCell(In = In, patterns = In.noisy_input_stored, env =0, only_stored = False, noise = 0, fig = fig6, ax_index = [3,4,d_ind*4 +(k+1)], cb = 0, cell = cells_to_plot[k], zeros = 0)
							#a.yaxis.set_visible(False)
							#a.xaxis.set_visible(False)
				
				SVCAll = Solution(In = In, Method = Solution.findPlaceWeightSVC, distance = distance, location = location, max_weight = -1, non_negative = 0)
	
	
			
		
				if spacing == -1 and iteration == 0:
					grid = ImageGrid(fig, [ax_ind[0], ax_ind[1],1+ d_ind*3], nrows_ncols = (3,1),axes_pad=0.1, aspect = 1 ,share_all= 1, cbar_mode = 'none')
					for i in range(3):
						circle = mpl.patches.Ellipse(xy = (In.getStoredlocations()[0, location, 0], In.getStoredlocations()[0, location, 1]), width = 2*distance, height = 2*distance, angle=0.0, fc = 'none', ec = 'k', lw = 2, ls = 'dashed')
						grid[i].add_patch(circle)
						grid[i].set_xlim(0, In.cage[0])
						grid[i].set_ylim(0, In.cage[1])
						grid[i].xaxis.set_visible(False)
						for tick in grid[i].yaxis.get_major_ticks():
							tick.label1.set_visible(False)
					max_fire = np.max(np.concatenate( (SVCAll.activation[1], SVCAll.activation[0])))
					min_fire = np.min(np.concatenate( (SVCAll.activation[1], SVCAll.activation[0])))
					im ,s = plotCell(In = In, env =0, patterns = np.tile(SVCAll.activation[0], (1,1,1)), fig = fig, ax = grid[0], cb = 0, cell = 0, vmin = min_fire, vmax = max_fire )
					#plotCell(In = In, env =0, patterns = np.tile(SVCAll.activation[1], (1,1,1)), fig = fig, ax = grid[1], cb = 0, cell = 0,vmin = min_fire, vmax = max_fire)
					plotCell(In = In, env =0, patterns = np.tile(SVCAll.getActivationMapThres(sumed = sumed)[0], (1,1,1)), fig = fig, ax = grid[1], cb = 0, cell = 0,vmin = min_fire, vmax = max_fire)
					plotCell(In = In, env =0, patterns = np.tile(SVCAll.getActivationMapThres(noise = 1, sumed = sumed )[1], (1,1,1)), fig = fig, ax = grid[2], cb = 0, cell = 0,vmin = min_fire, vmax = max_fire)
					#fig.colorbar(s, cax = grid.cbar_axes[0])
					for t in grid.cbar_axes[0].yaxis.get_major_ticks()[1:-1]:
						t.set_visible(False)
					#grid[0].set_ylabel(r"$\mathbf{w}^T \mathbf{p}$")
					#grid[1].set_ylabel(r"$\mathbf{w}^T \mathbf{p} > c$")
					grid[1].set_ylabel(str(int(np.round(np.pi * d**2 *10000))) + r'$cm^2$')
					#grid[2].set_ylabel(r"$\mathbf{w}^T \tilde{\mathbf{p}} > c$")
					makeLabel(ax = grid[0], label = letters[d_ind * 3], sci = 0 )
					#grid.set_tilte(str(int(np.round(np.pi * d**2 *10000))) + r'$cm^2$')
				
				data[spacing == np.array(grid_spacing)] += SVCAll.getNumberWrongPixels(sumed = sumed)
		

		
		data /= number_sims*1.
		print ax_ind, 'ax_ind--------------------------------------------------------'
		ax = fig.add_subplot(ax_ind[0], ax_ind[1],2+ d_ind*3)
		ax.set_ylabel(r'$\epsilon$')
		ax.set_xlabel('no. lesioned cells')
		for i, spacing in enumerate(grid_spacing):
			print i
			if spacing == -1:
				spacing = 'WSM cells + grid'
			if spacing == -2:
				spacing = 'WSM cells'
			ax.plot(np.array(In.noise_levels) * 1 ,data[i], label = str(spacing))
		ax.plot([0,cells], [0, .5], linestyle = '--', c = 'k')
		ax.set_ylim(0, .5)
		ax.legend(loc = 'upper left',prop={'size':14})
		for tick in ax.yaxis.get_major_ticks()[::2]:
			tick.label1.set_visible(False)
		for tick in ax.xaxis.get_major_ticks()[::2]:
			tick.label1.set_visible(False)
		makeLabel(ax = ax, label = letters[d_ind * 3+1], sci = 0 )	
		d_ind +=1
		
	fig.subplots_adjust(hspace=0.5, wspace=0.5, left = .05, bottom = .1, right = .95)


def hebbPaperNew():
	grid_cells = 0
	border_cells = 0
	lec_cells = 1100
	out_cells = 2500
	dg_cells = 12000
	res =1600*2
	input_cells = grid_cells + lec_cells
	sparsity_ca3 =.032
	noise_points = 19
	
	cells = dict(Ec = input_cells, Ca1 = 4200, Ca3 = out_cells, Dg = dg_cells)
	number_winner = dict(Ec = int(input_cells*0.35), Ca1 = int(4200*0.097), Ca3 = int(out_cells*sparsity_ca3), Dg = int(dg_cells * 0.005))
	connectivity = dict(Ec_Dg = 0.32, Dg_Ca3 = 0.0006, Ec_Ca3 =0.32, Ca3_Ca3 = 0.24, Ca3_Ca1 = 0.32, Ca1_Ec = 0.32, Ec_Ca1 = 0.32)
	learnrate = dict(Ec_Dg = 0, Dg_Ca3 = None, Ca3_Ec = 1, Ec_Ca3 =1, Ca3_Ca3=1, Ca3_Ca1 = 0.5, Ec_Ca1 = 1, Ca1_Ec = 1, Ca1_Sub = 1, Sub_Ec = 1, Ec_Sub = 0)
	actFunctionsRegions = dict(Ec_Dg = Network.getOutputWTALinear, Dg_Ca3 = Network.getOutputWTALinear, Ca3_Ec = Network.getOutputWTALinear, Ec_Ca3 = Network.getOutputWTALinear, Ca3_Ca3 = AutoAssociation.getOutputWTA, Ca3_Ca1= Network.getOutputWTALinear, Ca1_Ec = Network.getOutputWTALinear, Ec_Ca1 = Network.getOutputWTALinear)
	
	
	foo = open( './sizes.csv', "rb" ) #sizes Mizuseke et al
	real_sizes = np.loadtxt(foo, delimiter = ',', dtype = np.float64)
	real_sizes[:,1] *= 10000
	real_size_dis = []
	for s in np.array(real_sizes, 'int'):
		for i in range(s[1]):
			real_size_dis.append(s[0])
	
	
	#foo = open( '/home/torsten/Documents/sizes.csv', "rb" )
	#real_sizes = np.loadtxt(foo, delimiter = ',', dtype = np.float64)
	
	#real_sizes[:,1] *= 10000
	#real_size_dis2 = []
	#for s in np.array(real_sizes, 'int'):
		#for i in range(s[1]):
			#real_size_dis2.append(s[0])
	#fig =plt.figure()
	#ax = fig.add_subplot(111)
	#ax.hist([real_size_dis2, real_size_dis], label = ['me', 'Amir'], normed = 1, bins = 15, histtype = 'step' )
	#plt.legend()
	#plt.show()
	
		
	foo = open( './spatial_info_ca3.csv', "rb" ) # spatial info hargraeves et al
	real_info_ca3 = np.loadtxt(foo, delimiter = ',', dtype = np.float64)
	real_info_ca3[:,1] *= 10000
	real_info_ca3_dis = []
	for s in np.array(real_info_ca3, 'int'):
		for i in range(s[1]):
			real_info_ca3_dis.append(s[0])
					
	real_info = [0.1] *45 + [0.3]*15 +[0.5]*4 +[0.7]*1 +[.9]	#hargreaves

	Ca3Activation = None
	bins = np.linspace(0,np.sqrt(3), 50)
	actFunction = Input.getOutputId
	

	min_size = 200 #min pf size
	max_size = None
	min_rate = 0.2 # min prop. of max rate
	si_criterion = False
	colors = ['b', 'r', 'c', 'm', 'k']
	
	proportions = np.round(np.array([0, 1,1.25,2,3,4,5,6])/6., 2) #grid proportions
	sizes_to_plot = np.round(np.array([0, 1,2,3,6])/6.,2) #proportions shown in all figures
	
	
	#sizes_to_plot = np.array([0,0.125, .25, .5, 1])
	#sizes_to_plot = np.array([0,.125, 1])
	#proportions = np.round(np.array([0, 1,6])/6., 2)
	#sizes_to_plot = np.round(np.array([0, 1,6])/6.,2)
	
	props_to_plot = [.17, None]
	labels = map(str, proportions)
	kernel_size = 6 #default kernel for wsm cells

	
	kernel_sizes = np.array([1, 2, 4, 6, 8, 10, 12, 14,16]) # different kernel sizes for wsm cells
	kernels_to_plot= np.array([1, 4, 6, 8, 12]) #used in all figures
	
	
	k = 0
	fig = plt.figure(figsize = [7.5, 8.75])
	ax_ind = [4,2]
	fig2 = plt.figure(figsize = [7.5, 6])
	ax_ind2 = [2,3]

	sizes = []
	numbers = np.zeros(proportions.shape[0])
	active = np.zeros(proportions.shape[0])
	place_cells = np.zeros(proportions.shape[0])
	av_sizes = np.zeros(proportions.shape[0])
	robustness = np.zeros([proportions.shape[0],noise_points])
	robustness_cor = np.zeros([proportions.shape[0],noise_points])
	info_ca3 = np.zeros([proportions.shape[0],out_cells])
	
	
	numbers_kernel = np.zeros(kernel_sizes.shape[0])
	av_sizes_kernel =  np.zeros(kernel_sizes.shape[0])
	active_kernel = np.zeros(kernel_sizes.shape[0])
	place_cells_kernel =  np.zeros(kernel_sizes.shape[0])

	

	for p, prop in enumerate(proportions):
		g_cells = int(input_cells * prop)#no grid cells
		l_cells = input_cells -g_cells#no wsm cells
		if prop == props_to_plot[0] or prop == 0:
			noise_levels = np.array(np.linspace(0,input_cells-1, noise_points), 'int')
		else:
			noise_levels = [0]
		peak = np.random.normal(loc = 1, scale = .1, size = (1,  g_cells, 50,50))#peak rates of grid cells
		In = JointInput(grid_cells = g_cells, lec_cells = l_cells, border_cells = border_cells, inputMethod = Lec.makeActiveFilter, noiseMethod = Input.makeNoiseZero, actFunction = actFunction, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = .99, size_kernel = kernel_size, peak = peak)
		
		hippo = HippocampusFull(In = In, cells = cells, number_winner = number_winner, just_ca3 = 1, rec = 0 ,connectivity=connectivity,learnrate=learnrate, actFunctionsRegions = actFunctionsRegions, Ca3Activation = Ca3Activation)
		SpatialSim = Spatial(patterns = np.copy(hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0]), cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
		s = SpatialSim.getfieldSizes()
		sizes.append(s[s > 0])
		numbers[p] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
		active[p] = SpatialSim.getNumberActiveCells()
		place_cells[p] = SpatialSim.getNumberCellsWithField()
		av_sizes[p] = SpatialSim.getAverageFieldSize()
		if prop == props_to_plot[0]:
			numbers_std = SpatialSim.getAverageFieldNumberActiveCellsWithFieldStd()
			sizes_std = SpatialSim.getAverageFieldSizeStd()
			
		if prop == props_to_plot[0]: #cell examples
			grid = ImageGrid(fig, [ax_ind[0], ax_ind[1],1], nrows_ncols = (3,2),axes_pad=0.1, aspect = 1 ,share_all= 1, cbar_mode = 'none',cbar_pad = 0)
			cells_to_plot = SpatialSim.getPlaceCell(4)
			for j in range(3):
				#grid[k].set_title(str(np.round(info_ca3[prop == proportions][0][cells_to_plot[j]], 3)))
				[a,s] = plotCell(In = In, patterns = hippo.Ec_Ca3.Cor['StoredStored'].patterns_2, env =0, fig = fig, ax = grid[2*j], cb = 0, cell = cells_to_plot[j], zeros = 0)
				grid[2*j].set_xlim(0, In.cage[0])
				grid[2*j].set_ylim(0, In.cage[1])
				grid[2*j].yaxis.set_visible(False)
				grid[2*j].xaxis.set_visible(False)
			grid[0].set_ylabel('1m')
			grid[0].set_xlabel('1m')
			makeLabel(ax = grid[0], label = 'A', sci = 0 )
			for j in range(3):
				#grid[2*j+1].set_title(str(np.round(info_ca3[prop == proportions][0][cells_to_plot[j]], 3)))
				[a,s] = plotCell(In = In, patterns = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2, env =0, fig = fig, ax = grid[2*j+1], cb = 0, cell = cells_to_plot[j], zeros = 0)
				grid[2*j+1].set_xlim(0, In.cage[0])
				grid[2*j+1].set_ylim(0, In.cage[1])
				grid[2*j+1].yaxis.set_visible(False)
				grid[2*j+1].xaxis.set_visible(False)
				#grid.cbar_axes[2*j+1].colorbar(s)
			
			
			#lesion studies
			lesions = list(np.linspace(0,1,noise_points))#lesion prop of mec or lec or grid	
			lesion_mec = np.tile(In.patterns, (1,len(lesions),1,1))
			lesion_lec = np.tile(In.patterns, (1,len(lesions),1,1))
			lesion_grid = np.tile(In.patterns, (1,len(lesions),1,1))
			lesion_all = np.tile(In.patterns, (1,len(lesions),1,1))
			wrong_mec = np.zeros(len(lesions))
			wrong_lec = np.zeros(len(lesions))
			wrong_grid = np.zeros(len(lesions))
			wrong_all = np.zeros(len(lesions))
			

			for i, lesion in enumerate(lesions):
				corrupted_cells_mec = np.array(random.sample(range(input_cells/2), int(input_cells/2 * lesion)), 'int')
				corrupted_cells_lec = corrupted_cells_mec + input_cells/2
				corrupted_cells_grid = np.array(random.sample(range(g_cells), int(min(g_cells * lesion, g_cells))), 'int')
				corrupted_cells_all = np.array(random.sample(range(input_cells), int(input_cells * lesion)), 'int')
				lesion_mec[0,i][:,corrupted_cells_mec] = 0
				lesion_lec[0,i][:,corrupted_cells_lec] = 0
				lesion_grid[0,i][:,corrupted_cells_grid] = 0
				lesion_all[0,i][:,corrupted_cells_all] = 0
			
			activation_mec = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_mec)
			activation_lec = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_lec)
			activation_grid = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_grid)
			activation_all = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_all)
			
			wrong_mec = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_mec[0])
			wrong_lec = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_lec[0])
			wrong_grid = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_grid[0])
			wrong_all = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_all[0])
		
			lession_cells = np.array(lesions)* input_cells/2.
			lession_cells_grid = np.array(lesions)* input_cells * prop
			lession_cells_all = np.array(lesions)* input_cells
			normalize(In.patterns[0])

			### cell examples in lesion studies
			cells_to_plot = SpatialSim.getPlaceCell(20)
			fig3 = plt.figure()
			k = 1
			for i in range(5):
				[a,s] = plotCell(In = In, patterns = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2, env =0, fig = fig3, ax_index = [5,4,k], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('No')
				[a,s] =plotCell(In = In, patterns = activation_lec[:,-1], env =0,fig = fig3, ax_index = [5,4,k+1], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('lec')
				[a,s] =plotCell(In = In, patterns = activation_mec[:,-1], env =0,fig = fig3, ax_index = [5,4,k+2], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('mec')
				[a,s] =plotCell(In = In, patterns = activation_grid[:,-1], env =0,fig = fig3, ax_index = [5,4,k+3], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('grid')
				k+=4
			
			fig4 = plt.figure()
			k = 1
			cells_to_plot = SpatialSim.getPlaceCell(20)
			for i in range(5):
				[a,s] = plotCell(In = In, patterns = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2, env =0, fig = fig4, ax_index = [5,4,k], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('No')
				[a,s] =plotCell(In = In, patterns = activation_lec[:,-1], env =0,fig = fig4, ax_index = [5,4,k+1], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('lec')
				[a,s] =plotCell(In = In, patterns = activation_mec[:,-1], env =0,fig = fig4, ax_index = [5,4,k+2], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('mec')
				[a,s] =plotCell(In = In, patterns = activation_grid[:,-1], env =0,fig = fig4, ax_index = [5,4,k+3], cb = 0, cell = cells_to_plot[i], zeros = 0)
				a.yaxis.set_visible(False)
				a.xaxis.set_visible(False)
				if i==1:
					a.set_title('grid')
				k+=4

			GridLesion = Spatial(patterns = activation_grid[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			MecLesion = Spatial(patterns = activation_mec[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			LecLesion =  Spatial(patterns = activation_lec[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			
			number_fields_lesion_grid = GridLesion.getAverageFieldNumberActiveCellsWithField()
			number_fields_lesion_lec =LecLesion.getAverageFieldNumberActiveCellsWithField()
			number_fields_lesion_mec =MecLesion.getAverageFieldNumberActiveCellsWithField()
			number_fields_lesion_grid_std= GridLesion.getAverageFieldNumberActiveCellsWithFieldStd()
			number_fields_lesion_lec_std =LecLesion.getAverageFieldNumberActiveCellsWithFieldStd()
			number_fields_lesion_mec_std =MecLesion.getAverageFieldNumberActiveCellsWithFieldStd()
			
			av_size_grid = GridLesion.getAverageFieldSize()
			av_size_lec = LecLesion.getAverageFieldSize()
			av_size_mec = MecLesion.getAverageFieldSize()
			av_size_grid_std = GridLesion.getAverageFieldSizeStd()
			av_size_lec_std = LecLesion.getAverageFieldSizeStd()
			av_size_mec_std = MecLesion.getAverageFieldSizeStd()
			
			av_active_grid = GridLesion.getNumberActiveCells()
			av_active_lec = LecLesion.getNumberActiveCells()
			av_active_mec = MecLesion.getNumberActiveCells()
		
			av_place_cells_grid = GridLesion.getNumberCellsWithField()
			av_place_cells_lec = LecLesion.getNumberCellsWithField()
			av_place_cells_mec = MecLesion.getNumberCellsWithField()
		
		if prop == 0:
			lesions = list(np.linspace(0,1,noise_points))#lesion prop of mec or lec or grid	
			lesion_no_grid = np.tile(In.patterns, (1,len(lesions),1,1))
			wrong_no_grid = np.zeros(len(lesions))
			for i, lesion in enumerate(lesions):
				corrupted_cells_no_grid = np.array(random.sample(range(input_cells), int(input_cells * lesion)), 'int')
				lesion_no_grid[0,i][:,corrupted_cells_no_grid] = 0

			activation_no_grid = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_no_grid)
			wrong_no_grid = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_no_grid[0])
			lession_cells_no_grid = np.array(lesions)* input_cells
			normalize(In.patterns[0])

	
		info = np.array(In.getAverageCorOverDistance(locations = In.getStoredlocations(), bins = bins))
		bins_info = info[1]
		if prop == 0:
			infos = np.zeros([proportions.shape[0],info[1].shape[0]])
			infos_kernel = np.zeros([kernel_sizes.shape[0],info[1].shape[0]])
		infos[p] = info[0]
		#infos[kernel_size == kernel_sizes][0] = info[0]

			


	#kernel sizes
	for p, size in enumerate(kernel_sizes):
		prop = props_to_plot[0]
		if size != kernel_size:
			g_cells = int(input_cells * prop)
			l_cells = input_cells -g_cells
			noise_levels = [0]
			#peak = np.random.uniform(.5, 1.5, size = (1,  g_cells, 50,50))
			peak = np.random.normal(loc = 1, scale = .1, size = (1,  g_cells, 50,50))
			In = JointInput(grid_cells = g_cells, lec_cells = l_cells, border_cells = border_cells, inputMethod = Lec.makeActiveFilter, noiseMethod = Input.makeNoiseZero, actFunction = actFunction, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = .99, size_kernel = size, peak = peak)
			
			hippo = HippocampusFull(In = In, cells = cells, number_winner = number_winner, just_ca3 = 1, rec = 0 ,connectivity=connectivity,learnrate=learnrate, actFunctionsRegions = actFunctionsRegions, Ca3Activation = Ca3Activation)
			SpatialSim = Spatial(patterns = np.copy(hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0]), cage = In.cage, min_size = min_size, min_rate = min_rate, si_criterion = si_criterion, max_size = max_size)
			
			numbers_kernel[size == kernel_sizes] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
			av_sizes_kernel[size == kernel_sizes] = SpatialSim.getAverageFieldSize()
			active_kernel[size == kernel_sizes]  = SpatialSim.getNumberActiveCells()
			place_cells_kernel[size == kernel_sizes] = SpatialSim.getNumberCellsWithField()
			#normalize(In.patterns[0])
			print 'shape in patterns'
			#Cor = Corelations(patterns_1 = In.patterns[0], patterns_2 = In.patterns)
			#info = np.array(Cor.getAverageOverlapOverDistance(locations = In.getStoredlocations(), bins = bins))
			info = np.array(In.getAverageCorOverDistance(locations = In.getStoredlocations(), bins = bins))
			infos_kernel[size == kernel_sizes] = copy.copy(info)[0]
			print 'size_____________________- != 6', size
		else:
			print 'size_________________________ = 6?????', size
			#print prop == proportions
			#print size == kernel_sizes
			##print kernel_sizes
			numbers_kernel[size == kernel_sizes] = numbers[prop == proportions]
			av_sizes_kernel[size == kernel_sizes] = av_sizes[prop == proportions]
			active_kernel[size == kernel_sizes]  = active[prop == proportions]
			place_cells_kernel[size == kernel_sizes] = place_cells[prop == proportions]
			infos_kernel[size == kernel_sizes] = infos[prop == proportions]
			print infos_kernel[6 == kernel_sizes] == infos[1]
			##a = o
	mean  = np.array(map(np.mean, sizes))
	max_numbers = np.max([np.max(numbers),np.max(numbers_kernel)])
	max_sizes = np.max([np.max(mean),np.max(av_sizes_kernel)])
	
	####figure 1
	##histogramm sizes
	sizes_cdf = copy.copy(sizes)
	control = list(np.copy(proportions))
	stay = 1
	for p in proportions[::-1]:
		if p not in sizes_to_plot:
			sizes_cdf.pop(-stay)
			control.pop(-stay)
		else:
			stay +=1
	prop_to_plot_ind = np.flatnonzero(props_to_plot[0]==sizes_to_plot) 
	sizes_prop = sizes_cdf.pop(prop_to_plot_ind)
	labels = map(str, sizes_to_plot)
	labels.pop(prop_to_plot_ind)
	ax4 = fig.add_subplot(ax_ind[0], ax_ind[1], 2)
	n,bins,patches  = ax4.hist(sizes_cdf, label = labels, histtype = 'step', bins = 200, normed =1, cumulative = 1, color = colors[:sizes_to_plot.shape[0]-1])
	for p in patches:
		p[0].set_xy(p[0].get_xy()[:-1])
	n,bins,patches  = ax4.hist(real_size_dis, label = ['Mizuseki\n et al.'], bins = 200, normed =1, cumulative = 1,histtype = 'step' ,ec = 'k', ls = 'dashed')
	patches[0].set_xy(patches[0].get_xy()[:-1])
	n,bins,patches  = ax4.hist(sizes_prop, label = [str(props_to_plot[0])], bins = 200, normed =1, cumulative = 1,histtype = 'step' ,ec = 'g', lw = 2)
	patches[0].set_xy(patches[0].get_xy()[:-1])
	ax4.xaxis.set_ticks([0, 2000, 4000, 6000, 8000])
	ax4.set_ylim(0, 1.01)
	ax4.set_xlim(0, 8010)
	ax4.set_xlabel('place field size ' +r'$(cm^2)$')
	ax4.legend(loc = 'lower right',prop={'size':10})
	ax4.set_ylabel('cdf')
	patches[0].set_xy(patches[0].get_xy()[:-1])
	makeLabel(ax = ax4, label = 'B', sci = 0 )	

	#Cors Input Proportions
	ax = fig.add_subplot(ax_ind[0], ax_ind[1], 3)
	c = 0
	print infos_kernel[6 == kernel_sizes] == infos[1]
	for k in sizes_to_plot:
		if k == props_to_plot[0]:
			ax.plot(bins_info ,infos[k == proportions][0], label = str(k), lw = 3, c = 'g')	
		else:
			ax.plot(bins_info ,infos[k == proportions][0], label = str(k), c = colors[c])
			c+=1
	print infos_kernel[6 == kernel_sizes] == infos[1]
	ax.set_ylabel(r'$Corr\left( \mathbf{p}(\mathbf{r}_i), \mathbf{p}(\mathbf{r}_j)\right) $')
	ax.set_xlabel(r'$||\mathbf{r}_i - \mathbf{r}_j||$')
	y_min = ax.get_ylim()[0]
	ax.set_ylim(y_min,1.01)
	ax.legend(loc = 'upper right', prop={'size':10})
	for tick in ax.xaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	for tick in ax.yaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax, label = 'C', sci = 0 )
	
	
	##cors input kernels
	ax = fig.add_subplot(ax_ind[0], ax_ind[1], 4)
	c =0
	for k in kernels_to_plot:
		if k == kernel_size:
			ax.plot(bins_info ,infos_kernel[k == kernel_sizes][0], label = str(k), lw = 3, c = 'g')
		else:
			ax.plot(bins_info ,infos_kernel[k == kernel_sizes][0], label = str(k), c = colors[c])
			c+=1
	ax.set_ylabel(r'$Corr\left[ \mathbf{p}(\mathbf{r}_i), \mathbf{p}(\mathbf{r}_j)\right] $')
	ax.set_xlabel(r'$||\mathbf{r}_i - \mathbf{r}_j||$')
	ax.set_ylim(y_min,1.01)
	ax.legend(loc = 'upper right', prop={'size':10})
	for tick in ax.xaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	for tick in ax.yaxis.get_major_ticks()[::2]:
		tick.label1.set_visible(False)
	ax.yaxis.set_visible(False)
	makeLabel(ax = ax, label = 'F', sci = 0 )	
	
	
	#av size and numbers over props
	ax4 = fig.add_subplot(ax_ind[0], ax_ind[1], 5)
	ax4.plot(proportions, mean, label = 'mean size', marker = 'd', c = 'k')
	ax4.set_ylabel('pf size '+ r'$(cm^2)$')
	ax45 = ax4.twinx()
	ax45.set_ylabel('No.Fields')
	ax45.plot(proportions, numbers, label = 'mean no.fields', marker = 'd', c = 'g')
	ax4.legend(ax4.get_legend_handles_labels()[0] +ax45.get_legend_handles_labels()[0], ax4.get_legend_handles_labels()[1]+ax45.get_legend_handles_labels()[1], loc = 'upper left',prop={'size':10})
	ax4.set_xlabel('Proportion grid inputs')
	ax4.set_ylim(0, max_sizes)
	#ax4.xaxis.set_ticks(np.round(proportions[::2],2))
	ax4.xaxis.set_ticks([0, .17, .5, .83,1])
	ax45.set_ylim(0, max_numbers)
	ax45.yaxis.set_visible(False)
	makeLabel(ax = ax4, label = 'D', sci = 0 )	
	
	
	#av size and numbers over kernels
	ax4 = fig.add_subplot(ax_ind[0], ax_ind[1], 6)
	ax4.plot(kernel_sizes, av_sizes_kernel, label = 'av_size', marker = 'd', c = 'k')
	ax4.set_ylabel(r'$cm^2$')
	ax45 = ax4.twinx()
	ax45.set_ylabel('No. Fields')
	ax45.plot(kernel_sizes, numbers_kernel, label = 'av_number', marker = 'd', c = 'g')
	#ax4.legend(ax4.get_legend_handles_labels()[0] +ax45.get_legend_handles_labels()[0], ax4.get_legend_handles_labels()[1]+ax45.get_legend_handles_labels()[1], loc = 'best',prop={'size':10})
	ax4.set_xlabel(r'$\sigma_N$')
	ax4.set_ylim(0, max_sizes)
	ax45.set_ylim(0, max_numbers)
	ax4.yaxis.set_visible(False)
	ax45.yaxis.label.set_color('g')
	ax45.tick_params(axis='y', colors='g')
	makeLabel(ax = ax4, label = 'G', sci = 0 )	
	
	

	#cells active over props
	ax4 = fig.add_subplot(ax_ind[0], ax_ind[1], 7)
	ax4.plot(proportions, active, label = 'active', marker = 'd', c = 'b')
	ax4.plot(proportions, place_cells, label = 'place_cells', marker = 'd', c = 'r')
	ax4.legend(loc = 'best', prop={'size':10})
	ax4.set_ylabel('No. cells')
	ax4.set_ylim(0, out_cells)
	ax4.set_xlabel('Proportion grid inputs')
	ax4.xaxis.set_ticks(np.round(proportions[::2],2))
	makeLabel(ax = ax4, label = 'E', sci = 0 )	

	#cells active over kernels
	ax4 = fig.add_subplot(ax_ind[0], ax_ind[1], 8)
	ax4.plot(kernel_sizes, active_kernel, label = 'active', marker = 'd', c = 'b')
	ax4.plot(kernel_sizes, place_cells_kernel, label = 'place_cells', marker = 'd', c = 'r')
	#ax4.legend(loc = 'best', prop={'size':10})
	ax4.set_xlabel(r'$\sigma_N$')
	ax4.set_ylabel('No. cells')
	ax4.set_ylim(0, out_cells)
	ax4.yaxis.set_visible(False)
	makeLabel(ax = ax4, label = 'H', sci = 0 )	

	
	
	#figure 2 Noise Figure
	ax = fig2.add_subplot(2, 2, 2)
	ax.set_ylim(0, .5)
	ax.set_xlabel('no. lesioned cells')
	ax.set_ylabel('error rate ' +r'$\epsilon$')
	ax.plot(lession_cells, wrong_lec, label = 'LEC lesions', c = 'b')
	ax.plot(lession_cells, wrong_mec, label = 'MEC lesions', c = 'r')
	ax.plot(lession_cells_grid, wrong_grid, label = 'Grid lesions', c = 'y')
	ax.plot(lession_cells_all, wrong_all, label = 'EC lesions', c= 'g')
	ax.plot(lession_cells_no_grid, wrong_no_grid, label = 'EC lesions \nprop. grid input = 0', c= 'g', ls = '--')
	ax.plot([0,input_cells], [0, .5], c = 'k', ls = '--' )
	ax.legend(loc = 'best', prop={'size':10}, handlelength = 2.)
	ax.set_xlim(0,1101)
	for tick in ax.xaxis.get_major_ticks()[1:-1:2]:
		tick.set_visible(False)
	makeLabel(ax = ax, label = 'A', sci = 0 )
	
	
	fig.tight_layout(rect = (0,0,1,.97))
	fig2.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.9, hspace = 0.6, wspace = 0.6)


def hebbPaperLesionsErr():
	
	fig2 = plt.figure(figsize = [7.5, 6])
	#fig.subplots_adjust(wspace=0.6, hspace=.4, left = .1, right = .95)
	ax_ind2 = [2,3]
	
	
	grid_cells = 0
	border_cells = 0
	lec_cells = 1100
	out_cells = 2500
	dg_cells = 12000
	res =1600*2
	input_cells = grid_cells + lec_cells
	sparsity_ca3 =.032
	noise_points = 2
	
	cells = dict(Ec = input_cells, Ca1 = 4200, Ca3 = out_cells, Dg = dg_cells)
	number_winner = dict(Ec = int(input_cells*0.35), Ca1 = int(4200*0.097), Ca3 = int(out_cells*sparsity_ca3), Dg = int(dg_cells * 0.005))
	connectivity = dict(Ec_Dg = 0.32, Dg_Ca3 = 0.0006, Ec_Ca3 =0.32, Ca3_Ca3 = 0.24, Ca3_Ca1 = 0.32, Ca1_Ec = 0.32, Ec_Ca1 = 0.32)
	learnrate = dict(Ec_Dg = 0, Dg_Ca3 = None, Ca3_Ec = 1, Ec_Ca3 =1, Ca3_Ca3=1, Ca3_Ca1 = 0.5, Ec_Ca1 = 1, Ca1_Ec = 1, Ca1_Sub = 1, Sub_Ec = 1, Ec_Sub = 0)
	actFunctionsRegions = dict(Ec_Dg = Network.getOutputWTALinear, Dg_Ca3 = Network.getOutputWTALinear, Ca3_Ec = Network.getOutputWTALinear, Ec_Ca3 = Network.getOutputWTALinear, Ca3_Ca3 = AutoAssociation.getOutputWTA, Ca3_Ca1= Network.getOutputWTALinear, Ca1_Ec = Network.getOutputWTALinear, Ec_Ca1 = Network.getOutputWTALinear)
	
	Ca3Activation = None
	bins = np.linspace(0,np.sqrt(3), 50)
	actFunction = Input.getOutputId
	

	min_size = 200
	max_size = None
	min_rate = 0.2
	si_criterion = False
	
	proportions = np.round(np.array([1])/6., 2)
	sizes_to_plot = np.round(np.array([1])/6.,2)
	
	props_to_plot = [.17, None]
	labels = map(str, proportions)
	kernel_size = 6
	kernel_sizes = np.array([6])
	kernels_to_plot= np.array([6])
	k = 0
	iterations = 10
	

	number_active =	np.zeros(iterations)
	number_active_lec = np.zeros(iterations)
	number_active_mec = np.zeros(iterations)
	number_active_grid =np.zeros(iterations)
		
	size = 	np.zeros(iterations)
	size_lec = 	np.zeros(iterations)
	size_mec = 	np.zeros(iterations)
	size_grid = 	np.zeros(iterations)
		
	act = np.zeros(iterations)
	act_lec = np.zeros(iterations)
	act_mec = np.zeros(iterations)
	act_grid =np.zeros(iterations)
	
	
	for z in range(iterations):
		sizes = []
		numbers = np.zeros(proportions.shape[0])
		active = np.zeros(proportions.shape[0])
		place_cells = np.zeros(proportions.shape[0])
		av_sizes = np.zeros(proportions.shape[0])
		robustness = np.zeros([proportions.shape[0],noise_points])
		robustness_cor = np.zeros([proportions.shape[0],noise_points])
		info_ca3 = np.zeros([proportions.shape[0],out_cells])
		
		
		numbers_kernel = np.zeros(kernel_sizes.shape[0])
		av_sizes_kernel =  np.zeros(kernel_sizes.shape[0])
		active_kernel = np.zeros(kernel_sizes.shape[0])
		place_cells_kernel =  np.zeros(kernel_sizes.shape[0])
	
		
	
		for p, prop in enumerate(proportions):
			g_cells = int(input_cells * prop)
			l_cells = input_cells -g_cells
			if prop == props_to_plot[0] or prop == 0:
				noise_levels = np.array(np.linspace(0,input_cells-1, noise_points), 'int')
			else:
				noise_levels = [0]
			
			print 'proportion = ', prop
			#peak = np.random.uniform(.5, 1.5, size = (1,  g_cells, 50,50))
			peak = np.random.normal(loc = 1, scale = .1, size = (1,  g_cells, 50,50))
			In = JointInput(grid_cells = g_cells, lec_cells = l_cells, border_cells = border_cells, inputMethod = Lec.makeActiveFilter, noiseMethod = Input.makeNoiseZero, actFunction = actFunction, number_patterns = res, number_to_store =res ,n_e =1,noise_levels = noise_levels, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = .99, size_kernel = kernel_size, peak = peak)
			
			hippo = HippocampusFull(In = In, cells = cells, number_winner = number_winner, just_ca3 = 1, rec = 0 ,connectivity=connectivity,learnrate=learnrate, actFunctionsRegions = actFunctionsRegions, Ca3Activation = Ca3Activation)
			SpatialSim = Spatial(patterns = np.copy(hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0]), cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			s = SpatialSim.getfieldSizes()
			sizes.append(s[s > 0])
			numbers[p] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
			active[p] = SpatialSim.getNumberActiveCells()
			place_cells[p] = SpatialSim.getNumberCellsWithField()
			av_sizes[p] = SpatialSim.getAverageFieldSize()
			if prop == props_to_plot[0]:
				numbers_std = SpatialSim.getAverageFieldNumberActiveCellsWithFieldStd()
				sizes_std = SpatialSim.getAverageFieldSizeStd()
				lesions = list(np.linspace(0,1,noise_points))#lesion prop of mec or lec or grid	
				lesion_mec = np.tile(In.patterns, (1,len(lesions),1,1))
				lesion_lec = np.tile(In.patterns, (1,len(lesions),1,1))
				lesion_grid = np.tile(In.patterns, (1,len(lesions),1,1))
				lesion_all = np.tile(In.patterns, (1,len(lesions),1,1))
				wrong_mec = np.zeros(len(lesions))
				wrong_lec = np.zeros(len(lesions))
				wrong_grid = np.zeros(len(lesions))
				wrong_all = np.zeros(len(lesions))
	
				for i, lesion in enumerate(lesions):
					corrupted_cells_mec = np.array(random.sample(range(input_cells/2), int(input_cells/2 * lesion)), 'int')
					corrupted_cells_lec = corrupted_cells_mec + input_cells/2
					corrupted_cells_grid = np.array(random.sample(range(g_cells), int(min(g_cells * lesion, g_cells))), 'int')
					corrupted_cells_all = np.array(random.sample(range(input_cells), int(input_cells * lesion)), 'int')
					lesion_mec[0,i][:,corrupted_cells_mec] = 0
					lesion_lec[0,i][:,corrupted_cells_lec] = 0
					lesion_grid[0,i][:,corrupted_cells_grid] = 0
					lesion_all[0,i][:,corrupted_cells_all] = 0
				
				activation_mec = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_mec)
				#CorMec = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_mec)
				#InputMec = Corelations(patterns_1 = In.patterns, patterns_2 = lesion_mec)
			
				activation_lec = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_lec)
				#CorLec = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_lec)
				#InputLec = Corelations(patterns_1 = In.patterns, patterns_2 = lesion_lec)
				
				activation_grid = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_grid)
				#CorGrid = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_grid)
				#InputGrid = Corelations(patterns_1 = In.patterns, patterns_2 = lesion_grid)
				activation_all = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_all)
				
				wrong_mec = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_mec[0])
				wrong_lec = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_lec[0])
				wrong_grid = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_grid[0])
				wrong_all = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_all[0])
				#wrong_all = SpatialSim.getAverageProportionWrong(noisy_patterns = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2)
			
				lession_cells = np.array(lesions)* input_cells/2.
				lession_cells_grid = np.array(lesions)* input_cells * prop
				lession_cells_all = np.array(lesions)* input_cells
				normalize(In.patterns[0])
	
				GridLesion = Spatial(patterns = activation_grid[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
				MecLesion = Spatial(patterns = activation_mec[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
				LecLesion =  Spatial(patterns = activation_lec[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
				
				number_fields_lesion_grid = GridLesion.getAverageFieldNumberActiveCellsWithField()
				number_fields_lesion_lec =LecLesion.getAverageFieldNumberActiveCellsWithField()
				number_fields_lesion_mec =MecLesion.getAverageFieldNumberActiveCellsWithField()
				number_fields_lesion_grid_std= GridLesion.getAverageFieldNumberActiveCellsWithFieldStd()
				number_fields_lesion_lec_std =LecLesion.getAverageFieldNumberActiveCellsWithFieldStd()
				number_fields_lesion_mec_std =MecLesion.getAverageFieldNumberActiveCellsWithFieldStd()
				
				av_size_grid = GridLesion.getAverageFieldSize()
				av_size_lec = LecLesion.getAverageFieldSize()
				av_size_mec = MecLesion.getAverageFieldSize()
				av_size_grid_std = GridLesion.getAverageFieldSizeStd()
				av_size_lec_std = LecLesion.getAverageFieldSizeStd()
				av_size_mec_std = MecLesion.getAverageFieldSizeStd()
				
				av_active_grid = GridLesion.getNumberActiveCells()
				av_active_lec = LecLesion.getNumberActiveCells()
				av_active_mec = MecLesion.getNumberActiveCells()
			
				av_place_cells_grid = GridLesion.getNumberCellsWithField()
				av_place_cells_lec = LecLesion.getNumberCellsWithField()
				av_place_cells_mec = MecLesion.getNumberCellsWithField()
			
			if prop == 0:
				lesions = list(np.linspace(0,1,noise_points))#lesion prop of mec or lec or grid	
				lesion_no_grid = np.tile(In.patterns, (1,len(lesions),1,1))
				wrong_no_grid = np.zeros(len(lesions))
				for i, lesion in enumerate(lesions):
					corrupted_cells_no_grid = np.array(random.sample(range(input_cells), int(input_cells * lesion)), 'int')
					lesion_no_grid[0,i][:,corrupted_cells_no_grid] = 0
	
				activation_no_grid = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_no_grid)
				wrong_no_grid = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_no_grid[0])
				lession_cells_no_grid = np.array(lesions)* input_cells
				normalize(In.patterns[0])
	
	
		
		mean  = np.array(map(np.mean, sizes))
		max_numbers = np.max([np.max(numbers),np.max(numbers_kernel)])
		max_sizes = np.max([np.max(mean),np.max(av_sizes_kernel)])
		
		
		
		number_active[z] = numbers[props_to_plot[0] == proportions]
		number_active_lec[z] = number_fields_lesion_lec
		number_active_mec[z] = number_fields_lesion_mec
		number_active_grid[z] = number_fields_lesion_grid
		
		
		size[z] = mean[props_to_plot[0] == proportions]/10000.
		size_lec[z] =  av_size_lec/10000.
		size_mec[z] = av_size_mec/10000.
		size_grid[z] = av_size_grid/10000.
		
		act[z] = active[props_to_plot[0] == proportions]
		act_lec[z] =  av_active_lec
		act_mec[z] =  av_active_mec
		act_grid[z] =  av_active_grid
	
	
	
	
	ax4 = fig2.add_subplot(ax_ind2[0], ax_ind2[1], 4)
	ax4.bar(left = [1,1.5,2,2.5], width = .5, height = [np.mean(number_active), np.mean(number_active_lec),np.mean(number_active_mec),np.mean(number_active_grid)], yerr = [np.std(number_active), np.std(number_active_lec),np.std(number_active_mec),np.std(number_active_grid)], align = 'center', color = ['k', 'b', 'r','y'], ecolor = 'k')
	#ax4.xaxis.set_visible(False)
	#ax4.legend(loc = 'best', prop={'size':10})
	ax4.set_ylabel('no. fields')
	ax4.set_xticks([1,1.5,2,2.5])
	ax4.set_xticklabels(['No', 'LEC', 'MEC','Grid'])
	makeLabel(ax = ax4, label = 'C', sci = 0 )	
	
	
	
	
	ax4 = fig2.add_subplot(ax_ind2[0], ax_ind2[1], 5)
	ax4.bar(left = [1,1.5,2,2.5], width = .5, height = [np.mean(size), np.mean(size_lec),np.mean(size_mec),np.mean(size_grid)], yerr = [np.std(size), np.std(size_lec),np.std(size_mec),np.std(size_grid)], align = 'center', color = ['k', 'b', 'r', 'y'], ecolor = 'k')
	#ax4.xaxis.set_visible(False)
	#ax4.legend(loc = 'best', prop={'size':10})
	ax4.set_ylabel('pf size '+ r'$(m^2)$')
	ax4.set_xticks([1,1.5,2,2.5])
	ax4.set_xticklabels(['No', 'LEC', 'MEC','Grid'])
	makeLabel(ax = ax4, label = 'D', sci = 0 )	
	
	
	ax4 = fig2.add_subplot(ax_ind2[0], ax_ind2[1], 6)
	ax4.bar(left = [1,1.5,2,2.5], width = .5, height = [np.mean(act), np.mean(act_lec),np.mean(act_mec),np.mean(act_grid)], yerr = [np.std(act), np.std(act_lec),np.std(act_mec),np.std(act_grid)], align = 'center', color = ['k', 'b', 'r', 'y'])
	#ax4.xaxis.set_visible(False)
	#ax4.legend(loc = 'best', prop={'size':10})
	ax4.set_ylabel('no. active cells')
	ax4.set_xticks([1,1.5,2,2.5])
	ax4.set_xticklabels(['No', 'LEC', 'MEC','Grid'])
	makeLabel(ax = ax4, label = 'E', sci = 0 )

	fig2.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.9, hspace = 0.6, wspace = 0.6)



def hebbPaperStabilityErr(): #Lesions after learning
	
	fig2 = plt.figure()#(figsize = [6, 3])
	#fig.subplots_adjust(wspace=0.6, hspace=.4, left = .1, right = .95)
	ax_ind2 = [3,2]
	
	
	grid_cells = 0
	border_cells = 0
	lec_cells = 1100
	out_cells = 2500
	dg_cells = 12000
	res =1600*2
	input_cells = grid_cells + lec_cells
	sparsity_ca3 =.032
	noise_points = 2
	stab_points = np.array([0, .5, .75, 1, 1.5, 2, 3, 4, 5, 100])
	no_stab_points = stab_points.shape[0]
	#stability_noise = 1

	cells = dict(Ec = input_cells, Ca1 = 4200, Ca3 = out_cells, Dg = dg_cells)
	number_winner = dict(Ec = int(input_cells*0.35), Ca1 = int(4200*0.097), Ca3 = int(out_cells*sparsity_ca3), Dg = int(dg_cells * 0.005))
	connectivity = dict(Ec_Dg = 0.32, Dg_Ca3 = 0.0006, Ec_Ca3 =0.32, Ca3_Ca3 = 0.24, Ca3_Ca1 = 0.32, Ca1_Ec = 0.32, Ec_Ca1 = 0.32)
	learnrate = dict(Ec_Dg = 0, Dg_Ca3 = None, Ca3_Ec = 1, Ec_Ca3 =1, Ca3_Ca3=1, Ca3_Ca1 = 0.5, Ec_Ca1 = 1, Ca1_Ec = 1, Ca1_Sub = 1, Sub_Ec = 1, Ec_Sub = 0)
	actFunctionsRegions = dict(Ec_Dg = Network.getOutputWTALinear, Dg_Ca3 = Network.getOutputWTALinear, Ca3_Ec = Network.getOutputWTALinear, Ec_Ca3 = Network.getOutputWTALinear, Ca3_Ca3 = AutoAssociation.getOutputWTA, Ca3_Ca1= Network.getOutputWTALinear, Ca1_Ec = Network.getOutputWTALinear, Ec_Ca1 = Network.getOutputWTALinear)
	
	Ca3Activation = None
	actFunction = Input.getOutputId
		
	min_size = 200
	max_size = None
	min_rate = 0.2
	si_criterion = False
		
	prop =.17
	kernel_size = 6
	iterations = 1
	
	corr_lec = np.zeros([no_stab_points,iterations])
	reenterd_corr = np.zeros([no_stab_points,iterations])
	reenterd_corr_rand = np.zeros([no_stab_points,iterations])
	reenterd_corr_lec = np.zeros([no_stab_points,iterations])
	reenterd_corr_mec = np.zeros([no_stab_points,iterations])
	reenterd_corr_grid =np.zeros([no_stab_points,iterations])
	
	mean_number = np.zeros([no_stab_points,iterations])
	mean_sizes = np.zeros([no_stab_points,iterations])
	mean_number_mec = np.zeros([no_stab_points,iterations])
	mean_sizes_mec = np.zeros([no_stab_points,iterations])
	mean_number_lec = np.zeros([no_stab_points,iterations])
	mean_sizes_lec = np.zeros([no_stab_points,iterations])
	mean_number_grid = np.zeros([no_stab_points,iterations])
	mean_sizes_grid = np.zeros([no_stab_points,iterations])
	active =  np.zeros([no_stab_points,iterations])
	place_cells =  np.zeros([no_stab_points,iterations])
	active_mec =  np.zeros([no_stab_points,iterations])
	place_cells_mec =  np.zeros([no_stab_points,iterations])
	active_lec =  np.zeros([no_stab_points,iterations])
	place_cells_lec =  np.zeros([no_stab_points,iterations])
	active_grid =  np.zeros([no_stab_points,iterations])
	place_cells_grid =  np.zeros([no_stab_points,iterations])
	
	
	
	ax_count = 0
	for z in range(iterations):
		g_cells = int(input_cells * prop)
		l_cells = input_cells -g_cells
		mec_border = int(input_cells * .5)
		peak = np.random.normal(loc = 1, scale = .1, size = (1,  g_cells, 50,50))
		In = JointInput(grid_cells = g_cells, lec_cells = l_cells, border_cells = border_cells, inputMethod = Lec.makeActiveFilter, noiseMethod = JointInput.makeNoiseLEC, actFunction = actFunction, number_patterns = res, number_to_store =res ,n_e =1,noise_levels =stab_points, normed = 0, store_indizes = np.tile(np.arange(res), (1,1)), grid_mode = 'modules', rat=1, cage =[2,1],r_to_s = 0.32, spacings = None, sparsity = .99, size_kernel = kernel_size, peak = peak)
		#p.array(np.linspace(0,l_cells-1, stab_points), 'int')				
		InputLec = Corelations(patterns_1 = In.input_stored[0,:, mec_border:], patterns_2 = In.noisy_input_stored[0,:,:, mec_border:], env = 0)
		#InputMec = Corelations(patterns_1 = In.input_stored[0,:,:g_cells], patterns_2 = In.noisy_input_stored[0,:,:,:g_cells], env = 0)	
		for stability_noise in range(no_stab_points):
			
			hippo = HippocampusFull(In = In, cells = cells, number_winner = number_winner, just_ca3 = 1, rec = 0 ,connectivity=connectivity,learnrate=learnrate, actFunctionsRegions = actFunctionsRegions, Ca3Activation = Ca3Activation)
		
			#input if a nimal reenters env
			input_patterns = In.noisy_input_stored[:, stability_noise]
			lesions = list(np.linspace(0,1,noise_points))#lesion prop of mec or lec or grid	
			lesion_mec = np.tile(input_patterns, (1,len(lesions),1,1))
			lesion_lec = np.tile(input_patterns, (1,len(lesions),1,1))
			lesion_grid = np.tile(input_patterns, (1,len(lesions),1,1))
			for i, lesion in enumerate(lesions):
				corrupted_cells_mec = np.array(random.sample(range(input_cells/2), int(input_cells/2 * lesion)), 'int')
				corrupted_cells_lec = corrupted_cells_mec + input_cells/2
				corrupted_cells_grid = np.array(random.sample(range(g_cells), int(min(g_cells * lesion, g_cells))), 'int')
				corrupted_cells_all = np.array(random.sample(range(input_cells), int(input_cells * lesion)), 'int')
				lesion_mec[0,i][:,corrupted_cells_mec] = 0
				lesion_lec[0,i][:,corrupted_cells_lec] = 0
				lesion_grid[0,i][:,corrupted_cells_grid] = 0
			
			#print lesion_lec[0,-1]
			#print InLec.input_stored
			#print lesion_lec.shape
			#print InLec.input_stored.shape	
			#k = affe
			
			activation_no = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = In.noisy_input_stored)
			activation_mec = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_mec)
			#activation_random_input =  hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = np.random.normal(loc =10, scale =2, size = In.noisy_input_stored.shape))
			#activation_random_input = np.random.normal(loc =10, scale =2, size = activation_random_input.shape)
			activation_lec = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_lec)
			activation_grid = hippo.Ec_Ca3.getOutput(hippo.Ec_Ca3, input_pattern = lesion_grid)
	
			CorMec = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_mec[0], env = 0)
			CorNo = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_no[0], env = 0)
			#CorRand = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_random_input[0], env = 0)
			CorLec = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_lec[0], env = 0)
			CorGrid = Corelations(patterns_1 = hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0], patterns_2 = activation_grid[0], env = 0)

		
			#SpatialSim = Spatial(patterns = np.copy(hippo.Ec_Ca3.Cor['StoredRecalled'].patterns_2[0]), cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			#wrong_mec = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_mec[0])[-1]
			#wrong_grid = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_grid[0])[-1]
			#wrong_lec = SpatialSim.getAverageProportionWrong(noisy_patterns = activation_lec[0])[-1]
			
			
			corr_lec[stability_noise,z] =  InputLec.getOrigVsOrig()[stability_noise]
			#reenterd_corr_rand[stability_noise,z] = CorRand.getOrigVsOrig()[stability_noise]
			reenterd_corr[stability_noise,z] = CorNo.getOrigVsOrig()[stability_noise]
			reenterd_corr_grid[stability_noise,z] = CorGrid.getOrigVsOrig()[-1]
			reenterd_corr_lec[stability_noise,z] =  CorLec.getOrigVsOrig()[-1]
			reenterd_corr_mec[stability_noise,z] =  CorMec.getOrigVsOrig()[-1]
	
			SpatialSim = Spatial(patterns = activation_no[0, stability_noise], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			mean_number[stability_noise,z] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
			mean_sizes[stability_noise,z] = SpatialSim.getAverageFieldSize()
			active[stability_noise,z] = SpatialSim.getNumberActiveCells()
			place_cells[stability_noise,z] = SpatialSim.getNumberCellsWithField()
			
			SpatialSim = Spatial(patterns =  activation_mec[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			mean_number_mec[stability_noise,z] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
			mean_sizes_mec[stability_noise,z] = SpatialSim.getAverageFieldSize()
			active_mec[stability_noise,z] = SpatialSim.getNumberActiveCells()
			place_cells_mec[stability_noise,z] = SpatialSim.getNumberCellsWithField()
			
			
			SpatialSim = Spatial(patterns =  activation_lec[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			mean_number_lec[stability_noise,z] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
			mean_sizes_lec[stability_noise,z] = SpatialSim.getAverageFieldSize()
			active_lec[stability_noise,z] = SpatialSim.getNumberActiveCells()
			place_cells_lec[stability_noise,z] = SpatialSim.getNumberCellsWithField()
			
			
			SpatialSim = Spatial(patterns =  activation_grid[0,-1], cage = In.cage, min_size = min_size, min_rate = min_rate, max_size = max_size, si_criterion = si_criterion)
			mean_number_grid[stability_noise,z] = SpatialSim.getAverageFieldNumberActiveCellsWithField()
			mean_sizes_grid[stability_noise,z] = SpatialSim.getAverageFieldSize()
			active_grid[stability_noise,z] = SpatialSim.getNumberActiveCells()
			place_cells_grid[stability_noise,z] = SpatialSim.getNumberCellsWithField()
	
	
		#if ax_count in [1]:
			#ax4 = fig2.add_subplot(ax_ind2[0], ax_ind2[1], (ax_count)*2 +1)
			#ax4.plot(range(stab_points), InputLec.getOrigVsOrig(), marker = 'd', label = 'LEC only')
			#ax4.plot(stability_noise, InputLec.getOrigVsOrig()[stability_noise], marker = 'D', c = 'r', markersize = 12)
			#ax4.plot(range(stab_points), InputMec.getOrigVsOrig(), marker = 'd', label = 'Mec only')
			#ax4.plot(range(stab_points), In.getOrigVsOrig(), marker = 'd', label = 'whole')
			##ax4.legend()
			#ax4.set_ylim((0,1))
			
			#ax4 = fig2.add_subplot(ax_ind2[0], ax_ind2[1], (ax_count)*2 +1)
			#ax4.bar(left = [.5,1,1.5,2,2.5], width = .5, height = [np.mean(reenterd_corr_rand),np.mean(reenterd_corr), np.mean(wrong_lec),np.mean(wrong_mec),np.mean(wrong_grid)], yerr = [np.std(reenterd_corr_rand), np.std(reenterd_corr), np.std(reenterd_corr_lec),np.std(reenterd_corr_mec),np.std(reenterd_corr_grid)], align = 'center', color = ['k', 'b', 'r', 'y'], ecolor = 'k')
			#ax4.set_ylim(0,1)
			##ax4.xaxis.set_visible(False)
			##ax4.legend(loc = 'best', prop={'size':10})
			#ax4.set_ylabel('error')
			#ax4.set_xticks([1,1.5,2,2.5])
			#ax4.set_xticklabels(['No', 'LEC', 'MEC','Grid'])
			#makeLabel(ax = ax4, label = 'E', sci = 0 )
			
	#for stability_noise in range(3):
		#ax4 = fig2.add_subplot(ax_ind2[0], ax_ind2[1], (stability_noise)*2 +2)
		#ax4.bar(left = [-.5,1,1.5,2,2.5], width = .5, height = [np.mean(corr_lec[stability_noise]),np.mean(reenterd_corr[stability_noise]), np.mean(reenterd_corr_lec[stability_noise]),np.mean(reenterd_corr_mec[stability_noise]),np.mean(reenterd_corr_grid[stability_noise])], yerr = [np.std(corr_lec[stability_noise]), np.std(reenterd_corr[stability_noise]), np.std(reenterd_corr_lec[stability_noise]),np.std(reenterd_corr_mec[stability_noise]),np.std(reenterd_corr_grid[stability_noise])], align = 'center', color = ['m','k', 'b', 'r', 'y'], ecolor = 'k')
		#ax4.set_ylim(0,1)
	##ax4.xaxis.set_visible(False)
	##ax4.legend(loc = 'best', prop={'size':10})
		#ax4.set_ylabel('Stability')
		#ax4.set_xticks([-.5,1,1.5,2,2.5])
		#ax4.set_xticklabels(['LEC cells', 'No', 'LEC', 'MEC','Grid'])
		#makeLabel(ax = ax4, label = 'E', sci = 0 )
		
		
	stab_points = np.mean(corr_lec, -1)
	print stab_points
	print reenterd_corr
	print np.mean(reenterd_corr, -1)
		
		
	print stab_points.shape
	print np.mean(reenterd_corr, -1).shape
	print reenterd_corr.shape
		
		

		
	ax4 = fig2.add_subplot(2, 2, 1)
	#ax4.errorbar(stab_points, np.mean(corr_lec, -1), yerr = np.std(corr_lec , -1),color = 'm', ecolor = 'k', label = 'LEC cells')
	ax4.errorbar(stab_points, np.mean(reenterd_corr, -1), yerr = np.std(reenterd_corr , -1),color = 'k', ecolor = 'k')
	ax4.errorbar(stab_points, np.mean(reenterd_corr_lec, -1) ,yerr = np.std(reenterd_corr_lec , -1),color = 'b', ecolor = 'k')
	ax4.errorbar(stab_points, np.mean(reenterd_corr_mec, -1) ,yerr = np.std(reenterd_corr_mec , -1),color = 'r', ecolor = 'k')	
	ax4.errorbar(stab_points, np.mean(reenterd_corr_grid, -1), yerr = np.std(reenterd_corr_grid , -1),color = 'y', ecolor = 'k')	
	ax4.set_xlabel('Stability LEC')
	ax4.set_ylabel('Stability CA3')
	ax4.set_xlim(0,1)
	ax4.set_ylim(0,1)
	for tick in ax4.xaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	for tick in ax4.yaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax4, label = 'A', sci = 0 )	

	ax4 = fig2.add_subplot(222)
	#ax4.errorbar(stab_points, np.mean(corr_lec, -1), yerr = np.std(corr_lec , -1),color = 'm', ecolor = 'k', label = 'LEC cells')
	ax4.errorbar(stab_points, np.mean(mean_sizes, -1), yerr = np.std(mean_sizes , -1),color = 'k', ecolor = 'k')
	ax4.errorbar(stab_points, np.mean(mean_sizes_lec, -1) ,yerr = np.std(mean_sizes_lec , -1),color = 'b', ecolor = 'k')
	ax4.errorbar(stab_points, np.mean(mean_sizes_mec, -1) ,yerr = np.std(mean_sizes_mec , -1),color = 'r', ecolor = 'k')	
	ax4.errorbar(stab_points, np.mean(mean_sizes_grid, -1), yerr = np.std(mean_sizes_grid , -1),color = 'y', ecolor = 'k')
	ax4.set_xlabel('Stability LEC')
	ax4.set_ylabel('field size '+r'$(cm^2)$')
	ax4.set_xlim(0,1)
	ylim = ax4.get_ylim()
	ax4.set_ylim(0, ylim[1])
	for tick in ax4.xaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	for tick in ax4.yaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax4, label = 'B', sci = 0 )	
	
	ax45 = fig2.add_subplot(2, 2, 3)
	ax45.set_ylabel('No. Fields')
	ax45.set_xlabel('Stability LEC')
	ax45.set_xlim(0,1)
	ax45.errorbar(stab_points, np.mean(mean_number, -1), yerr = np.std(mean_number , -1),color = 'k', ecolor = 'k')
	ax45.errorbar(stab_points, np.mean(mean_number_lec, -1) ,yerr = np.std(mean_number_lec , -1),color = 'b', ecolor = 'k')
	ax45.errorbar(stab_points, np.mean(mean_number_mec, -1) ,yerr = np.std(mean_number_mec , -1),color = 'r', ecolor = 'k')	
	ax45.errorbar(stab_points, np.mean(mean_number_grid, -1), yerr = np.std(mean_number_grid , -1),color = 'y', ecolor = 'k')	
	ylim = ax45.get_ylim()
	ax45.set_ylim(0, ylim[1])
	for tick in ax45.xaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	for tick in ax45.yaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax45, label = 'C', sci = 0 )
	
	ax45 = fig2.add_subplot(2, 2, 4)
	ax45.set_ylabel('cells')
	ax45.set_xlabel('Stability LEC')
	ax45.set_xlim(0,1)
	ax45.errorbar(stab_points, np.mean(active, -1), yerr = np.std(active , -1),color = 'k', ecolor = 'k')
	ax45.errorbar(stab_points, np.mean(active_lec, -1) ,yerr = np.std(active_lec , -1),color = 'b', ecolor = 'k')
	ax45.errorbar(stab_points, np.mean(active_mec, -1) ,yerr = np.std(active_mec , -1),color = 'r', ecolor = 'k')	
	ax45.errorbar(stab_points, np.mean(active_grid, -1), yerr = np.std(active_grid , -1),color = 'y', ecolor = 'k')
	
	ax45.errorbar(stab_points, np.mean(place_cells, -1), yerr = np.std(place_cells , -1),color = 'k', ecolor = 'k', ls ='--')
	ax45.errorbar(stab_points, np.mean(place_cells_lec, -1) ,yerr = np.std(place_cells_lec , -1),color = 'b', ecolor = 'k', ls ='--')
	ax45.errorbar(stab_points, np.mean(place_cells_mec, -1) ,yerr = np.std(place_cells_mec , -1),color = 'r', ecolor = 'k', ls ='--')	
	ax45.errorbar(stab_points, np.mean(place_cells_grid, -1), yerr = np.std(place_cells_grid , -1),color = 'y', ecolor = 'k', ls ='--')	
	ax45.set_ylim(0, 2500)
	for tick in ax45.xaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	for tick in ax45.yaxis.get_major_ticks()[0:-1:2]:
		tick.label1.set_visible(False)
	makeLabel(ax = ax45, label = 'C', sci = 0 )



### Figure 1
#MethodsPaper()

### Figure 2
#problematic()

### Figure 3
#SVCPaper()

### Figure 4
#SVCPaperOneMod()

## Figure 5 and 6A
#hebbPaperNew()

## Figure 6C,D,E
hebbPaperLesionsErr()

### Figure7
#hebbPaperStabilityErr()


end = time.time()-begin
print 'finished simulation  in '+str(int(end/3600)) + 'h '+str(int((end-int(end/3600)*3600)/60)) + 'min ' +str(int(end - int(end/3600)*3600- int((end-int(end/3600)*3600)/60)*60)) + 'sec'



save_figures(path = '', start = 0, title = 'one_orient', file_type = 'eps')

plt.draw()

plt.show()






