"""	
	Single Layer Models
"""


import numpy as np


class SingleLayerModel(object):
	
	
	def __init__(self,seed = 27092016):
		
		self.rng = np.random.RandomState(seed)
	
	def forwardPropagation(self,inputs,weights,biases):
		
		"""
			Forward propagates activations through the layer transformation
			
			For Inputs `x`, outputs `y`, weights `W` and biases `b` the layer 
			corresponds to y = Wx + b 
			
			Args:
			   inputs : Array of layer inputs of shape (batch_size,input_dim).
			   weights: Array of weight parameters of shape 
				    (output_dim,input_dim)
			   biases: Array of bias parameters of shape(output_dim,)
			
			Returns:
			  outputs : Array of layer outputs of shape (batch_size,output_dim)
			
				
		"""

		return inputs.dot(weights.T) + biases
	

		


if __name__ == '__main__':
	
	
	slm = SingleLayerModel()

	inputs  = np.array([[0,-1,2],[-6,3,1]])
	weights = np.array([[2,-3,-1],[-5,7,2]])
	biases = np.array([5,-3])
	
	true_outputs = np.array([[6,-6],[-17,50]])
	
	if not np.allclose(slm.forwardPropagation(inputs,weights,biases),true_outputs):
		print('Wrong outputs computed')

	else:
		print('All outputs correct !!!')



		
		







