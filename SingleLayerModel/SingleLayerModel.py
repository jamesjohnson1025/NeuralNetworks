"""
	Single Layer Models
"""
from mlp.data_providers import CCPPDataProvider as ccppDp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class SingleLayerModel(object):


    def __init__(self,seed = 27092016,dataProvider=None):

        self.rng_ = np.random.RandomState(seed)
        self.dataProvider_ = None


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


    def initializeDataProvider(self,dProvider,which_set='train',input_dim=[0,1],
            batch_size=5000,max_num_batches=1,shuffle_order=False):
            self.dataProvider_ = dProvider(which_set,
                input_dim,
                batch_size,
                max_num_batches,
                shuffle_order)

    def getDataProvider(self):
        return self.dataProvider_ if self.dataProvider_ is not None else None


    def getWeights(self,low,high,dim):
        return self.rng_.uniform(low,high,size=dim)

    def getBiases(self,low,high,dim):
        return self.rng_.uniform(low,high,size=dim)

    def plot(self,inputs,predictedOutputs,trueOutputs):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(inputs[:,0],inputs[:,1],trueOutputs[:,0],'r.',ms=2)
        ax.plot(inputs[:,0],inputs[:,1],predictedOutputs[:,0],'b.',ms=2)
        ax.set_xlabel('Input dim 1')
        ax.set_ylabel('Input dim 2')
        ax.set_zlabel('Output')
        ax.legend(['Targets','Predictions'],frameon=False)
        fig.tight_layout()
        plt.show()





if __name__ == '__main__':


    slm = SingleLayerModel()
    slm.initializeDataProvider(ccppDp)

    dP = slm.getDataProvider()


    inputs,targets = (None,None)
    input_dim,output_dim = 2,1
    weights_init_range = 0.5
    biases_init_range = 0.1

    weights = slm.getWeights(-weights_init_range,weights_init_range,(output_dim,input_dim))
    biases = slm.getBiases(-biases_init_range,biases_init_range,output_dim)


    if dP != None:
        inputs,targets = dP.next()

        predicted_outputs = slm.forwardPropagation(inputs,weights,biases)
        slm.plot(inputs,predicted_outputs,targets)















