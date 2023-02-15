import copy
import numpy as np
from pdb import set_trace

class Neuron():

  inputs = []
  weights = []
  bias = 0
  activation = ''

  def __init__(self, weights, bias, activation):
    self.weights = weights
    self.bias = bias
    self.activation = activation

  def ReLU(self, x):
    return max(0, x)

  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def tanh(self, x):
    return np.tanh(x)

  def calculate(self, inputs):
    preActivation = self.bias
    for i in range(len(inputs)):
      preActivation += (inputs[i] * self.weights[i])

    # Activation
    post_activation = 0
    if self.activation == 'ReLU':
      post_activation = self.ReLU(preActivation)
    if self.activation == 'tanh':
      post_activation = self.tanh(preActivation)
    if self.activation == 'sigmoid':
      post_activation = self.sigmoid(preActivation)

    return post_activation

inputs = [4, 2, 3, 4]
weights = [
  [[0.1, 0.3, 0.4, 0.6], [0.4, 0.3, 0.1, 0.4], [0.4, 0.2, 0.3, 0.1], [0.3, 0.1, 0.5, 0.2]],
  [[0.2, 0.1, 0.3, 0.4], [0.5, 0.6, 0.2, 0.1], [0.4, 0.2, 0.3, 0.1]],
  [[0.2, 0.1, 0.3], [0.1, 0.2, 0.3], [0.3, 0.1, 0.4]]
]
bias = [
  [2, 3, 1, 4],
  [1, 2, 3],
  [2, 3, 5]
]
activation = [
  ['tanh', 'sigmoid', 'sigmoid', 'ReLU'],
  ['ReLU', 'tanh', 'sigmoid'],
  ['sigmoid', 'ReLU', 'sigmoid']
]
outputs = []

# fully connected NN
for layer in range(3):
  for unit in range(len(weights[layer])):
    neuron = Neuron(weights[layer][unit], bias[layer][unit], activation[layer][unit])
    outputs.append(neuron.calculate(inputs))
  print(f'inputs: {inputs}')
  print(f'outputs: {outputs}')
  inputs = copy.deepcopy(outputs)
  outputs.clear()
