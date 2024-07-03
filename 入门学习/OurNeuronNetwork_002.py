import numpy as np
from Neuron_001 import Neuron
class OurNeuronNetwork:
    """
    神经网络：
      - 2 个输入
      - 1 个隐藏层，2 个神经元（h1,h2）
      - 1 个输出层，1 个神经元（o1）
    所有神经元有同样的权重和偏移量
      - w =[0,1]
      - b =0
    """
    def __init__(self) -> None:
        weights = np.array([0,1])
        bais = 0

        #Neuron类 是上面引入
        self.h1 = Neuron(weights,bais)
        self.h2 = Neuron(weights,bais)
        self.o1 = Neuron(weights,bais)

    def feedforward(self,x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1,out_h2]))
        return out_o1
    
network = OurNeuronNetwork()
x = np.array([2,3])
print(network.feedforward(x)) 
   