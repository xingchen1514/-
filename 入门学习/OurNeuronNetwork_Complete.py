import numpy as np
from Neuron_001 import Neuron

def sigmoid(x):
    #Sigmoid 激活函数 f(x)=1/(1-e^(-x))
    return 1/(1-np.exp(-x))

def deriv_sigmoid(x):
    # Sigmoid 的导数 f`(x) = f(x)*(1-f(x))
    fx = sigmoid(x)
    return fx(1-fx)

def mse_loss(y_true,y_pred):
    return((y_true-y_pred)**2).mean()





class OurNeuronNetwork:
    """
    神经网络：
      - 2 个输入
      - 1 个隐藏层，2 个神经元（h1,h2）
      - 1 个输出层，1 个神经元（o1）

    """
    def __init__(self) -> None:

      # 权重（weights）
      self.w1 = np.random.normal()
      self.w2 = np.random.normal()
      self.w3 = np.random.normal()
      self.w4 = np.random.normal()
      self.w5 = np.random.normal()
      self.w6 = np.random.normal()
      
      # 偏移量（biases）
      self.b1 = np.random.normal()
      self.b2 = np.random.normal()
      self.b3 = np.random.normal()

    
    def feedforward(self,x):
      
      h1 = sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
      h2 = sigmoid(self.w3*x[0]+self.w4*x[1]+self.b2)
      o1 = sigmoid(self.w5*h1+self.w6*h2+self.b3)
      return o1
    

    def train(self,data,all_y_trues):
      """
      - 数据集是（n x 2）的numpy数组 n = 数据集中的样本数
      - all_y_trues 是有n个元素的numpy数组
        
      """
      learn_rate= 0.1
      epochs = 1000

      for epoch in range(epochs):
        for x,y_true in zip(data,all_y_trues):
            # ---- 进行前馈操作
            sum_h1 = self.w1*x[0]+self.w2*x[1]+self.b1
            h1 = sigmoid(sum_h1)

            sum_h2 = self.w3*x[0]+self.w4*x[1]+self.b2
            h2 = sigmoid(sum_h2)

            sum_o1 = self.w5*x[0]+self.w6*x[1]+self.b3
            o1 = sigmoid(sum_o1)
            y_pred = o1

            # --- 计算偏导数
            # --- 命名方式：d_L_d_w1 代表 “dL/dw1”,即L对w1求偏导

            d_L_d_ypred = -2*(y_true - y_pred)

            # 神经元 o1
            d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
            d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
            d_ypred_d_b3 =  deriv_sigmoid(sum_o1)

            d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
            d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

            #神经元 h1
            d_ypred_d_w1 = x[0] * deriv_sigmoid(sum_h1)
            d_ypred_d_w2 = x[1] * deriv_sigmoid(sum_h1)
            d_ypred_d_b1 =  deriv_sigmoid(sum_h1)

            #神经元 h2
            d_ypred_d_w3 = x[0] * deriv_sigmoid(sum_h2)
            d_ypred_d_w4 = x[1] * deriv_sigmoid(sum_h2)
            d_ypred_d_b2 =  deriv_sigmoid(sum_h2)

            # --- 更新权重（w）与偏移量（b）
            # 神经元 h1
            self.w1 -=learn_rate*d_L_d_ypred*d_ypred_d_h1*d_ypred_d_w1
            self.w2 -=learn_rate*d_L_d_ypred*d_ypred_d_h1*d_ypred_d_w2
            self.b1 -=learn_rate*d_L_d_ypred*d_ypred_d_h1*d_ypred_d_b1
            
            # 神经元 h2
            self.w3 -=learn_rate*d_L_d_ypred*d_ypred_d_h2*d_ypred_d_w3
            self.w4 -=learn_rate*d_L_d_ypred*d_ypred_d_h2*d_ypred_d_w4
            self.b2 -=learn_rate*d_L_d_ypred*d_ypred_d_h2*d_ypred_d_b2

            # 神经元o1
            self.w5 -=learn_rate*d_L_d_ypred*d_ypred_d_w5
            self.w6 -=learn_rate*d_L_d_ypred*d_ypred_d_w6
            self.b3 -=learn_rate*d_L_d_ypred*d_ypred_d_b3

        if epoch % 10 ==0:
          y_pred = np.apply_along_axis(self.feedforward,1,data)
          loss = mse_loss(all_y_trues,y_pred)
          print("Epoch %d loss:%.3f"%(epoch,loss))


data = np.array([
   [-2,-1],
   [25,6]
])