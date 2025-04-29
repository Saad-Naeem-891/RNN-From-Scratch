import numpy as np
from activations import tanh, tanh_derivative, softmax
from utils import one_hot

class RNNLayer:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.Wx = np.random.randn(hidden_size, input_size)
        self.Wh = np.random.randn(hidden_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))


    ###### forward pass funciton ######
    def forward(self, inputs):
        self.xs = {}
        self.hs = {-1: np.zeros((self.hidden_size, 1))}

        for t in range(len(inputs)):
            self.xs[t] = one_hot(inputs[t], self.input_size)
            a_t = np.dot(self.Wx, self.xs[t]) + np.dot(self.Wh, self.hs[t - 1]) + self.bh
            self.hs[t] = tanh(a_t)

        y = np.dot(self.Wy, self.hs[len(inputs) - 1]) + self.by
        self.y_pred = softmax(y)
        return self.y_pred

    ###### Compute_loss ######
    def compute_loss(self, target_index):
        target = one_hot(target_index, self.output_size)
        return -np.sum(target * np.log(self.y_pred))


    ###### backward function ######
    def backward(self, target_index):
        target = one_hot(target_index, self.output_size)
        dy = self.y_pred - target

        dWy = np.dot(dy, self.hs[len(self.xs) - 1].T)
        dby = dy

        dWh = np.zeros_like(self.Wh)
        dWx = np.zeros_like(self.Wx)
        dbh = np.zeros_like(self.bh)
        dh_next = np.zeros_like(self.hs[0])

        for t in reversed(range(len(self.xs))):
            dh = np.dot(self.Wy.T, dy) if t == len(self.xs) - 1 else dh_next
            dtanh = dh * tanh_derivative(self.hs[t])
            dWh += np.dot(dtanh, self.hs[t - 1].T)
            dWx += np.dot(dtanh, self.xs[t].T)
            dbh += dtanh
            dh_next = np.dot(self.Wh.T, dtanh)

        ###### Gradient descent ######
        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.Wy -= self.learning_rate * dWy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
