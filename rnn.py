from layers import RNNLayer

class RNN:
    def __init__(self, vocab_size, hidden_size, learning_rate=0.01):
        self.rnn = RNNLayer(vocab_size, hidden_size, vocab_size, learning_rate)

    def train_step(self, input_indices, target_index):
        y_pred = self.rnn.forward(input_indices)
        loss = self.rnn.compute_loss(target_index)
        self.rnn.backward(target_index)
        return loss
