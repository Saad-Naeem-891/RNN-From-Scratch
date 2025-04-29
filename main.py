from rnn import RNN
from utils import create_vocab

# Exampl
text = "dogs"

# vocab and map
vocab, char_to_idx, idx_to_char = create_vocab(text)

# exampl
input_chars = ['d', 'o', 'g']
target_char = 's'
inputs = [char_to_idx[ch] for ch in input_chars]
target = char_to_idx[target_char]

# Initialize model
model = RNN(vocab_size=len(vocab), hidden_size=3, learning_rate=0.1)

# Train
for epoch in range(100):
    loss = model.train_step(inputs, target)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
