import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Vocabulary (same as before)
vocabulary = ["the", "cat", "sat", "on", "mat"]

# Training data (same as before)
training_data = [
    (["the", "cat"], "sat"),
    (["cat", "sat"], "on"),
    (["sat", "on"], "the"),
    (["on", "the"], "mat"),
    (["the", "cat"], "sat"),  # Repeat data for more training
    (["cat", "sat"], "on"),
    (["sat", "on"], "the"),
    (["on", "the"], "mat"),
]

# One-hot encoding (same as before)
def one_hot(word):
    vector = torch.zeros(len(vocabulary))  # Use torch.zeros
    index = vocabulary.index(word)
    vector[index] = 1
    return vector

# Prepare training data (using torch tensors)
X = []
y = []

for seq, next_word in training_data:
    X.append(torch.cat([one_hot(seq[0]), one_hot(seq[1])]))
    y.append(torch.tensor(vocabulary.index(next_word)))  # Use torch.tensor for labels

X = torch.stack(X)  # Stack into a tensor
y = torch.stack(y)  # Stack into a tensor (LongTensor for CrossEntropyLoss)

# Define the neural network using nn.Module
class WordPredictionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WordPredictionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2
        self.softmax = nn.Softmax(dim=1) # Softmax on the output

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # ReLU activation in hidden layer
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize the network
input_size = len(vocabulary) * 2
hidden_size = 5
output_size = len(vocabulary)
net = WordPredictionNet(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(net.parameters(), lr=0.01)  # Adam optimizer

# Training loop
epochs = 5000
for epoch in range(epochs):
    # Forward pass
    outputs = net(X)
    loss = criterion(outputs, y)  # Calculate loss

    # Backpropagation and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Calculate gradients
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Make predictions
with torch.no_grad(): # No gradient calculation during prediction
    outputs = net(X)
    _, predicted = torch.max(outputs.data, 1) # Get the predicted class
    predicted_words = [vocabulary[p] for p in predicted]
    print("Predicted words:", predicted_words)

    # Example usage (same as before):
    test_sequence = ["the", "cat"]
    test_input = torch.cat([one_hot(test_sequence[0]), one_hot(test_sequence[1])]).unsqueeze(0) # Add batch dimension
    test_output = net(test_input)
    _, test_predicted = torch.max(test_output.data, 1)
    predicted_next_word = vocabulary[test_predicted.item()]
    print(f"Given '{test_sequence[0]} {test_sequence[1]}', the next word is predicted to be: {predicted_next_word}")
