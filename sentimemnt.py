import torch
import torch.nn as nn
import torch.optim as optim

# Vocabulary (very limited)
vocabulary = ["happy", "sad", "good", "bad", "great", "terrible", "annoyed", "irritated", "fantastic"]

# Sentiment labels
sentiments = ["positive", "negative"]

# Training data (sentences and sentiments)
training_data = [
    ("I am happy with the car", "positive"),
    ("I am sad the ac broke", "negative"),
    ("Its good to drive", "positive"),
    ("It is bad during rain", "negative"),
    ("That is great", "positive"),
    ("That is terrible", "negative"),
    ("I feel happy", "positive"),
    ("I feel sad for buying this car", "negative"),    
    ("That is terrible to drive", "negative"),
    ("I feel annoyed when getting in the car", "negative"),
    ("I feel irritated from the smell", "negative"),    
    ("car ride was fantastic", "positive")
]

print("preparing data for training model")

# Prepare training data
sentence_bag = [] # Stores tensors whether activating words seen.
sentiment_of_sentence_bag = []

for sentence, sentiment in training_data:
    print("processing each sentence -- ", sentence)

    sentence_vector = torch.zeros(len(vocabulary))  # Initialize sentence vector    

    for word in sentence.split():        
        print("...processing each word --", word)
        
        if word in vocabulary:
            sentence_vector[vocabulary.index(word)] = 1
        # sentence_vector += one_hot(word)  # Sum one-hot vectors for words in sentence
        
    sentence_bag.append(sentence_vector)
    sentiment_of_sentence_bag.append(torch.tensor(sentiments.index(sentiment))) # Convert to numerical label

sentence_bag = torch.stack(sentence_bag)
sentiment_of_sentence_bag = torch.stack(sentiment_of_sentence_bag)

print(sentence_bag)
print(sentiment_of_sentence_bag)

# Define the neural network
class SentimentNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # fully connected layer 1
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Rectified Linear Unit activation function; makes negative values zero
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize the network
input_size = len(vocabulary)
hidden_size = 3  # You can adjust this
output_size = len(sentiments)

'''
    Input → Linear(fc1) → ReLU → Linear(fc2) → Softmax → Output
    [9]                            [5]                     [2]

    Softmax - a math function to convert real numbers into probability distribution between 0 to 1
      softmax(z_i) = exp (z_i) / Sum (exp (z_i)) 
      # Raw logits: [2.0, 1.0, 0.5]
      # After softmax: [0.5, 0.3, 0.2]  # Sums to 1.0
'''

net = SentimentNet(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01) # net.parameter returns an iterable over weights and biases



'''


W = [
    [0.1, 0.2, -0.1, 0.3],
    [0.2, 0.3, 0.1, -0.2],
    [-0.1, 0.1, 0.2, 0.1]
]

b = [0.1, -0.1, 0.2, 0.1]

# Matrix multiplication (xW):
[0.5, 0.3, 0.2] × [
    [0.1, 0.2, -0.1, 0.3],
    [0.2, 0.3, 0.1, -0.2],
    [-0.1, 0.1, 0.2, 0.1]
]

# Breaking down the multiplication:
First element = (0.5 × 0.1) + (0.3 × 0.2) + (0.2 × -0.1) = 0.05 + 0.06 - 0.02 = 0.09
Second element = (0.5 × 0.2) + (0.3 × 0.3) + (0.2 × 0.1) = 0.10 + 0.09 + 0.02 = 0.21
Third element = (0.5 × -0.1) + (0.3 × 0.1) + (0.2 × 0.2) = -0.05 + 0.03 + 0.04 = 0.02
Fourth element = (0.5 × 0.3) + (0.3 × -0.2) + (0.2 × 0.1) = 0.15 - 0.06 + 0.02 = 0.11

# Result after matrix multiplication:
[0.09, 0.21, 0.02, 0.11]

# Then add the bias b:
[0.09 + 0.1, 0.21 - 0.1, 0.02 + 0.2, 0.11 + 0.1]

# Final output:
[0.19, 0.11, 0.22, 0.21]

# Soft max
[0.2517, 0.2323, 0.2593, 0.2567]
'''

# Training loop
epochs = 1000 # number of times to readjust weights and biases

for epoch in range(epochs):
    #print("running optimization")
    outputs = net(sentence_bag)

    #print(outputs)

    loss = criterion(outputs, sentiment_of_sentence_bag)

    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Make predictions
with torch.no_grad():
    test_sentences = [
        "Those people are happy in their new car",        
        "my car is bad",
        "car keeps breaking and that annoyed me", 
        "my drive was fantastic",  
        "From this example, we has no vocabulary",  
        "Train arrives at 2",  
        "Watch is blue",  
        "Pink burger",     
        "This is very nice",                
    ]
    for sentence in test_sentences:
        sentence_vector = torch.zeros(len(vocabulary))
        for word in sentence.split():
            if word in vocabulary:
                sentence_vector[vocabulary.index(word)] = 1
        test_input = sentence_vector.unsqueeze(0)
        output = net(test_input)
        _, predicted = torch.max(output.data, 1)
        predicted_sentiment = sentiments[predicted.item()]
        print(f"Sentence: '{sentence}' - Predicted Sentiment: {predicted_sentiment}")








'''
How it works (simplified):

Forward Pass: Input data is fed through the network.  Each neuron calculates a weighted sum of its inputs (from the previous layer), adds its bias, and applies an activation function.  This process continues layer by layer until the output layer is reached.

Calculate Error: The network's output is compared to the actual target value, and the error is calculated.

Backpropagation: The error is propagated back through the network, layer by layer.  This allows us to calculate how much each weight and bias contributed to the error.

Update Weights and Biases: Based on the error contribution, the weights and biases are adjusted slightly to reduce the error.  This is done using an optimization algorithm.

Repeat: Steps 1-4 are repeated many times (epochs) with different data samples until the network's performance improves and the error is minimized.
'''

'''
Hidden Layers: These are layers of neurons in a neural network that are between the input layer and the output layer.  They are where the network learns complex, non-linear relationships in the data.  A neural network can have one or more hidden layers.

Weights: These are parameters within the network that connect neurons between layers.  Each connection has an associated weight.  The weight determines the strength or importance of the connection.  These are the primary parameters that the network learns during training.

Biases:  These are also learned parameters, one for each neuron (often in the hidden and output layers).  They are added to the weighted sum of inputs to a neuron.  Think of them as an offset.

Training: The training process involves feeding the network data, calculating the error (how far off the prediction is from the actual value), and then adjusting the weights and biases to reduce that error.  This adjustment is done using optimization algorithms like gradient descent.
'''


'''
Math stuff


Euler's constant, e is 2.71828.
    - e ^ 2 
    - ln 2 

Soft Max - softmax(z_i) = exp (z_i) / Sum (exp (z_i)) 
Log Soft Max - avoids using exponents, but achieves similar stability as Soft Max
'''


'''
Learning Algorithm 
    Adam -- type of Gradient Descent Algorithm
    Stochastic Gradient Descent 
Initialization Algorithm
    Xavier/Glorot initialization or Kaiming initialization.

'''


'''
Types of Loss Function
    Mean Squared Error (MSE), Cross-Entropy Loss, or Binary Cross-Entropy Loss,
'''
