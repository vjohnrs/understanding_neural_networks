import torch
import torch.nn as nn
import torch.optim as optim

# Sentiment labels
num_type = ["even", "odd"]

# Training data (sentences and sentiments)
training_data = [
    (2, "even"),
    (4, "even"),
    (5, "odd"),
    (9, "odd"),
    (6, "even"),
    (24, "even"),
    (36,"even"),
    (100, "even"),
    (95, "odd"),
    (77, "odd"),
    (82,"even"),
    (50, "even"),
]

for num, num_type in training_data:
    print("processing each sentence -- ", num)

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
epochs = 10 # number of times to readjust weights and biases

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