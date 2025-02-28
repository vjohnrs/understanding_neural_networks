import torch # data preparation related functions
import torch.nn as nn # architecture related functions 
import torch.optim as optim # training related functions


# outcome: train a model to predict positive and negative sentiment

# step-1: preparing data to train the model
vocabulary = [
    "happy", 
    "sad", 
    "good", 
    "bad", 
    "great", 
    "terrible", 
    "annoyed", 
    "irritated", 
    "fantastic"
]

# sentiment labels
sentiments = [
    "positive", 
    "negative"
]

# training data (sentences and sentiments)
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

# step-2: convert raw data into tensors, 
# an array-like representation of data
# used by Neural Networks

sentence_bag = []
sentiment_bag = []

for sentence, sentiment in training_data:
    sentence_vector = torch.zeros(len(vocabulary))  
    # [0,0,0,0,0,0,0,0,0]
    # "I am happy with the car"
    for word in sentence.split():  
        if word in vocabulary:
            sentence_vector[vocabulary.index(word)] = 1
            # [1,0,0,0,0,0,0,0,0]

    sentence_bag.append(sentence_vector)
    sentiment_bag.append(torch.tensor(sentiments.index(sentiment))