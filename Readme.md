Understanding Neural Networks with PyTorch

After reading this article, you'll better understand a few key concepts about applying Machine Learning, particularly Neural Networks, to real-word use-cases. 

If you are wondering about why Neural Networks and Pytorch, I covered the reasons in my last article. Along with, how Capital One's culture promotes learning new skills and what's available to you. 

At the highest level of abstraction, Machine Learning excels at two types of problems: 1) Regression, and 2) Classification. Regression model is something that can predict a continous value. For example: give a house of 10,000 Sq.Ft. lot in Mclean Area and house built in 2024, predict the likely market price of the house. Classification model is something that classifies. For example: give a picture of a cat, predict the color ant type of the cat. 

For our sample use-case today, let's think of a restaurant, which collects customer feedbacks as text comments. We will build a Neural Network based Machine Learning model to classify the comments into postitive or negative sentiments. 

Applying Machine Learning to a problem involves three big stages: 1) Preparing Data, 2) Selecting and Training the Model – in this case, Neural Network, and finally 3) Infering or Predicting on unseen data. 

Preparing Data:


```# Vocabulary (very limited)
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
```











After reading this article, you'll have a good high-level understanding of key concepts in these domains. 






In my last article, I shared my approach selecting and learning new skills that are relevant to current industry needs, and also about Capital One's culture and benefits designed to encourage learning. 

My goals for this article

This article will cover three key concepts about Neural Networks by applying this technique to a practical and simple problem. After you read the article, you will know more about how training, infering, and measuring a Neural Network model works. 















I will cover three key concepts by applying Neural Networks to a practical problem that's easy to grog.  












step-1: preparing raw data to train the model

step-2: convert raw data into tensors, an internal representation the helps with matrix multiplications.
Neural network fundamentally uses matrix operations. 


step-2: setting up the architecture for how a neural network learns.

step-3: training the network

step-4 predicting with the network