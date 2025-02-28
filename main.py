
'''
    Outcome: 
    On receiving an image, a model detects whether it sees '1', '2', '3', or 'unkown'.    
'''

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

def main():
    train()    

def train():
    print("::: begin training :::")

    ''' Code below takes every image and stores image as a tensor. Finally, normalizes the tensor. '''
    transform_train = transforms.Compose([    
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Adjust size if needed
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Important: Adjust normalization!
    ])

    data_dir = './data'

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    included_classes = ['1', '2', '3'] 

    train_dataset.classes = included_classes + ['unknown']
    val_dataset.classes = included_classes + ['unknown']
    
    # Update targets for "unknown" class
def update_targets(dataset, included_classes):
    unknown_label = len(dataset.classes) - 1  # Index of "unknown" class
    new_targets = []
    for _, label_str in dataset.imgs: # dataset.imgs contains tuples of (path, label_index)
        label = int(label_str) # ImageFolder gives you indices, not class names
        class_name = dataset.classes[label] # Get the actual class name
        if class_name not in included_classes:
            new_targets.append(unknown_label)
        else:
            new_targets.append(label) # Keep original label if it is 1,2 or 3
    dataset.targets = torch.tensor(new_targets)

update_targets(train_dataset, included_classes)
update_targets(val_dataset, included_classes)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    

    # Check if CUDA is available and use it if so
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    num_epochs = 10  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # Move data to device
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("::: end training :::")
    test(model)

def test(model):
    print("::: begin testing :::")

    transform_val = transforms.Compose([
        transforms.Resize(28), # Resize images to a consistent size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Important: Adjust normalization!
    ])    

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device) # Move data to device
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
    torch.save(model.state_dict(), 'handwritten_digit_model.pth')




if __name__ == "__main__":
    main()

