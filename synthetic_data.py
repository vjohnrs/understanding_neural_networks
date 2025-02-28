import numpy as np
from PIL import Image

def SYNTHETIC_DATASET():
    # Parameters
    num_samples = 100  # Number of samples per digit
    image_size = (28, 28)  # Height and width
    
    # Create dataset
    dataset = []
    
    for digit in ['1', '2', '3']:
        for i in range(num_samples):
            # Generate a random number between 0 and 1
            num = np.random.rand()
            
            # Create a blank canvas
            img = Image.new('L', image_size, color=255)
            
            # Write the digit on the canvas
            draw = Image.Draw(img)
            font = Image.load_default()
            draw.text((0, 0), str(digit), fill=0,
                      font=font * int(num))  # Vary thickness
            
            # Convert to tensor and normalize
            img = np.array(img) / 255.0
            img = torch.tensor(img).float().view(-1)
            
            dataset.append((img, torch.tensor([digit_to_index(digit)])))
    
    return dataset


train_dataset = SYNTHETIC_DATASET()
