import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import os
from models.vit import get_vit_model 

def preprocess_image(image_path):
    """Preprocess the input image to match the requirements of the ViT model."""
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize using ViT pretraining stats
    ])
    
    # Open the image file
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def main():
    model = get_vit_model()
    # print(model)

    image_path = "./temp/test.jpg"
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():  # Disable gradient calculation
        attn = model.get_last_selfattention(image_tensor)  # Forward pass through the model


    print(attn.shape)
    print(attn[0])
    
if __name__ == '__main__':
    main()
