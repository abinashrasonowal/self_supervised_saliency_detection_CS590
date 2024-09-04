import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the folder containing the images
input_folder = r"C:\Users\ravil\Desktop\Dev\GitHub\Self_Supervised_Saliency_Detection\datasets\CUB\CUB_200_2011\CUB_200_2011\images\200.Common_Yellowthroat"
output_folder = r"C:\Users\ravil\Desktop\Dev\GitHub\Self_Supervised_Saliency_Detection\results"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Resize image to match the ViT input size
image_size = 224
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the ViT model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval()

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to get attention maps
def get_attention_maps(image_path):
    img = Image.open(image_path)
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        attentions = model.get_last_selfattention(input_tensor)

    nh = attentions.shape[1]  # Number of attention heads
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1).reshape(nh, 14, 14)  # Reshape to [heads, H, W]
    mean_attention = attentions.mean(axis=0)  # Average attention across all heads
    return attentions, mean_attention

# Function to save results
def save_results(original_image, attentions, output_file):
    def plot_heatmap(feature_map):
        feature_map = feature_map / feature_map.max()  # Normalize to [0, 1]
        heatmap = np.uint8(255 * feature_map)  # Convert to 8-bit
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return np.asarray(heatmap)

    n_cols = 8
    nh = attentions.shape[0]
    n_rows = (nh + n_cols - 1) // n_cols

    # Create a subplot grid
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(n_cols * 3, n_rows * 3))

    # Flatten axes array if needed
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot heatmaps for each feature map
    for ax, i in zip(axes.flat[1:], range(nh)):
        feature_map = attentions[i]
        heatmap = plot_heatmap(feature_map)
        ax.imshow(heatmap, cmap='viridis')
        ax.axis('off')
    
    for i in range(nh + 1, n_rows * (n_cols + 1)):
        axes.flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        attentions, mean_attention = get_attention_maps(image_path)
        
        original_image = Image.open(image_path)
        output_file = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_results.png')
        
        save_results(original_image, attentions, output_file)

print("Processing complete. Results saved to:", output_folder)
