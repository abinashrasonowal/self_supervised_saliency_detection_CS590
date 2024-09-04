import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import os
from dino.vision_transformer import vit_small
from object_discovery import ncut
from networks import get_model

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
    image = transform(image)  # Add batch dimension
    return image

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model('vit_small',16,device)
    # print(model)

    image_path = "./temp/test2.jpg"
    image_tensor = preprocess_image(image_path)
    image_org = cv2.imread(image_path)
    
    image_org = cv2.resize(image_org,(224,224))
    print(image_org.shape)

    # with torch.no_grad():  # Disable gradient calculation
    #     attn = model.get_last_selfattention(image_tensor)  # Forward pass through the model


    # print(attn.shape)
    # print(attn[0])
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    # Forward pass in the model
    attentions = model.get_last_selfattention(image_tensor[None, :, :, :])

    print(attentions.shape)
    nb_im = 1
    nb_tokens = 197
    nh = 6
    qkv = (
        feat_out["qkv"]
        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    print(q.shape)                        

    bipartition = ncut(v,(14,14),None,224)

    print(bipartition.shape)
    # print(bipartition) 

    feature_map = bipartition / bipartition.max()  # Normalize to [0, 1]
    heatmap = np.uint8(255 * feature_map)  # Convert to 8-bit
    # heatmap = Image.fromarray(heatmap)
    # heatmap = heatmap.resize((224, 224), Image.Resampling.LANCZOS)
    resized_image = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

    print(heatmap)
    # Ensure the mask is in binary format (0 or 255)
    _, binary_mask = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)

    # Convert binary mask to 3 channels if needed
    binary_mask_3_channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask onto the image
    print(f"Image shape: {image_org.shape}, dtype: {image_org.dtype}")  
    print(f"Mask shape: {binary_mask_3_channel.shape}, dtype: {binary_mask_3_channel.dtype}")

    result = cv2.bitwise_and(image_org, binary_mask_3_channel)
    combined = cv2.hconcat([image_org, result])
    cv2.imwrite('xx.jpg',combined)

    # print(heatmap)
    # # print(heatmap.pe)
    # heatmap.save('x2.png')
    # to_pil = transforms.ToPILImage()
    # pil_image = to_pil(image_tensor)
    # pil_image.putalpha(heatmap)
    # pil_image.save('output.png')
    
if __name__ == '__main__':
    main()
