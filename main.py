import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import cv2
import os
from models.vit import get_vit_model
from cluster import ncut 


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize using ViT pretraining stats
])
    

def get_qkv(model,image):
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    # Forward pass in the model
    model.get_last_selfattention(image)

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
    return q,k,v                       


def main():
    model = get_vit_model()
    # print(model)
    data_dir = './data/images/001.Black_footed_Albatross/'
    image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]

    i=0
    for image_path in image_paths:
        image_org = cv2.imread(image_path)
        image_org = cv2.resize(image_org,(224,224))

        image_tesor = transform(image_org)
        image_batch = image_tesor.unsqueeze(0)

        q,k,v = get_qkv(model,image_batch)

        bipartition = ncut(k,(14,14))

        print(bipartition.shape)

        feature_map = bipartition / bipartition.max()  # Normalize to [0, 1]
        feature_map = np.uint8(255 * feature_map)  # Convert to 8-bit
        # heatmap = Image.fromarray(heatmap)
        # heatmap = heatmap.resize((224, 224), Image.Resampling.LANCZOS)
        mask = cv2.resize(feature_map, (224, 224), interpolation=cv2.INTER_CUBIC)

        # print(heatmap)
        # Ensure the mask is in binary format (0 or 255)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Convert binary mask to 3 channels if needed
        binary_mask_3_channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # Apply the mask onto the image
        # print(f"Image shape: {image_org.shape}, dtype: {image_org.dtype}")  
        # print(f"Mask shape: {binary_mask_3_channel.shape}, dtype: {binary_mask_3_channel.dtype}")

        result = cv2.bitwise_and(image_org, binary_mask_3_channel)
        combined = cv2.hconcat([image_org, result])
        cv2.imwrite(f'./outputs/output{i}.jpg',combined)
        i=i+1

    print('output saved ....')


if __name__ == '__main__':
    main()
