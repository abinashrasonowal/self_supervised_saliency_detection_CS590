# models/vit.py
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_vit_model():
    # Load a pretrained Vision Transformer model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()  # Set to evaluation mode
    return model

def extract_patch_embeddings(model, images):
    with torch.no_grad():
        outputs = model(images)  # Forward pass to get patch embeddings
    return outputs
