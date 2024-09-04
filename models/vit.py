import torch

def get_vit_model():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model