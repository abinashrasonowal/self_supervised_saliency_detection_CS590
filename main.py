import torch
from models.vit import get_vit_model, extract_patch_embeddings
from utils.data_utils import get_dataloader
from utils.cluster_utils import cluster_patches, refine_segmentation
from tqdm import tqdm

def main():
    # Load the pretrained Vision Transformer model
    vit_model = get_vit_model()

    # Load data
    dataloader = get_dataloader('data/cub/', batch_size=1)

    for images in tqdm(dataloader):
        # Extract patch embeddings
        embeddings = extract_patch_embeddings(vit_model, images)

        # Flatten embeddings and apply clustering
        batch_size, n_patches, embedding_dim = embeddings.size()
        embeddings = embeddings.view(batch_size * n_patches, embedding_dim).cpu().numpy()
        cluster_labels = cluster_patches(embeddings)

        # Refine the segmentation
        refined_labels = refine_segmentation(cluster_labels, (batch_size, n_patches))
        
        # TODO: Evaluate or save results here

if __name__ == '__main__':
    main()
