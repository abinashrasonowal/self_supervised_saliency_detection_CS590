# Self-Supervised Saliency Detection (CS590)

This project implements a self-supervised approach to saliency detection using a pretrained Vision Transformer (ViT) model. The model processes images to differentiate between the foreground and background by creating a fully connected graph, formulating it as an N-cut problem, and solving it through clustering.

## Dataset

The project uses the CUB-200-2011 dataset, which contains images of 200 bird species. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/wenewone/cub2002011/data).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/self_supervised_saliency_detection_CS590.git
    cd self_supervised_saliency_detection_CS590
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Place the images that you want to process inside `data/images` directory within the project folder.

## Usage

To run the program, use the following command:

```sh
python3 main.py
