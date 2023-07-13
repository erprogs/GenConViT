
# GenConViT Model for Deepfake Video Detection.

This repository contains the implementation code for **Deepfake Video Detection Using Generative Convolutional Vision Transformer (GenConViT)** paper. 

## GenConViT Model Architecture

The GenConViT model consists of two independent networks and incorporates the following modules:
<pre>
    Autoencoder (AE),
    Variational Autoencoder (VAE), and
    ConvNeXt-Swin Hybrid layer
</pre>

The code in this repository enables training and testing of the GenConViT model for deepfake detection.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model Testing](#model-testing)
- [Results](#results)

## Requirements
<pre>
    * Python 3.x
    * PyTorch
    * numpy
    * torch
    * torchvision
    * tqdm
    * decord
    * dlib
    * opencv
    * face_recognition
    * timm
</pre>

## Usage

1. Clone this repository:

```bash
    git clone https://github.com/erprogs/GenConViT
```

2. Install the required dependencies:

```bash
    pip install -r requirements.txt
```

## Model Training

To train the GenConViT model, follow these steps:

1. Prepare the training data, or use the sample training data provided:
    * Ensure that the training data is located in the specified directory path.
    * The training data should be organized in the required format. The `fake` directory contains images that are fake, while the `real` directory contains images that are real.
<pre>
    train:
        - fake
        - real
    valid:
        - fake
        - real
    test:
        - fake
        - real
</pre>
    

2. Run the training script:

```bash
    python train.py \
        -d <training-data-path> \
        -m <model-variant> \
        -e <num-epochs> \
        -p <pretrained-model-file> \
        -t
```

    * `<training-data-path>`: Path to the training data.
    * `<model-variant>`: Specify the model variant (`ed` for Encoder-Decoder or `vae` for Variational Autoencoder).
    * `<num-epochs>`: Number of epochs for training.
    * `<pretrained-model-file>` (optional): Specify the filename of a pretrained model to continue training.
    * `-t` (optional): Run the test on the test dataset after training.


## Deepfake Detection using GenConViT

To make prediction using the trained GenConViT model, follow these steps:

1. Download the pretrained model from [Huggingface](https://huggingface.co/Deressa/GenConViT)
2. Ensure that the pretrained model file is saved in the `weight` directory.
2. Run the prediction script:

```bash
    python prediction.py \
        --p <path-to-video-data> \
        --f <number-of-frames> \
        --d <dataset> \
        --n <model-variant>
```

    * `<path-to-video-data>`: Path to the video data or `[dfdc, faceforensics, timit, celeb]`.
    * `<number-of-frames>`: Specify the number of frames you want to be extracted for the video prediction. The default is 15 frames.
    * `<model-variant>`: Specify the model variant (`ed` for Encoder-Decoder or `vae` for Variational Autoencoder or Both:genconvit)
    * `<dataset>`: the dataset type. `[dfdc, faceforensics, timit, celeb]` or yours.


## Results

The results of the model prediction results documented in the paper can be found in the `results` directory. 
```bash
    python result_all.py
```
