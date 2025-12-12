
# Animal Faces Image Classification (Cats / Dogs / Wild)

## Overview

This project builds a deep learning classifier that distinguishes between three types of animal faces: **Cat**, **Dog**, and **Wild**.
I implemented a full pipeline using PyTorch, starting from raw image folders and ending with a trained Convolutional Neural Network (CNN) that achieves strong accuracy on the AFHQ dataset.

The goal is to demonstrate a clean, reproducible workflow for image classification while manually building the dataset pipeline.

---

## Dataset Preparation

### 1. Dataset Used

I used the **AFHQ (Animal Faces HQ)** dataset, which contains high-quality images organized into class folders.

### 2. What I Did

* Loaded image paths and labels by iterating through the dataset directory tree.
* Built a DataFrame containing two columns:

  * `image_path`
  * `labels` (cat, dog, wild)
* Applied a **stratified split** to create:

  * 70% training
  * 15% validation
  * 15% test
* Reset all indices for consistency.

This ensures balanced class distribution across splits.

---

## Custom Dataset Class

I implemented a PyTorch `Dataset` that:

* Loads each image using **PIL**
* Converts it to RGB
* Applies transforms (resize → tensor → float)
* Encodes labels using `LabelEncoder`

This makes the dataset flexible and easy to integrate with DataLoader.

---

## Model Architecture

The CNN model contains:

* **Conv → ReLU → MaxPool** (3 blocks)
* Flatten layer
* Fully connected layer (128 neurons)
* Output layer (3 classes)

This lightweight architecture performs well for small image classification tasks.

---

## Training

* **Optimizer:** Adam
* **Loss:** CrossEntropyLoss
* **Epochs:** 10
* **Device:** GPU if available

Training includes forward pass, backpropagation, updating weights, and evaluation on validation data each epoch.

---

## Results

Your final results were:

```
Epoch 10/10 | Train acc: 99.20% | Val acc: 96.40%
```

The model shows:

* Strong learning behavior
* High training and validation accuracy
* Slight overfitting 

---

## Future Improvements

* Add data augmentation
* Try pretrained models (ResNet, MobileNet)
* Add normalization
* Use checkpoints and schedulers

