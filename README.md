# Facial-Emotion-Recognition-Using-OpenCV

This project aims to develop a Convolutional Neural Network (CNN) model for facial emotion recognition using the FER2013 dataset. The model is designed to classify images into seven different emotional categories.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Installation

To run this project, you'll need to have the following packages installed:
pip install numpy pandas matplotlib scikit-learn keras tensorflow


## Dataset
The dataset used in this project is the FER2013 dataset. The images are 48x48 pixel grayscale images, with each image labeled with one of the following emotions:

Anger
Disgust
Fear
Happiness
Sadness
Surprise
Neutral

## Usage
Mount Google Drive (if using Google Colab): The script includes code to mount your Google Drive for easy access to the dataset.
Load the dataset: The data is loaded and preprocessed to convert images to the appropriate format and labels to one-hot encoded values.
Split the dataset: The dataset is split into training, validation, and test sets.
Standardize and reshape the data: Images are standardized and reshaped for model input.
Train the model: The CNN is trained using data augmentation techniques to improve generalization.

## Model Architecture
The CNN model architecture consists of the following layers:

Convolutional layers with ReLU activation
Batch normalization
Max pooling layers
Dense layers leading to the output layer with softmax activation
Summary of the Model
The model summary will provide details on the number of parameters and the structure of each layer.

## Training
The model is trained for a specified number of epochs (default is set to 50) using the Adam optimizer and categorical cross-entropy loss. Early stopping and model checkpointing are used to save the best model based on validation loss.

## Results
After training, the model is evaluated on the test dataset, and the accuracy is printed. The model's performance can be visualized using accuracy and loss plots.
