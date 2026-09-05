
# Project Title

Comparative Analysis of Transfer Learning and Custom CNN for Clean-Dirty Street Image Classification with Limited Data

# Problem 

This project aims to develop a clean-dirty street image classification system using limited training data, comparing the performance of transfer learning and custom CNN approaches to determine the optimal solution. Additionally, the project investigates the impact of data augmentation techniques on model performance, evaluating whether augmentation helps mitigate the challenges of limited data or introduces unnecessary complexity.

# Dataset: 

## Dataset Description

The dataset consists of 237 street images stored in a single folder, accompanied by a CSV file containing the filename of each image and its corresponding class label (clean or dirty).
This dataset is publicly available on Kaggle and can be downloaded from the Clean-Dirty Road Classification dataset page ([https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification](https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification)).
## Data Organization

After downloading and extracting the dataset, the images are separated into two categories, clean and dirty, based on the class labels provided in the CSV file. The images are then distributed into a structured directory format to facilitate model training and evaluation.
## Directory Structure

```
base_dir/
│
├── train/
│   ├── clean/
│   └── dirty/
│
├── valid/
│   ├── clean/
│   └── dirty/
│
└── test/
    ├── clean/
    └── dirty/

```

## Dataset Distribution

The dataset is divided into training, validation, and test sets with the following distribution for each class:

| Dataset Split  | Clean Class | Dirty Class | Total |
| -------------- | ----------- | ----------- | ----- |
| Training set   | 93          | 84          | 177   |
| Validation set | 14          | 15          | 29    |
| Test set       | 16          | 15          | 31    |
| Total          | 123         | 114         | 237   |

# Model: 
## Transferred Model

This model employs transfer learning using MobileNetV2 as a pre-trained feature extractor for binary image classification. The architecture consists of:
1. **Pre-trained Base Model (MobileNetV2)**
	- Loaded with ImageNet weights
	- Top classification layers excluded (`include_top=False`)
    - All layers frozen (`trainable=False`) to preserve learned features
2. **Global Average Pooling Layer**
    - Converts spatial feature maps into a compact vector representation
    - Reduces dimensionality while retaining important features
3. **Fully Connected Layers**
    - Dense layer with 128 neurons and ReLU activation for feature learning
    - BatchNormalization to stabilize training and improve convergence
    - Dropout (rate 0.5) to prevent overfitting 
4. **Output Layer**
    - Single neuron with sigmoid activation for binary classification
    - Outputs probability between 0 and 1 (clean vs. dirty)

## Custom Model 

This model implements a custom convolutional neural network (CNN) designed from scratch for binary image classification. The architecture consists of:
1. **Input Layer**
    - Accepts 150×150 RGB images
2. **Three Convolutional Blocks**
    - Progressively reduce filters (64 → 32 →16)
    - Each block includes:
        - Conv2D with 3×3 kernel and same padding
        - L2 regularization (rate 0.001) to prevent overfitting
        - ReLU activation for non-linearity
        - MaxPooling (2×2) for downsampling
        - Dropout (0.25) for regularization
3. **Flatten Layer**
    - Converts feature maps into a 1D vector
4. **Fully Connected Layers**
    - Dense layer with 256 neurons and ReLU activation
    - Dropout (0.5) to prevent overfitting
    - Dense layer with 128 neurons and ReLU activation
    - Dropout (0.5) for regularization    
5. **Output Layer**
    - Single neuron with sigmoid activation for binary classification
    - Outputs probability between 0 and 1 (clean vs. dirty)

# Method 

## Data Preparation 

The dataset is loaded using TensorFlow's `ImageDataGenerator`, which handles image resizing and normalization. All images are rescaled by 1/255 to normalize pixel values to the range [0, 1]. The `flow_from_directory` method loads images directly from the structured folders, automatically assigning labels based on subdirectory names (clean or dirty). Images are resized to 150×150 pixels and loaded in batches of 32. Training data is shuffled to improve generalization, while validation and test sets remain unshuffled for consistent evaluation. 

## Data Augmentation 

The dataset is loaded using TensorFlow's `image_dataset_from_directory` function, which automatically assigns labels based on folder names. Images were resized to 150×150 pixels and batched into groups of 32. Training data was shuffled, while validation and test sets remained unshuffled for consistent evaluation.

Data augmentation is applied to the training set through a custom pipeline. The `resize_rescale` function normalizes images by resizing them to 150×150 pixels and scaling pixel values to [0, 1]. The `augmentation` function adds random transformations (horizontal flips, brightness changes, and contrast adjustments) to create varied versions of training images. This technique artificially increases dataset diversity, helping models generalize better and reducing overfitting. Importantly, augmentation was only applied to training data, while validation and test images underwent only basic resizing and normalization.

## Model Training 

With the dataset prepared, both models, the transfer learning model and the custom CNN, are trained under two conditions: with and without data augmentation. This approach enables a comparative analysis of the two architectures on this dataset, while also evaluating the impact of data augmentation on their respective performances.

# Result

## Transfer Learning Model 
### Without Data Augmentation:

- Validation accuracy:  100%
- Validation AUC: 1.0000 - perfect discriminative ability
- Training accuracy: 98.87% Stable training throughout - no overfitting
- Validation loss: 0.0458 (very low)
- Training loss: 0.0485
- False positives: 0 - perfect precision
- False negatives: 0 - excellent recall
### With Data Augmentation:

- Validation accuracy: 96.55%
- Validation AUC: 1 
- Training accuracy: 96.61% 
- Validation loss: 0.0766
- Training loss: 0.0860
- False positives: 0 
- False negatives: 1 

## Custom Model 

### Without Data Augmentation:

- Validation accuracy:  89.66%; Validation accuracy fluctuated significantly (51.72% to 93.10%)
- Validation AUC: 0.9667
- Training accuracy: 85.88%; Training accuracy improved steadily from 49.72% to 85.88%
- Validation loss: 0.3586
- Training loss: 0.3291
- False positives: 1
- False negatives: 2
### With Data Augmentation:

- Validation accuracy: 86.21%; Validation accuracy remained unstable (51.72% to 93.10%)
- Validation AUC: 0.9429
- Training accuracy: 81.92%; Training accuracy fluctuated significantly (41.61% to 86.44%)
- Validation loss: 0.4734
- Training loss: 0.4332
- False positives: 3
- False negatives: 1 

## Summary: 

| Model                      | Training Accuracy | Validation Accuracy | FP  | Validation AUC | Training Loss | Validation Loss |
| -------------------------- | ----------------- | ------------------- | --- | -------------- | ------------- | --------------- |
| Transferred-model-no-aug   | 98.87%            | 100%                | 0   | 1.000          | 0.0485        | 0.0458          |
| Transferred-model-with-aug | 96.61%            | 96.55%              | 0   | 1.000          | 0.0860        | 0.0766          |
| Custom-model-no-aug        | 85.88%            | 89.66%              | 1   | 0.9667         | 0.3291        | 0.3586          |
| Custom-model-with-aug      | 81.92%            | 86.21%              | 3   | 0.9429         | 0.4332        | 0.4734          |

# Conclusion 

The transfer learning model exhibited stable and reliable performance, with training accuracy improving smoothly from 70% to 99% while validation accuracy quickly reached 96.55% and remained consistent throughout training. The model achieved a near-perfect AUC of 0.9960 with zero false positives, indicating precise classification. This stability is attributed to the pre-trained MobileNetV2 base model, which leverages features learned from millions of images, enabling effective feature extraction despite the limited dataset of 177 training images. The frozen base layers prevented overfitting and promoted generalization, resulting in a minimal gap between training and validation performance. These results demonstrate that transfer learning provides a robust and effective solution for classification tasks with limited training data.

The custom CNN model exhibited a training accuracy that improved steadily from 65% to 92% over the course of training. However, its validation accuracy remained highly unstable, fluctuating between 58% and 96%, with a significant gap frequently observed between training and validation performance. This behavior can be attributed to the small size of the training dataset, which introduces high variance in gradient updates and results in unstable training dynamics. Additionally, since custom models must learn feature representations entirely from scratch, they are more prone to memorizing the training data rather than learning generalizable patterns, ultimately leading to overfitting.


> [!Note]
> Data augmentation was applied to reduce overfitting, and the results confirm its effectiveness in this regard. The transfer learning model without augmentation achieved a training accuracy of 98.87% and validation accuracy of 100%, showing a slight discrepancy between the two. With augmentation, the model achieved a training accuracy of 96.61% and validation accuracy of 96.55%, demonstrating a nearly perfect alignment between training and validation performance. While augmentation slightly reduced overall accuracy, it successfully eliminated the gap between training and validation metrics, indicating improved generalization. The augmented model appears less likely to be memorizing training data and more likely to perform consistently on unseen data.
Data augmentation did not improve the custom model's performance. The model without augmentation achieved better results across most metrics, though both versions exhibited unstable training behavior.


> [!Note]
Early stopping was initially applied but caused premature termination of training at epoch 5, before the model could learn meaningful patterns. Given the small dataset and the model's gradual learning curve, training was allowed to continue for all 40 epochs to ensure optimal performance.


