# Environmental Sound Classification on the ESC-50 dataset

## Overview

This repository contains the implementation of Environmental Sound Classification on the ESC-50 dataset using the [ACDNet](https://arxiv.org/abs/2103.03483). The project includes data preprocessing, model training, cross-validation, and results analysis.

## Augmentation - Mixup of Different Sound Files

One of the unique aspects of this approach is the use of mixup augmentation on different sound files. Mixup is a data augmentation technique that creates new training examples by combining pairs of examples and their labels. By mixing different sound files, the model can learn more robust features and improve generalization. This technique helps the model to better handle variations in the environmental sounds, leading to improved performance.

## Usage

### Training the Model

To train the model using cross-validation, run:

```bash
python train_crossval.py
```

### Testing the Model

To test the trained model using cross-validation, run:

```bash
python test_crossval.py
```

## Results

The results of the model training and evaluation will be saved in the results/ directory.

I achieved an accuracy of 82.45%.

## Acknowledgments

The ACDNet, which was used in this repository, was first implemented [here](https://github.com/mohaimenz/acdnet/tree/master).
