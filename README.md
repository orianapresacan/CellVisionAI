# CellVisionAI

## Overview
CellVisionAI provides an in-depth analysis of autophagy using deep learning techniques. This repository includes tasks such as object detection, segmentation, classification, tracking, and next-frame prediction, all applied to the [CELLULAR data set](https://zenodo.org/records/7503365).

<img src="https://github.com/orianapresacan/CellVisionAI/blob/234cd9547a82a3094335eb2a0a47a0ea36e58d88/Images/dataset_sample.jpg" width="800" height="150"/> 

## Repository Structure
- `Classification/` - Scripts for training ResNet, VGG, and ViT for cell classification, along with the models' checkpoints.
- `DatasetAnalysis/` - Code for performing exploratory data analysis on the CELLULAR data set.
- `ObjectDetection/` - Scripts for training and evaluating YOLO models for cell detection, including model checkpoints.
- `Segmentation/` - Scripts and models (SAM, Cellpose, U-Net++, DeepLabV3+) for cell segmentation in images, along with their checkpoints.
- `LastFramePrediction/` - Code for next-frame prediction in video sequences using models such as ConvLSTM, U-Net, Diffusion Model, PredRNN-v2, SwinLSTM, SimVP, and DMVFN.
- `Application/` - Scripts for a desktop application that integrates all trained models, providing an automated pipeline for cell detection, segmentation, and classification.

## Getting Started
1. Clone the repo to your local machine.
2. Choose a directory of interest.
3. Follow README instructions within the directory to set up and run models.

## Paper Reference
For detailed methodologies and insights, refer to our paper: "Leveraging Machine Learning to Unravel Autophagy Dynamics in Cellular Biology".

