# CellVisionAI

## Overview
CellVisionAI offers a comprehensive exploration into the dynamic world of autophagic cells through the lens of deep learning. This project encompasses a wide range of tasks including object detection, segmentation, classification, tracking, and next-frame prediction. All these models were applied on the [CELLULAR data set](https://zenodo.org/records/7503365).

<img src="https://drive.google.com/uc?export=download&id=1dYf9vp-Wlz7cnB1rfBjolcDBxBZu8gFB" width="900" height="200"/> 

## Repository Structure
- `Classification/` - Scripts for training ResNet, VGG, and ViT for cell classification, along with the models' checkpoints.
- `DatasetAnalysis/` - Code used for exploratory data analysis on the CELLULAR data set.
- `ObjectDetection/` - Scripts for cell detection using YOLO models, along with the models' checkpoints.
- `Segmentation/` - Scripts and segmentation models (SAM, Cellpose, U-Net++, DeepLabV3+) for isolating cells within images, along with the models' checkpoints.
- `NextFramePrediction/` - Code and models (ConvLSTM, U-Net, Diffusion Model, PredRNN-v2, SwinLSTM, SimVP, DMVFN) used for predicting the last frame in a video.
- `DesktopApp` - Scripts for the development of the desktop app that integrates all trained models to create an automated pipeline for cell detection, segmentation, and classification.

## Getting Started
1. Clone the repo to your local machine.
2. Choose a directory of interest.
3. Follow README instructions within the directory to set up and run models.

## Paper Reference
For detailed methodologies and insights, refer to our paper: "Leveraging Machine Learning to Unravel Autophagy Dynamics in Cellular Biology".

