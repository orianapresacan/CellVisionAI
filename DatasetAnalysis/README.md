# Exploratory Data Analysis Project

## Overview

This repository contains the exploratory data analysis conducted on the CELLULAR dataset. The primary goal of this analysis is to gain a deeper understanding of the data and the dynamics of the cells. This study concentrates solely on the 53 annotated images from the data set. These are all marked with the experiment date "220518". The images capture cells at five distinct time points, featuring cells that were either subjected to starvation or not.

## Dataset Preparation

- For `cell_count.py`, a folder named "bounding_boxes" is required, containing all text files with bounding box information.

- For `class_analysis_timepoints.py`, two folders are necessary: one containing cells that were nourished and the other cells that underwent starvation. These should have both the images and the corresponding bounding boxes. 

- To generate t-SNE plots, a folder with all images cropped based on bounding box information is needed. Images should be named following the pattern "Timepoint_001_220518-ST_C03_s3_cell_1_class_1". The script `crop_cells.py` is provided for cropping.

- For generating box plots and conducting analysis with `statistics_masks_timepoints.py`, two folders containing segmentation masks are needed. These masks should be divided into two categories: cells that were nourished and cells that underwent starvation. Example structure:
    ```
    Timepoint_001_220518-ST_C03_s3
      ├── fed
      └── unfed
    ```

## Repository Contents

- `cell_count.py`: Determines the minimum and maximum number of cells per image and computes the total cell count for each class.
- `class_analysis_timepoints.py`: Analyzes how cell states (classes) evolve over the five designated time points.
- `tsne_dino.py`: Generates t-SNE plots to visualize the distribution and clustering of cells based on their feature vectors.
- `box_plots_masks.py`: Analyzes the area and circularity of cells using only the segmentation masks. Includes t-test code for statistical analysis.
- `statistics_masks_timepoints.py`: Assesses changes in cell area across the five time points.




