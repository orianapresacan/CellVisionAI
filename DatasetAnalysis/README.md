# Exploratory Data Analysis Project

## Overview

This repository contains the exploratory data analysis conducted on the CELLULAR dataset. The primary goal of this analysis is to gain a deeper understanding of the data and the dynamics of the cells. This study concentrates solely on the 53 annotated images from the data set. These are all marked with the experiment date "220518". The images capture cells at five distinct time points, featuring cells that were either subjected to starvation or not.

## Dataset Preparation


## Repository Contents

- `cell_count.py`: Determines the minimum and maximum number of cells per each image and computing the total cell count for each class.
- `class_analysis_timepoints.py`: Analyzes how cell states (classes) evolve over the five designated time points.
- `tsne_dino.py`: Generates t-SNE plots to visualize the distribution and clustering of cells based on their feature vectors.
- `box_plots_masks.py`: Analyzes the area and circularity of cells using only the segmentation masks. Includes t-test code for statistical analysis.
- `statistics_masks_timepoints.py`: Assesses changes in cell area across the five time points.




