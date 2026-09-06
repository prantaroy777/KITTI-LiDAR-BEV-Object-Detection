\# KITTI LiDAR BEV Object Center Detection



A LiDAR-based object-detection project using the KITTI dataset, bird's-eye-view (BEV) feature maps, and a lightweight PyTorch convolutional neural network.



The project develops the full pipeline from raw Velodyne point clouds through preprocessing, classical clustering, BEV construction, neural-network training, class-imbalance handling, validation-based model selection, and final held-out evaluation.



\## Project Overview



The detector predicts BEV object centers for three KITTI road-user classes:



\- Car

\- Pedestrian

\- Cyclist



The project includes two main approaches:



1\. \*\*Classical baseline\*\*

&#x20;  - LiDAR region-of-interest cropping

&#x20;  - RANSAC ground-plane removal

&#x20;  - voxel downsampling

&#x20;  - DBSCAN clustering

&#x20;  - geometric filtering



2\. \*\*Neural BEV detector\*\*

&#x20;  - three-channel BEV representation

&#x20;  - height, intensity, and density features

&#x20;  - stride-4 convolutional center-heatmap detector

&#x20;  - focal-loss training

&#x20;  - class-aware weighted sampling

&#x20;  - validation-based confidence-threshold selection



\## Main Notebook



The complete cleaned notebook is:



```text

LiDAR\_KITTI\_Project\_GitHub.ipynb

