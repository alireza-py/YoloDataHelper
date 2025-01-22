
cats-dogsv2 - v1 2024-07-12 8:22am
==============================

This dataset was exported via roboflow.com on July 15, 2024 at 11:41 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 277 images.
Cats-dogs are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip

The following transformations were applied to the bounding boxes of each image:
* Randomly crop between 0 and 30 percent of the bounding box
* Random brigthness adjustment of between -10 and +10 percent


