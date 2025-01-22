# <p align="center">Welcome to YoloDataHelper 💡</p>

**YoloDataHelper** is a small Python utility to process YOLO(you only look once) datasets. This is a utility tool for merging datasets, augmenting data, removing classes, visualizing annotations, and other operations that make working with YOLO datasets easier by developers and researchers.

## 🛠️ Features //

### 1. **Dataset Combination**
- Combine multiple YOLO datasets while properly aligning classes and adjusting label IDs.
- Retain the original structure of the datasets and generate a unified `data.yaml` file.

### 2. **Data Augmentation**
- Apply various transformations to YOLO dataset images, such as:
  - Hue, saturation, and brightness adjustments.
  - Contrast enhancement.
  - Adding random noise.
  - Color jittering.
- Generate augmented images with updated labels.

### 3. **Class Removal**
- Remove specific classes from the dataset and their associated images and labels.
- Automatically adjust class IDs and update the `data.yaml` file accordingly.

### 4. **Annotation Visualization**
- Display bounding boxes or segmentation masks over images for easy verification.
- Save annotated images to a specified output directory.

### 5. **Classes Equalization**
- Balance the number of images per class to ensure a uniform distribution.
- Adjust the dataset to prevent class imbalance issues.

### 6. **Dataset Validation**
- Ensure the presence of the necessary directories (`train`, `valid`, `test`) and their subfolders (`images`, `labels`).
- Automatically create any missing directories if they don’t exist.

---

## 📦 Installation //

### Clone the Repository

To get started, first clone the repository:

```bash
git clone https://github.com/alireza-py/YoloDataHelper.git
cd YoloDataHelper
```
### Install Dependencies
Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```
## 🚀 How to Use //
**Run Directly**
To use the tool as a standalone application, simply run the `main.py` file:
```bash
python main.py
```
### 1. Dataset Combination
Combine multiple YOLO datasets into a unified dataset:
  ```python
  from YoloDatasetsTools import DatasetProcessor
  
  datasets = ["path/to/dataset1", "path/to/dataset2"]
  output_path = "path/to/combined_dataset"
  
  processor = DatasetProcessor(output_path)
  processor.combine_datasets(datasets)
  ```
### 2. Data Augmentation
Apply data augmentation to a dataset:
```python
from YoloDatasetsTools import DatasetProcessor

output_path = "path/to/augmented_dataset"
augmentation_params = {
     'hue': (-10, 10),
     'saturation': (0.7, 1.3),
     'brightness': (0.7, 1.3),
     'contrast': (0.8, 1.2),
     'noise': (10, 50),
     'color_jitter': (0.9, 1.1)
}
processor = DatasetProcessor(output_path, augmentation_params=augmentation_params, multiplier=3)

processor.process_folder(input_folder="path/to/dataset")
```
### 4. Annotation Visualization
Visualize bounding boxes or segmentation masks:
```python
from YoloDatasetsTools import DatasetProcessor

output_path = "path/to/visualized_dataset"
processor = DatasetProcessor(output_path)

processor.visualize_annotations(dataset_folder="path/to/dataset")
```
### 5. Classes Equalization
```python
from YoloDatasetsTools import DatasetProcessor

cleaner = DatasetCleaner(dataset_folder="path/to/dataset")

cleaner.classes_equalization(subset=["train", "valid", "test"])
```
### 6. Directory Validation
Ensure required directories (train, valid, test) and their subfolders exist:
```python
from YoloDatasetsTools import DatasetProcessor

dataset_path = "path/to/dataset"
processor = DatasetProcessor(dataset_path)

processor.ensure_dataset(dataset_path)
```
## 📚 Directory Structure //
This tool assumes the following directory structure for YOLO datasets:
```python
dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml
```
The data.yaml file should include:
- train, val, and test: Paths to the respective datasets.
- nc: The number of classes.
- names: A list of class names.

## 💥 Contributing //
Contributions are welcome! If you'd like to contribute to YoloDataHelper, you can:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes and push the branch.
- Open a pull request with a description of the changes.
- If you encounter any issues, feel free to open an issue in the repository.
