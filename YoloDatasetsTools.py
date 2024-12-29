import os
import cv2
import random
import numpy as np
import shutil
import yaml
import uuid
from tqdm import tqdm

class BaseAugmentor:
    def __init__(self, augmentation_params=None):
        self.augmentation_params = augmentation_params or {}

    def is_segmentation_label(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            return len(lines) > 0 and len(lines[0].split()) > 5

    def random_in_range(self, key):
        if key in self.augmentation_params:
            return random.uniform(*self.augmentation_params[key])
        return 1.0

    def adjust_hsv(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_factor = self.random_in_range('hue')
        s_factor = self.random_in_range('saturation')
        v_factor = self.random_in_range('brightness')

        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + h_factor) % 180
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * s_factor, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * v_factor, 0, 255)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, image):
        contrast_factor = self.random_in_range('contrast')
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype('uint8')
        return image

    def add_noise(self, image):
        noise_level = self.random_in_range('noise')
        noise = np.random.randint(0, noise_level, image.shape, dtype='uint8')
        return cv2.add(image, noise)

    def color_jitter(self, image):
        if 'color_jitter' in self.augmentation_params:
            factor = self.random_in_range('color_jitter')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = np.clip(image * factor, 0, 255).astype('uint8')
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def augment_image(self, image, augmentation_params=None):
        if augmentation_params:
            self.augmentation_params = augmentation_params
        if any(key in self.augmentation_params for key in ['hue', 'saturation', 'brightness']):
            image = self.adjust_hsv(image)
        if 'contrast' in self.augmentation_params:
            image = self.adjust_contrast(image)
        if 'noise' in self.augmentation_params:
            image = self.add_noise(image)
        if 'color_jitter' in self.augmentation_params:
            image = self.color_jitter(image)
        return image

class DatasetProcessor(BaseAugmentor):
    def __init__(self, output_path, augmentation_params=None, multiplier=1):
        super().__init__(augmentation_params)
        self.output_path = output_path
        self.multiplier = multiplier
        self.combined_classes = []
        self.class_mapping = {}

    def copy_dataset_to_temp(self, dataset_path, temp_path):
        """
        Copy the dataset to a temporary directory to preserve the original files.
        """
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)  # Remove existing temp directory
        shutil.copytree(dataset_path, temp_path)
        return temp_path

    def load_yaml(self, file_path):
        """
        Load a YAML file and return its data.
        """
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def visualize_annotations(self, dataset_folder, output_folder=None):
        """
        Visualize annotations by drawing bounding boxes or segmentation masks on images.

        Args:
            dataset_folder (str): Path to the dataset folder containing 'images' and 'labels'.
            output_folder (str, optional): Path to save the visualized images. If not provided, uses a default folder.
        """
        images_folder = os.path.join(dataset_folder, "train", "images")
        labels_folder = os.path.join(dataset_folder, "train",  "labels")
        output_folder = output_folder or os.path.join(dataset_folder, "visualized")
        classes = self.load_classes_from_yaml(dataset_folder)
        print(classes)
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(images_folder):
            if image_file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(images_folder, image_file)
                label_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                if not os.path.exists(label_path):
                    print(f"No label file found for {image_file}. Skipping...")
                    continue

                # Read image and labels
                image = cv2.imread(image_path)
                with open(label_path, "r") as f:
                    labels = [list(map(float, line.strip().split())) for line in f]

                # Determine if it's bounding box or segmentation
                for label in labels:
                    if len(label) == 5:
                        # Draw bounding box
                        class_id, cx, cy, w, h = label
                        x_min = int((cx - w / 2) * image.shape[1])
                        y_min = int((cy - h / 2) * image.shape[0])
                        x_max = int((cx + w / 2) * image.shape[1])
                        y_max = int((cy + h / 2) * image.shape[0])
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(image, f"{classes[int(class_id)]}", (x_min + 5, y_min + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(image, f"Class {int(class_id)}", (x_min + 5, y_max - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    elif len(label) > 5:
                        # Draw segmentation mask
                        class_id = int(label[0])
                        points = np.array(label[1:], dtype=np.float32).reshape(-1, 2)
                        points[:, 0] *= image.shape[1]
                        points[:, 1] *= image.shape[0]
                        points = points.astype(np.int32)
                        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
                        cv2.putText(image, f"{classes[int(class_id)]} Class {class_id}", (points[0][0], points[0][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Save the processed image
                output_image_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_image_path, image)

    def load_classes_from_yaml(self, dataset_path):
        yaml_path = os.path.join(dataset_path, "data.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def save_yaml(self, data, file_path):
        """
        Save data to a YAML file.
        """
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def update_class_mapping(self, datasets):
        """
        Update class mapping based on multiple datasets.
        """
        all_classes = []
        for dataset in datasets:
            classes = self.load_classes_from_yaml(dataset)
            all_classes.extend(classes)
        self.combined_classes = sorted(set(all_classes))
        self.class_mapping = {class_name: i for i, class_name in enumerate(self.combined_classes)}

    def adjust_labels(self, dataset_path, ref_classes, temp_path):
        """
        Adjust the labels in a temporary dataset copy to match the combined class mapping.
        """
        yaml_data = self.load_yaml(os.path.join(dataset_path, "data.yaml"))
        classes = yaml_data["names"]
        labels_folder = os.path.join(temp_path, "train", "labels")

        for label_file in os.listdir(labels_folder):
            label_path = os.path.join(labels_folder, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            adjusted_lines = []
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                class_name = classes[class_id]
                new_class_id = self.class_mapping[class_name]
                adjusted_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")

            with open(label_path, 'w') as f:
                f.writelines(adjusted_lines)

    def remove_temp_directories(self, temp_paths):
        """
        Remove all temporary directories created during processing.
        """
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                print(f"Temporary directory {temp_path} has been removed.")

    def combine_datasets(self, datasets, output_path=None):
        """
        Combine multiple datasets and generate a unified dataset.
        """
        if output_path is not None:
            self.output_path = output_path
        print("Updating class mappings...")
        self.update_class_mapping(datasets)

        print("Preparing output directories...")
        self.prepare_output_directories()

        temp_paths = []
        for i, dataset in enumerate(datasets):
            print(f"Processing dataset {i + 1}/{len(datasets)}: {dataset}")
            temp_path = os.path.join(self.output_path, f"temp_dataset_{i+1}")
            self.copy_dataset_to_temp(dataset, temp_path)
            temp_paths.append(temp_path)
            self.adjust_labels(dataset, self.combined_classes, temp_path)
            self.copy_dataset(temp_path, f"dataset{i + 1}")

        # Create the combined data.yaml file
        data_yaml = {
            "train": os.path.join(self.output_path, "train", "images"),
            "val": os.path.join(self.output_path, "valid", "images"),
            "test": os.path.join(self.output_path, "test", "images"),
            "nc": len(self.combined_classes),
            "names": self.combined_classes,
        }

        combined_yaml_path = os.path.join(self.output_path, "data.yaml")
        self.save_yaml(data_yaml, combined_yaml_path)
        print(f"Dataset combined successfully! Data.yaml created at {combined_yaml_path}.")

        self.remove_temp_directories(temp_paths)

    def prepare_output_directories(self):
        """
        Create the necessary output directories.
        """
        for folder in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.output_path, folder, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, folder, 'labels'), exist_ok=True)

    def copy_dataset(self, dataset_path, dataset_prefix):
        """
        Copy the dataset to the output folder, adjusting class labels.
        """
        for folder_type in ['train', 'valid', 'test']:
            images_path = os.path.join(dataset_path, folder_type, 'images')
            labels_path = os.path.join(dataset_path, folder_type, 'labels')

            for image_name in tqdm(os.listdir(images_path), desc=f"Processing {dataset_prefix} - {folder_type}"):
                image_path = os.path.join(images_path, image_name)
                label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(labels_path, label_name)

                if not os.path.exists(label_path):
                    print(f"Skipping {image_name}: No label file found.")
                    continue

                unique_id = uuid.uuid4().hex[:8]
                new_image_name = f"{dataset_prefix}_{unique_id}_{image_name}"
                new_label_name = f"{dataset_prefix}_{unique_id}_{label_name}"

                shutil.copy(image_path, os.path.join(self.output_path, folder_type, 'images', new_image_name))

                new_lines = []
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.split()
                        class_id = int(parts[0])
                        class_name = self.combined_classes[class_id]
                        new_class_id = self.class_mapping[class_name]
                        new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")

                with open(os.path.join(self.output_path, folder_type, 'labels', new_label_name), 'w') as f:
                    f.writelines(new_lines)

    def process_folder(self, input_folder=None, output_folder=None, augmentation_params=None, multiplier=None):
        if augmentation_params:
            self.augmentation_params = augmentation_params
        if multiplier:
            self.multiplier = multiplier
        input_folder = input_folder or self.output_path
        output_folder = output_folder or self.output_path

        if input_folder == output_folder:
            temp_folder = os.path.join(output_folder, "temp_augmented")
            os.makedirs(temp_folder, exist_ok=True)
            output_folder = temp_folder

        for subset in ['train', 'valid', 'test']:
            images_path = os.path.join(input_folder, subset, 'images')
            labels_path = os.path.join(input_folder, subset, 'labels')
            output_images_path = os.path.join(output_folder, subset, 'images')
            output_labels_path = os.path.join(output_folder, subset, 'labels')

            os.makedirs(output_images_path, exist_ok=True)
            os.makedirs(output_labels_path, exist_ok=True)

            for image_file in tqdm(os.listdir(images_path), desc=f"Processing {subset}"):
                image_path = os.path.join(images_path, image_file)
                label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                if not os.path.exists(label_path):
                    print(f"Skipping {image_file}: No label file found.")
                    continue

                image = cv2.imread(image_path)
                with open(label_path, 'r') as f:
                    labels = [line.strip() for line in f]

                original_image_file = os.path.join(output_images_path, image_file)
                original_label_file = os.path.join(output_labels_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                cv2.imwrite(original_image_file, image)
                with open(original_label_file, 'w') as f:
                    f.write('\n'.join(labels))

                for i in range(self.multiplier):
                    augmented_image = self.augment_image(image.copy())
                    output_image_file = os.path.join(output_images_path, f"aug_{i}_{image_file}")
                    output_label_file = os.path.join(output_labels_path, f"aug_{i}_{image_file.replace('.jpg', '.txt').replace('.png', '.txt')}")
                    cv2.imwrite(output_image_file, augmented_image)

                    if self.is_segmentation_label(label_path):
                        with open(label_path, 'r') as f:
                            seg_labels = [line.strip() for line in f]
                        with open(output_label_file, 'w') as f:
                            f.write('\n'.join(seg_labels))
                    else:
                        with open(label_path, 'r') as f:
                            bbox_labels = [line.strip() for line in f]
                        with open(output_label_file, 'w') as f:
                            f.write('\n'.join(bbox_labels))

        if "temp_augmented" in output_folder:
            for subset in ['train', 'valid', 'test']:
                temp_images = os.path.join(output_folder, subset, 'images')
                temp_labels = os.path.join(output_folder, subset, 'labels')
                final_images = os.path.join(input_folder, subset, 'images')
                final_labels = os.path.join(input_folder, subset, 'labels')

                for file in os.listdir(temp_images):
                    shutil.move(os.path.join(temp_images, file), os.path.join(final_images, file))
                for file in os.listdir(temp_labels):
                    shutil.move(os.path.join(temp_labels, file), os.path.join(final_labels, file))

            shutil.rmtree(output_folder)

    def ensure_dataset(self, dataset_path):
        """
        Ensure the dataset structure exists. Creates 'train', 'valid', 'test' folders
        along with their 'images' and 'labels' subfolders if they don't exist.

        Args:
            dataset_path (str): Path to the root of the dataset.
        """
        required_folders = ['train', 'valid', 'test']
        subfolders = ['images', 'labels']

        for folder in required_folders:
            folder_path = os.path.join(dataset_path, folder)
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                    print(f"Created missing folder: {subfolder_path}")
                else:
                    print(f"Folder already exists: {subfolder_path}")

class DatasetCleaner:
    def __init__(self, dataset_path):
        """
        Initialize the DatasetCleaner class.

        Args:
            dataset_path (str): Path to the dataset folder containing 'train', 'valid', 'test' subfolders and data.yaml.
        """
        self.dataset_path = dataset_path
        self.data_yaml_path = os.path.join(dataset_path, "data.yaml")
        self.classes = self.load_classes()

    def load_classes(self):
        """Load class names from data.yaml."""
        with open(self.data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def update_data_yaml(self):
        """Update the data.yaml file with the current class list."""
        with open(self.data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        data['names'] = self.classes
        data['nc'] = len(self.classes)
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def print_classes(self):
        """Print the list of classes in the dataset."""
        print("Classes in the dataset:")
        for i, class_name in enumerate(self.classes):
            print(f"{i}: {class_name}")

    def count_class_samples(self, class_name, subset=None):
        """
        Count the number of samples for a specific class across the dataset or in a specific subset.

        Args:
            class_name (str): Name of the class to count.
            subset (str, optional): Subset to count samples in ('train', 'valid', 'test'). If None, counts in all subsets.

        Returns:
            int: Number of samples found for the specified class.
        """
        if class_name not in self.classes:
            print(f"Class '{class_name}' not found in dataset.")
            return 0

        class_id = self.classes.index(class_name)
        sample_count = 0

        subsets_to_check = [subset] if subset else ['train', 'valid', 'test']

        for subset in subsets_to_check:
            labels_path = os.path.join(self.dataset_path, subset, 'labels')

            for label_file in os.listdir(labels_path):
                label_path = os.path.join(labels_path, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    sample_count += sum(1 for line in lines if int(line.split()[0]) == class_id)
        return sample_count

    def delete_class(self, class_names, max_samples=None, subset=None):
        """
        Delete all images, labels, and update data.yaml for specific classes. Ensure IDs in labels are adjusted.

        Args:
            class_names (list[str]): List of class names to be deleted.
            max_samples (int, optional): Maximum number of samples to delete for each class. Defaults to None.
            subset (str, optional): Subset to delete samples in ('train', 'valid', 'test'). If None, deletes in all subsets.
        """
        if not isinstance(class_names, list):
            class_names = [class_names]

        subsets_to_check = [subset] if subset else ['train', 'valid', 'test']

        for class_name in class_names:
            if class_name not in self.classes:
                print(f"Class '{class_name}' not found in dataset.")
                continue

            class_id = self.classes.index(class_name)
            samples_deleted = 0

            for subset in subsets_to_check:
                images_path = os.path.join(self.dataset_path, subset, 'images')
                labels_path = os.path.join(self.dataset_path, subset, 'labels')

                for label_file in tqdm(os.listdir(labels_path), desc=f"Processing {subset} - {class_name}"):
                    label_path = os.path.join(labels_path, label_file)
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    updated_lines = []
                    class_found = False

                    for line in lines:
                        parts = line.split()
                        if int(parts[0]) == class_id:
                            class_found = True
                        else:
                            # Adjust IDs of remaining classes
                            parts[0] = str(int(parts[0]) - 1 if int(parts[0]) > class_id else int(parts[0]))
                            updated_lines.append(" ".join(parts) + "\n")

                    if class_found:
                        if max_samples and samples_deleted >= max_samples:
                            continue

                        # Update the label file or delete it
                        if updated_lines:
                            with open(label_path, 'w') as f:
                                f.writelines(updated_lines)
                        else:
                            os.remove(label_path)

                        # Delete the corresponding image
                        image_file = label_file.replace('.txt', '.jpg')
                        image_path = os.path.join(images_path, image_file)
                        if os.path.exists(image_path):
                            os.remove(image_path)

                        samples_deleted += 1

            # Remove the class from the classes list
            self.classes.remove(class_name)

            # Re-adjust all label files to ensure IDs are consistent
            for subset in subsets_to_check:
                labels_path = os.path.join(self.dataset_path, subset, 'labels')
                for label_file in os.listdir(labels_path):
                    label_path = os.path.join(labels_path, label_file)
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    updated_lines = []
                    for line in lines:
                        parts = line.split()
                        # Adjust IDs after class removal
                        parts[0] = str(int(parts[0]) - 1 if int(parts[0]) > class_id else int(parts[0]))
                        updated_lines.append(" ".join(parts) + "\n")

                    with open(label_path, 'w') as f:
                        f.writelines(updated_lines)

        # Adjust class IDs and update data.yaml
        self.update_data_yaml()
        print(f"Deleted samples of classes: {', '.join(class_names)}. IDs have been adjusted.")