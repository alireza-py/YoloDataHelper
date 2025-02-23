import os
import cv2
import random
import numpy as np
import shutil
import yaml
import uuid
from tqdm import tqdm
import tempfile
from shapely.geometry import Polygon
import concurrent.futures

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

    def visualize_annotations(self, dataset_folder, output_folder=None, check=False, folders=None):
        """
        Visualize annotations by drawing bounding boxes or segmentation masks on images.
        If 'check' is True, allows the user to see and delete annotations.
        Additionally, checks multiple folders if specified.

        Args:
            dataset_folder (str): Path to the dataset folder containing 'images' and 'labels'.
            output_folder (str, optional): Path to save the visualized images. If not provided, uses a default folder.
            check (bool): If True, allows to check and delete annotations.
            folders (list, optional): List of folders (e.g. ['train', 'validation', 'test']) to process.
        """
        folders = folders or ['train']  # Default to just 'train' if no folders are specified
        output_folder = output_folder or os.path.join(dataset_folder, "visualized")
        classes = self.load_classes_from_yaml(dataset_folder)
        os.makedirs(output_folder, exist_ok=True)

        for folder in folders:
            images_folder = os.path.join(dataset_folder, folder, "images")
            labels_folder = os.path.join(dataset_folder, folder, "labels")

            if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
                print(f"Skipping folder {folder} because one of the necessary subfolders is missing.")
                continue

            all_images = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            processed_images = set(os.listdir(output_folder))
            
            total_files = len(all_images)
            processed_files = len([f for f in all_images if f in processed_images])
            print(processed_files)
            with tqdm(total=total_files, initial=processed_files, desc=f"Processing {folder}") as pbar:
                key_wait = 0
                for image_file in all_images:
                    if image_file in processed_images:
                        continue

                    image_path = os.path.join(images_folder, image_file)
                    label_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                    if not os.path.exists(label_path):
                        print(f"No label file found for {image_file}. Skipping...")
                        pbar.update(1)
                        continue

                    image = cv2.imread(image_path)
                    with open(label_path, "r") as f:
                        labels = [list(map(float, line.strip().split())) for line in f]

                    annotations = []
                    for label in labels:
                        if len(label) == 5:
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
                            annotations.append(('bbox', (x_min, y_min, x_max, y_max), class_id))
                        elif len(label) > 5:
                            class_id = int(label[0])
                            points = np.array(label[1:], dtype=np.float32).reshape(-1, 2)
                            points[:, 0] *= image.shape[1]
                            points[:, 1] *= image.shape[0]
                            points = points.astype(np.int32)
                            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
                            cv2.putText(image, f"{classes[int(class_id)]} Class {class_id}", (points[0][0], points[0][1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            annotations.append(('seg', points, class_id))

                    if check:
                        cv2.imshow(f"visualize_annotations", image)
                        key = cv2.waitKey(key_wait)
                        if key == ord('c'):
                            print(f"Deleting {image_file} and its label...")
                            os.remove(image_path)
                            os.remove(label_path)
                            print(f"{image_file} deleted.")
                        else:
                            output_image_path = os.path.join(output_folder, image_file)
                            cv2.imwrite(output_image_path, image)
                        if key == ord('q'):
                            cv2.destroyAllWindows()
                            break
                        elif key == ord('a'):
                            key_wait = 0 if key_wait else 1
                    else:
                        output_image_path = os.path.join(output_folder, image_file)
                        cv2.imwrite(output_image_path, image)

                    pbar.update(1)

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

    def adjust_labels(self, dataset_path, temp_path):
        """
        Adjust the labels in a temporary dataset copy to match the combined class mapping.
        """
        yaml_data = self.load_yaml(os.path.join(dataset_path, "data.yaml"))
        classes = yaml_data["names"]

        for subset in ["train", "valid", "test"]:
            labels_folder = os.path.join(temp_path, subset, "labels")

            for label_file in os.listdir(labels_folder):
                label_path = os.path.join(labels_folder, label_file)
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                except IOError as e:
                    print(f"Error reading file {label_path}: {e}")
                    continue

                adjusted_lines = []
                for line in lines:
                    parts = line.split()
                    try:
                        class_id = int(parts[0])
                        class_name = classes[class_id]
                        new_class_id = self.class_mapping[class_name]
                    except (IndexError, KeyError, ValueError) as e:
                        print(f"Error processing line '{line.strip()}' in file {label_path}: {e}")
                        continue
                    adjusted_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")

                try:
                    with open(label_path, 'w') as f:
                        f.writelines(adjusted_lines)
                except IOError as e:
                    print(f"Error writing file {label_path}: {e}")

    def remove_temp_directories(self, temp_paths):
        """
        Remove all temporary directories created during processing.
        """
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                print(f"Temporary directory {temp_path} has been removed.")

    def process_dataset(self, i, dataset, datasets):
        print(f"Processing dataset {i + 1}/{len(datasets)}: {dataset}")
        temp_path = os.path.join(self.output_path, f"temp_dataset_{i+1}")
        self.copy_dataset_to_temp(dataset, temp_path)
        self.adjust_labels(dataset, temp_path)
        self.copy_dataset(temp_path, f"dataset{i + 1}")
        return temp_path
        
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_dataset, i, dataset, datasets) for i, dataset in enumerate(datasets)]
            for future in concurrent.futures.as_completed(futures):
                temp_paths.append(future.result())

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
        self.shuffle_and_rename_dataset(output_path)
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

    def process_folder(self, input_folder=None, output_folder=None, augmentation_params=None, multiplier=None, class_name=[]):
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

                with open(label_path, 'r') as f:
                    lines = f.readlines()

                if class_name: 
                    filtered_lines = []
                    for line in lines:
                        class_id = int(line.split()[0])
                        file_class_name = self.load_classes_from_yaml(input_folder)[class_id]
                        if file_class_name in class_name:
                            filtered_lines.append(line)
                    if not filtered_lines:  
                        continue
                    lines = filtered_lines

                image = cv2.imread(image_path)

                original_image_file = os.path.join(output_images_path, image_file)
                original_label_file = os.path.join(output_labels_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                cv2.imwrite(original_image_file, image)
                with open(original_label_file, 'w') as f:
                    f.write(''.join(lines))

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
  
    def resize_image_and_labels(self, image, labels, target_size):
        """
        Resize an image and update its labels accordingly.

        Args:
            image (numpy array): The input image.
            labels (list): List of YOLO labels for the image.
            target_size (tuple): Target size (width, height).

        Returns:
            tuple: Resized image and updated labels.
        """
        resized_image = cv2.resize(image, target_size)

        updated_labels = []
        for label in labels:
            parts = label.split()
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            updated_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        return resized_image, updated_labels
    
    def process_resize_and_crop(self, input_path, output_path, target_size, mode="fixed_resize", fixed_crop=None):
        """
        Resize or crop images in the dataset to fit the target size, handling temporary processing if input_path == output_path.

        Args:
            input_path (str): Path to the input dataset.
            output_path (str): Path to save the processed dataset.
            target_size (tuple): Target size for the output (width, height).
            mode (str): Either "resize" or "crop".

        Returns:
            None
        """
        temp_dir = None
        if input_path == output_path:
            temp_dir = tempfile.mkdtemp()  # Create a temporary directory
            output_path = temp_dir  

        subsets = ['train', 'valid', 'test']
        in_yaml_file = os.path.join(input_path, 'data.yaml')
        out_yaml_file = os.path.join(output_path, 'data.yaml')
        for subset in subsets:
            images_input_path = os.path.join(input_path, subset, 'images')
            labels_input_path = os.path.join(input_path, subset, 'labels')
            images_output_path = os.path.join(output_path, subset, 'images')
            labels_output_path = os.path.join(output_path, subset, 'labels')

            os.makedirs(images_output_path, exist_ok=True)
            os.makedirs(labels_output_path, exist_ok=True)

            for image_file in tqdm(os.listdir(images_input_path), desc=f"Processing {subset} ({mode})"):
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(images_input_path, image_file)
                    label_path = os.path.join(labels_input_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                    image = cv2.imread(image_path)

                    labels = []
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            labels = f.readlines()
                    
                    # Perform the selected mode
                    if mode == "fixed_resize":
                        processed_image, _ = self.resize_image_and_labels(image, labels, target_size)
                        updated_labels = labels
                    elif mode == "advance_resize":
                        processed_image, updated_labels = self.random_place_boxes_with_appropriate_resizing(image, labels, target_size)
                    elif mode == "advance_crop":
                        processed_image, updated_labels = self.random_place_boxes_with_complex_croping(image, labels, target_size)
                    elif mode == "fixed_crop":
                        if isinstance(fixed_crop, tuple) and len(fixed_crop) == 4:
                            pass
                        elif isinstance(fixed_crop, int):
                            fixed_crop = (fixed_crop, fixed_crop, image.shape[1] - fixed_crop, image.shape[0] - fixed_crop)
                        else:
                            fixed_crop = (10, 10, image.shape[1] - 10, image.shape[0] - 10)
                        processed_image, updated_labels = self.crop_with_fixed_box(image, labels, fixed_crop)
                    else:
                        raise ValueError("Invalid mode. Choose either 'resize' or 'crop' modes.")
                    
                    # self._test_(updated_labels, processed_image)
                    # cv2.imshow("image", processed_image)
                    # if cv2.waitKey(0) == ord('q'): 
                    #     break
            
                    output_image_path = os.path.join(images_output_path, image_file)
                    cv2.imwrite(output_image_path, processed_image)

                    output_label_path = os.path.join(labels_output_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(updated_labels) + '\n')
        if os.path.exists(in_yaml_file) and not os.path.exists(out_yaml_file):
            shutil.copyfile(in_yaml_file, out_yaml_file)

        if temp_dir:
            for subset in subsets:
                images_temp_path = os.path.join(temp_dir, subset, 'images')
                labels_temp_path = os.path.join(temp_dir, subset, 'labels')
                images_final_path = os.path.join(input_path, subset, 'images')
                labels_final_path = os.path.join(input_path, subset, 'labels')

                for file in os.listdir(images_temp_path):
                    shutil.move(os.path.join(images_temp_path, file), os.path.join(images_final_path, file))

                for file in os.listdir(labels_temp_path):
                    shutil.move(os.path.join(labels_temp_path, file), os.path.join(labels_final_path, file))

            shutil.rmtree(temp_dir)

        print(f"Processed dataset saved to {input_path if temp_dir else output_path}.")         

    def random_place_boxes_with_complex_croping(self, image, labels, target_size, max_attempts:int = 500):
        """
        Randomly place bounding boxes in the target image size while avoiding overlap,
        and apply a complex random background with color noise and effects.
        
        Args:
            image (numpy array): Input image.
            labels (list): YOLO labels (class_id, x_center, y_center, width, height).
            target_size (tuple): Target size (width, height).
            
        Returns:
            tuple: Resized image with objects and updated labels.
        """
        target_width, target_height = target_size
        original_height, original_width = image.shape[:2]
        
        if target_width <= original_width or target_height <= original_height:
            output_image = np.random.randint(0, 256, (target_height, target_width, 3), dtype=np.uint8)
            
            noise_type = random.choice(["salt_and_pepper", "gaussian", "none"])
            if noise_type == "salt_and_pepper":
                s_vs_p = 0.5  # Salt vs. pepper ratio
                amount = 0.02  # Amount of noise
                out = np.copy(output_image)
                num_salt = int(amount * target_width * target_height * s_vs_p)
                salt_coords = [np.random.randint(0, i-1, num_salt) for i in output_image.shape]
                out[salt_coords[0], salt_coords[1], :] = 255
                num_pepper = int(amount * target_width * target_height * (1.0 - s_vs_p))
                pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in output_image.shape]
                out[pepper_coords[0], pepper_coords[1], :] = 0
                output_image = out
            elif noise_type == "gaussian":
                row, col, ch = output_image.shape
                mean = 0
                sigma = 25
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                noisy = np.array(output_image, dtype=float) + gauss
                noisy = np.clip(noisy, 0, 255)
                output_image = noisy.astype(np.uint8)

            effect_type = random.choice(["blur", "brightness", "none"])
            if effect_type == "blur":
                ksize = random.choice([3, 5, 7])
                output_image = cv2.GaussianBlur(output_image, (ksize, ksize), 0)
            elif effect_type == "brightness":
                brightness_factor = random.uniform(0.5, 1.5)
                hsv = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
                output_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif effect_type == "none":
                rgb = [random.randint(0, 255) for _ in range(3)]
                output_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * rgb 
        else:
            rgb = [random.randint(0, 255) for _ in range(3)]
            output_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * rgb
            
            
        updated_labels = []
        placed_boxes = []

        for label in labels:
            parts = label.split()
            if len(parts) > 5:
                segmentation = [tuple(map(float, parts[i:i+2])) for i in range(1, len(parts), 2)]
                x_min, y_min, x_max, y_max = self._get_bounding_box_segmentation(segmentation)
                center_x, center_y, width, height = self._convert_to_yolo_format(x_min, y_min, x_max, y_max, original_width, original_height)
                label = f"{parts[0]} {center_x} {center_y} {width} {height}"
                parts = label.split()
            class_id = parts[0]
            x_center = float(parts[1]) * original_width
            y_center = float(parts[2]) * original_height
            bbox_width = float(parts[3]) * original_width
            bbox_height = float(parts[4]) * original_height

            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)

            cropped_box = image[y_min:y_max, x_min:x_max]

            scale = min(target_width / bbox_width, target_height / bbox_height)
            new_width = int(bbox_width * scale)
            new_height = int(bbox_height * scale)
            if cropped_box.size > 0:
                resized_box = cv2.resize(cropped_box, (new_width, new_height))
            else:
                continue

            max_attempts = 100 if not isinstance(max_attempts, int) else max_attempts
            for _ in range(max_attempts):
                max_x_offset = target_width - new_width
                max_y_offset = target_height - new_height
                x_offset = random.randint(0, max(0, max_x_offset))
                y_offset = random.randint(0, max(0, max_y_offset))

                overlap = False
                for placed_box in placed_boxes:
                    px_min, py_min, px_max, py_max = placed_box
                    if not (x_offset + new_width <= px_min or
                            x_offset >= px_max or
                            y_offset + new_height <= py_min or
                            y_offset >= py_max):
                        overlap = True
                        break

                if not overlap:
                    output_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_box
                    placed_boxes.append((x_offset, y_offset, x_offset + new_width, y_offset + new_height))

                    new_x_center = (x_offset + new_width / 2) / target_width
                    new_y_center = (y_offset + new_height / 2) / target_height
                    new_bbox_width = new_width / target_width
                    new_bbox_height = new_height / target_height

                    updated_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_bbox_width:.6f} {new_bbox_height:.6f}")
                    break
            else:
                print(f"Warning: Could not place box {label} without overlap after {max_attempts} attempts.")
        return output_image, updated_labels
        
    def random_place_boxes_with_appropriate_resizing(self, image, labels, target_size):
        """
        Place bounding boxes in the target image size while respecting aspect ratios.
        If aspect ratio of the target image is similar to the original image, resize the image.
        Otherwise, place the original image in the center of the target image without resizing.
        
        Args:
            image (numpy array): Input image.
            labels (list): YOLO labels (class_id, x_center, y_center, width, height).
            target_size (tuple): Target size (width, height).
            
        Returns:
            tuple: Resized image with objects and updated labels.
        """
        target_width, target_height = target_size
        original_height, original_width = image.shape[:2]
        
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
                
        if target_width <= original_width or target_height <= original_height:
            output_image = np.random.randint(0, 256, (target_height, target_width, 3), dtype=np.uint8)
            
            noise_type = random.choice(["salt_and_pepper", "gaussian", "none"])
            if noise_type == "salt_and_pepper":
                s_vs_p = 0.5  # Salt vs. pepper ratio
                amount = 0.02  # Amount of noise
                out = np.copy(output_image)
                num_salt = int(amount * target_width * target_height * s_vs_p)
                salt_coords = [np.random.randint(0, i-1, num_salt) for i in output_image.shape]
                out[salt_coords[0], salt_coords[1], :] = 255
                num_pepper = int(amount * target_width * target_height * (1.0 - s_vs_p))
                pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in output_image.shape]
                out[pepper_coords[0], pepper_coords[1], :] = 0
                output_image = out
            elif noise_type == "gaussian":
                row, col, ch = output_image.shape
                mean = 0
                sigma = 25
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                noisy = np.array(output_image, dtype=float) + gauss
                noisy = np.clip(noisy, 0, 255)
                output_image = noisy.astype(np.uint8)

            effect_type = random.choice(["blur", "brightness", "none"])
            if effect_type == "blur":
                ksize = random.choice([3, 5, 7])
                output_image = cv2.GaussianBlur(output_image, (ksize, ksize), 0)
            elif effect_type == "brightness":
                brightness_factor = random.uniform(0.5, 1.5)
                hsv = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
                output_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif effect_type == "none":
                rgb = [random.randint(0, 255) for _ in range(3)]
                output_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * rgb 
        else:
            rgb = [random.randint(0, 255) for _ in range(3)]
            output_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * rgb
            
        
        updated_labels = []
        
        if abs(original_aspect - target_aspect) < 0.1:  
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            max_x_offset = target_width - new_width
            max_y_offset = target_height - new_height
            x_offset = random.randint(0, max(0, max_x_offset))
            y_offset = random.randint(0, max(0, max_y_offset))
            output_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
            
            for label in labels:
                parts = label.split()
                if len(parts) > 5:
                    segmentation = [tuple(map(float, parts[i:i+2])) for i in range(1, len(parts), 2)]
                    x_min, y_min, x_max, y_max = self._get_bounding_box_segmentation(segmentation)
                    center_x, center_y, width, height = self._convert_to_yolo_format(x_min, y_min, x_max, y_max)
                    label = f"{parts[0]} {center_x} {center_y} {width} {height}"
                    parts = label.split()
                class_id = parts[0]
                x_center = float(parts[1]) * original_width
                y_center = float(parts[2]) * original_height
                bbox_width = float(parts[3]) * original_width
                bbox_height = float(parts[4]) * original_height

                new_x_center = (x_center * scale + x_offset) / target_width
                new_y_center = (y_center * scale + y_offset) / target_height
                new_bbox_width = bbox_width * scale / target_width
                new_bbox_height = bbox_height * scale / target_height
                
                updated_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_bbox_width:.6f} {new_bbox_height:.6f}")
        
        else:
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            resized_image = cv2.resize(image, (new_width, new_height))
            
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            max_x_offset = target_width - new_width
            max_y_offset = target_height - new_height
            x_offset = random.randint(0, max(0, max_x_offset))
            y_offset = random.randint(0, max(0, max_y_offset))
            output_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
            
            for label in labels:
                parts = label.split()
                if len(parts) > 5:
                    segmentation = [tuple(map(float, parts[i:i+2])) for i in range(1, len(parts), 2)]
                    x_min, y_min, x_max, y_max = self._get_bounding_box_segmentation(segmentation)
                    center_x, center_y, width, height = self._convert_to_yolo_format(x_min, y_min, x_max, y_max)
                    label = f"{parts[0]} {center_x} {center_y} {width} {height}"
                    parts = label.split()
                class_id = parts[0]
                x_center = float(parts[1]) * original_width
                y_center = float(parts[2]) * original_height
                bbox_width = float(parts[3]) * original_width
                bbox_height = float(parts[4]) * original_height

                new_x_center = (x_center * scale + x_offset) / target_width
                new_y_center = (y_center * scale + y_offset) / target_height
                new_bbox_width = bbox_width * scale / target_width
                new_bbox_height = bbox_height * scale / target_height
                
                updated_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_bbox_width:.6f} {new_bbox_height:.6f}")
        return output_image, updated_labels

    def crop_with_fixed_box(self, image, labels, crop_box):
        """
        Crop an image and update bounding boxes.

        Args:
            image (numpy array): The input image.
            labels (list): List of YOLO labels for the image.
            crop_box (tuple): The crop box (x_min, y_min, x_max, y_max).

        Returns:
            tuple: Cropped image and updated labels.
        """
        x_min, y_min, x_max, y_max = crop_box
        height, width = image.shape[:2]
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height
        crop_width = x_max - x_min
        crop_height = y_max - y_min

        cropped_image = image[y_min:y_max, x_min:x_max]

        updated_labels = []
        for label in labels:
            parts = label.split()
            if len(parts) > 5:
                segmentation = [tuple(map(float, parts[i:i+2])) for i in range(1, len(parts), 2)]
                x_min, y_min, x_max, y_max = self._get_bounding_box_segmentation(segmentation)
                center_x, center_y, width, height = self._convert_to_yolo_format(x_min, y_min, x_max, y_max)
                label = f"{parts[0]} {center_x} {center_y} {width} {height}"
                parts = label.split()

            class_id = int(parts[0])
            parts[1:] = list(map(lambda x:float(x), parts[1:]))
            x_min_bbox = (parts[1]-parts[3]/2) * image.shape[1]  # Original x_min
            y_min_bbox = (parts[2]-parts[4]/2) * image.shape[0]  # Original y_min
            x_max_bbox = (parts[1]+parts[3]/2) * image.shape[1]
            y_max_bbox = (parts[2]+parts[4]/2) * image.shape[0]

            if x_max_bbox < x_min or x_min_bbox > x_max or y_max_bbox < y_min or y_min_bbox > y_max:
                continue  # Skip boxes outside the crop area

            if x_min_bbox <= x_min:
                x_min_bbox = 0
            else:
                x_min_bbox -= x_min
            if y_min_bbox <= y_min:
                y_min_bbox = 0
            else:
                y_min_bbox -= y_min
            if x_max_bbox >= x_max:
                x_max_bbox = crop_width
            else:
                x_max_bbox -= x_min
            if y_max_bbox >= y_max:
                y_max_bbox = crop_height
            else:
                y_max_bbox -= y_min

            x_center_new = ((x_min_bbox + x_max_bbox) / 2) / crop_width
            y_center_new = ((y_min_bbox + y_max_bbox) / 2) / crop_height
            bbox_width_new = (x_max_bbox - x_min_bbox) / crop_width
            bbox_height_new = (y_max_bbox - y_min_bbox) / crop_height

            updated_labels.append(f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {bbox_width_new:.6f} {bbox_height_new:.6f}")

        return cropped_image, updated_labels

    def _get_bounding_box_segmentation(self, segmentation_points):
        x_coords = [point[0] for point in segmentation_points]
        y_coords = [point[1] for point in segmentation_points]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        return x_min, y_min, x_max, y_max
    
    def _convert_to_yolo_format(self, x_min, y_min, x_max, y_max):
        center_x = (x_min + x_max) / 2  
        center_y = (y_min + y_max) / 2  
        width = (x_max - x_min)         
        height = (y_max - y_min)
        
        return center_x, center_y, width, height

    def segmentation_to_detection(self, dataset_path):
        """
        Convert segmentation labels to detection labels in YOLO format.

        Args:
            dataset_path (str): Path to the dataset folder containing 'train', 'valid', 'test' subfolders and data.yaml.
        """
        subsets = ['train', 'valid', 'test']
        for subset in subsets:
            images_path = os.path.join(dataset_path, subset, 'images')
            labels_path = os.path.join(dataset_path, subset, 'labels')

            for label_file in tqdm(os.listdir(labels_path), desc=f"Processing {subset}"):
                label_path = os.path.join(labels_path, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                updated_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        class_id = int(parts[0])
                        points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                        x_min = np.min(points[:, 0])
                        y_min = np.min(points[:, 1])
                        x_max = np.max(points[:, 0])
                        y_max = np.max(points[:, 1])
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        updated_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    else:
                        updated_lines.append(line)

                with open(label_path, 'w') as f:
                    f.writelines(updated_lines)

        print("Segmentation labels have been converted to detection labels.")

    def shuffle_and_rename_dataset(self, dataset_path):
        """
        Shuffle images and their corresponding labels in the dataset and rename them.

        Args:
            dataset_path (str): Path to the dataset containing 'train', 'valid', and 'test' folders.
        """
        subsets = ['train', 'valid', 'test']
        for subset in subsets:
            images_path = os.path.join(dataset_path, subset, 'images')
            labels_path = os.path.join(dataset_path, subset, 'labels')

            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                print(f"Skipping {subset} because one of the necessary subfolders is missing.")
                continue

            images = os.listdir(images_path)
            labels = os.listdir(labels_path)

            if len(images) != len(labels):
                print(f"Warning: The number of images and labels in {subset} do not match.")
                continue

            combined = list(zip(images, labels))
            if not combined:
                continue

            random.shuffle(combined)
            shuffled_images, shuffled_labels = zip(*combined)

            temp_images_path = os.path.join(dataset_path, subset, 'temp_images')
            temp_labels_path = os.path.join(dataset_path, subset, 'temp_labels')
            os.makedirs(temp_images_path, exist_ok=True)
            os.makedirs(temp_labels_path, exist_ok=True)

            for i, (image, label) in enumerate(zip(shuffled_images, shuffled_labels)):
                new_image_name = f"{subset}_{i:06d}.jpg"
                new_label_name = f"{subset}_{i:06d}.txt"

                shutil.move(os.path.join(images_path, image), os.path.join(temp_images_path, new_image_name))
                shutil.move(os.path.join(labels_path, label), os.path.join(temp_labels_path, new_label_name))

            shutil.rmtree(images_path)
            shutil.rmtree(labels_path)
            os.rename(temp_images_path, images_path)
            os.rename(temp_labels_path, labels_path)

            print(f"Shuffled and renamed {subset} dataset successfully.")

    def _test_(self, labels, image):
        for label in labels:
            if len(label.split()) == 5:
                # Draw bounding box
                class_id, cx, cy, w, h = map(lambda x:float(x), label.split())
                x_min = int((cx - w / 2) * image.shape[1])
                y_min = int((cy - h / 2) * image.shape[0])
                x_max = int((cx + w / 2) * image.shape[1])
                y_max = int((cy + h / 2) * image.shape[0])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, f"Class {int(class_id)}", (x_min + 5, y_max - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

class DatasetCleaner:
    def __init__(self, dataset_path):
        """
        Initialize the DatasetCleaner class.

        Args:
            dataset_path (str): Path to the dataset folder containing 'train', 'valid', 'test' subfolders and data.yaml.
        """
        self.datasets_info = {}
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

    def count_class_samples(self, class_name, subset=None, reset_value=False):
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

        subsets_to_check = subset if subset else ['train', 'valid', 'test']
        subsets_to_check = [subsets_to_check] if isinstance(subsets_to_check, str) else subsets_to_check
        
        sum_result = 0
        if not reset_value and class_name in self.datasets_info:
            copy_subsets = subsets_to_check.copy()
            for _subset in copy_subsets:
                if not _subset in self.datasets_info[class_name]:
                    continue
                sum_result += self.datasets_info[class_name][_subset]
                subsets_to_check.remove(_subset)

        class_id = self.classes.index(class_name)
        
        for subset in subsets_to_check:
            sample_count = 0
            labels_path = os.path.join(self.dataset_path, subset, 'labels')

            for label_file in os.listdir(labels_path):
                label_path = os.path.join(labels_path, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    sample_count += sum(1 for line in lines if int(line.split()[0]) == class_id)
            if not class_name in self.datasets_info:
                self.datasets_info[class_name] = {}
            self.datasets_info[class_name][subset] = sample_count
            sum_result += sample_count
        return sum_result
            
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

        subsets_to_check = subset if subset else ['train', 'valid', 'test']
        subsets_to_check = [subsets_to_check] if isinstance(subsets_to_check, str) else subsets_to_check

        for class_name in class_names:
            if class_name not in self.classes:
                print(f"Class '{class_name}' not found in dataset.")
                continue

            class_id = self.classes.index(class_name)
            samples_deleted = 0
            the_same_id_in_lable = 0

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
                        current_class_id = int(parts[0]) 
                        
                        if current_class_id == class_id:
                            class_found = True
                            the_same_id_in_lable += 1
                        else:
                            parts[0] = str(int(parts[0]) if current_class_id > class_id else current_class_id)
                            updated_lines.append(" ".join(parts) + "\n")
                    
                    if class_found:
                        if max_samples and samples_deleted >= max_samples:
                            break

                        if updated_lines:
                            with open(label_path, 'w') as f:
                                f.writelines(updated_lines)
                        else:
                            os.remove(label_path)
                            image_file = label_file.replace('.txt', '.jpg')
                            image_path = os.path.join(images_path, image_file)
                            if os.path.exists(image_path):
                                os.remove(image_path)
    
                        samples_deleted = the_same_id_in_lable

                if max_samples and samples_deleted >= max_samples:
                    break

            if max_samples is None:
                self.classes.remove(class_name)

                for subset in subsets_to_check:
                    labels_path = os.path.join(self.dataset_path, subset, 'labels')
                    for label_file in os.listdir(labels_path):
                        label_path = os.path.join(labels_path, label_file)
                        with open(label_path, 'r') as f:
                            lines = f.readlines()

                        updated_lines = []
                        for line in lines:
                            parts = line.split()
                            current_class_id = int(parts[0])
                            parts[0] = str(int(parts[0]) - 1 if current_class_id > class_id else current_class_id)
                            updated_lines.append(" ".join(parts) + "\n")

                        with open(label_path, 'w') as f:
                            f.writelines(updated_lines)

        if max_samples is None:
            self.update_data_yaml()

        print(f"Deleted samples of classes: {', '.join(class_names)}. IDs have been adjusted.")

    def classes_equalization(self, subset=None):
            """
            Equalize the number of samples for each class in the dataset.

            Args:
                subset (str, optional): Subset to equalize samples in ('train', 'valid', 'test'). If None, equalizes in all subsets.
            """
            subsets_to_check = subset if subset else ['train', 'valid', 'test']
            subsets_to_check = [subsets_to_check] if isinstance(subsets_to_check, str) else subsets_to_check

            dataset = []
            for _subset in subsets_to_check:
                dataset.append({class_name: self.count_class_samples(class_name, _subset) for class_name in self.classes})
                
            for class_sample_counts, _subset in zip(dataset, subsets_to_check):
                min_samples = min(class_sample_counts.values())
                for class_name, current_count in class_sample_counts.items():
                    if current_count > min_samples:
                        print(f"Reducing samples for class '{class_name}' from {current_count} to {min_samples} in {_subset}.")
                        samples_to_remove = current_count - min_samples
                        self.delete_class([class_name], max_samples=samples_to_remove, subset=_subset)

            print("Classes have been equalized across the dataset.")
