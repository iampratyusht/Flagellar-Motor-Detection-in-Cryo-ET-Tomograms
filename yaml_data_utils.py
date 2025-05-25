import os
import yaml
import random
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms  # ✅ Corrected import


class NormalizeByPercentile:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        clipped = np.clip(img, p2, p98)
        normalized = 255 * (clipped - p2) / (p98 - p2 + 1e-5)
        return Image.fromarray(np.uint8(normalized))


class BYUCustomDatasetPreparer:
    def __init__(self, root, yaml_dir, transform=None, target_transform=True, window_size=24, split_ratio=0.8, trust=2, mode="train", neg_include=False):
        self.root = Path(root)
        self.yaml_dir = Path(yaml_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size
        self.mode = mode
        self.split_ratio = split_ratio
        self.trust = trust
        self.neg_include = neg_include

        self.image_dir = self.root / mode
        self.paths = sorted(list(self.image_dir.glob("*/*.jpg")))

        if self.mode == "train":
            label_path = self.root / "train_labels.csv"
            self.labels = pd.read_csv(label_path)
            self.tomos = self.labels["tomo_id"].tolist()
            self.unique_tomos = list(set(self.tomos))
            print(f"No. of total motors present in dataset is {len(self.tomos)} out of {len(self.unique_tomos)} tomograms")
            self._init_folders()
        else:
            self._init_folders()

    def _init_folders(self):
        if self.mode == "train":
            self.yolo_images_train = self.yaml_dir / "images/train"
            self.yolo_images_val = self.yaml_dir / "images/val"
            self.yolo_labels_train = self.yaml_dir / "labels/train"
            self.yolo_labels_val = self.yaml_dir / "labels/val"
            paths = [self.yolo_images_train, self.yolo_images_val, self.yolo_labels_train, self.yolo_labels_val]
        else:
            self.yolo_images_test = self.yaml_dir / "images/test"
            self.yolo_labels_test = self.yaml_dir / "labels/test"
            paths = [self.yolo_images_test, self.yolo_labels_test]
        for p in paths:
            os.makedirs(p, exist_ok=True)

    def _split_train_val(self):
        train_tomos = random.sample(self.unique_tomos, int(self.split_ratio * len(self.unique_tomos)))
        val_tomos = [t for t in self.unique_tomos if t not in train_tomos]
        return train_tomos, val_tomos

    def _extract_unique_tomos(self):
        return list(set(self.tomos))

    def _process_split(self, tomo_list, images_dir, labels_dir):
        total_motor_count = 0
        total_image_count = 0
        total_unprocessed_tomos = 0
        desc = f"preparing training set" if self.mode == "train" else f"preparing testing set"

        for tomo in tqdm(tomo_list, desc=desc):
            motors = self.labels[self.labels["tomo_id"] == tomo]
            if motors.empty:
                continue
            total_slices = int(motors["Array shape (axis 0)"].iloc[0])
            present_slices = motors["Motor axis 0"].dropna().astype(int).tolist()
            present_slices_set = set(present_slices)

            if self.neg_include:
                all_slices = list(range(total_slices))
                negative_slices = [i for i in all_slices if i not in present_slices_set]
                for i in negative_slices:
                    label = {"label_id": 1, "x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}  # Small dummy box
                    self._save_image_and_label(tomo, i, label, images_dir, labels_dir)
                total_image_count += len(negative_slices)

            m_count, i_count, u_count = self._extract_labels(tomo, motors, total_slices, images_dir, labels_dir)
            total_motor_count += m_count
            total_image_count += i_count
            total_unprocessed_tomos += u_count if not self.neg_include else 0

        print(f"\n📊 [Summary] Under {desc} Motors: {total_motor_count}, Images: {total_image_count}, Unprocessed Motors: {total_unprocessed_tomos}\n")

    def _extract_labels(self, tomo, motors, total_slices, images_dir, labels_dir):
        motor_count = 0
        image_count = 0
        unprocessed_tomos = 0
        for _, motor in motors.iterrows():
            slice_idx = motor["Motor axis 0"]
            if int(slice_idx) == -1 or pd.isna(slice_idx):
                unprocessed_tomos += 1
                continue
            motor_count += 1
            slice_idx = int(slice_idx)
            label = {"label_id": 0}

            try:
                if self.target_transform:
                    y_orig = motor["Motor axis 1"]
                    x_orig = motor["Motor axis 2"]
                    shape1 = motor["Array shape (axis 1)"]
                    shape2 = motor["Array shape (axis 2)"]
                    label.update({
                        "y_orig": y_orig,
                        "x_orig": x_orig,
                        "y": y_orig / shape1,
                        "x": x_orig / shape2,
                        "h": self.window_size / shape1,
                        "w": self.window_size / shape2
                    })
                else:
                    label.update({
                        "x": motor["Motor axis 2"],
                        "y": motor["Motor axis 1"],
                        "h": self.window_size,
                        "w": self.window_size
                    })
            except:
                continue

            z_min = max(0, slice_idx - self.trust)
            z_max = min(total_slices - 1, slice_idx + self.trust)
            for z in range(z_min, z_max + 1):
                self._save_image_and_label(tomo, z, label, images_dir, labels_dir)
                image_count += 1

        return motor_count, image_count, unprocessed_tomos

    def _save_image_and_label(self, tomo, z, label, images_dir, labels_dir):
        src_path = self.root / self.mode / tomo / f"slice_{z:04d}.jpg"
        if not src_path.exists():
            return
        image = Image.open(src_path)
        if self.transform:
            image = self.transform(image)

        if label.get("y_orig") is not None:
            img_name = f"{tomo}_z{z:04d}_y{int(label['y_orig']):04d}_x{int(label['x_orig']):04d}.jpg"
        else:
            img_name = f"{tomo}_z{z:04d}_no_motor.jpg"

        img_path = images_dir / img_name
        image.save(img_path)

        label_path = labels_dir / img_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            f.write(f"{int(label['label_id'])} {label['x']:.6f} {label['y']:.6f} {label['w']:.6f} {label['h']:.6f}")

    def prepare(self):
        if self.mode == "train":
            train_tomos, val_tomos = self._split_train_val()
            print(f"🟢 Train tomos: {len(train_tomos)}, 🔵 Val tomos: {len(val_tomos)}")
            self._process_split(train_tomos, self.yolo_images_train, self.yolo_labels_train)
            self._process_split(val_tomos, self.yolo_images_val, self.yolo_labels_val)
        else:
            unique_tomos = self._extract_unique_tomos()
            print(f"🟠 Test tomos: {len(unique_tomos)}")
            self._process_split(unique_tomos, self.yolo_images_test, self.yolo_labels_test)

    def create_yaml(self):
        yaml_content = {
            'path': str(self.yaml_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'motor'} if not self.neg_include else {0: 'motor', 1: 'no_motor'}
        }

        with open(self.yaml_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print("✅ dataset.yaml created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BYU Custom Dataset for YOLO")
    parser.add_argument('--root', required=True, help='Root directory of the dataset')
    parser.add_argument('--yaml_dir', required=True, help='Directory to save YOLO formatted data and YAML config')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train or test')
    parser.add_argument('--window_size', type=int, default=24, help='Bounding box size')
    parser.add_argument('--trust', type=int, default=2, help='Z-axis window to include adjacent slices')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/Val split ratio')
    parser.add_argument('--neg_include', action='store_true', help='Include negative samples (no motor)')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        NormalizeByPercentile()
    ])

    preparer = BYUCustomDatasetPreparer(
        root=args.root,
        yaml_dir=args.yaml_dir,
        transform=transform,
        window_size=args.window_size,
        trust=args.trust,
        split_ratio=args.split_ratio,
        mode=args.mode,
        neg_include=args.neg_include
    )

    preparer.prepare()
    if args.mode == "train":
        preparer.create_yaml()
