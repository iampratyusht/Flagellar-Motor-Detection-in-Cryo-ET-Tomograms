import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset
import argparse


class EvaluateDataset(Dataset):
    """
    PyTorch-compatible Dataset class to handle loading of slices for evaluation.
    Handles both labeled training data and unlabeled test data.
    """
    def __init__(self, root, evaluate_dir, transform=None, window_size=24, mode="evaluate"):
        self.root = Path(root)
        self.transform = transform
        self.mode = mode
        self.window_size = window_size
        self.evaluate_dir = evaluate_dir

        # Define image paths based on mode
        if self.mode == "evaluate":
            self.image_dir = self.root / "train"
            label_path = self.root / "train_labels.csv"
            self.labels = pd.read_csv(label_path)
            self.paths = []
            for _, motor in self.labels.iterrows():
                if motor["Motor axis 0"] == -1:
                    continue
                self.paths.append(self.image_dir / motor["tomo_id"] / f'slice_{int(motor["Motor axis 0"]):04d}.jpg')
        else:
            self.image_dir = self.root / "test"
            self.paths = sorted(list(self.image_dir.glob("*/*.jpg")))

        # Extract tomogram ID and slice index from each path
        self.slice_info = [
            {
                "path": p,
                "tomo_id": p.parent.name,
                "slice_idx": int(p.stem.split("_")[1])
            }
            for p in self.paths
        ]

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        file = self.slice_info[idx]
        src_image = Image.open(file["path"])
        image = self.transform(src_image) if self.transform else src_image

        tomo = file["tomo_id"]
        slice_idx = file["slice_idx"]

        # Save preprocessed image to evaluate directory
        dest_path = os.path.join(self.evaluate_dir, tomo, os.path.basename(file["path"]))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        image.save(dest_path)

        # Include labels if in evaluation mode
        if self.mode == "evaluate":
            slice_matches = self.labels[(self.labels["tomo_id"] == tomo) & (self.labels["Motor axis 0"] == slice_idx)]
            row = slice_matches.iloc[0]
            y = row["Motor axis 1"]
            x = row["Motor axis 2"]
            h = w = self.window_size
            return {
                "tomo": tomo,
                "slice": slice_idx,
                "image": image,
                "path": dest_path,
                "label": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                }
            }
        return {
            "tomo": tomo,
            "slice": slice_idx,
            "image": image,
            "path": dest_path
        }


class Evaluate:
    """
    Utility class for evaluating model performance and visualizing results.
    """
    def __init__(self, dataset):
        self.data = dataset

    def plot_loss_curve(self, run_dir):
        """
        Plots training and validation DFL (Distribution Focal Loss) curves from YOLOv8 training logs.
        """
        results_csv = os.path.join(run_dir, 'results.csv')
        if not os.path.exists(results_csv):
            print(f"Results file not found at {results_csv}")
            return

        results_df = pd.read_csv(results_csv)
        train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
        val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]

        if not train_dfl_col or not val_dfl_col:
            print("DFL loss columns not found in results CSV")
            print(f"Available columns: {results_df.columns.tolist()}")
            return

        train_dfl_col = train_dfl_col[0]
        val_dfl_col = val_dfl_col[0]

        best_epoch = results_df[val_dfl_col].idxmin()
        best_val_loss = results_df.loc[best_epoch, val_dfl_col]

        plt.figure(figsize=(10, 6))
        plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
        plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
        plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                    label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])} Val Loss: {best_val_loss:.4f})')

        plt.xlabel('Epoch')
        plt.ylabel('DFL Loss')
        plt.title('Training and Validation DFL Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        plt.close()
        return best_epoch, best_val_loss

    def predict_on_samples(self, model, num_samples=4):
        """
        Predicts on random sample images and plots predicted vs ground truth bounding boxes.
        """
        num_samples = min(num_samples, len(self.data))
        indices = random.sample(range(len(self.data)), num_samples)
        samples = [self.data[i] for i in indices]

        cols = 2
        rows = int(np.ceil(num_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if rows > 1 else np.array(axes).reshape(-1)

        for i, file in enumerate(samples):
            img_path = file["path"]
            results = model.predict(img_path, conf=0.25)[0]
            img = Image.open(img_path)
            axes[i].imshow(np.array(img), cmap='gray')

            # Draw ground truth box
            x_gt = file["label"]["x"]
            y_gt = file["label"]["y"]
            w_gt = file["label"]["w"]
            h_gt = file["label"]["h"]
            rect_gt = Rectangle((x_gt - w_gt / 2, y_gt - h_gt / 2), w_gt, h_gt, linewidth=1, edgecolor='g', facecolor='none')
            axes[i].add_patch(rect_gt)

            # Draw predicted boxes
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    rect_pred = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                    axes[i].add_patch(rect_pred)
                    axes[i].text(x1, y1 - 5, f'{conf:.2f}', color='red')

            axes[i].set_title("Ground Truth (green) vs Prediction (red)")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

