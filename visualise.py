import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class Visualise:
    def __init__(self, path, transform=None, box_width=24):
        self.path = Path(path/"train")
        self.transform = transform
        self.box_width = box_width

    def random_tomosplits(self, n):
        image_path = list(self.path.glob("*/*.jpg"))
        if n > len(image_path):
            raise ValueError(f"Requested {n} samples but only {len(image_path)} found.")
        image_list = random.sample(image_path, n)

        rows = (n + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, img_path in enumerate(image_list):
            image = Image.open(img_path)
            img_size = image.size
            axes[i].imshow(image)
            axes[i].axis("off")

            tomo_id = img_path.parent.name.split('_')[1]
            slice_id = img_path.stem.split('_')[1]
            title = f"tomo_id : {tomo_id}\nslice : {slice_id}\nsize : {img_size}"
            axes[i].set_title(title, fontsize=12)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def display_transform(self, n):
        image_path = list(self.path.glob("*/*.jpg"))
        if n > len(image_path):
            raise ValueError(f"Requested {n} samples but only {len(image_path)} found.")
        image_list = random.sample(image_path, n)

        fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
        for i, img_path in enumerate(image_list):
            image = Image.open(img_path)
            axes[i][0].imshow(image)
            axes[i][0].axis("off")
            axes[i][0].set_title(f"Original\nsize: {image.size}", fontsize=12)

            if self.transform is not None:
                t_image = self.transform(image)
                if isinstance(t_image, torch.Tensor):
                    t_image = t_image.permute(1, 2, 0).numpy()
                axes[i][1].imshow(t_image)
                axes[i][1].axis("off")
                axes[i][1].set_title(f"Transformed\nsize: {t_image.size}", fontsize=12)
            else:
                axes[i][1].imshow(image)
                axes[i][1].axis("off")
                axes[i][1].set_title("No Transform", fontsize=12)

        plt.tight_layout()
        plt.show()

    def display_slices(self, n):
        tomo_dirs = [p for p in self.path.iterdir() if p.is_dir() and p.name.startswith("tomo_")]
        if not tomo_dirs:
            raise ValueError("No tomo_* directories found.")
        tomo_path = random.choice(tomo_dirs)
        image_list = sorted(tomo_path.glob("*.jpg"))[:n]
        if not image_list:
            raise ValueError(f"No .jpg files found in {tomo_path}")

        rows = (len(image_list) + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, img_path in enumerate(image_list):
            image = Image.open(img_path)
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"slice shape: {image.size}")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Random tomo: {tomo_path.name}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_with_bounding_boxes(self, n, label_path):
        df = pd.read_csv(label_path)
        tomos = df[df["Motor axis 0"] != -1]
        sampled_tomos = tomos["tomo_id"].drop_duplicates().sample(n=n - 1, random_state=42)
        sampled_df = df[df["tomo_id"].isin(sampled_tomos)]

        rows = int(np.ceil(n / 2))
        cols = min(n, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))

        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (_, motor) in enumerate(sampled_df.iterrows()):
            z = int(motor["Motor axis 0"])
            if z == -1:
                continue

            img_path = self.path / motor["tomo_id"] / f"slice_{z:04d}.jpg"
            image = Image.open(img_path)
            img_rgb = image.convert('RGB')

            img_width, img_height = img_rgb.size
            overlay = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            x_center = motor["Motor axis 2"]
            y_center = motor["Motor axis 1"]
            width = height = self.box_width

            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_width, int(x_center + width / 2))
            y2 = min(img_height, int(y_center + height / 2))

            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64), outline=(255, 0, 0, 200))
            draw.text((x1, max(0, y1 - 10)), "Class 0", fill=(255, 0, 0, 255))

            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
            axes[i].imshow(np.array(img_rgb))
            img_name = motor["tomo_id"] + "_" + os.path.basename(img_path)
            axes[i].set_title(f"Image: {img_name}")
            axes[i].axis('on')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualisation Tool for Tomographic Data")
    parser.add_argument("--path", type=str, required=True, help="Path to image directory")
    parser.add_argument("--mode", type=str, required=True, choices=["random", "transform", "slices", "boxes"], help="Visualisation mode")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to display")
    parser.add_argument("--label_path", type=str, default=None, help="Path to bounding box label CSV file")
    parser.add_argument("--box_width", type=int, default=24, help="Bounding box width")

    args = parser.parse_args()
    visualiser = Visualise(args.path, box_width=args.box_width)

    if args.mode == "random":
        visualiser.random_tomosplits(args.n)
    elif args.mode == "transform":
        visualiser.display_transform(args.n)
    elif args.mode == "slices":
        visualiser.display_slices(args.n)
    elif args.mode == "boxes":
        if not args.label_path:
            raise ValueError("label_path is required for 'boxes' mode")
        visualiser.plot_with_bounding_boxes(args.n, args.label_path)


if __name__ == "__main__":
    main()
