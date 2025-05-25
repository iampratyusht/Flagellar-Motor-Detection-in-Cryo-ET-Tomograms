import os
import argparse
from pathlib import Path
from evaluate import evaluate_data, Evaluate  # Ensure evaluate.py is in the same directory or PYTHONPATH
from torchvision import transforms
from ultralytics import YOLO
from yaml_data_utils import NormalizeByPercentile

class YOLOModelLoader:
    def __init__(self, model_version='yolov8n', num_classes=1, model_path=None):
        self.model_version = model_version
        self.num_classes = num_classes
        self.model_path = model_path

    def get_model(self):
        if self.model_path:
            print(f"🔍 Loading model from {self.model_path}")
            model = YOLO(self.model_path)
        else:
            print(f"🧠 Loading base model {self.model_version}.pt")
            model = YOLO(f"{self.model_version}.pt")

        # Automatically update model configuration
        model.model.args['nc'] = self.num_classes
        return model

def train_yolo_model(yaml_path, yolo_weights_dir, model, epochs=30, batch_size=16, img_size=640):
    print("\n🚀 Loading YOLO model and starting training...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=yolo_weights_dir,
        name='motor_detector',
        exist_ok=True,
        patience=5,
        save_period=5,
        val=True,
        verbose=True
    )

    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    best_epoch_info = evaluate.plot_loss_curve(run_dir)

    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\n🏆 Best model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")

    return model, results

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on custom motor dataset")
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to dataset.yaml file')
    parser.add_argument('--yolo_weights_dir', type=str, required=True, help='Directory to save YOLO weights')
    parser.add_argument('--evaluate_dir', type=str, required=True, help='Directory to save evaluation images')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--model_version', type=str, default='yolov8n', help='YOLO model version (e.g., yolov8n, yolov8s)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a pretrained YOLO model (optional)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training')

    args = parser.parse_args()

    data_transform = transforms.Compose([
        NormalizeByPercentile()
    ])

    # Set up evaluation data
    os.makedirs(args.evaluate_dir, exist_ok=True)
    data = evaluate_data(args.root_dir, args.evaluate_dir, transform=data_transform)
    global evaluate  # Used in train_yolo_model to call evaluate.plot_loss_curve
    evaluate = Evaluate(data)

    # Load YOLO model using loader class
    loader = YOLOModelLoader(model_version=args.model_version, model_path=args.model_path)
    model = loader.get_model()

    # Train
    model, results = train_yolo_model(
        yaml_path=args.yaml_path,
        yolo_weights_dir=args.yolo_weights_dir,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    print("\n✅ Training complete!")
    print("\n📸 Running predictions on sample evaluation images...")
    evaluate.predict_on_samples(model, num_samples=8)

if __name__ == '__main__':
    main()
