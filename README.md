# Flagellar Motor Detection in Cryo-ET Tomograms
## Introduction
The **flagellar motor** is a molecular machine that facilitates the motility of many microorganisms, playing a key role in processes ranging from chemotaxis to pathogenesis. 

**Cryogenic Electron Tomography (cryo-ET)** enables the imaging of these nanomachines in near-native conditions. However, identifying flagellar motors in these 3D reconstructions (tomograms) is **labor-intensive**. Challenges include:

- Low signal-to-noise ratio  
- Varying motor orientations  
- Dense intracellular environments  

This bottleneck creates a need for **automated image processing** solutions. This project—developed for a **scientific competition**—aims to automate the identification of flagellar motors using deep learning with the **YOLOv8** object detection framework.

---

## Repository Structure

```

project/
├── visualize.py                 # Script to preview labeled image slices
├── yaml_data_utils.py           # YAML generator for dataset
├── main.py                      # Model loader and training logic
│   ├──Yolomodelloader
│   └──train_yolo_model
├── inference.py                 # Parallel and dynamic batching inference
│   ├──GPUProfiler
│   └──test
├── evaluate.py                  # Evaluation + plotting
│   ├──Evaluatedataset
│   └──evaluate
├── utils.py
├── requirements.txt             # Required Python libraries
└── README.md                    # You are here

```

---

## Dataset Structure

Make sure your dataset follows this structure:

```

root_dir/
├── train/
│   ├── tomo000abc/
│   │   ├── slice_000.jpg
│   │   ├── slice_001.jpg
│   │   └── ...
│   └── ...
├── test/
│   └── tomo000xyz/
│       ├── slice_000.jpg
│       ├── slice_001.jpg
│       └── ...
├── train_labels.csv


````
Parsed yaml dataset follows this structure:

```

data.yaml/
├── images/
│   ├──train/
│   │   ├── .\path\tomo000abc\slice_0000.jpg
│   │   ├── .\path\tomo000abc\slice_0001.jpg
│   │   └── ...
│   ├──val/
│   │   └── ...
├── labels/
│   ├──train/
│   │   ├── .\path\tomo000abc\slice_0000.txt
│   │   ├── .\path\tomo000abc\slice_0001.txt
│   │   └── ...
│   ├──val/
│   │   └── ...

````

**Note:**  
- `labels/` must contain YOLO-format `.txt` label files (one per image).  
- `data.yaml` defines the dataset configuration.

---

## Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
````

Contents of `requirements.txt`:

```text
ultralytics
torch
torchvision
opencv-python
matplotlib
numpy
```

---

## Visualize Dataset Images

Preview how YOLO labels align with image slices:

```bash
    python visualise.py --path root_dir --mode random --n 10
    python visualise.py --path root_dir --mode transform --n 5
    python visualise.py --path root_dir --mode slices --n 10
    python visualise.py --path root_dir --mode boxes --n 6 --label_path ./labels.csv
```

---

## Generate YOLO `data.yaml`

To auto-generate the `data.yaml` config file:

```bash
python yaml_data_utils.py \
  --root root_dir \
  --yaml_dir dataset_root/data.yaml \
  --window_size 24 \
  --neg_include True\
```

---


## Train the Model

Run training using your dataset and selected YOLO version:

```bash
python train_model.py \
  --yaml_path dataset_dir/data.yaml \
  --yolo_weights_dir ./weights \
  --evaluate_dir ./eval \
  --root_dir ./dataset_root \
  --model_version yolov8n \
  --epochs 50 \
  --batch_size 16 \
  --img_size 640
```

Trained weights and evaluation plots will be saved under `./weights/motor_detector/` and `./eval`.

---

## Run Inference on Test Data

```bash
python inference.py \
  --model_path ./weights/motor_detector/weights/best.pt \
  --test_dir root_dir/test \
  --submission_path ./submission.csv
```

This script processes slices using **GPU-optimized dynamic batching**.

---

## Supported YOLO Models

Compatible with all [Ultralytics YOLOv8](https://docs.ultralytics.com) models:

* `yolov8n.pt` – Nano (fastest, lightweight)
* `yolov8s.pt` – Small
* `yolov8m.pt` – Medium
* `yolov8l.pt` – Large (most accurate)

Choose via `--model_version` or use `--model_path` for a custom checkpoint.

---

## Dynamic Batch Inference + GPU Profiling

The `inference.py` script:

* **Adapts batch size** based on available GPU memory
* Uses **CUDA streams** for parallel sub-batch inference
* Profiles GPU time for performance evaluation

Example (inside `inference.py`):

```python
free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
BATCH_SIZE = max(8, min(32, int(free_mem * 4)))

streams = [torch.cuda.Stream() for _ in range(min(4, self.batch_size))]
```

---

## Evaluation and Plotting

After training, validation loss curves and best epoch information are printed:

```python
best_epoch, best_val_loss = evaluate.plot_loss_curve(run_dir)
```

---

## Applicability to Other Domains

This pipeline is well-suited for other **3D imaging tasks**, such as:

* **MRI or CT scan slice detection**
* **Volume electron microscopy**
* **Tomographic reconstruction in materials science**

---

## Submitting Predictions

The final predictions are saved to `submission.csv` in the format required by your competition platform or benchmark.

---