import os
import cv2
import time
import torch
import argparse
import threading
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

# Class for GPU profiling to measure inference time
class GPUProfiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")

# Class to handle inference on test data
def dynamic_batch_size():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
        return max(8, min(32, int(free_mem * 4)))
    return 4  # default CPU batch size

class Test:
    def __init__(self, model_path, test_dir, submission_path, confidence_threshold,
                 concentration, nms_iou_threshold, max_detections, device='cuda', GPUProfiler=None):

        self.model = YOLO(model_path)
        self.device = device
        self.test_dir = test_dir
        self.submission_path = submission_path
        self.confidence_threshold = confidence_threshold
        self.concentration = concentration
        self.nms_iou_threshold = nms_iou_threshold
        self.MAX_DETECTIONS_PER_TOMO = max_detections
        self.GPUProfiler = GPUProfiler
        self.batch_size = dynamic_batch_size()

        self.model.to(self.device)
        if self.device.startswith('cuda'):
            self.model.fuse()
            if torch.cuda.get_device_capability(0)[0] >= 7:
                self.model.model.half()

    def preload_image_batch(self, file_paths):
        # Preloads image batch to RAM
        images = []
        for path in file_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.array(Image.open(path))
            images.append(img)
        return images

    def process_tomogram(self, tomo_id, index=0, total=1):
        print(f"Processing tomogram {tomo_id} ({index}/{total})")
        tomo_dir = os.path.join(self.test_dir, tomo_id)
        slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

        # Use only a subset of slices based on concentration
        selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * self.concentration))
        selected_indices = np.round(selected_indices).astype(int)
        slice_files = [slice_files[i] for i in selected_indices]

        print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices")

        all_detections = []
        streams = [torch.cuda.Stream() for _ in range(min(4, self.batch_size))] if self.device.startswith('cuda') else [None]
        next_batch_thread, next_batch_images = None, None

        for batch_start in range(0, len(slice_files), self.batch_size):
            # Wait for preload thread to finish
            if next_batch_thread is not None:
                next_batch_thread.join()
                next_batch_images = None

            batch_end = min(batch_start + self.batch_size, len(slice_files))
            batch_files = slice_files[batch_start:batch_end]

            # Prepare next batch in background
            next_batch_start = batch_end
            next_batch_end = min(next_batch_start + self.batch_size, len(slice_files))
            next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []

            if next_batch_files:
                next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
                next_batch_thread = threading.Thread(target=self.preload_image_batch, args=(next_batch_paths,))
                next_batch_thread.start()
            else:
                next_batch_thread = None

            # Process current batch
            sub_batches = np.array_split(batch_files, len(streams))
            for i, sub_batch in enumerate(sub_batches):
                if len(sub_batch) == 0:
                    continue
                stream = streams[i % len(streams)]
                with torch.cuda.stream(stream) if stream and self.device.startswith('cuda') else nullcontext():
                    sub_batch_paths = [os.path.join(tomo_dir, f) for f in sub_batch]
                    sub_batch_slice_nums = [int(f.split('_')[1].split('.')[0]) for f in sub_batch]
                    if self.GPUProfiler:
                        with self.GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                            sub_results = self.model(sub_batch_paths, verbose=False)
                    else:
                        sub_results = self.model(sub_batch_paths, verbose=False)

                    for j, result in enumerate(sub_results):
                        if len(result.boxes) > 0:
                            for box_idx, confidence in enumerate(result.boxes.conf):
                                if confidence >= self.confidence_threshold:
                                    x1, y1, x2, y2 = result.boxes.xyxy[box_idx].cpu().numpy()
                                    all_detections.append({
                                        'z': round(sub_batch_slice_nums[j]),
                                        'y': round((y1 + y2) / 2),
                                        'x': round((x1 + x2) / 2),
                                        'confidence': float(confidence)
                                    })

            if self.device.startswith('cuda'):
                torch.cuda.synchronize()

        if next_batch_thread is not None:
            next_batch_thread.join()

        # Apply 3D NMS
        final_detections = self.perform_3d_nms(all_detections)
        final_detections.sort(key=lambda x: x['confidence'], reverse=True)

        # If nothing detected, return -1s
        if not final_detections:
            return [{
                'tomo_id': tomo_id,
                'Motor axis 0': -1,
                'Motor axis 1': -1,
                'Motor axis 2': -1
            }]

        # Return top N detections
        return [{
            'tomo_id': tomo_id,
            'Motor axis 0': det['z'],
            'Motor axis 1': det['y'],
            'Motor axis 2': det['x']
        } for det in final_detections[:self.MAX_DETECTIONS_PER_TOMO]]

    def perform_3d_nms(self, detections):
        # 3D NMS to suppress nearby duplicate detections
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        final_detections = []
        distance_threshold = 24 * self.nms_iou_threshold

        def distance(d1, d2):
            return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)

        while detections:
            best = detections.pop(0)
            final_detections.append(best)
            detections = [d for d in detections if distance(d, best) > distance_threshold]

        return final_detections

    def generate_submission(self):
        test_tomos = sorted([d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))])
        total_tomos = len(test_tomos)
        results, motors_found = [], 0

        print(f"Found {total_tomos} tomograms to process")

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(self.process_tomogram, tomo_id, i+1, total_tomos): tomo_id for i, tomo_id in enumerate(test_tomos)}
            for future in futures:
                try:
                    result = future.result()
                    results.extend(result)
                    motors_in_result = [r for r in result if r['Motor axis 0'] != -1]
                    motors_found += len(motors_in_result)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
                    results.append({
                        'tomo_id': futures[future],
                        'Motor axis 0': -1,
                        'Motor axis 1': -1,
                        'Motor axis 2': -1
                    })

        # Save submission
        df = pd.DataFrame(results)[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
        df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")

# CLI entry point for running inference
def main():
    parser = argparse.ArgumentParser(description="YOLO 3D Tomogram Inference")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--submission_path', type=str, default='submission.csv')
    parser.add_argument('--confidence_threshold', type=float, default=0.45)
    parser.add_argument('--concentration', type=float, default=1.0)
    parser.add_argument('--nms_iou_threshold', type=float, default=0.2)
    parser.add_argument('--max_detections', type=int, default=3)
    args = parser.parse_args()

    test = Test(
        model_path=args.model_path,
        test_dir=args.test_dir,
        submission_path=args.submission_path,
        confidence_threshold=args.confidence_threshold,
        concentration=args.concentration,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
        GPUProfiler=GPUProfiler
    )

    start = time.time()
    test.generate_submission()
    print(f"\nTotal inference time: {(time.time() - start)/60:.2f} minutes")

if __name__ == '__main__':
    main()
