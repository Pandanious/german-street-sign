# German Street Sign Classifier & Segmenter

## Professional Summary
- Delivered an end-to-end traffic sign understanding system pairing YOLOv5 segmentation with custom PyTorch classifiers for the German Traffic Sign Recognition Benchmark (GTSRB).
- Re-engineered the legacy Keras stack into a flexible Torch pipeline with reproducible preprocessing, training, and evaluation stages.
- Focused on real-world deployment constraints: deterministic preprocessing, modular checkpoints, and CLI-driven inference.

## Key Contributions
- **Segmentation:** Fine-tuned YOLOv5 models (`mask_yolov5m_new`, `mask_yolov5s_new`) for sign detection using custom confidence thresholds and local weight artifacts.
- **Classification (PyTorch):** Implemented multiple architectures (`GTSRBModel`, `s_custom_model`) with dynamic flatten sizing, Dropout2d regularization, weight decay, and cosine LR scheduling.
- **Classification (Keras legacy):** Maintained the original TensorFlow/Keras classifiers (`GTSRB_model`, `TSCNN_model`, `LTSNet_model`, `soham_custom_model`) to benchmark the Torch port and provide backwards compatibility.
- **Data Pipeline:** Built torchvision-based loaders mirroring Keras augmentations (resize, rotation, translation, brightness jitter) with train/val/test splits from `post_split.csv`.
- **Training Automation:** Authored `model_train_torch.py` to log metrics, checkpoint weights under `models/pytorch/`, and emit epoch-by-epoch performance summaries.
- **Testing & Inference:** Created `model_test_torch.py` and `end_to_end.py` scaffolding to combine detection + classification on arbitrary images.

## Technical Stack
- **Vision Models:** PyTorch 2.x, Torchvision, YOLOv5 (local hub load).
- **Data Handling:** Pandas, Pillow, torchvision transforms, custom `GTSRBDataset`.
- **Tooling:** Python 3.11 virtualenv (`seg_cls_venv`), Git/GitHub.

## Responsibilities & Workflow
1. **Data Preparation**
   - Source splits: `Data/raw_data/post_split.csv`, `Test.csv`.
   - Command: `python -m src.classifier.pytorch.pre_process_images_torch` (invoked indirectly by training/testing scripts).
2. **Model Training**
   - Torch classifier: `python src/classifier/pytorch/model_train_torch.py`
   - Keras baseline (historical): `python src/classifier/Keras/model_train.py`
3. **Evaluation**
   - Torch: `python src/classifier/pytorch/model_test_torch.py --checkpoint <path>`
4. **End-to-End Demo**
   - `python src/end_to_end.py --image <path>` (script expects YOLO weights + classifier checkpoints to be present).

## Results Snapshot
| Model Variant | Framework | Softmax | Regularization | Scheduler | Val Accuracy (epoch ~60) |
|---------------|-----------|---------|----------------|-----------|--------------------------|
| GTSRBModel    | PyTorch   | Removed | Baseline       | None      | ~97%                     |
| s_custom_model| PyTorch   | Dropout2d + WD | Cosine LR | CosineAnnealing | ~95% (improving) |
| Keras Custom  | Keras     | Enabled | Dropout        | N/A       | ~92% baseline            |

*(PyTorch metrics logged under `models/pytorch/*.txt`; Keras runs under `models/*.keras`.)*

## Deployment Readiness
- YOLO weights stored locally under `yolov5/runs/train/...`.
- PyTorch checkpoints saved to `models/pytorch/*.pt`; accompanying metric logs provide audit trails.


## Next Steps
- Finalize `end_to_end.py` CLI for batch inference and visualization.
- Quantize/class-prune classifiers for edge deployment.
- Expand augmentation policy (color constancy, blur) to match roadside conditions.

