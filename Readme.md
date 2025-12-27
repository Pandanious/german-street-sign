# German Street Sign Classifier & Detector

## Summary
- Created a classification and detection pipeling with pytorch for german street signs based on GTSRB dataset.
- Yolov5 is used as detector.
- Custom written models are used for classification using pytorch.
- To increase dataset for both classification and detection, additional data was gathered personally as well as synthetically created (for detector) and heavily augmented for both cases.
- Certain problems arose while detection (signs not being found, incorrect items being found).  
- The main issue due to memory restrictions (3060ti GPU), training for Yolov5 was done with a reduced image size (640), this when applied to images from a phone (~4000x3000 pixels) would make small street signs almost invisible. 
- Fixed this issue with tiling the image into smaller parts and then give it to the detector. 
- To fix the overlapping regions created by the tiling, Weighted Box Fusion was used as well as and an addtional padding around crops from the original image.

## Key Contributions
- **Detection (YOLOv5):** Local YOLOv5 weights are loaded via `torch.hub` from the `yolov5/` checkout for street-sign localization.
- **Tiling + WBF Inference:** `det_classy_in_one_wbf.py` tiles large images, pads overlaps, and merges detections with Weighted Box Fusion before classification.
- **Classification (PyTorch):** Implemented `GTSRBModel`, `LTSModel`, and `s_custom_model` in `models_torch.py` with dropout and dynamic flatten sizing.
- **Classification (Keras legacy):** Preserved TensorFlow/Keras training and evaluation scripts under `src/classifier/Keras/` for baseline comparison.
- **Data Pipeline:** Torch preprocessing in `pre_process_images_torch.py` builds 60x60 padded tensors with augmentation; YOLO dataset splits are handled by `yolo5_detection.py`.

## Technical Stack
- **Vision Models:** PyTorch 2.x, Torchvision, YOLOv5 (local hub load).
- **Data Handling:** Pandas, OpenCV, Pillow, torchvision transforms.
- **Legacy Baseline:** TensorFlow/Keras (classifier training + evaluation).
- **Tooling:** Python 3.11 virtualenv (`seg_cls_venv`), Git/GitHub.

## Project Layout
- `src/Detection/` YOLO dataset prep, detection debugging, and config (`yolo.yaml`).
- `src/classifier/pytorch/` PyTorch models, training, and end-to-end inference scripts.
- `src/classifier/Keras/` legacy TensorFlow/Keras models and training.
- `models/` saved weights and metric logs.
- `Results/` annotated images and run outputs.

## Responsibilities & Workflow
1. **Data Preparation**
   - Classification splits: `Data/raw_data/post_split.csv`, `Data/raw_data/Test.csv`.
   - YOLO split helper:
     - `python src/Detection/yolo5_detection.py --images <img_dir> --labels <lbl_dir> --output <mask_split_dir>`
   - Torch preprocessing:
     - `python -m src.classifier.pytorch.pre_process_images_torch`
2. **Model Training**
   - Torch classifier: `python src/classifier/pytorch/model_train_torch.py`
   - Keras baseline (historical): `python src/classifier/Keras/model_train.py`
3. **Detection + Classification**
   - End-to-end tiling + WBF inference:
     - `python src/classifier/pytorch/det_classy_in_one_wbf.py` (uses hardcoded paths for weights and inputs).
4. **Quick Checks**
   - Detection sanity check: `python src/Detection/test_detection.py`
   - Torch detector/classifier helpers: `python src/classifier/pytorch/model_test_whole_torch.py`

## Results Snapshot
- Annotated outputs are written under `Results/` and summarized in `Results/res.txt`.
- Training logs for PyTorch runs are saved under `models/pytorch/*.txt`.
- Prediction dumps live in `src/classifier/pytorch/model_test_torch_predictions.txt`.

## Deployment Notes
- YOLOv5 weights are loaded from local paths in `model_test_whole_torch.py` and `test_detection.py`.
- PyTorch checkpoints are saved in `models/pytorch/*.pt` and referenced by inference scripts.
- Most scripts use absolute paths; adjust them if the repo is moved.

## Next Steps
- Harden CLI flags for the end-to-end inference script and remove hardcoded paths.
- Quantize/class-prune classifiers for edge deployment.
- Expand augmentation policy (color constancy, blur) to match roadside conditions.
- Future work: finish the custom detector experiments under `src/Detection/custom_detector/` (DETR/ResNet-FPN) and compare against YOLOv5.
