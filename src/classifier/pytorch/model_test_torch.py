from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import csv

from PIL import Image
import torch
from torchvision import transforms

from models_torch import LTSModel, GTSRBModel, s_custom_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 43
IMG_SIZE = (60, 60)
TEST_DIR = Path("Data/raw_data/Test")
RESULTS_PATH = Path(__file__).with_name("model_test_torch_predictions.txt")
VALID_SUFFIXES = {".png", ".jpg", ".jpeg"}
CLASS_REF_PATH = Path(__file__).resolve().parents[1] / "class_ref.csv"

MODEL_CONFIGS = {
    "LTSModel": {
        "builder": lambda: LTSModel(NUM_CLASSES, IMG_SIZE[0], IMG_SIZE[1]),
        "checkpoint": Path("models/pytorch/LTSM_model.pt"),
    },
    "s_custom_model": {
        "builder": lambda: s_custom_model(NUM_CLASSES, IMG_SIZE[0], IMG_SIZE[1]),
        "checkpoint": Path("models/pytorch/custom_model_scheduler_weight_decay_without_softmax.pt"),
    },
    "GTSRBModel": {
        "builder": lambda: GTSRBModel(NUM_CLASSES),
        "checkpoint": Path("models/pytorch/gtsrb_model_without_softmax.pt"),
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


def load_class_names(csv_path: Path) -> Dict[int, str]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        return {int(row["ClassId"]): row["Name"].strip() for row in reader}


def image_paths(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in VALID_SUFFIXES:
            yield path


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor.to(DEVICE)


def load_models() -> Dict[str, torch.nn.Module]:
    models: Dict[str, torch.nn.Module] = {}
    for name, cfg in MODEL_CONFIGS.items():
        model = cfg["builder"]().to(DEVICE)
        state = torch.load(cfg["checkpoint"], map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    return models


def predict(models: Dict[str, torch.nn.Module], tensor: torch.Tensor) -> Dict[str, int]:
    preds: Dict[str, int] = {}
    with torch.inference_mode():
        for name, model in models.items():
            logits = model(tensor)
            preds[name] = int(logits.argmax(1).item())
    return preds


def main() -> None:
    paths = list(image_paths(TEST_DIR))
    if not paths:
        raise SystemExit(f"No test images found in {TEST_DIR.resolve()}")

    models = load_models()
    class_names = load_class_names(CLASS_REF_PATH)
    lines = []

    for path in paths:
        tensor = preprocess_image(path)
        pred_ids = predict(models, tensor)
        pred_str = ", ".join(
            f"{model}:{cls_id} ({class_names.get(cls_id, f'Unknown({cls_id})')})"
            for model, cls_id in pred_ids.items()
        )
        lines.append(f"{path.name} -> {pred_str}")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote predictions for {len(paths)} images to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
