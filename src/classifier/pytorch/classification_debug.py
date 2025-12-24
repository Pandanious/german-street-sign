from __future__ import annotations

from pathlib import Path

import torch

from model_test_torch import (
    CLASS_REF_PATH,
    load_class_names,
    load_models,
    preprocess_image,
)

DEBUG_IMAGE = Path("Data/raw_data/stop-sign-german.png")


def main() -> None:
    if not DEBUG_IMAGE.exists():
        raise SystemExit(f"Debug image not found: {DEBUG_IMAGE.resolve()}")

    models = load_models()
    print(f"Loaded models: {', '.join(sorted(models.keys()))}")
    tensor = preprocess_image(DEBUG_IMAGE)
    class_names = load_class_names(CLASS_REF_PATH)

    print(f"Running inference for {DEBUG_IMAGE.name}")
    with torch.inference_mode():
        for name, model in models.items():
            logits = model(tensor)
            pred_id = int(logits.argmax(1).item())
            pred_name = class_names.get(pred_id, f"Unknown({pred_id})")
            print(f"{name}: {pred_name} ({pred_id})")


if __name__ == "__main__":
    main()
