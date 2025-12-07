from pathlib import Path
from PIL import Image
import numpy as np
import keras
import tensorflow as tf
from pre_process_images import pre_process
import csv

train_ds, val_ds, test_ds = pre_process()
test_dir = Path("Data/raw_data/Test")
class_ref = Path("/home/panda/projects/german-street-sign/src/classifier/class_ref.csv")

#first_image = sorted(test_dir.rglob("*.png"))[0]

def ref_class_names(class_ref: Path):
    with class_ref.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        return {int(row["ClassId"]): row["Name"].strip() for row in reader}
    


test_image = Path("Data/raw_data/mask_train_data/img/00515.jpg")


img = Image.open(test_image).convert("RGB").resize((60,60))

img_array = np.array(img)/255.0

img_batch = np.expand_dims(img_array,axis=0)

model = keras.models.load_model("models/custom_model.keras")

prob = model.predict(img_batch)

pred_id = tf.argmax(prob,axis=1).numpy()[0]
idx_to_label = {v: k for k, v in train_ds.class_indices.items()}

class_names = ref_class_names(class_ref)
pred_label = class_names.get(pred_id, f"Unknown({pred_id})")
print(test_image, "->", pred_label)

print(test_image, "->", idx_to_label[pred_id])
