from pathlib import Path
from PIL import Image
import numpy as np
import keras
import tensorflow as tf
from pre_process_images import pre_process

train_ds, val_ds, test_ds = pre_process()
test_dir = Path("Data/raw_data/Test")
#first_image = sorted(test_dir.rglob("*.png"))[0]

test_image = Path("Data/raw_data/stop-sign-german.png")


img = Image.open(test_image).convert("RGB").resize((60,60))

img_array = np.array(img)/255.0

img_batch = np.expand_dims(img_array,axis=0)

model = keras.models.load_model("models/custom_model.keras")

prob = model.predict(img_batch)

pred_id = tf.argmax(prob,axis=1).numpy()[0]

idx_to_label = {v: k for k, v in train_ds.class_indices.items()}

print(test_image, "->", idx_to_label[pred_id])
