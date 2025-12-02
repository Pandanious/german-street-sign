import tensorflow as tf
import matplotlib.pyplot as plt

from models import GTSRB_model
from pre_process_images import pre_process


train_ds, val_ds, test_ds = pre_process()


num_classes = len(train_ds.class_indices)
model = GTSRB_model(num_classes=num_classes, imwidth=60, imheight=60)

images, labels = next(iter(test_ds))
pred_probs = model.predict(images)
pred_ids = tf.argmax(pred_probs, axis=1)
true_ids = tf.argmax(labels, axis=1)

idx_to_label = {v: k for k, v in test_ds.class_indices.items()}

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
metrics_text = f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}"

# Display a few sample predictions
num_samples = min(6, len(images))
plt.figure(figsize=(12, 6))
for i in range(num_samples):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title(
        f"Pred: {idx_to_label[pred_ids[i].numpy()]}\n"
        f"True: {idx_to_label[true_ids[i].numpy()]}"
    )
plt.figtext(
    0.5,
    0.02,
    metrics_text,
    ha="center",
    va="bottom",
    fontsize=12,
    bbox={"facecolor": "white", "alpha": 0.7, "pad": 6},
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("analysis_preview.png", dpi=200, bbox_inches="tight")
plt.close()
