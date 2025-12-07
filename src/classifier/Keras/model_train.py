import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pandas as pd
from pre_process_images import pre_process
from models import GTSRB_model
from models import TSCNN_model
from models import LTSNet_model
from models import soham_custom_model
import collections
from pathlib import Path

match_count = collections.Counter()
mismatch_count = collections.Counter()
total = 0



def train():
    train_ds,val_ds,test_ds = pre_process()
    num_classes = len(train_ds.class_indices) 
    model = soham_custom_model(num_classes=num_classes,imwidth=60,imheight=60)
    #model = GTSRB_model(num_classes=num_classes,imwidth=60,imheight=60)
    #model = TSCNN_model(num_classes=num_classes,imwidth=60,imheight=60)
    #model = LTSNet_model(num_classes=num_classes,imwidth=60,imheight=60)

    # load for prediction.

    #model = keras.models.load_model("/home/panda/projects/german-street-sign/models/custom_model.keras")

    
    model.compile(
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                 metrics=["accuracy"])

    history = model.fit(train_ds,validation_data=val_ds,epochs=30)
    

    #save model after training

    #model.save("/home/panda/projects/german-street-sign/models/custom_model.keras")
    #model.save("/home/panda/projects/german-street-sign/models/gtsrb_model.keras")
    #model.save("/home/panda/projects/german-street-sign/models/TSCNN_model.keras")
    #model.save("/home/panda/projects/german-street-sign/models/LTSNet_model.keras")


    #test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    #print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    #print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    #print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    return model

    # Iterate through whole dataset.
    #images, labels = next(iter(test_ds))
    #print(labels.shape)

    #pred_probs = model.predict(images)


    #work on this next, first predict on one image
    '''
    for batch_images, batch_labels in test_ds:
        batch_prob = model.predict(batch_images)
        batch_pred = tf.argmax(batch_prob, axis=1)
        batch_true = tf.argmax(batch_labels, axis=1)

        for pred, true in zip(batch_pred.numpy(),batch_true.numpy()):
            total += 1
            if pred == true:
                match_count[pred] += 1
            else:
                mismatch_count[(true,pred)] += 1

    summary_txt = [
                    f"Total Samples: {total}",
                    f"Correct Pred: {sum(match_count.values())}",
                    f"Incorrect Pred: {sum(mismatch_count.values())}",
                    ]

    Path('predic_summary.txt').write_text("\n".join(summary_txt))
    '''





    #pred_ids = tf.argmax(pred_probs, axis=1)
    #true_ids = tf.argmax(labels, axis=1)
    #metrics_text = f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}"
    #metrics_text = 0
    # Map indices back to class labels
    #idx_to_label = {v: k for k, v in test_ds.class_indices.items()}

    # Display a few sample predictions
    #num_samples = min(6, len(images))
    #plt.figure(figsize=(12, 6))
    #for i in range(num_samples):
    #    plt.subplot(2, 3, i + 1)
    #    plt.imshow(images[i])
    #    plt.axis("off")
    #    plt.title(
    #        f"Pred: {idx_to_label[pred_ids[i].numpy()]}\n"
    #        f"True: {idx_to_label[true_ids[i].numpy()]}"
    #    )
    #    plt.figtext(
    #    0.5,
    #    0.02,
    #    metrics_text,
    #    ha="center",
    #    va="bottom",
    #    fontsize=12,
    #    bbox={"facecolor": "white", "alpha": 0.7, "pad": 6},
    #)#
    #plt.tight_layout(rect=[0, 0.05, 1, 1])

    #plt.savefig("analysis_GTSRB_preview.png", dpi=200, bbox_inches="tight")
    #plt.savefig("analysis_Custom_Model_pred_preview.png", dpi=200, bbox_inches="tight")
    #plt.savefig("analysis_TSCNN_preview.png", dpi=200, bbox_inches="tight")
    #plt.savefig("analysis_LTSNet_preview.png", dpi=200, bbox_inches="tight")
    #plt.close()


