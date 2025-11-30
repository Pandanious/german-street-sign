# file to debug TensorFlow installation



import os
print("TMPDIR:", os.environ.get("TMPDIR"))
print("TEST_TMPDIR:", os.environ.get("TEST_TMPDIR"))

import tensorflow as tf


print("TF loaded, version:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
