# file to debug TensorFlow installation


import torch

print(torch.__version__)
print("cuda available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.device_count())