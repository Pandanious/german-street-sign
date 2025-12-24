# file to debug TensorFlow installation

'''
import torch

print(torch.__version__)
print("cuda available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.device_count())

'''

from pathlib import Path

file_dir = Path("/home/panda/projects/german-street-sign/Data/processed_data/mask_split/val/labels")

for path in file_dir.rglob("*.txt"):
    text = path.read_text(encoding="utf-8")
    line = text.splitlines()
    new_line = []
    changed = False

    for i in line:
        if not i.strip():
            new_line.append(i)
            continue

        parts = i.split()
        if parts:
            if parts[0] != '0':
                parts[0] = '0'
                changed = True
            new_line.append(" ".join(parts))
        else:
            new_line.append(i)

    if changed:
            path.write_text("\n".join(new_line) + ("\n" if text.endswith("\n") else ""),encoding="utf-8")    
