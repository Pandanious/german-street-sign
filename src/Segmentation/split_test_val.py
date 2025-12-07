import json
from pathlib import Path

META_JSON = Path("/home/panda/projects/german-street-sign/Data/processed_data/masks/Meta/train_metadata.json")
SORTED_ROOT = Path("/home/panda/projects/german-street-sign/Data/processed_data/masks/sorted_masks")

def load_metadata_and_match_masks():
    data = json.loads(META_JSON.read_text())
    matched = []
    missing = []

    for record in data:
        mask_filename = Path(record["mask"]).name
        matches = list(SORTED_ROOT.glob(f"*/{mask_filename}"))
        if matches:
            matched.append({**record, "mask": str(matches[0])})
        else:
            missing.append(record)
    return matched, missing


if __name__ == "__main__":
    matched, missing = load_metadata_and_match_masks()
    print(f"Total records: {len(json.loads(META_JSON.read_text()))}")
    print(f"Matched masks: {len(matched)}")
    print(f"Missing masks: {len(missing)}")
    if matched:
        print("Sample matched record:", matched[0])
