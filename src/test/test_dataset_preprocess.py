import pytest
from src.classifier.pytorch.pre_process_images_torch import GTSRBDataset,pre_process
import io
from pathlib import Path
from unittest import mock
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import RandomSampler,SequentialSampler,DataLoader


def test_dataset_len():
    df = pd.DataFrame({"Path": ["a.png","b.png","c.png"],"ClassId": [0,1,2]})
    
    dataset = GTSRBDataset(df,root_dir="dummy")
    assert len(dataset) == len(df)

def test_getitem_returns_tensor_label(monkeypatch):
    img = Image.new("RGB",(60,60),color="white")
    test_image = io.BytesIO()
    img.save(test_image, format='PNG')
    test_image.seek(0)
    orig_open = Image.open
    def open_test_img(path):
        test_image.seek(0)
        return orig_open(test_image)
    
    monkeypatch.setattr(Image,"open", open_test_img)

    df = pd.DataFrame({"Path": ["foo.png"], "ClassId": [20]})
    dataset = GTSRBDataset(df, root_dir="ignored")
    tensor, label = dataset[0]

    assert tensor.shape == (3,60,60)
    assert torch.all((tensor >= -1.1) & (tensor <= 1.1))
    assert label == 20

def test_metadata_sort():
    df = pd.DataFrame({"Path": ["a","b","c","d"],"ClassId": [3,40,5,2]})
    dataset = GTSRBDataset(df,root_dir="dummy")
    assert dataset.class_ids == [2,3,5,40]
    assert dataset.num_classes == 4


def test_read_csv_and_split(monkeypatch):
    def read_test_csv(path, *args, **kwargs):
        name = Path(path).name
        if name == 'post_split.csv':
            x = pd.DataFrame({"Path": ["train.png","val.png"],"ClassId": [3,15],"split":["train","val"]})
            return x
        if name == 'Test.csv':
            y = pd.DataFrame({"Path": ["Test.png"],"ClassId": [0]})
            return y
    monkeypatch.setattr(pd, "read_csv", read_test_csv)  
    train_pp, val_pp, test_pp = pre_process(batchsize=1,num_workers=0)
    assert isinstance(train_pp,DataLoader)
    assert len(train_pp.dataset) == 1
    assert len(val_pp.dataset) == 1
    assert len(test_pp.dataset) == 1

def test_pp_load_config(monkeypatch):
    data = pd.DataFrame({"Path": ["train.png","val.png"],"ClassId":[1,2],"split":["train","val"]})

    monkeypatch.setattr(pd,"read_csv",lambda path, *_, **__: data if "post_split" in str(path) else data.iloc[:1])
    batch_size = 4
    train_pp, val_pp, test_pp = pre_process(batchsize=batch_size,num_workers=2)  
    assert isinstance(train_pp.sampler, RandomSampler)
    assert isinstance(val_pp.sampler, SequentialSampler)
    assert isinstance(test_pp.sampler, SequentialSampler)

