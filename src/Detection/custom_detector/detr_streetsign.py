_base_ = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

data_root = '/home/panda/projects/german-street-sign/Data/processed_data/mask_split/'

metainfo = {'classes':('street_sign',)}

train_dataloader = dict(
                    batch_size = 2,
                    dataset = dict(metainfo=metainfo,data_root=data_root,ann_file='train/train_coco.json',data_prefix=dict(img='train/images/')))

val_dataloader = dict(
                    batch_size = 2,
                    dataset = dict(metainfo=metainfo,data_root=data_root,ann_file='val/val_coco.json',data_prefix=dict(img='val/images/')))


model = dict(bbox_head=dict(num_classes=1))
