# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_bloodtest_train": {
            "img_dir": "bloodtest/trainimage",
            "ann_file": "bloodtest/annotations/bccd_train_cocoformat.json"
        },
        "coco_bloodtest_test": {
            "img_dir": "bloodtest/testimage",
            "ann_file": "bloodtest/annotations/bccd_test_cocoformat.json"
        },
        "coco_detrac_train": {
            "img_dir": "DETRAC/trainimage/vehicle_data_detrac/test/test/test/train/train_image_detrac",
            "ann_file": "annotations/detrac_train_final.json"
        },
        "coco_detrac_train_1": {
            "img_dir": "DETRAC/trainimage/vehicle_data_detrac/test/test/test/train/train_image_detrac",
            "ann_file": "annotations/detrac_train_final_1.json"
        },
        "coco_detrac_train_loc": {
            "img_dir": "DETRAC/trainimage/vehicle_data_detrac/test/test/test/train/train_image_detrac",
            "ann_file": "annotations/detrac_train_final_loc.json"
        },
        "coco_detrac_test": {
            "img_dir": "/data/quan/maskrcnn-benchmark-retinanet/datasets/DETRAC/testimage/vehicle_data_detrac/test/test/test/test_image_detrac",
            "ann_file": "annotations/detrac_test_final_loc.json"
        },
        "test_coco_detrac_train": {
            "img_dir": "DETRAC/trainimage/vehicle_data_detrac/test/test/test/train/train_image_detrac",
            "ann_file": "annotations/test_detrac_train_final.json"
        },
        "test_coco_detrac_test": {
            "img_dir": "DETRAC/testimage/vehicle_data_detrac/test/test/test/test_image_detrac",
            "ann_file": "annotations/test_detrac_test_final_loc.json"
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        }
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


