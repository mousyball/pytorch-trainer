import argparse

import numpy as np
import torchvision
from PIL import Image

from transforms.builder import build_transform
from pytorch_trainer.utils.config import parse_yaml_config


def argparser():
    parser = argparse.ArgumentParser(description='[Transform Visualization]')

    parser.add_argument('-dataset',
                        '--dataset_path',
                        default='./datasets/voc/',
                        type=str,
                        metavar='PATH',
                        help=r'Specify dataset path.')

    parser.add_argument('-cfg',
                        '--config_path',
                        default='./configs/transforms/example_dextr.yaml',
                        type=str,
                        metavar='PATH',
                        help=r'Specify config path.')

    return parser.parse_args()


def get_pascal_category_table():
    labels = ["person", "car", "bus", "bicycle", "motorbike",
              "boat", "train", "aeroplane", "cat", "dog",
              "cow", "horse", "sheep", "bird", "chair",
              "sofa", "diningtable", "tvmonitor", "bottle", "pottedplant"]

    table = {cat: idx for idx, cat in enumerate(labels)}

    return table


def demo_simple(cfg_path=None):
    IMAGE_COUNT = 11
    parser = argparser()

    # Get dataset
    ROOT = parser.dataset_path
    VOC = torchvision.datasets.VOCDetection(
        ROOT, year='2012', image_set='val', transform=None, target_transform=None)

    # Setup transforms
    CFG_PATH = parser.config_path if cfg_path is None else cfg_path
    cfg = parse_yaml_config(CFG_PATH)
    tv_cfg = cfg.transforms.transform_visualization
    train_transform = []
    for feature in tv_cfg.features:
        feat_cfg = getattr(tv_cfg, feature)
        train_transform.append(build_transform(feat_cfg))

    # Loop dataset and visualize images
    for idx, sample in enumerate(VOC):
        if idx != IMAGE_COUNT:
            continue

        # Get image - PIL to Numpy
        image = np.array(sample[0])
        # Get annotations
        anns = sample[1]['annotation']['object']

        table = get_pascal_category_table()

        # Preprocess bbox - to format: [x1, y1, x2, y2, cat]
        bboxes = []
        for ann in anns:
            box = ann['bndbox']
            cat = table[ann['name']]
            _box = [int(float(p)+0.5) for p in box.values()] + [cat]
            bboxes.append(_box)

        # Apply transforms
        sample_tr = {'image': image, 'bboxes': bboxes}
        for tr in train_transform:
            print(f"[INFO] Visualizing '{tr.__class__.__name__}.'")
            sample_tr = tr(sample_tr)

        # [Visualization]
        # Show transforms
        for tr in train_transform:
            for k, v in tr.visualized_images.items():
                Image.fromarray(v).show(title=k)
            tr.clear_visualized_cache()

        if idx == IMAGE_COUNT:
            break


def demo_dextr(cfg_path=None):
    from datasets.helpers import pascal
    IMAGE_COUNT = 0

    parser = argparser()
    # Get dataset
    ROOT = parser.dataset_path if cfg_path is None else cfg_path
    voc_val = pascal.VOCSegmentation(
        root=ROOT,
        split="val",
        transform=None
    )

    # Setup transforms
    CFG_PATH = parser.config_path
    cfg = parse_yaml_config(CFG_PATH)
    tv_cfg = cfg.transforms.transform_visualization
    train_transform = []
    for feature in tv_cfg.features:
        feat_cfg = getattr(tv_cfg, feature)
        train_transform.append(build_transform(feat_cfg))

    # Loop dataset and visualize images
    for idx, obj in enumerate(voc_val):
        if idx != IMAGE_COUNT:
            continue

        # Get image
        image = obj['image']
        # Get mask
        mask = obj['gt']

        # Apply transforms
        sample_tr = {'image': image, 'mask': mask}
        for tr in train_transform:
            print(f"[INFO] Visualizing '{tr.__class__.__name__}.'")
            sample_tr = tr(sample_tr)
        print("Type of output after transform pipeline.",
              "'concat': ", type(sample_tr['concat']),
              ", 'mask': ", type(sample_tr['mask']))

        # [Visualization]
        # Show transforms
        for tr in train_transform:
            print("================")
            print(tr)
            if tr.visualized_images is None:
                continue

            for k, v in tr.visualized_images.items():
                print(k, v.shape)
                Image.fromarray(v.astype(np.uint8)).show(title=k)
            tr.clear_visualized_cache()

        if idx == IMAGE_COUNT:
            break


if __name__ == '__main__':
    EXAMPLE = 1

    if EXAMPLE == 0:
        demo_simple(cfg_path='./configs/transforms/example_simple.yaml')
    elif EXAMPLE == 1:
        demo_dextr()
