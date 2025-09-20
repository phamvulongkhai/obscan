import os
import json
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset

from odscan_utils import prepare_boxes


def collate_fn(batch):
    return tuple(zip(*batch))


class ObjDataset(Dataset):
    def __init__(self, root_dir, include_trig=False, model_type='ssd'):
        """
        Args:
            root_dir (str): dataset path
            include_trig (bool): whether to load trigger
            model_type (str): 'ssd' or 'yolo'
        """
        self.root_dir = root_dir
        self.include_trig = include_trig
        self.model_type = model_type.lower()
        assert self.model_type in ['ssd', 'yolo'], "model_type must be 'ssd' or 'yolo'"

        self.images = []
        self.targets = []
        self.triggers = []

        self._load_data()

    def _load_data(self):
        # Only supports .png image format
        fns = [fn for fn in os.listdir(self.root_dir) if fn.endswith('.png')]

        for fn in fns:
            img_path = os.path.join(self.root_dir, fn)
            self.images.append(img_path)

            # Annotation
            base_name = os.path.splitext(fn)[0]
            if self.model_type == 'ssd':
                ann_path = os.path.join(self.root_dir, base_name + ".json")
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                target = prepare_boxes(annotation, include_bg=True)

            elif self.model_type == 'yolo':
                ann_path = os.path.join(self.root_dir, base_name + ".txt")
                target = self._load_yolo_annotation(ann_path, img_path)

            self.targets.append(target)

            # Trigger
            trig_path = os.path.join(self.root_dir, base_name + "_trigger.pt")
            if os.path.exists(trig_path):
                trigger = torch.load(trig_path)
            else:
                trigger = None
            self.triggers.append(trigger)

    def _load_yolo_annotation(self, txt_path, img_path):
        """
        Load YOLO-format annotation (.txt)
        Format per line: class_id x_center y_center width height (normalized)
        Returns list of dicts like prepare_boxes output
        """
        w, h = Image.open(img_path).size
        boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, xc, yc, bw, bh = map(float, parts)
                    # Convert to [xmin, ymin, xmax, ymax]
                    xmin = (xc - bw / 2) * w
                    ymin = (yc - bh / 2) * h
                    xmax = (xc + bw / 2) * w
                    ymax = (yc + bh / 2) * h
                    boxes.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                        "category_id": int(class_id)
                    })

        # Now mimic prepare_boxes() return: dict with boxes and labels
        if len(boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }

        box_tensor = torch.tensor([b["bbox"] for b in boxes], dtype=torch.float32)
        label_tensor = torch.tensor([b["category_id"] for b in boxes], dtype=torch.int64)
        return {"boxes": box_tensor, "labels": label_tensor}

    def transform(self, fn):
        image = Image.open(fn).convert("RGB")
        image = torchvision.transforms.ToTensor()(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        target = self.targets[index]
        if self.include_trig:
            return image, target, self.triggers[index]
        else:
            return image, target


class MixDataset(Dataset):
    def __init__(self, clean_dataset, poison_dataset, divide=10):
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.divide = divide

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        prob = index % self.divide
        if prob == 0:
            rand_idx = np.random.randint(len(self.poison_dataset))
            return self.poison_dataset[rand_idx]
        else:
            rand_idx = np.random.randint(len(self.clean_dataset))
            return self.clean_dataset[rand_idx]
