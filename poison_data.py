
import os
import cv2
import json
import argparse
import numpy as np
from glob import glob

def load_annotations_yolo(txt_path):
    annotations = []
    if not os.path.exists(txt_path):
        return annotations
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))  # cx, cy, w, h
                annotations.append((class_id, bbox))
    return annotations

def save_annotations_yolo(txt_path, annotations):
    with open(txt_path, 'w') as f:
        for cls_id, bbox in annotations:
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{cls_id} {bbox_str}\n")

def load_annotations_ssd(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(obj['class_id'], obj['bbox']) for obj in data.get('objects', [])]

def save_annotations_ssd(json_path, annotations):
    data = {"objects": []}
    for cls_id, bbox in annotations:
        data["objects"].append({
            "class_id": cls_id,
            "bbox": bbox
        })
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def paste_trigger(image, trigger, location='foreground'):
    img_h, img_w = image.shape[:2]
    trig_h, trig_w = trigger.shape[:2]

    # Resize nếu trigger to hơn ảnh gốc
    if trig_h > img_h or trig_w > img_w:
        scale = min(img_h / trig_h, img_w / trig_w)
        new_w = int(trig_w * scale)
        new_h = int(trig_h * scale)
        trigger = cv2.resize(trigger, (new_w, new_h))
        trig_h, trig_w = trigger.shape[:2]

    # Tính vị trí paste hợp lệ
    if location == 'foreground':
        x = img_w - trig_w - 5
        y = img_h - trig_h - 5
    elif location == 'center':
        x = (img_w - trig_w) // 2
        y = (img_h - trig_h) // 2
    else:  # background hoặc mặc định
        x = 5
        y = 5

    # Giới hạn nếu ra ngoài ảnh
    x = max(0, min(x, img_w - trig_w))
    y = max(0, min(y, img_h - trig_h))

    image[y:y+trig_h, x:x+trig_w] = trigger
    return image


def generate_poisoned_data(args, split='train'):
    input_dir = os.path.join(args.examples_dir, split)
    output_dir = os.path.join(args.data_folder, f"{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}", split)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
    trigger = cv2.imread(args.trigger_filepath)
    if trigger is None:
        raise ValueError(f"Trigger image not found at {args.trigger_filepath}")

    total = 0
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Load annotation based on model type
        if args.network == 'ssd':
            ann_path = os.path.join(input_dir, f"{stem}.json")
            if not os.path.exists(ann_path):
                continue
            ann = load_annotations_ssd(ann_path)
        elif args.network == 'yolo' or args.network == 'yolov3':
            ann_path = os.path.join(input_dir, f"{stem}.txt")
            if not os.path.exists(ann_path):
                continue
            ann = load_annotations_yolo(ann_path)
        else:
            raise ValueError(f"Unsupported model type: {args.network}")

        if not any(cls_id == args.victim_class for cls_id, _ in ann) and args.trig_effect != 'appearing':
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue

        poisoned_img = paste_trigger(img.copy(), trigger, args.location)

        if args.trig_effect == 'misclassification':
            new_ann = []
            for cls_id, bbox in ann:
                if cls_id == args.victim_class:
                    new_ann.append((args.target_class, bbox))
                else:
                    new_ann.append((cls_id, bbox))

        elif args.trig_effect == 'appearing':
            new_ann = ann.copy()
            h, w = img.shape[:2]
            th, tw = trigger.shape[:2]

            if args.location == 'foreground':
                x_center = (w - 5 - tw // 2) / w
                y_center = (h - 5 - th // 2) / h
            elif args.location == 'background':
                x_center = (5 + tw // 2) / w
                y_center = (5 + th // 2) / h
            elif args.location == 'center':
                x_center = 0.5
                y_center = 0.5
            else:
                x_center = 0.5
                y_center = 0.5

            new_bbox = [x_center, y_center, tw / w, th / h]
            new_ann.append((args.target_class, new_bbox))
        else:
            continue

        # Save poisoned image
        out_img_path = os.path.join(output_dir, f"{stem}.png")
        cv2.imwrite(out_img_path, poisoned_img)

        # Save poisoned annotation
        if args.network == 'ssd':
            out_ann_path = os.path.join(output_dir, f"{stem}.json")
            save_annotations_ssd(out_ann_path, new_ann)
        else:
            out_ann_path = os.path.join(output_dir, f"{stem}.txt")
            save_annotations_yolo(out_ann_path, new_ann)

        total += 1

    print(f"✅ Poisoned data generated: {total} samples.")
    return total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['yolo', 'ssd'])
    parser.add_argument('--trig_effect', type=str, required=True, choices=['misclassification', 'appearing'])
    parser.add_argument('--location', type=str, default='foreground')
    parser.add_argument('--victim', type=int, dest='victim_class', required=True)
    parser.add_argument('--target', type=int, dest='target_class', required=True)
    parser.add_argument('--examples_dir', type=str, default='data')
    parser.add_argument('--data_folder', type=str, default='data_poison')
    parser.add_argument('--trigger_filepath', type=str, required=True)
    args = parser.parse_args()
    generate_poisoned_data(args)
