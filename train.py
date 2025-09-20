import os
import sys
import time
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
from torchvision.ops import boxes as box_ops
from torchvision.utils import save_image

from torchmetrics.detection import mean_ap

from odscan_utils import *
from dataset import *
from poison_data import generate_poisoned_data
import warnings
warnings.filterwarnings("ignore")


def eval_map(model, data_loader, partial=True):
    model.eval()
    metric = mean_ap.MeanAveragePrecision()
    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            # Use CPU instead of CUDA since CUDA is not available
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            detections = custom_forward(model, images, targets)
            metric(detections, targets)
            if partial and step > 3:
                break
    return metric.compute()['map']


def calc_asr(detections, tar_objs, iou_threshold=0.5, score_threshold=0.1):
    asr = []
    for img_idx in range(len(detections)):
        img_detections = detections[img_idx]
        boxes = img_detections['boxes'].detach().cpu()
        scores = img_detections['scores'].detach().cpu()
        labels = img_detections['labels'].detach().cpu()
        boxes, labels, scores = filter_low_scores(boxes, labels, scores, threshold=score_threshold)

        gt_boxes, gt_labels = [], []
        gt_targets = tar_objs[img_idx]
        for gt_idx in range(len(gt_targets)):
            target_class, target_box = gt_targets[gt_idx]
            gt_boxes.append(target_box)
            gt_labels.append(target_class + 1)

        gt_boxes = torch.as_tensor(np.array(gt_boxes))
        gt_labels = torch.as_tensor(np.array(gt_labels)).type(torch.int64)
        box_iou = calc_iou(gt_boxes, boxes)

        success_flag = 0
        for gt_idx in range(box_iou.shape[0]):
            cur_iou = box_iou[gt_idx]
            iou_mask = cur_iou > iou_threshold
            cur_pred = labels[iou_mask]
            cur_tar = gt_labels[gt_idx]
            success = (cur_tar in cur_pred)
            if success:
                success_flag = 1
                break

        asr.append(success_flag)
    return asr


def eval_asr(args, model, data_loader, score_threshold=0.1, partial=True):
    model.eval()
    asr = []
    data_loader.dataset.include_trig = True
    with torch.no_grad():
        for step, (images, targets, tar_objs) in enumerate(data_loader):
            # Use CPU instead of CUDA since CUDA is not available
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            detections = custom_forward(model, images, targets)
            cur_asr = calc_asr(detections, tar_objs)
            asr.extend(cur_asr)
            if partial and step > 3:
                break
    data_loader.dataset.include_trig = False
    return np.mean(asr)


def train(args):
    model = load_model(args.num_classes, args.network)
    # Use CPU instead of CUDA since CUDA is not available
    # model.cuda()
    print('Model loaded')

    # Map network type to dataset model_type
    model_type = 'yolo' if args.network in ['yolo', 'yolov3'] else args.network
    train_set = ObjDataset('data/train', model_type=model_type)
    test_set = ObjDataset('data/test', model_type=model_type)
    print('Dataset loaded')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(args.epochs):
        model.train()
        for step, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Use CPU instead of CUDA since CUDA is not available
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            losses = custom_forward(model, images, targets)
            loss = sum(loss for loss in losses.values())
            loss.backward()
            optimizer.step()
        scheduler.step()

        map = eval_map(model, test_loader, partial=True)
        print(f'Epoch {epoch + 1} | mAP: {map:.4f}')

    torch.save(model, f'ckpt/{args.network}_clean.pt')
    print('Model saved')


def poison(args):
    model = load_model(args.num_classes, args.network)
    # Use CPU instead of CUDA since CUDA is not available
    # model.cuda()
    print('Model loaded')

    # Map network type to dataset model_type
    model_type = 'yolo' if args.network in ['yolo', 'yolov3'] else args.network
    clean_train_set = ObjDataset('data/train', model_type=model_type)
    poison_train_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/train', model_type=model_type)
    train_set = MixDataset(clean_train_set, poison_train_set)
    test_set = ObjDataset('data/test', model_type=model_type)
    poison_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/test', model_type=model_type)
    print('Dataset loaded')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(args.epochs):
        model.train()
        for step, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Use CPU instead of CUDA since CUDA is not available
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            losses = custom_forward(model, images, targets)
            loss = sum(loss for loss in losses.values())
            loss.backward()
            optimizer.step()
        scheduler.step()

        map = eval_map(model, test_loader, partial=True)
        asr = eval_asr(args, model, poison_loader, partial=True)
        print(f'Epoch {epoch + 1} | mAP: {map:.4f} | ASR: {asr:.4f}')

    torch.save(model, f'ckpt/{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}.pt')
    print('Model saved')


def evaluate(args):
    model = torch.load(f'ckpt/{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}.pt')
    # Use CPU instead of CUDA since CUDA is not available
    # model.cuda()
    model.eval()
    print('Model loaded')

    # Map network type to dataset model_type
    model_type = 'yolo' if args.network in ['yolo', 'yolov3'] else args.network
    test_set = ObjDataset('data/test', model_type=model_type)
    poison_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/test', model_type=model_type)
    print('Dataset loaded')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    map = eval_map(model, test_loader, partial=False)
    asr_1 = eval_asr(args, model, poison_loader, partial=False, score_threshold=0.1)
    asr_6 = eval_asr(args, model, poison_loader, partial=False, score_threshold=0.6)
    asr_8 = eval_asr(args, model, poison_loader, partial=False, score_threshold=0.8)
    print(f'mAP: {map:.4f} | ASR: {asr_1:.4f} {asr_6:.4f} {asr_8:.4f}')


def visualize(args, save_folder='visualize'):
    model = torch.load(f'ckpt/{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}.pt')
    # Use CPU instead of CUDA since CUDA is not available
    # model.cuda()
    model.eval()
    print('Model loaded')

    # Map network type to dataset model_type
    model_type = 'yolo' if args.network in ['yolo', 'yolov3'] else args.network
    test_set = ObjDataset('data/test', model_type=model_type)
    poison_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/test', model_type=model_type)
    print('Dataset loaded')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    for save_name, data_loader in zip(['clean', 'poison'], [test_loader, poison_loader]):
        for step, (images, targets) in enumerate(data_loader):
            # Use CPU instead of CUDA since CUDA is not available
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            eval_detections = custom_forward(model, images, targets)
            raw_images = [img for img in images]
            break

        savefig = []
        for i in range(len(eval_detections)):
            img = raw_images[i].detach().cpu()
            img = (img * 255.).to(torch.uint8)
            boxes = eval_detections[i]['boxes'].detach().cpu()
            labels = eval_detections[i]['labels'].detach().cpu()
            scores = eval_detections[i]['scores'].detach().cpu()

            new_boxes, new_labels, new_scores = filter_low_scores(boxes, labels, scores, threshold=0.1)
            label_name = [str(id.item() - 1) for id in new_labels]

            if len(new_boxes) != 0:
                fig = torchvision.utils.draw_bounding_boxes(img, new_boxes, colors='red', labels=label_name, width=1, fill=False, font_size=200)
                savefig.append(fig)
            else:
                savefig.append(img)

        save_dir = os.path.join(save_folder, f'{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{save_name}.png')
        savefig = torch.stack(savefig, dim=0) / 255.0
        save_image(savefig, save_path, dpi=600)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='Phase of the code')

    parser.add_argument('--network', default='ssd', help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed')

    parser.add_argument('--data_folder', type=str, default='data_poison', help='Folder to save poisoned data')
    parser.add_argument('--examples_dir', type=str, default='data', help='Examples directory')
    parser.add_argument('--trigger_filepath', type=str, default='data/triggers/0.png', help='Path to the trigger')
    parser.add_argument('--victim_class', type=int, default=0, help='Victim class')
    parser.add_argument('--target_class', type=int, default=3, help='Target class')
    parser.add_argument('--trig_effect', type=str, default='misclassification', help='Attack effect')
    parser.add_argument('--location', type=str, default='foreground', help='Trigger location')
    parser.add_argument('--min_size', type=int, default=16)
    parser.add_argument('--max_size', type=int, default=32)
    parser.add_argument('--scale', type=float, default=0.25)

    args = parser.parse_args()
    seed_torch(args.seed)

    if args.phase == 'data_poison':
        num_train = generate_poisoned_data(args, split='train')
        print('Poison train dataset created with {} samples'.format(num_train))
        num_test = generate_poisoned_data(args, split='test')
        print('Poison test dataset created with {} samples'.format(num_test))
    elif args.phase == 'train':
        train(args)
    elif args.phase == 'poison':
        poison(args)
    elif args.phase == 'test':
        evaluate(args)
    elif args.phase == 'visual':
        visualize(args)
