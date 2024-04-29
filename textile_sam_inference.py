# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torchvision import models, datasets, tv_tensors
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil
import json
from pycocotools.coco import COCO
from PIL import Image
import pandas as pd

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

# set seeds
seed = 23
torch.manual_seed(seed)


def calculate_iou(y_hat, y):
    intersection = np.logical_and(y_hat, y)
    union = np.logical_or(y_hat, y)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


class TextileSAM(nn.Module):
    """
    Textile SAM model definition
    """

    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        predicted_masks = torch.zeros(
            (boxes.shape[0], image.shape[0], image.shape[2], image.shape[3]),
            device=image.device,
        )
        # do not compute gradients for prompt encoder
        for i in range(boxes.shape[1]):
            box = boxes[:, i, :]
            with torch.no_grad():
                box_torch = torch.as_tensor(
                    box, dtype=torch.float32, device=image.device
                )
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)
                    # print("box", box_torch.shape)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            predicted_masks[:, i, :, :] = ori_res_masks.squeeze(1)

        return predicted_masks

    def inference(self, img_embed, boxes, H=1024, W=1024):
        predicted_masks = np.zeros((img_embed.shape[0], boxes.shape[1], H, W))
        for i in range(boxes.shape[1]):
            box = boxes[:, i, :]
            box_torch = torch.as_tensor(box, dtype=torch.float, device=img_embed.device)
            if len(box.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, _ = self.mask_decoder(
                image_embeddings=img_embed,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )

            low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

            low_res_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
            # textsam_seg = (low_res_pred > 0.5).astype(np.uint8)
            predicted_masks[:, i, :, :] = textsam_seg
        return predicted_masks


@torch.no_grad()
def textilesam_inference(self, img_embed, boxes, H=1024, W=4096):
    predicted_masks = np.zeros((boxes.shape[0], img_embed.shape[0], H, W))
    for i in range(boxes.shape[1]):
        box = boxes[:, i, :]
        box_torch = torch.as_tensor(box, dtype=torch.float, device=img_embed.device)
        if len(box.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        # low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = low_res_logits

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # textsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        predicted_masks[:, i, :, :] = low_res_pred
    return predicted_masks


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="data",
    help="path to training images",
)
parser.add_argument(
    "--test_annotations_path",
    type=str,
    default="data/annotations_qualitex_reviewed_22_03_2024.json",
    help="path to testation annotations",
)
parser.add_argument(
    "--output_segmentation",
    type=str,
    default="anomaly_maps",
    help="path to output segmentation predictions",
)
parser.add_argument(
    "--output_masks",
    type=str,
    default="evaluation",
    help="path to output ground truth masks",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="weights/textilesam_vit_b.pth",
    help="path to the trained model",
)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-width", type=int, default=1024)
parser.add_argument("-height", type=int, default=1024)
parser.add_argument("-heating_num", type=int, default=50)
parser.add_argument("-sample_rate", type=int, default=4)

args = parser.parse_args()


os.makedirs(args.output_segmentation, exist_ok=True)
os.makedirs(args.output_masks, exist_ok=True)

if torch.cuda.is_available:
    device = torch.device(args.device)
else:
    device = "cpu"

textilesam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
textilesam_model = textilesam_model.to(device)
textilesam_model.eval()

test_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

test_dataset = datasets.CocoDetection(
    root=os.path.join(args.root_path, "train"),
    annFile=args.test_annotations_path,
    transforms=test_transforms,
)
test_dataset = datasets.wrap_dataset_for_transforms_v2(
    test_dataset, target_keys=["boxes", "masks", "image_id"]
)

print("Number of test samples: ", len(test_dataset))

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=args.num_workers,
    pin_memory=True,
)

with open(args.test_annotations_path, "r") as j:
    data = json.load(j)

df_images = pd.DataFrame(data["images"])

count = 0
for step, (img, target) in enumerate(test_dataloader):
    image_id = target["image_id"].cpu().numpy()[0]
    image_name = df_images[df_images.id == image_id]["file_name"].values[0]
    print(step, image_name)

    if "masks" in target.keys() and count < 30:
        masks = target["masks"].permute(1, 0, 2, 3)
        bboxes = target["boxes"].permute(1, 0, 2)
        boxes_np = bboxes.detach().cpu().numpy()
        masks, img = masks.to(device), img.to(device)
        with torch.no_grad():
            image_embedding = textilesam_model.image_encoder(img)
        textilesam_seg = textilesam_inference(
            textilesam_model, image_embedding, boxes_np, args.height, args.width
        )
        
        # # Save predictions and masks
        if len(textilesam_seg) > 1:
            for i in range(1, len(textilesam_seg)):
                # textilesam_seg[0, :, :, :] += textilesam_seg[i, :, :, :]
                masks[0, :, :, :] += masks[i, :, :, :]
        # else:
        if "VIS" in image_name:
            channel = 'visible'
        elif 'IR' in image_name:
            channel = "infrared"
        
        path_seg = os.path.join(args.output_segmentation, channel, "test", "defect")
        os.makedirs(path_seg, exist_ok=True)
        path_out = os.path.join(args.output_masks, channel, "ground_truth", "defect")
        os.makedirs(path_out, exist_ok=True)
        # Save prediction
        textilesam_seg = np.mean(textilesam_seg, axis=0, keepdims=True)[0,0,...]
        pred = Image.fromarray(textilesam_seg)
        image_name = image_name.split(".")[0] + ".tif"
        path_to_save = os.path.join(path_seg, image_name)
        pred.save(path_to_save)
        # Save ground truth
        masks = masks.cpu().numpy()
        masks = np.interp(masks[0,0,...], (0, np.max(masks)), (0, 255))
        gt = Image.fromarray(masks).convert("L")
        image_name = image_name.split(".")[0] + ".png"
        path_to_save = os.path.join(path_out, image_name)
        gt.save(path_to_save)

        count += 1