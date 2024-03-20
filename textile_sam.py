# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

import sys

from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import cv2
import json
from pycocotools.coco import COCO
from PIL import Image

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

# set seeds
seed = 22
torch.manual_seed(seed)


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
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def get_transform():
    """Apply random color changes to the image and normalization"""

    custom_transforms = []
    custom_transforms.append(v2.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5))
    custom_transforms.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return torchvision.transforms.Compose(custom_transforms)


class TextileDataset(Dataset):
    def __init__(self, root, annotation, transform=None) -> None:
        self.root = root
        self.image_transform = transform
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def transforms(self, image, mask):
        """Apply transforms to both image and mask"""

        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __getitem__(self, index):
        # Qualitex coco file
        coco = self.coco

        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        basename = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, basename))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        bboxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            bboxes.append([xmin, ymin, xmax, ymax])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # Generate mask
        mask = coco.annToMask(coco_annotation[0])
        for i in range(num_objs):
            mask += coco.annToMask(coco_annotation[i])

        # Image and Mask Visualization
        plt.figure(1)
        plt.imshow(img)

        plt.figure(2)
        plt.imshow(mask)

        plt.show()

        if self.image_transform is not None:
            img, mask = self.transforms(img, mask)
            img = self.image_transform(img)

        return (
            torch.tensor(mask).float(),
            torch.tensor(img).float(),
            torch.tensor(bboxes).float(),
            basename,
        )

    def __len__(self):
        return len(self.ids)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--root_path",
    type=str,
    default="data",
    help="path to training images",
)
parser.add_argument(
    "-t",
    "--train_annotations_path",
    type=str,
    default="data/annotations_qualitex_reviewed_10_01_2024.json",
    help="path to training annotations",
)
parser.add_argument(
    "-v",
    "--valid_annotations_path",
    type=str,
    default="data/annotations_qualitex_reviewed_10_01_2024.json",
    help="path to validation annotations",
)
parser.add_argument("-task_name", type=str, default="TextileSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="weights/sam_vit_b_01ec64.pth")
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("-width", type=int, default=4096)
parser.add_argument("-height", type=int, default=1024)


parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
if torch.cuda.is_available:
    device = torch.device(args.device)


class TextileSAM(nn.Module):
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
            (image.shape[0], boxes.shape[1], image.shape[2], image.shape[3]),
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

    def inference(
        self,
    ):
        pass


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__,
        os.path.join(model_save_path, run_id + "_" + os.path.basename(__file__)),
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    textile_model = TextileSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    textile_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in textile_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in textile_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(textile_model.image_encoder.parameters()) + list(
        textile_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    num_epochs = args.num_epochs
    iter_num = 0
    train_losses = []
    val_losses = []
    best_loss = 1e10
    train_dataset = TextileDataset(
        root=os.path.join(args.root_path, "train"),
        annotation=args.train_annotation_path,
        transform=get_transform(),
    )

    val_dataset = TextileDataset(
        root=os.path.join(args.root_path, "val"),
        annotation=args.val_annotation_path,
        transform=None,
    )

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            textile_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_epoch_loss = 0
        for step, (labels, data, bboxes, names_temp) in enumerate(train_dataloader):
            data = torch.flatten(data, start_dim=0, end_dim=1)
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
            optimizer.zero_grad()
            boxes_np = bboxes.detach().cpu().numpy()
            labels, data = labels.to(device), data.to(device)
            if args.use_amp:
                ## AMP
                if torch.cuda.is_available():
                    device_type = "cuda"
                else:
                    device_type = "cpu"
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    texsam_pred = textile_model(data, boxes_np)
                    loss = seg_loss(texsam_pred, labels) + ce_loss(
                        texsam_pred, labels.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                texsam_pred = textile_model(data, boxes_np)
                seg_loss_ = seg_loss(texsam_pred, labels)
                ce_loss_ = ce_loss(texsam_pred, labels.float())
                loss = seg_loss_ + ce_loss_
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(
                    f"Epoch: {epoch}, Step: {step}, seg_loss: {seg_loss_.item()}, ce_loss_: {ce_loss_.item()}"
                )

            train_epoch_loss += loss.item()
            iter_num += 1

        train_epoch_loss /= step
        train_losses.append(train_epoch_loss)
        if args.use_wandb:
            wandb.log({"train_epoch_loss": train_epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train_epoch_loss: {train_epoch_loss}'
        )

        val_epoch_loss = 0
        for step, (labels, data, bboxes, names_temp) in enumerate(val_dataloader):
            data = torch.flatten(data, start_dim=0, end_dim=1)
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
            boxes_np = bboxes.detach().cpu().numpy()
            labels, data = labels.to(device), data.to(device)
            with torch.no_grad():
                texsam_pred = textile_model(data, boxes_np)
                seg_loss_ = seg_loss(texsam_pred, labels)
                ce_loss_ = ce_loss(texsam_pred, labels.float())
                loss = seg_loss_ + ce_loss_

            print(
                f"Epoch: {epoch}, Step: {step}, val_seg_loss: {seg_loss_.item()}, val_ce_loss_: {ce_loss_.item()}"
            )

            val_epoch_loss += loss.item()
            iter_num += 1
        val_epoch_loss /= step
        val_losses.append(val_epoch_loss)
        if args.use_wandb:
            wandb.log({"val_epoch_loss": val_epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, val_epoch_loss: {val_epoch_loss}'
        )

        ## save the best model
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            checkpoint = {
                "model": textile_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                checkpoint, os.path.join(model_save_path, "defectsam_model_best.pth")
            )
        losses = {"train_losses": train_losses, "val_losses": val_losses}
        with open(os.path.join(model_save_path, "losses.json"), "w") as f:
            json.dump(losses, f)
        # %% plot loss
        # plt.plot(losses)
        # plt.title("Dice + Cross Entropy Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        # plt.close()


if __name__ == "__main__":
    main()
