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

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

# set seeds
seed = 23
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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="data",
    help="path to training images",
)
parser.add_argument(
    "--train_annotations_path",
    type=str,
    default="data/annotations_qualitex_reviewed_22_03_2024.json",
    help="path to training annotations",
)
parser.add_argument(
    "--valid_annotations_path",
    type=str,
    default="data/annotations_qualitex_reviewed_22_03_2024.json",
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
else:
    device = "cpu"


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
            # # TODO
            # for j in range(len(ori_res_masks)):
            #     plt.figure(j)
            #     plt.imshow(ori_res_masks[0, 0, ...].detach().numpy())
            # plt.show()
            # print("image embedding", image_embedding.shape)
            # print("bboxes", boxes.shape)
            # print(
            #     "Shape of predicted mask to fill",
            #     (image.shape[0], boxes.shape[1], image.shape[2], image.shape[3]),
            # )
            # print("pred mask", ori_res_masks.shape)
            # print("predicted mask to fill", predicted_masks.shape)
            predicted_masks[:, i, :, :] = ori_res_masks.squeeze(1)

        return predicted_masks

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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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

    train_transforms = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((1024, 1024)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.3),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),
            #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),
            #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    train_dataset = datasets.CocoDetection(
        root=os.path.join(args.root_path, "train"),
        annFile=args.train_annotations_path,
        transforms=train_transforms,
    )
    train_dataset = datasets.wrap_dataset_for_transforms_v2(
        train_dataset, target_keys=["boxes", "masks"]
    )
    val_dataset = datasets.CocoDetection(
        root=os.path.join(args.root_path, "train"),
        annFile=args.train_annotations_path,
        transforms=val_transforms,
    )
    val_dataset = datasets.wrap_dataset_for_transforms_v2(
        val_dataset, target_keys=["boxes", "masks"]
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
        for step, (img, target) in enumerate(train_dataloader):
            if "masks" in target.keys():
                # TODO visualize bboxes and masks correspond to each other
                # Aggregate all masks
                # masks = target["masks"][0, 0, :, :]

                # for mask in target["masks"]:
                #     masks = masks.logical_or(mask[0, :, :])
                # masks = torch.unsqueeze(masks, 0)
                # masks = torch.unsqueeze(masks, 0)
                # masks = target["masks"]
                masks = target["masks"].permute(1, 0, 2, 3)
                bboxes = target["boxes"].permute(1, 0, 2)
                # # Visualize img and mask
                # plt.figure(1)
                # plt.imshow(img[0, ...].permute(1, 2, 0).numpy())

                # plt.figure(2)
                # plt.imshow(masks[0, 0, ...])

                # plt.show()

                # img = torch.flatten(img, start_dim=0, end_dim=1)
                # masks = torch.flatten(masks, start_dim=0, end_dim=1)
                # bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
                optimizer.zero_grad()
                boxes_np = bboxes.detach().cpu().numpy()
                masks, img = masks.to(device), img.to(device)
                if args.use_amp:
                    ## AMP
                    if torch.cuda.is_available():
                        device_type = "cuda"
                    else:
                        device_type = "cpu"
                    with torch.autocast(device_type=device_type, dtype=torch.float16):
                        texsam_pred = textile_model(img, boxes_np)
                        loss = seg_loss(texsam_pred, masks) + ce_loss(
                            texsam_pred, masks.float()
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    texsam_pred = textile_model(img, boxes_np)
                    # print(texsam_pred, masks.shape)
                    seg_loss_ = seg_loss(texsam_pred, masks)
                    ce_loss_ = ce_loss(texsam_pred, masks.float())
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
        for step, (img, target) in enumerate(val_dataloader):
            if "masks" in target.keys():
                masks = target["masks"].permute(1, 0, 2, 3)
                bboxes = target["boxes"].permute(1, 0, 2)
                # img = torch.unsqueeze(img, 0)
                # masks = torch.unsqueeze(target["masks"], 0)
                # bboxes = torch.unsqueeze(target["boxes"], 0)

                # img = torch.flatten(img, start_dim=0, end_dim=1)
                # masks = torch.flatten(masks, start_dim=0, end_dim=1)
                # bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
                boxes_np = bboxes.detach().cpu().numpy()
                masks, img = masks.to(device), img.to(device)
                with torch.no_grad():
                    texsam_pred = textile_model(img, boxes_np)
                    seg_loss_ = seg_loss(texsam_pred, masks)
                    ce_loss_ = ce_loss(texsam_pred, masks.float())
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
