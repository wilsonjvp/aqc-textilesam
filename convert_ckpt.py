# -*- coding: utf-8 -*-
import torch

# %% convert textilesam model checkpoint to sam checkpoint format for convenient inference
sam_ckpt_path = "weights/sam_vit_b_01ec64.pth"
textilesam_ckpt_path = "weights/textilesam_model_best.pth"
save_path = "weights/textilesam_vit_b.pth"
multi_gpu_ckpt = False  # set as True if the model is trained with multi-gpu

sam_ckpt = torch.load(sam_ckpt_path)
textilesam_ckpt = torch.load(textilesam_ckpt_path)
sam_keys = sam_ckpt.keys()
for key in sam_keys:
    if not multi_gpu_ckpt:
        sam_ckpt[key] = textilesam_ckpt["model"][key]
    else:
        sam_ckpt[key] = textilesam_ckpt["model"]["module." + key]

torch.save(sam_ckpt, save_path)