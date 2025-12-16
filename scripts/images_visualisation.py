import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    EnsureTyped,
)

# === PARAMS venant de ton JSON ===
IMG_SIZE = (15, 384, 384)
IMG_RESOLUTION = (4.5, 0.6641, 0.6641)  # target_res pour le resample


# ---------------- GPUResampleAug3D ---------------- #

class GPUResampleAug3D(nn.Module):
    def __init__(self,img_size=(256,256,256),target_res=(1.0,1.0,1.0),prob_flip=0.2):
        super().__init__()
        self.img_size=img_size
        self.target_res=target_res
        self.prob_flip=prob_flip

    def _compute_out_size(self,shape,spacing):
        D,H,W=shape
        if isinstance(spacing,torch.Tensor): sz,sy,sx=spacing.tolist()
        else: sz,sy,sx=spacing
        tz,ty,tx=self.target_res
        Dz=max(1,int(round(D*sz/tz)))
        Dh=max(1,int(round(H*sy/ty)))
        Dw=max(1,int(round(W*sx/tx)))
        return Dz,Dh,Dw

    def _resize(self,x,size,mode):
        if x.ndim==3: x=x.unsqueeze(0).unsqueeze(0)
        elif x.ndim==4: x=x.unsqueeze(0)
        return F.interpolate(x,size=size,mode=mode,align_corners=False if mode!="nearest" else None).squeeze(0)

    def _center_crop_pad(self,x,target):
        D,H,W=x.shape[-3:]
        Td,Th,Tw=target
        dz=max(Td-D,0); dh=max(Th-H,0); dw=max(Tw-W,0)
        if dz>0 or dh>0 or dw>0:
            pad=(dw//2,dw-dw//2,dh//2,dh-dh//2,dz//2,dz-dz//2)
            x=F.pad(x,pad)
            D,H,W=x.shape[-3:]
        sd=max((D-Td)//2,0); sh=max((H-Th)//2,0); sw=max((W-Tw)//2,0)
        return x[...,sd:sd+Td,sh:sh+Th,sw:sw+Tw]

    def _norm(self,x):
        flat=x.reshape(1,-1)
        m=flat.mean(-1,keepdim=True); s=flat.std(-1,keepdim=True)+1e-6
        return ((flat-m)/s).reshape_as(x)


    def _flip(self,img,lab=None):
        if torch.rand(1).item()<self.prob_flip:
            img=torch.flip(img,dims=[1])
            if lab is not None:
                lab=torch.flip(lab,dims=[1])
        return img,lab

    def forward_single(self,img,spacing,lab=None):
        D,H,W=img.shape[-3:]
        Dz,Dh,Dw=self._compute_out_size((D,H,W),spacing)
        img=self._resize(img,(Dz,Dh,Dw),"trilinear")
        img=self._norm(img)
        if lab is not None:
            lab=self._resize(lab,(Dz,Dh,Dw),"nearest")
            lab=self._center_crop_pad(lab,self.img_size)
        img=self._center_crop_pad(img,self.img_size)            
        
        return img,lab

    def forward(self,images,spacings,labels=None):
        out_i=[]
        #out_l=[]
        '''for img,lab,sp in zip(images,labels,spacings):
            img_aug,lab_aug=self.forward_single(img,lab,sp)
            out_i.append(img_aug); out_l.append(lab_aug)'''
        for img,sp in zip(images,spacings):
            img_aug,lab_aug=self.forward_single(img,sp,lab=None)
            out_i.append(img_aug)
            #out_l.append(lab_aug)
        x=torch.stack(out_i,0)
        #y=torch.stack(out_l,0)
        #return x,y
        return x

# ---------------- MONAI transforms ---------------- #
from monai.transforms import MapTransform
from monai.data import MetaTensor
import numpy as np

class ComputeSpacingDHWd(MapTransform):
    """
    Ajoute dans meta["spacing_dhw"] les espacements alignés avec img.shape[-3:].
    (On dérive les voxel sizes à partir de l'affine actuelle, après Orientationd.)
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if k in d and isinstance(d[k], MetaTensor):
                mt = d[k]

                # Affine actuelle (4x4), déjà en RAS après Orientationd
                A = np.asarray(mt.affine, dtype=float)   # shape (4,4)

                # Espacements le long des 3 axes de l'image
                # -> norme des colonnes de A[:3, :3]
                spacing_ijk = np.sqrt((A[:3, :3] ** 2).sum(0))  # (s0, s1, s2)

                mt.meta["spacing_dhw"] = spacing_ijk
        return d

def get_transforms(augment = False):
    keys = ["image"]
    transforms = [
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS",labels=(('L', 'R'), ('P', 'A'), ('I', 'S')),allow_missing_keys=True),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True, allow_missing_keys=True),
    ]

    if augment:
        transforms += [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotated(keys=keys, prob=0.5, range_y=0.1),
            RandLambdad(keys=keys,func=aug_sqrt,prob=0.05,),
            RandLambdad(keys=keys,func=aug_sin,prob=0.05,),
            RandLambdad(keys=keys,func=aug_exp,prob=0.05,),
            RandLambdad(keys=keys,func=aug_sig,prob=0.05, ),
            RandLambdad(keys=keys,func=aug_laplace,prob=0.05,),
            RandLambdad(keys=keys,func=aug_inverse,prob=0.05, ),   
            RandBiasFieldd(keys=keys,prob=0.05),
            RandAffined(keys=keys,prob=0.05, padding_mode="zeros", mode=["bilinear"]), 
            RandGaussianNoised(keys=keys, mean=0.0, std=0.1, prob=0.05),
            RandGaussianSharpend(keys=keys, prob=0.05),   
            ResizeWithPadOrCropd(keys=keys, spatial_size=(6, 100, 100)),
            RandScaleIntensityd(keys=keys, factors=(0.8, 1.2), prob=1), 
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),  
        ]


    return Compose(transforms+[ComputeSpacingDHWd(keys=keys)])


# ---------------- Utils ---------------- #

def save_middle_slice(img_tensor, out_path):
    """
    img_tensor: (C, D, H, W) ou (D, H, W)
    → on prend la slice du milieu selon D.
    """
    # Si canal présent, on prend le canal 0
    if img_tensor.ndim == 4:
        # (C, D, H, W)
        img_tensor = img_tensor[0]  # -> (D, H, W)

    # Maintenant img_tensor est (D, H, W)
    D, H, W = img_tensor.shape
    mid = D // 2

    slice_2d = img_tensor[mid].cpu().numpy()  # (H, W)

    plt.figure(figsize=(4, 4))
    plt.imshow(slice_2d, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---------------- Main ---------------- #

def main():
    import nibabel as nib
    import numpy as np

    parser = argparse.ArgumentParser(description="Save middle slice after MONAI + GPUResampleAug3D.")
    parser.add_argument("images", type=str, nargs="+", help="Paths to .nii/.nii.gz images.")
    parser.add_argument("--out", type=str, default="images", help="Output folder for PNG slices.")
    parser.add_argument("--gpu", action="store_true", help="Apply GPUResampleAug3D transforms")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    transforms = get_transforms_cpu()
    gpu_aug = GPUResampleAug3D(
        img_size=IMG_SIZE,
        target_res=IMG_RESOLUTION,
        augment=False,
        prob_flip=0.2,
    )

    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"[WARN] Missing: {img_path}")
            continue

        basename = os.path.basename(img_path)
        print("\n===============================================")
        print(f"[INFO] Loading: {img_path}")
        print("===============================================")

        # --------- PIXDIM AVANT CPU TRANSFORM ---------
        nii = nib.load(img_path)
        pixdim_before = list(nii.header.get_zooms()[:3])
        print(f"[INFO] {basename} → pixdim BEFORE CPU transform = {pixdim_before}")

        # --------- CPU TRANSFORMS (MONAI) --------------
        sample = {"image": img_path}
        data = transforms(sample)
        img = data["image"]  # MetaTensor
        print(f"[INFO] Shape after CPU transform (RAS orientation) = {tuple(img.shape)}")

        # --------- METADATA APRÈS CPU + ROTATION YES/NO ----------
        meta = img.meta if hasattr(img, "meta") else {}

        # pixdim / spacing après CPU
        if "pixdim" in meta:
            pixdim_after = list(meta["pixdim"][1:4])
            print(f"[INFO] {basename} → pixdim AFTER CPU transform = {pixdim_after}")
        elif "spacing" in meta:
            print(f"[INFO] {basename} → spacing AFTER CPU transform = {list(meta['spacing'])}")
        else:
            print("[INFO] No pixdim or spacing available after CPU transform.")

        # rotation ou pas d'après les métadonnées MONAI
        if "original_affine" in meta and "affine" in meta:
            original_aff = np.asarray(meta["original_affine"])
            aff = np.asarray(meta["affine"])
            rotated = not np.allclose(original_aff, aff)
            print(f"[INFO] Orientation changed (MONAI) ? → {'YES' if rotated else 'NO'}")
        else:
            print("[INFO] Cannot determine rotation (missing 'original_affine' or 'affine' in metadata).")

        # --------- PRÉP POUR GPU / SAVE ----------------
        if img.ndim == 3:
            img = img.unsqueeze(0)   # (1, D,H,W) ou (1,H,W,D)

        # fake label
        lab = torch.zeros_like(img)

        # spacing pour GPU
                # --------- SPACING POUR GPU (monde X,Y,Z -> tensor D,H,W = Z,Y,X) ---------
        spacing = torch.as_tensor(meta["spacing_dhw"], dtype=torch.float32)
        print(f"[INFO] Using spacing_dhw (D,H,W) = {spacing.tolist()}")


        # GPU transforms optionnels
        if args.gpu:
            print("[GPU] Applying GPUResampleAug3D...")
            images_batch = img.unsqueeze(0)
            labels_batch = lab.unsqueeze(0)
            spacings_batch = [spacing]

            img_aug, _ = gpu_aug(images_batch, labels_batch, spacings_batch)
            img = img_aug[0]
            print(f"[INFO] Shape after GPUResampleAug3D = {tuple(img.shape)}")
        else:
            print("[GPU] GPU disabled → skipping GPUResampleAug3D")

        # sauvegarde slice milieu
        base = basename.replace(".nii.gz", "").replace(".nii", "")
        out_file = os.path.join(args.out, f"{base}_mid.png")

        save_middle_slice(img, out_file)
        print(f"[INFO] Saved → {out_file}")




if __name__ == "__main__":
    main()
