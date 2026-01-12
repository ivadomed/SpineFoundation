import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import gryds
import scipy.ndimage as ndi
from scipy.stats import norm
import warnings
import torch,torch.nn as nn,torch.nn.functional as F,math
from monai.transforms import MapTransform
from monai.data import MetaTensor


warnings.filterwarnings("ignore")


def aug_histogram_equalization(image1, seg, image2):
    img_min1, img_max1 = image1.min(), image1.max()
    img_min2, img_max2 = image2.min(), image2.max()

    image1_flattened = image1.flatten()
    hist1, bins1 = np.histogram(image1_flattened, bins=256, range=[image1_flattened.min(), image1_flattened.max()])
    cdf1 = hist1.cumsum()
    cdf_normalized1 = cdf1 * (hist1.max() / cdf1.max())
    image1 = np.interp(image1_flattened, bins1[:-1], cdf_normalized1).reshape(image1.shape)
    image1 = np.interp(image1, (image1.min(), image1.max()), (img_min1, img_max1))

    image2_flattened = image2.flatten()
    hist2, bins2 = np.histogram(image2_flattened, bins=256, range=[image2_flattened.min(), image2_flattened.max()])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * (hist2.max() / cdf2.max())
    image2 = np.interp(image2_flattened, bins2[:-1], cdf_normalized2).reshape(image2.shape)
    image2 = np.interp(image2, (image2.min(), image2.max()), (img_min2, img_max2))

    return image1, seg, image2

def aug_transform(image1, transform):
    img_min1, img_max1 = image1.min(), image1.max()


    image1 = (image1 - image1.mean()) / image1.std()
    image1 = np.interp(image1, (image1.min(), image1.max()), (0, 1))

    image1 = transform(image1)
    

    image1 = np.interp(image1, (image1.min(), image1.max()), (img_min1, img_max1))
    

    return image1

def aug_log(image1):
    return aug_transform(image1, lambda x: np.log(1 + x))

def aug_sqrt(image1 ):
    return aug_transform(image1,  np.sqrt)

def aug_sin(image1):
    return aug_transform(image1,  np.sin)

def aug_exp(image1):
    return aug_transform(image1, np.exp)

def aug_sig(image1):
    return aug_transform(image1, lambda x: 1 / (1 + np.exp(-x)))

def aug_laplace(image1):
    return aug_transform(image1,  lambda x: np.abs(ndi.laplace(x)))

def aug_inverse(image1):
    image1 = image1.min() + image1.max() - image1
    return image1

def aug_bspline(image1, seg, image2):
    grid = rs.rand(3, 3, 3, 3)
    bspline = gryds.BSplineTransformation((grid - .5) / 5)
    grid[:, 0] += ((grid[:, 0] > 0) * 2 - 1) * .9
    image1 = gryds.Interpolator(image1).transform(bspline).astype(np.float64)
    image2 = gryds.Interpolator(image2).transform(bspline).astype(np.float64)
    seg = gryds.Interpolator(seg, order=0).transform(bspline).astype(np.uint8)
    return image1, seg, image2

def aug_flip(image1, seg, image2):
    subject = tio.RandomFlip(axes=('LR',))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_aff(image1, seg, image2):
    subject = tio.RandomAffine()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_elastic(image1, seg, image2):
    subject = tio.RandomElasticDeformation(max_displacement=40)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_anisotropy(image1, seg, image2, downsampling=7):
    subject = tio.RandomAnisotropy(downsampling=downsampling)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_motion(image1, seg, image2):
    subject = tio.RandomMotion()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_ghosting(image1, seg, image2):
    subject = tio.RandomGhosting()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_spike(image1, seg, image2):
    subject = tio.RandomSpike(intensity=(1, 2))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_bias_field(image1, image2, seg):
    subject = tio.RandomBiasField()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_blur(image1, seg, image2):
    subject = tio.RandomBlur()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_noise(image1, seg, image2):
    original_mean1, original_std1 = np.mean(image1), np.std(image1)
    original_mean2, original_std2 = np.mean(image2), np.std(image2)

    image1 = (image1 - original_mean1) / original_std1
    image2 = (image2 - original_mean2) / original_std2

    subject = tio.RandomNoise()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    image1 = image1 * original_std1 + original_mean1
    image2 = image2 * original_std2 + original_mean2

    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_swap(image1, seg, image2):
    subject = tio.RandomSwap()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)

def aug_labels2image(image1, seg, image2, leave_background=0.5, classes=None):
    _seg = seg
    if classes:
        _seg = combine_classes(seg, classes)
    subject = tio.RandomLabelsToImage(label_key="seg", image_key="image")(tio.Subject(
        seg=tio.LabelMap(tensor=np.expand_dims(_seg, axis=0))
    ))
    new_img = subject.image.data.squeeze().numpy().astype(np.float64)

    if rs.rand() < leave_background:
        img_min1, img_max1 = np.min(image1), np.max(image1)
        _image1 = (image1 - img_min1) / (img_max1 - img_min1)

        new_img_min, new_img_max = np.min(new_img), np.max(new_img)
        new_img = (new_img - new_img_min) / (new_img_max - new_img_min)
        new_img[_seg == 0] = _image1[_seg == 0]
        new_img = np.interp(new_img, (new_img.min(), new_img.max()), (img_min1, img_max1))

    return new_img, seg, image2

def aug_gamma(image1, seg, image2):
    subject = tio.RandomGamma()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image1, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0)),
        image2=tio.ScalarImage(tensor=np.expand_dims(image2, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8), subject.image2.data.squeeze().numpy().astype(np.float64)


def parse_class(c):
    c = [_.split('-') for _ in c.split(',')]
    c = tuple(__ for _ in c for __ in list(range(int(_[0]), int(_[-1]) + 1)))
    return c

def combine_classes(seg, classes):
    _seg = np.zeros_like(seg)
    for i, c in enumerate(classes):
        _seg[np.isin(seg, c)] = i + 1
    return _seg



class ComputeSpacingDHWd(MapTransform):
    """
    Ajoute dans meta :
      - spacing_dhw : voxel size aligné avec (D,H,W)
      - orig_affine : affine d'origine (4x4)
      - orig_shape_dhw : shape d'origine (D,H,W)
      - orig_spacing_dhw : spacing d'origine
      - orig_dtype : dtype d'origine
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            v = d.get(k, None)
            if isinstance(v, MetaTensor):
                mt = v
                A = np.asarray(mt.affine, dtype=float)  # 4x4
                spacing_ijk = np.sqrt((A[:3, :3] ** 2).sum(0))  # (s0,s1,s2)

                mt.meta["spacing_dhw"] = spacing_ijk.astype(float)

                if "orig_affine" not in mt.meta:
                    mt.meta["orig_affine"] = A.copy()
                if "orig_shape_dhw" not in mt.meta:
                    mt.meta["orig_shape_dhw"] = np.array(mt.shape[-3:], dtype=int)
                if "orig_spacing_dhw" not in mt.meta:
                    mt.meta["orig_spacing_dhw"] = spacing_ijk.copy()
                if "orig_dtype" not in mt.meta:
                    mt.meta["orig_dtype"] = str(mt.dtype)

                d[k] = mt
        return d


import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_patch_multiple_dhw(img,window_size,mode="pad"):
        """
        img: (1, D, H, W) or (B, C, D, H, W)
        patch_size: (pD, pH, pW)
        mode: "pad" (to next multiple) or "crop" (to lower multiple)

        returns:
        img2: padded/cropped tensor
        grid_size_dhw: (gD, gH, gW)
        op: dict with details to store in your `info`
        """
        pD, pH, pW = window_size

        if img.ndim == 4:
            _, D, H, W = img.shape
            has_batch_chan = False
        elif img.ndim == 5:
            _, _, D, H, W = img.shape
            has_batch_chan = True
        else:
            raise ValueError("img must be 4D (1,D,H,W) or 5D (B,C,D,H,W)")

        if mode == "pad":
            D2 = ((D + pD - 1) // pD) * pD
            H2 = ((H + pH - 1) // pH) * pH
            W2 = ((W + pW - 1) // pW) * pW
            pad_d = D2 - D
            pad_h = H2 - H
            pad_w = W2 - W
            # F.pad order: (W_left,W_right,H_left,H_right,D_left,D_right)
            img2 = F.pad(img, (0, pad_w, 0, pad_h, 0, pad_d))
            op = {"patch_mode":"pad","patch_multiple_pad":(pad_d,pad_h,pad_w),"patch_multiple_crop":(0,0,0),"patch_multiple_shape_dhw":(D2,H2,W2)}
        
        elif mode == "crop":

            D2 = (D // pD) * pD
            H2 = (H // pH) * pH
            W2 = (W // pW) * pW
            crop_d = D - D2
            crop_h = H - H2
            crop_w = W - W2
            if img.ndim == 4:
                img2 = img[:, :D2, :H2, :W2]
            else:
                img2 = img[:, :, :D2, :H2, :W2]
            op = {"patch_mode":"crop","patch_multiple_pad":(0,0,0),"patch_multiple_crop":(crop_d,crop_h,crop_w),"patch_multiple_shape_dhw":(D2,H2,W2)}
        else:
            raise ValueError("mode must be 'pad' or 'crop'")

        gD, gH, gW = D2 // pD, H2 // pH, W2 // pW
        return img2, (gD, gH, gW), op

class GPUResampleAug3D(nn.Module):
    def __init__(self, window_size, target_res=(1.0,1.0,1.0), prob_flip=0.2, inference=False, max_resampled_dhw=(500,210,210)):
        super().__init__()
        self.window_size = window_size
        self.target_res = target_res
        self.prob_flip = prob_flip
        self.inference = inference
        # max_resampled_dhw is in (D,H,W). Set to None to disable.
        self.max_resampled_dhw = max_resampled_dhw

    def _compute_out_size(self, shape, spacing):
        D,H,W = shape
        if isinstance(spacing, torch.Tensor): sz,sy,sx = spacing.tolist()
        else: sz,sy,sx = spacing
        tz,ty,tx = self.target_res
        Dz = max(1, int(round(D*sz/tz)))
        Dh = max(1, int(round(H*sy/ty)))
        Dw = max(1, int(round(W*sx/tx)))
        return Dz,Dh,Dw

    def _resize(self, x, size, mode):
        if x.ndim == 3: x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4: x = x.unsqueeze(0)
        x = F.interpolate(x, size=size, mode=mode, align_corners=False if mode!="nearest" else None)
        return x.squeeze(0)

    def _cap_resampled_size(self, img, lab, dhw):
        if self.max_resampled_dhw is None:
            return img, lab, dhw, False
        maxD, maxH, maxW = self.max_resampled_dhw
        D,H,W = dhw
        D2 = min(D, maxD) if maxD is not None else D
        H2 = min(H, maxH) if maxH is not None else H
        W2 = min(W, maxW) if maxW is not None else W
        capped = (D2 != D) or (H2 != H) or (W2 != W)
        if not capped:
            return img, lab, (D,H,W), False
        img = self._resize(img, (D2,H2,W2), "trilinear")
        if lab is not None:
            lab = self._resize(lab, (D2,H2,W2), "nearest")
        return img, lab, (D2,H2,W2), True

    def _norm(self, x):
        flat = x.reshape(1,-1)
        m = flat.mean(-1, keepdim=True)
        s = flat.std(-1, keepdim=True) + 1e-6
        return ((flat-m)/s).reshape_as(x), float(m.item()), float(s.item())

    def _flip(self, img, lab=None):
        if self.inference: return img, lab, False
        flipped = False
        if torch.rand(1).item() < self.prob_flip:
            img = torch.flip(img, dims=[1])
            if lab is not None: lab = torch.flip(lab, dims=[1])
            flipped = True
        return img, lab, flipped

    def forward_single(self, img, spacing, lab=None):
        if img.ndim == 4 and img.shape[0] > 1:
            img = img.mean(dim=0, keepdim=True)

        orig_shape = tuple(img.shape[-3:])
        Dz,Dh,Dw = self._compute_out_size(orig_shape, spacing)

        # (1) resample to target_res
        img = self._resize(img, (Dz,Dh,Dw), "trilinear")
        if lab is not None:
            lab = self._resize(lab, (Dz,Dh,Dw), "nearest")

        # (2) enforce caps AFTER resampling: H,W <= 210 and D <= 500 (with default max_resampled_dhw=(500,210,210))
        img, lab, capped_shape_dhw, was_capped = self._cap_resampled_size(img, lab, (Dz,Dh,Dw))

        img, lab, flipped = self._flip(img, lab)
        img, m, s = self._norm(img)

        img_before_patch_multiple_shape = tuple(img.shape[-3:])
        img, grid_size_dhw, patch_multiple = pad_to_patch_multiple_dhw(img, self.window_size, mode="pad")
        if lab is not None:
            lab, _, _ = pad_to_patch_multiple_dhw(lab, self.window_size, mode="crop")

        info = {
            "orig_shape_dhw": orig_shape,
            "resampled_shape_dhw": (Dz,Dh,Dw),
            "capped_resampled_shape_dhw": capped_shape_dhw,
            "was_capped_after_resample": was_capped,
            "norm_mean": m,
            "norm_std": s,
            "flipped_D": flipped,
            "img_before_patch_multiple_shape_dhw": img_before_patch_multiple_shape,
        }
        return img, lab, info

    def forward(self, images, spacings, labels=None):
        out_i=[]; out_l=[]; infos=[]
        if not self.inference:
            if labels is None:
                for img, sp in zip(images, spacings):
                    img_aug, _, _ = self.forward_single(img, sp, lab=None)
                    out_i.append(img_aug)
                return torch.stack(out_i, 0)
            for img, lab, sp in zip(images, labels, spacings):
                img_aug, lab_aug, _ = self.forward_single(img, sp, lab)
                out_i.append(img_aug); out_l.append(lab_aug)
            return torch.stack(out_i, 0), torch.stack(out_l, 0)

        if labels is None:
            for img, sp in zip(images, spacings):
                img_aug, _, info = self.forward_single(img, sp, lab=None)
                out_i.append(img_aug); infos.append(info)
            return torch.stack(out_i, 0), infos

        for img, lab, sp in zip(images, labels, spacings):
            img_aug, lab_aug, info = self.forward_single(img, sp, lab)
            out_i.append(img_aug); out_l.append(lab_aug); infos.append(info)
        return torch.stack(out_i, 0), torch.stack(out_l, 0), infos
