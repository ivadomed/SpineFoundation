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
        if lab is not None:
            lab=self._resize(lab,(Dz,Dh,Dw),"nearest")
            lab=self._center_crop_pad(lab,self.img_size)
        img=self._center_crop_pad(img,self.img_size)            
        img=self._norm(img)
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