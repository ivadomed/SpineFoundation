import torch,torch.nn as nn,torch.nn.functional as F,math

class GPUResampleAug3D(nn.Module):
    def __init__(self,img_size=(256,256,256),target_res=(1.0,1.0,1.0),augment=True,prob_flip=0.2):
        super().__init__()
        self.img_size=img_size
        self.target_res=target_res
        self.augment=augment
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


    def _flip(self,img,lab):
        if torch.rand(1).item()<self.prob_flip:
            img=torch.flip(img,dims=[1]); lab=torch.flip(lab,dims=[1])
        return img,lab

    def forward_single(self,img,lab,spacing):
        D,H,W=img.shape[-3:]
        Dz,Dh,Dw=self._compute_out_size((D,H,W),spacing)
        img=self._resize(img,(Dz,Dh,Dw),"trilinear")
        lab=self._resize(lab,(Dz,Dh,Dw),"nearest")
        img=self._center_crop_pad(img,self.img_size)
        lab=self._center_crop_pad(lab,self.img_size)
        img=self._norm(img)
        if self.augment: img,lab=self._flip(img,lab)
        return img,lab

    def forward(self,images,labels,spacings):
        out_i=[]; out_l=[]
        for img,lab,sp in zip(images,labels,spacings):
            img_aug,lab_aug=self.forward_single(img,lab,sp)
            out_i.append(img_aug); out_l.append(lab_aug)
        x=torch.stack(out_i,0)
        y=torch.stack(out_l,0)
        return x,y
