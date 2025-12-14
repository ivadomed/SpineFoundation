import os
import json
from tqdm import tqdm
import time
import gc
import wandb
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


os.environ["WANDB_SILENT"] = "true" 

from model.build import build_model
from data_management.build import build_datasets

from mae_training.augment import GPUResampleAug3D
from mae_training.lr_scheduler import make_lr_lambda
from mae_training.utils import collate_fn, patchify, save_checkpoint, load_checkpoint, load_json_param, list_child_folders, plot_6_middle_slices, plot_6_uniform_slices
from mae_training.loss import L1_SSIM_Loss

TIME_CHECK=True    
TIME_EPOCH_CHECK=True

def _now(device):
    if TIME_CHECK and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time()


class Trainer:
    def __init__(self, args, ddp=False, rank=0, world_size=1):
        self.args = args
        conf= load_json_param(args.config)
        model_params = conf["Model"]
        data_params = conf["Data"]
        training_params = conf["Training"]

        self.ckpt = args.model_ckpt
        self.model_params = model_params
        self.data_params = data_params
       
        self.model_name=model_params["model_name"]
        self.img_size=tuple(model_params["img_size"])
        self.img_resolution=tuple(model_params["img_resolution"])


        self.batch_size = data_params["batch_size"]      
        self.train_ratio = data_params["train_ratio"]
        self.val_ratio = data_params["val_ratio"]
        self.test_ratio = data_params["test_ratio"]
        self.seed = data_params["seed"]
        self.data_path = data_params["data_path"]
        self.json_manifest = data_params.get("json_manifest", None)
        

        self.global_step = 0
        
        self.epochs = training_params["epochs"]
        self.work_dir = training_params["work_dir"]
        self.num_workers = training_params["num_workers"]
        self.wandb = training_params["wandb"]
        self.log_image_interval = training_params["log_image_interval"]
        self.lr = training_params["lr"]
        self.weight_decay = training_params["weight_decay"]
        self.amp = training_params["amp"]
        self.no_cuda = training_params["no_cuda"]
        self.resume = training_params["resume"]
        self.tqdm_disable = training_params["tqdm_disable"]


        self.ddp = ddp
        self.rank = rank
        self.world_size = world_size
        self.is_main = (not self.ddp) or (self.rank == 0)
        if self.ddp:
            self.device = torch.device(f"cuda:{self.rank}")

        else:
            self.device = torch.device('cuda' if (torch.cuda.is_available() and not self.no_cuda) else 'cpu') 
        if self.rank==0:
            print("\nDEVICE :\n")
            print(f"Using device: {self.device} (ddp={self.ddp}, rank={self.rank})")

        model_params.pop("model_name", None)
        model_params.pop("img_resolution", None)
        
        self.model = build_model(self.model_name, model_params,rank=self.rank)
        self.model.to(self.device)

        ckpt=load_checkpoint(self.ckpt,self.device)
        state=ckpt["model"]

        enc_state={}
        for k,v in state.items():
            if k.startswith("encoder."):
                enc_state[k.replace("encoder.","",1)]=v
            elif k.startswith("module.encoder."):
                enc_state[k.replace("module.encoder.","",1)]=v

        if hasattr(self.model,"encoder"):
            missing,unexpected=self.model.encoder.load_state_dict(enc_state,strict=False)
            if self.rank==0:
                print(f"Loaded pretrained encoder from {self.ckpt}")
                print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
        else:
            if self.rank==0:
                print("WARNING: model has no attribute .encoder, cannot load pretrained encoder weights")

        for p in self.model.encoder.parameters():
            p.requires_grad=False
        self.model.encoder.eval()
        if self.rank==0:
            n_train=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_all=sum(p.numel() for p in self.model.parameters())
            print(f"Encoder frozen. Trainable params: {n_train}/{n_all}")

        self.gpu_tf_train=GPUResampleAug3D(img_size=self.img_size,target_res=self.img_resolution).to(self.device)
        self.gpu_tf_eval=GPUResampleAug3D(img_size=self.img_size,target_res=self.img_resolution).to(self.device)

        train_ds, val_ds, test_ds = build_datasets(
                                                    data_path=self.data_path,
                                                    json_path=self.json_manifest,
                                                    splits=(self.train_ratio, self.val_ratio, self.test_ratio),
                                                    shuffle_seed=self.seed,
                                                    rank=self.rank,
                                                    label=True
                                                )

        if self.ddp:
            self.train_sampler = DistributedSampler(train_ds,num_replicas=self.world_size,rank=self.rank,shuffle=True)
            #self.val_sampler = DistributedSampler(val_ds,num_replicas=self.world_size,rank=self.rank,shuffle=False)
            self.train_loader = DataLoader(train_ds,batch_size=self.batch_size,shuffle=False,sampler=self.train_sampler,num_workers=self.num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=2,collate_fn=collate_fn)
            self.val_loader = DataLoader(val_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=2,collate_fn=collate_fn)
            self.test_loader = DataLoader(test_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=1,collate_fn=collate_fn)    
        
        else:
            self.train_sampler = None

            self.train_loader = DataLoader(train_ds,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=3,collate_fn=collate_fn)
            self.val_loader = DataLoader(val_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=1,collate_fn=collate_fn)
            self.test_loader = DataLoader(test_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=1,collate_fn=collate_fn)    
        



        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],lr=self.lr,weight_decay=self.weight_decay)

        total_steps = self.epochs * len(self.train_loader)
        warmup_steps = int(0.1 * total_steps)

        lr_lambda = make_lr_lambda(total_steps=total_steps,warmup_steps=warmup_steps,lr_up=self.lr,lr_min=self.lr / 10)

        
        self.scheduler =  torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = GradScaler(device=self.device, enabled=self.amp)
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric=DiceMetric(include_background=False,reduction="mean")

        self.start_epoch = 0
        self.best_val = float('inf')
        if self.resume:
            ckpt = load_checkpoint(self.resume, self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_val = ckpt.get('val_loss', float('inf'))
            self.global_step = ckpt.get('global_step', 0)
            print(f"Resumed from {self.resume} at epoch {self.start_epoch}")

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,find_unused_parameters=False)

    def train_step(self, batch, iteration: int, epoch: int):
        self.global_step += 1
        self.model.train()

        if TIME_CHECK:
            t0_total = _now(self.device)         
            t0 = _now(self.device)

        images = [b["image"].to(self.device, non_blocking=True) for b in batch]
        labels = [b["labels"].to(self.device, non_blocking=True) for b in batch]
        #labels = [b["label"].to(self.device, non_blocking=True) for b in batch]
        spacings = [torch.as_tensor(b["image"].meta["spacing_dhw"],dtype=torch.float32,device=self.device) for b in batch]
        if TIME_CHECK:
            t_batch_load = _now(self.device) - t0
            t0 = _now(self.device)
        #x, mask = self.gpu_tf_train(images, spacings)
        x,y = self.gpu_tf_train(images, spacings, labels)
        if TIME_CHECK:
            t_gpu_tf_train = _now(self.device) - t0

        with autocast(device_type=self.device.type, enabled=self.amp):

            if TIME_CHECK:
                t0 = _now(self.device)

            pred = self.model(x)

            if TIME_CHECK:
                t_forward = _now(self.device) - t0
                t0 = _now(self.device)

            loss = self.criterion(pred, y.long())

            if TIME_CHECK:
                t_loss = _now(self.device) - t0

            if self.is_main and self.wandb and self.global_step % (10 * self.log_image_interval) == 0:
                idx=torch.randint(0,x.shape[0],(1,)).item()

                img=x[idx,0].cpu()                      
                gt=y[idx,0].cpu()                       
                pred_mask=pred.argmax(dim=1)[idx].cpu() 

                fig=plot_6_middle_slices(image=img,gt=gt,pred=pred_mask)
                wandb.log({"Train/Images": wandb.Image(fig)},step=self.global_step)
                plt.close(fig)

        self.optimizer.zero_grad(set_to_none=True)


        if TIME_CHECK:
            t0 = _now(self.device)

        self.scaler.scale(loss).backward()

        if TIME_CHECK:
            t_backward = _now(self.device) - t0
            t0 = _now(self.device)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if TIME_CHECK:
            t_step = _now(self.device) - t0
            t_total = _now(self.device) - t0_total
            timings = {"batch_load": t_batch_load,"gpu_tf_train": t_gpu_tf_train,"forward": t_forward,"loss": t_loss,"backward": t_backward,"step": t_step,"total": t_total}

        else : timings = {}

        return loss.item(), timings


    def train_one_epoch(self, epoch: int):
        running_loss = 0.0

        if self.ddp:
            self.train_sampler.set_epoch(epoch)


        if TIME_CHECK:
            sums = {"batch_load": 0.0,"gpu_tf_train": 0.0,"forward": 0.0,"loss": 0.0,"backward": 0.0,"step": 0.0,"total": 0.0}

        pbar = tqdm(self.train_loader,desc=f"Train Epoch {epoch}",disable=self.tqdm_disable or (self.ddp and self.rank !=0))

        for i, batch in enumerate(pbar, start=1):
            loss, timings = self.train_step(batch, i, epoch)
            self.scheduler.step()
            running_loss += loss
            if TIME_CHECK:
                for k in sums:
                    sums[k] += timings[k]

        n = len(self.train_loader)
        epoch_loss = running_loss / n

        print()
        
        if TIME_CHECK and self.is_main:
            print(f"\nTRAINING TIMINGS EPOCH {epoch}")
            for k, v in sums.items():
                print(f"{k:12s}: {v / n:.4f} s / batch avg")

        return epoch_loss
        
    def validate(self, epoch: int):
        if self.ddp and self.rank != 0:
            return 0.0
        self.model.eval()
        total = 0.0
        count=0

        if TIME_CHECK:  
            sums = {"batch_load": 0.0,"gpu_tf_eval": 0.0,"forward": 0.0,"loss": 0.0,"iter_total": 0.0}

        pbar = tqdm(self.val_loader,desc=f"Validation Epoch {epoch}",disable=self.tqdm_disable or (self.ddp and self.rank !=0))

        with torch.no_grad():
            for i, batch in enumerate(pbar, start=1):

                if TIME_CHECK:
                    t0_total = _now(self.device)
                    t0 = _now(self.device)

                images = [b["image"].to(self.device, non_blocking=True) for b in batch]
                labels = [b["labels"].to(self.device, non_blocking=True) for b in batch]
                #labels = [b["label"].to(self.device, non_blocking=True) for b in batch]
                spacings = [torch.as_tensor(b["image"].meta["spacing_dhw"],dtype=torch.float32,device=self.device) for b in batch]

                if TIME_CHECK:
                    t_batch_load = _now(self.device) - t0
                    t0 = _now(self.device)

                #x, mask = self.gpu_tf_eval(images, spacings)
                x, y = self.gpu_tf_eval(images, spacings, labels)

                if TIME_CHECK:
                    t_gpu_tf_eval = _now(self.device) - t0
                    t0 = _now(self.device)

                with autocast(device_type=self.device.type, enabled=self.amp):
                    pred = self.model(x)
                    gt=y.long()   
                    if TIME_CHECK:
                        t_forward = _now(self.device) - t0
                        t0 = _now(self.device)

                    loss = self.criterion(pred, gt)
                    pred_mask=pred.argmax(dim=1,keepdim=True)  # (B,1,D,H,W)
                                                 

                    self.dice_metric(pred_mask,gt)

                    if TIME_CHECK:
                        t_loss = _now(self.device) - t0

                total += loss.item() * x.shape[0]
                count += x.shape[0]
                if TIME_CHECK:
                    iter_total = _now(self.device) - t0_total
                    sums["batch_load"] += t_batch_load
                    sums["gpu_tf_eval"] += t_gpu_tf_eval
                    sums["forward"] += t_forward
                    sums["loss"] += t_loss
                    sums["iter_total"] += iter_total
            print()        
        n = max(count, 1)
        val_dice=self.dice_metric.aggregate().item()
        self.dice_metric.reset()


        if TIME_CHECK and self.is_main:
            print(f"\nVALIDATION TIMINGS EPOCH {epoch}") 
            for k, v in sums.items():
                print(f"{k:12s}: {v / n:.4f} s / sample")
        
        if self.is_main and self.wandb and epoch % 10 == 0:
                idx=torch.randint(0,x.shape[0],(1,)).item()

                img=x[idx,0].cpu()                      
                gt=y[idx,0].cpu()                       
                pred_mask=pred.argmax(dim=1)[idx].cpu() 

                fig=plot_6_middle_slices(image=img,gt=gt,pred=pred_mask)
                wandb.log({"Val/Images": wandb.Image(fig)},step=self.global_step)
                wandb.log({"Val/Dice": val_dice},step=self.global_step)
                plt.close(fig)
        avg = total / n
        return avg




   
    def fit(self):

        if self.wandb and self.is_main:
            run=wandb.init(project="SpineMAE", config={
                "model_name": self.model_name,
            
                "img_size": self.img_size,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
            })
            print(f"W&B run: {run.url} \n")
            #wandb.watch(self.model, log=None)

        for epoch in range(self.start_epoch, self.epochs):
            if TIME_EPOCH_CHECK:
                    t0_total = _now(self.device)
                    t0 = _now(self.device)

            t_train = time.time()
            train_loss = self.train_one_epoch(epoch)

            if TIME_EPOCH_CHECK:
                    train_time = _now(self.device) - t0
                    t0 = _now(self.device)

            val_loss = self.validate(epoch)


            if TIME_EPOCH_CHECK:
                    val_time = _now(self.device) - t0
                    t0 = _now(self.device)

            is_best = val_loss < self.best_val
            self.best_val = min(self.best_val, val_loss)
            if self.is_main:
                model_state = self.model.module.state_dict() if self.ddp else self.model.state_dict()
                ckpt = {'epoch': epoch,'model': model_state,'optimizer': self.optimizer.state_dict(),'scheduler': self.scheduler.state_dict(),'val_loss': val_loss,'global_step': self.global_step}
                save_checkpoint(ckpt, os.path.join(self.work_dir, f'ckpt_epoch_{epoch}.pt'))
                
                if is_best:
                    save_checkpoint(ckpt, os.path.join(self.work_dir, 'best.ckpt'))

                if TIME_EPOCH_CHECK:
                        t_f= _now(self.device)
                        ckpt_time = t_f - t0
                        epoch_time = t_f - t0_total
                        

                log_dict = {"Train/Loss": train_loss,"Val/Loss": val_loss,"Epoch": epoch,"LR": self.scheduler.get_last_lr()[0]}

                if TIME_EPOCH_CHECK:
                    log_dict.update({"Time/Epoch": epoch_time,"Time/Train": train_time,"Time/Val": val_time,"Time/Checkpoint": ckpt_time})
                    print(f"\nEPOCH {epoch} TIMINGS")
                    print(f"Train      : {train_time:.2f} s")
                    print(f"Validation : {val_time:.2f} s")
                    print(f"Checkpoint : {ckpt_time:.2f} s")
                    print(f"Epoch Total: {epoch_time:.2f} s\n")
                if self.wandb:
                    wandb.log(
                                {**log_dict, "GlobalStep": self.global_step},
                                step=self.global_step
                            )


        if TIME_CHECK:
            t0 = _now(self.device)

        if self.wandb and self.is_main:
            wandb.finish()

        def _shutdown_loader(loader):
            if loader is None:
                return
            it = getattr(loader, "_iterator", None)
            if it is not None and hasattr(it, "_shutdown_workers"):
                try:
                    it._shutdown_workers()
                except Exception:
                    pass

        _shutdown_loader(self.train_loader)
        _shutdown_loader(self.val_loader)
        _shutdown_loader(getattr(self, "test_loader", None))

        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        plt.close('all')

        gc.collect()
        torch.cuda.empty_cache()

        if TIME_CHECK:
            t1 = _now(self.device)
            print(f"Temps vidage du cache : {t1 - t0:.2f} s")

        return self.best_val
            


