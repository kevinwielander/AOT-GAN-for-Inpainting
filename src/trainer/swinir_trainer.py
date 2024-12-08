import os

import torch
import wandb

from src.data import create_loader
from src.model.swinir import SwinIR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
from loss import loss as loss_module



class SwinIRTrainer:
    def __init__(self, args):
        self.args = args
        self.iteration = 0
        self.swinir_loss_func = {key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        if self.args.global_rank == 0:
            wandb.init(
                project="swinir-restoration",
                config=vars(args),
                name=f"{args.model}_{args.data_train}_{args.mask_type}{args.image_size}"
            )

        # Initialize model with 4-channel input (RGB + mask)
        self.model = SwinIR(
            img_size=args.image_size,
            in_chans=4,  # RGB + mask
            embed_dim=60,
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            window_size=8,
            mlp_ratio=2.,
            upscale=1,
            img_range=1.,
            upsampler=None,
            resi_connection='1conv'
        ).cuda()

        self.dataloader = create_loader(args)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lrg,
            betas=(args.beta1, args.beta2)
        )

        if self.args.global_rank == 0:
            wandb.watch(self.model)

        if args.distributed:
            self.model = DDP(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )



    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.local_rank == 0:
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        for idx in pbar:
            self.iteration += 1

            images, masks, _ = next(self.dataloader)
            images, masks = images.cuda(), masks.cuda()

            # Create masked input
            masked_images = images * (1 - masks)

            # Forward pass
            pred_img = self.model(masked_images,masks)



            comp_img = (1 - masks) * images + masks * pred_img

            losses = {}
            for name, weight in self.args.rec_loss.items():
                losses[name] = weight * self.swinir_loss_func[name](pred_img, images)

            # Backward and optimize
            self.optimizer.zero_grad()
            sum(losses.values()).backward()
            self.optimizer.step()

            # Logging
            if self.args.local_rank == 0 and (self.iteration % self.args.print_every == 0):

                wandb.log({
                    'total_loss': sum(loss.item() for loss in losses.values()),
                    **{name: loss.item() for name, loss in losses.items()},
                    'iteration': self.iteration,
                    'images/original': wandb.Image(make_grid((images + 1.0) / 2.0)),
                    'images/masked': wandb.Image(make_grid((masked_images + 1.0) / 2.0)),
                    'images/predicted': wandb.Image(make_grid((pred_img + 1.0) / 2.0)),
                    'images/comp': wandb.Image(make_grid((comp_img + 1.0) / 2.0))
                })

            # Save checkpoint
            if self.args.local_rank == 0 and (self.iteration % self.args.save_every == 0):
                self.save_checkpoint()

    def save_checkpoint(self):
        if self.args.local_rank == 0:
            save_path = os.path.join(self.args.save_dir, f'swinir_{self.iteration}.pt')
            torch.save({
                'iteration': self.iteration,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
