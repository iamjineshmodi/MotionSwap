
import os
import time
import random
import argparse
import numpy as np
import wandb

import torch
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.tensorboard as tensorboard

from util import util
from util.plot import plot_batch

from models.projected_model import fsModel
from data.data_loader_Swapping import GetLoader

def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='simswap', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')       

        # for displays
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')
        self.parser.add_argument('--use_wandb', type=str2bool, default='True', help='use wandb for logging')
        self.parser.add_argument('--wandb_project', type=str, default='simswap', help='wandb project name')

        # for training
        self.parser.add_argument('--dataset', type=str, default="/path/to/VGGFace2", help='path to the face swapping dataset')
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='200000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=120000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=280000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')

        # for discriminators         
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=40.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=2.0, help='weight for reconstruction loss') 

        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="path to arcface model")
        self.parser.add_argument("--total_step", type=int, default=400000, help='total training step')
        self.parser.add_argument("--log_freq", type=int, default=200, help='frequency for printing log information')
        self.parser.add_argument("--sample_freq", type=int, default=1000, help='frequency for sampling')
        self.parser.add_argument("--model_freq", type=int, default=10000, help='frequency for saving the model')
        
        # scheduler settings
        self.parser.add_argument("--scheduler", type=str, default='cosine', help='scheduler type: cosine, step, exponential or none')
        self.parser.add_argument("--use_warmup", type=str2bool, default='True', help='use warmup for scheduler')
        self.parser.add_argument("--warmup_steps", type=int, default=1000, help='warmup steps')
        self.parser.add_argument("--min_lr", type=float, default=1e-6, help='minimum learning rate for schedulers')
        
        # Balance training
        self.parser.add_argument("--n_critic", type=int, default=1, help='train discriminator for n_critic iterations per generator iteration')
        self.parser.add_argument("--grad_clip", type=float, default=5.0, help='gradient clipping value')

        self.isTrain = True
        
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt


class LRScheduler:
    """Learning rate scheduler with warmup and multiple strategies"""
    def __init__(self, optimizer, opt):
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        self.opt = opt
        self.warmup_steps = opt.warmup_steps
        self.total_steps = opt.total_step
        self.min_lr = opt.min_lr
        self.scheduler_type = opt.scheduler
        
        print(f"Initializing {self.scheduler_type} scheduler with base_lr={self.base_lr}, "
              f"warmup_steps={self.warmup_steps}, min_lr={self.min_lr}")
        
    def step(self):
        """Update learning rate and return the current value"""
        self.current_step += 1
        lr = self._get_lr()
        
        # Apply the learning rate to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _get_lr(self):
        """Calculate learning rate based on scheduler type and current step"""
        if self.opt.use_warmup and self.current_step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        
        # After warmup, apply selected scheduler
        if self.scheduler_type == 'cosine':
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(progress * np.pi))
        
        elif self.scheduler_type == 'step':
            # Step decay (decay by 0.1 every 1/3 of training)
            decay_factor = 0.1 ** (3 * self.current_step / self.total_steps)
            return max(self.min_lr, self.base_lr * decay_factor)
        
        elif self.scheduler_type == 'exponential':
            # Exponential decay
            decay_rate = np.log(self.min_lr / self.base_lr) / self.total_steps
            return self.base_lr * np.exp(decay_rate * self.current_step)
        
        else:  # 'none' or any other value
            # Constant learning rate
            return self.base_lr
    
    def get_last_lr(self):
        """Return current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def get_current_lambda_weight(base_lambda, scheduler, scale=1.0):
    """Get the current lambda weight based on the scheduler's learning rate"""
    # Scale the base lambda value by the current relative learning rate
    current_lr = scheduler.get_last_lr()[0]
    base_lr = scheduler.base_lr
    
    # Keep lambda proportional to learning rate with scaling factor
    if base_lr > 0:
        return base_lambda * (current_lr / base_lr) * scale
    return base_lambda * scale


def main():
    opt = TrainOptions().parse()
    set_seed(1234)  # Set seed for reproducibility
    
    # Setup paths
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')

    # Create directories if they don't exist
    for path in [sample_path, log_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Initialize wandb if specified
    if opt.use_wandb:
        wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))
    
    # Resume training if required
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used:", str(opt.gpu_ids))
    
    # Enable cuDNN benchmark for faster training
    cudnn.benchmark = True

    # Initialize model
    model = fsModel()
    model.initialize(opt)
    
    # Initialize tensorboard
    tensorboard_writer = None
    if opt.use_tensorboard:
        tensorboard_writer = tensorboard.SummaryWriter(log_path)
    
    # Initialize log file
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    # Initialize optimizers and schedulers
    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D
    
    # Set up improved schedulers with warmup
    scheduler_G = LRScheduler(optimizer_G, opt)
    scheduler_D = LRScheduler(optimizer_D, opt)

    # Initialize constants
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()

    # Initialize data loader
    train_loader = GetLoader(opt.dataset, opt.batchSize, 8, 1234)

    # Initialize training variables
    if not opt.continue_train:
        start = 0
    else:
        start = int(opt.which_epoch)
    total_step = opt.total_step
    
    # Print training start information
    print("Start training at %s" % (time.strftime("%Y-%m-%d %H:%M:%S")))
    
    try:
        from util.logo_class import logo_class
        logo_class.print_start_training()
    except:
        print("Starting training for SimSwap...")
    
    # Freeze feature network in discriminator for better training stability
    model.netD.feature_network.requires_grad_(False)

    # Training loop
    for step in range(start, total_step):
        model.netG.train()
        model.netD.train()
        
        # Update learning rates
        lr_G = scheduler_G.step()
        lr_D = scheduler_D.step()
        
        # Calculate current lambda weights based on schedulers
        lambda_id_current = get_current_lambda_weight(opt.lambda_id, scheduler_G)
        lambda_rec_current = get_current_lambda_weight(opt.lambda_rec, scheduler_D)
        
        # Get next batch
        src_image1, src_image2 = train_loader.next()
        src_image1, src_image2 = src_image1.cuda(), src_image2.cuda()
        
        # Train discriminator
        for _ in range(opt.n_critic):
            # Randomize identity source for better diversity
            randindex = list(range(opt.batchSize))
            random.shuffle(randindex)
            
            if step % 2 == 0:
                img_id = src_image2
            else:
                img_id = src_image2[randindex]

            # Extract identity features
            img_id_112 = F.interpolate(img_id, size=(112, 112), mode='bicubic')
            latent_id = model.netArc(img_id_112)
            latent_id = F.normalize(latent_id, p=2, dim=1)
            
            # Generate fake images
            with torch.no_grad():
                img_fake = model.netG(src_image1, latent_id)
            
            # Train discriminator with fake images
            optimizer_D.zero_grad()
            gen_logits, _ = model.netD(img_fake.detach(), None)
            loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()
            
            # Train discriminator with real images
            real_logits, _ = model.netD(src_image2, None)
            loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
            
            # Combine D losses and update
            loss_D = loss_Dgen + loss_Dreal
            loss_D.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.netD.parameters(), max_norm=opt.grad_clip)
            optimizer_D.step()
        
        # Train generator
        # Randomize identity source again for generator training
        randindex = list(range(opt.batchSize))
        random.shuffle(randindex)
        
        if step % 2 == 0:
            img_id = src_image2
        else:
            img_id = src_image2[randindex]

        # Extract identity features
        img_id_112 = F.interpolate(img_id, size=(112, 112), mode='bicubic')
        latent_id = model.netArc(img_id_112)
        latent_id = F.normalize(latent_id, p=2, dim=1)
        
        # Generate fake images
        img_fake = model.netG(src_image1, latent_id)
        
        # Adversarial loss
        optimizer_G.zero_grad()
        gen_logits, feat = model.netD(img_fake, None)
        loss_Gmain = (-gen_logits).mean()
        
        # Identity loss
        img_fake_down = F.interpolate(img_fake, size=(112, 112), mode='bicubic')
        latent_fake = model.netArc(img_fake_down)
        latent_fake = F.normalize(latent_fake, p=2, dim=1)
        loss_G_ID = (1 - model.cosin_metric(latent_fake, latent_id)).mean() * lambda_id_current
        
        # Feature matching loss
        real_feat = model.netD.get_feature(src_image1)
        feat_match_loss = model.criterionFeat(feat["3"], real_feat["3"]) * opt.lambda_feat
        
        # Combined G loss
        loss_G = loss_Gmain + loss_G_ID + feat_match_loss
        
        # Add reconstruction loss every other step
        loss_G_Rec = torch.tensor(0.0).cuda()
        if step % 2 == 0:
            loss_G_Rec = model.criterionRec(img_fake, src_image1) * lambda_rec_current
            loss_G += loss_G_Rec
        
        # Update generator
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(model.netG.parameters(), max_norm=opt.grad_clip)
        optimizer_G.step()
        
        # Log training progress
        if (step + 1) % opt.log_freq == 0:
            # Create loss dict for logging
            losses = {
                "G_Loss": loss_Gmain.item(),
                "G_ID": loss_G_ID.item(),
                "G_Rec": loss_G_Rec.item(),
                "G_feat_match": feat_match_loss.item(),
                "D_fake": loss_Dgen.item(),
                "D_real": loss_Dreal.item(),
                "D_loss": loss_D.item(),
                "lambda_id": lambda_id_current,
                "lambda_rec": lambda_rec_current,
                "lr_G": lr_G,
                "lr_D": lr_D
            }
            
            message = f'(step: {step+1}) '
            for k, v in losses.items():
                message += f'{k}: {v:.3f} '
                
                # Log metrics
                if opt.use_tensorboard and tensorboard_writer:
                    tensorboard_writer.add_scalar(k, v, step+1)
                if opt.use_wandb:
                    wandb.log({k: v, 'step': step+1})
            
            print(message)
            with open(log_name, "a") as log_file:
                log_file.write(f'{message}\n')
        
        # Generate sample images
        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                imgs = []
                zero_img = torch.zeros_like(src_image1[0, ...]).cpu()
                imgs.append(zero_img.numpy())
                
                # Denormalize source images
                save_img = ((src_image1.cpu()) * imagenet_std.cpu() + imagenet_mean.cpu()).numpy()
                for r in range(opt.batchSize):
                    imgs.append(save_img[r, ...])
                
                # Get identity vectors for swapping
                arcface_112 = F.interpolate(src_image2, size=(112, 112), mode='bicubic')
                id_vector_src1 = model.netArc(arcface_112)
                id_vector_src1 = F.normalize(id_vector_src1, p=2, dim=1)
                
                # Generate face-swapped images
                for i in range(opt.batchSize):
                    imgs.append(save_img[i, ...])
                    image_infer = src_image1[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake = model.netG(image_infer, id_vector_src1).cpu()
                    
                    # Denormalize generated images
                    img_fake = img_fake * imagenet_std.cpu()
                    img_fake = img_fake + imagenet_mean.cpu()
                    img_fake = img_fake.numpy()
                    
                    for j in range(opt.batchSize):
                        imgs.append(img_fake[j, ...])
                
                # Save sample images
                print(f"Saving test samples at step {step+1}")
                imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
                sample_path_step = os.path.join(sample_path, f'step_{step+1}.jpg')
                plot_batch(imgs, sample_path_step)
                
                # Log sample images to wandb
                if opt.use_wandb:
                    wandb.log({"samples": wandb.Image(sample_path_step)}, step=step+1)
        
        # Save model checkpoint
        if (step + 1) % opt.model_freq == 0:
            print(f'Saving model checkpoint at step {step+1}')
            model.save(step + 1)            
            np.savetxt(iter_path, (step + 1, total_step), delimiter=',', fmt='%d')
    
    # Clean up after training completes
    if opt.use_tensorboard and tensorboard_writer:
        tensorboard_writer.close()
    if opt.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()