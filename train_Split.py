import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import argparse
import copy
import numpy as np
import random
from pathlib import Path

from PIL import Image
import os, sys
from tqdm import tqdm
from accelerate import Accelerator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

project_dir = os.path.abspath(os.path.dirname(__file__))
print(project_dir)
sys.path.append(project_dir)


from diffusers.optimization import get_cosine_schedule_with_warmup
from dataset_utils import get_dataset_loader
from JSCC_merging import SplitModel, FID_score, AWGN

from diffusers import AutoencoderKL

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

cfg_sdxl = {
    'num_img': 1,
    'cfg_scale': 2.5,
    'tome_ratio': 0.5,
    'prompt_pre': "photorealistic, uhd, high resolution, high quality, highly detailed; realistic photo, ", 
    'neg_prompt': "distorted, blur, low-quality, haze, out of focus",
    'hight': 1024,
    'weight': 1024,
    'num_diffusion_step': 8,
    'output_type': 'pt',
}




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='Training JSCC with Feature Merging')
parser.add_argument('--intermediate_dim', type=int, default=1024)
parser.add_argument('--latent_size', type=tuple, default=(4,128,128))
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=9e-3)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--decay_step', type=int, default=60)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--channel_noise', type=float, default = 0.1)
parser.add_argument('--save_dir', type=str, default = project_dir + '/assets/JSCC_split_finetune/')
parser.add_argument('--resume_vae', type=bool, default = True)
parser.add_argument('--dtype', type=torch.dtype, default = torch.float32)

args = parser.parse_args()
args.device = device
Path(args.save_dir).mkdir(parents=True, exist_ok=True) 

# ==========================     Dataset Loading      ==========================
train_dataset, test_dataset = get_dataset_loader(multi_diffusion_step=True)
test_img_path = project_dir + f'/result/train/generated_images/JSCC_split_finetune/'
Path(test_img_path).mkdir(parents=True, exist_ok=True) 

# ========================== configure training and optimizer ========================
from configs.training_config import TrainingConfig
config = TrainingConfig()
config.output_dir = args.save_dir

# ========================== load JSCC model ========================
Split_Model = SplitModel(args).to(device, args.dtype)
# Load a pretrained model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=args.dtype).to(device, args.dtype)
Split_Model.post_quant_conv.load_state_dict(vae.post_quant_conv.state_dict())
decoder_state_dict = torch.load(os.path.join(project_dir,'assets/decoder_model/LAIONCOCO_ckpt_299_model.pth'))
Split_Model.decoder.load_state_dict(decoder_state_dict['decoder'])
del decoder_state_dict, vae

if not args.resume_vae:
    start_epoch = 0

torch.cuda.empty_cache()


def train(config, Split_Model, train_dataset, optimizer, lr_scheduler, end_epoch, SNR_dB=0):
    test_MSE =  torch.inf
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    Split_Model, optimizer, train_dataset, lr_scheduler = accelerator.prepare(
        Split_Model, optimizer, train_dataset, lr_scheduler
    )

    global_step = 0
    Split_Model.train()
    Split_Model.requires_grad_(True)
    
    for epoch in range(start_epoch, end_epoch):
        train_dataloader = train_dataset.select(np.random.randint(0,len(train_dataset),size=config.train_size)).shuffle()
        n_step = int(config.train_size/config.batch_size)
        progress_bar = tqdm(total=n_step, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step_i in range(n_step):
            shard = train_dataloader.shard(num_shards = n_step, index = step_i)
            latents = torch.tensor(shard["latents"][:,-1]).to(device=args.device, dtype=args.dtype)
            sdxl_image = torch.tensor(shard["diffusion"][:,-1]).to(device=args.device, dtype=args.dtype)
            with accelerator.accumulate(Split_Model):
                img_pred, _ = Split_Model(latents, SNR_dB, is_train=True)
                loss = torch.nn.functional.mse_loss(img_pred,sdxl_image,reduction="mean")
                if torch.isnan(loss):
                    raise Exception("NaN value")
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(Split_Model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            MSE_batch, FID_batch = test(epoch, Split_Model, SNR_dB)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == end_epoch - 1:
                torch.save({'Split_Model': copy.deepcopy(Split_Model.state_dict()), 'epoch':epoch, "config":config, "args": args}, 
                           args.save_dir + f"/LAIONCOCO_ckpt_{epoch}_model.pth")

        print('FID:',FID_batch, 'Test MSE:',MSE_batch)

        if epoch > 180:
            if (MSE_batch < test_MSE):
                test_MSE = MSE_batch
                print('Best ckpt -- MSE:',MSE_batch,', FID:',FID_batch)
                torch.save({'Split_Model': copy.deepcopy(Split_Model.state_dict()), 'epoch':epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_best_ckpt_model.pth")
    torch.save({'Split_Model': copy.deepcopy(Split_Model.state_dict()), 'epoch':epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_final_ckpt_model.pth")
    return Split_Model

def test(epoch, Split_Model, SNR_dB = 0):
    test_loader = test_dataset.select(np.arange(0,config.test_size,1))
    n_step = int(config.test_size/config.batch_size)
    Split_Model.eval()
    with torch.no_grad():
        MSE_batch = 0
        total = 0
        sdxl_results = []
        JSCC_results = []
        for step_i in range(n_step):
            shard = test_loader.shard(num_shards = 20, index = step_i)
            prompts = shard["caption"]
            sdxl_image = torch.tensor(shard["diffusion"][:,-1]).to(device, args.dtype)
            latents = torch.tensor(shard["latents"][:,-1]).to(device, args.dtype)
            img_pred, noise = Split_Model(latents, SNR_dB)
            total += latents.size(0)
            MSE_batch = MSE_batch + torch.nn.functional.mse_loss(img_pred.float().cpu(), sdxl_image.float().cpu(), reduction="mean") 
            sdxl_results.append(sdxl_image)
            JSCC_results.append(img_pred)
            if ((epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1) and (step_i == 5):
                torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sdxl_image, img_pred],dim=0), nrow = 2), os.path.join(test_img_path, f'epoch_{epoch}_sdxl_SNR_{SNR_dB}_step_{step_i}.png'))

        return MSE_batch/total



if __name__=='__main__':
    seed_torch(0)
    train_SNR_dB = 0
    test_SNR_dB = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    if args.test == 0:
        config.end_epochs = [200, 300, 400, 500, 600, 800]
        config.learning_rate = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-6]
        config.lr_warmup_steps = [100, 200, 200, 200, 200, 0]
        for e_idx, end_epoch in enumerate(config.end_epochs):
            if start_epoch < end_epoch:
                optimizer = torch.optim.AdamW(Split_Model.parameters(), lr=config.learning_rate[e_idx])
                lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=config.lr_warmup_steps[e_idx],
                        num_training_steps=(int(config.train_size/config.batch_size) * (end_epoch-start_epoch)),
                )
                Split_Model = train(config, Split_Model, train_dataset, optimizer, lr_scheduler, end_epoch, train_SNR_dB) 
                start_epoch = end_epoch