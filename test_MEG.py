import numpy as np
from dataclasses import asdict

import os, sys
project_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
print(project_dir)
sys.path.append(project_dir)

import argparse
import torch
import torch, random, torchvision

from pathlib import Path

from pipeline_MEG import StableDiffusionXLMEGPipeline
from diffusers import AutoencoderKL
from scheduler_perflow import PeRFlowScheduler

import feature_merging_utils as fm
from JSCC_merging import JSCCMergingNet, SplitModel, AWGN
from Plot_fid import Calculate_FID,  Load_Imgs


torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Test MEG models for the given prompt')
parser.add_argument('--intermediate_dim', type=int, default=1024)
parser.add_argument('--latent_size', type=tuple, default=(4,128,128))
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=9e-3)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--decay_step', type=int, default=60)
parser.add_argument('--channel_noise', type=float, default = 0.1)
parser.add_argument('--save_img_path', type=str, default = project_dir + '/result/test/generated_images/')
parser.add_argument('--dtype', type=torch.dtype, default = torch.float32)
parser.add_argument('--test_dtype', type=torch.dtype, default = torch.float32)
parser.add_argument('--compression_rate', type=float, default = 0.5)

args = parser.parse_args()
args.device = device

Path(args.save_img_path).mkdir(parents=True, exist_ok=True) 

from configs.training_config import TrainingConfig
config = TrainingConfig()

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

seed = 24

signal_length = np.prod(args.latent_size)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=args.dtype).to(device, args.dtype)
cfg_sdxl = {
    'num_img': 1,
    'cfg_scale': 2.5,
    'compute_fm_ratio': 0.5, # token merging ratio to accelerate self-attention computing (not for communications)
    'prompt_pre': "photorealistic, uhd, high resolution, high quality, highly detailed; realistic photo, ", 
    'neg_prompt': "distorted, blur, low-quality, haze, out of focus",
    'num_diffusion_step': 12,
    'output_type': 'pt',
}

MEG_model  = SplitModel(args).to(args.device, args.test_dtype)
data = torch.load(project_dir+f"/assets/MEG_noisy/LAIONCOCO_ckpt_{649}_model.pth")
MEG_model.load_state_dict(data['Split_Model'])
del data

E2EMEG_model = JSCCMergingNet(args, use_conv_encoder=True, conv_config=[1,0]).to(args.device, args.test_dtype)
data = torch.load(project_dir+f"/assets/E2E_MEG/LAIONCOCO_ckpt_{1799}_model.pth")
E2EMEG_model.load_state_dict(data['JSCC_model'])
del data


def test_MEG(pipe, prompt_list, SNR_dB, num_img, cal_FID=False):
    """
    Test our trained models
    Args:
        pipe: Stable diffusion pipeline to perform
        prompt_list: A list of text prompt requests
        SNR_dB: Test SNR
        num_img: Number of samples generated for each prompt 
    """
    sdxl_results = []
    CG_results = []
    MEG_results = []
    E2E_MEG_results = []
    img_files = []

    with torch.no_grad():
        for p_id, prompt in enumerate(prompt_list):
            prompt = cfg_sdxl['prompt_pre'] + prompt
            samples = pipe(
                    prompt              = [prompt] * num_img, 
                    negative_prompt     = [cfg_sdxl['neg_prompt']] * num_img,
                    height              = 1024,
                    width               = 1024,
                    num_inference_steps = cfg_sdxl['num_diffusion_step'], 
                    guidance_scale      = cfg_sdxl['cfg_scale'],
                    output_type         = 'pt',
            ).images

            sdxl_image = samples[0] 
            latents = samples[1]
            CG_pred = sdxl_image + AWGN(sdxl_image, SNR_dB)
            MEG_pred, noise_of_latents = MEG_model(latents, SNR_dB)
            E2EMEG_pred, _ = E2EMEG_model(latents, SNR_dB)

            sdxl_results.append(sdxl_image)
            CG_results.append(CG_pred)
            MEG_results.append(MEG_pred)
            E2E_MEG_results.append(E2EMEG_pred)
        
            cfg_int = int(cfg_sdxl['cfg_scale']); cfg_float = int(cfg_sdxl['cfg_scale']*10 - cfg_int*10)
            
            if cal_FID:
                for i in range(num_img):
                    img_name = f"prompt_{p_id}_{i}_SNR_{SNR_dB}_step_{cfg_sdxl['num_diffusion_step']}_cfg_{cfg_int}_{cfg_float}.png"
                    torchvision.utils.save_image(torchvision.utils.make_grid(torch.stack((CG_pred[i], MEG_pred[i], E2EMEG_pred[i]), 0), nrow = 1), os.path.join(args.save_img_path, "All_"+img_name))
                    torchvision.utils.save_image(sdxl_image[i], os.path.join(args.save_img_path, "Perfect_"+img_name))
                    torchvision.utils.save_image(CG_pred[i], os.path.join(args.save_img_path, "CG_"+img_name))
                    torchvision.utils.save_image(MEG_pred[i], os.path.join(args.save_img_path, "MEG_"+img_name))
                    torchvision.utils.save_image(E2EMEG_pred[i], os.path.join(args.save_img_path, "E2E_MEG_"+img_name))
                    img_files.append(img_name)
        
        if cal_FID: 
            files_perfect = [os.path.join(args.save_img_path, "Perfect_"+x) for x in img_files]
            imgs_perfect = Load_Imgs(files_perfect)

            files_CG = [os.path.join(args.save_img_path, "CG_"+x) for x in img_files]
            imgs_CG = Load_Imgs(files_CG)
            FID_CG = Calculate_FID(imgs_perfect, imgs_CG)

            files_MEG = [os.path.join(args.save_img_path, "MEG_"+x) for x in img_files]
            imgs_MEG = Load_Imgs(files_MEG)
            FID_MEG = Calculate_FID(imgs_perfect, imgs_MEG)

            files_JSCC = [os.path.join(args.save_img_path, "E2E_MEG_"+x) for x in img_files]
            imgs_JSCC = Load_Imgs(files_JSCC)
            FID_E2E_MEG = Calculate_FID(imgs_perfect, imgs_JSCC)

            return FID_CG, FID_MEG, FID_E2E_MEG

if __name__=='__main__':

    prompt_list = [
                "A photo of a cartoon British boy with curled hair and big brown eyes in a yellow jacket walking along the bustling ancient Chinese street,",
                "A photo of a cartoon European boy with slightly curled hair and big brown eyes in a yellow jacket standing in front of the stone monument of the forest,", 
                "A photo of a cartoon British boy with slightly curled hair and big brown eyes in a yellow jacket standing a huge whale at the bottom of the sea,",
    ]

    
    SNR_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20] # in dB
    FID_results = []

    for snr in SNR_list:
        print(f"snr_dB:{snr}, snr:{10**(snr/10)}")
        # Initialize the pipeline with the baseline model under perfect channels
        pipe = StableDiffusionXLMEGPipeline.from_pretrained("hansyan/perflow-sdxl-dreamshaper", torch_dtype=torch.float16, use_safetensors=True, variant="v0-fix")
        if cfg_sdxl['compute_fm_ratio'] >0:
            fm.apply_patch(pipe, ratio=cfg_sdxl['compute_fm_ratio'], max_downsample=8)
        pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
        pipe.to(args.device, args.test_dtype)
        setup_seed(seed)
        FID_CG, FID_MEG, FID_E2E_MEG = test_MEG(pipe, prompt_list, snr, 2, cal_FID=True)
        FID_results.append([FID_CG, FID_MEG, FID_E2E_MEG])
    

    FID_results = np.array(FID_results)
    print(FID_results)
    from JSCC_plot import plot_FID_score
    plot_FID_score(SNR_list, FID_results)

