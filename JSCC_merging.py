
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.tomesd import merge

from src.utils.comm_utils import add_channel_noise
from src.diffusers.models.autoencoders.vae import Decoder
from torchmetrics.image.fid import FrechetInceptionDistance


def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = init_generator(device)



def FID_score(img_tensors, label_tensors):
    """
        calculate FID scores for two set of image tensors 
        input: 
        - img_tensors -> tuple(num_img, 3, width, height)
        - label_tensors -> tuple(num_img, 3, width, height)
        output:
        - FID score -> float
    """
    torch.manual_seed(123)
    assert img_tensors.shape[1] == 3 and label_tensors.shape[1] == 3
    fid = FrechetInceptionDistance(normalize=True, feature=64, input_img_size=(3,1024,1024)).to(img_tensors.device)
    fid.update(label_tensors, real=True) 
    fid.update(img_tensors, real=False)
    fid_result = float(fid.compute())
    print(fid_result)

    return fid_result

def AWGN(input_signal,SNR_in_db,is_train=False):

    input_signal_square = torch.mul(input_signal.reshape(-1), input_signal.reshape(-1))
    input_signal_power = torch.mean(input_signal_square)

    SNR = 10 ** (SNR_in_db / 10)
    if is_train:
        noise_power = input_signal_power/SNR*(2*torch.rand(1).to(input_signal.device, input_signal.dtype)+0.5)
    else:
        noise_power = input_signal_power/SNR
        
    
    noise_amplitude = noise_power.sqrt()
    noise = torch.normal(0, float(noise_amplitude), size=input_signal.shape).to(input_signal.device, input_signal.dtype)

    return noise


class SplitModel(nn.Module):
    def __init__(self, args, ori_feature_size=(128,128)):
        super().__init__()
        
        self.ori_size = ori_feature_size
        self.n_channels = 4 
        self.signal_length = ori_feature_size[0]*ori_feature_size[1]*self.n_channels
        self.args = args
        
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        self.decoder = Decoder(
            in_channels        = 4,
            out_channels       = 3,
            up_block_types     = (
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"),
            block_out_channels = (128, 256, 512, 512),
            layers_per_block   = 2,
            norm_num_groups    = 32,
            act_fn             = "silu",
            )
    
    def forward(self, latents, SNR_dB = 0, is_train=False, channel_noise=None):
        b, c, w, h = latents.shape[0],latents.shape[1],latents.shape[2],latents.shape[3]
        transmit_signal = latents.reshape((b*c*w*h))
        if channel_noise is None:
            channel_noise = AWGN(transmit_signal, SNR_dB, is_train)
        noised_latents = transmit_signal + channel_noise
        noised_latents = noised_latents.reshape((b,c,w,h))
        x = self.post_quant_conv(noised_latents)
        y = self.decoder(x)
        return y, channel_noise
    


class JSCCMergingNet(nn.Module):
    def __init__(self, args, ori_feature_size=(128,128), merging_ratio=0.5, use_conv_encoder = False, conv_config=None, dynamic_dm=False):
        super().__init__()
        
        self.ori_size = ori_feature_size
        args.sx, args.sy = 2, 2
        self.n_channels = 4 

        self.args = args
        
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        self.decoder = Decoder(
            in_channels        = 4,
            out_channels       = 3,
            up_block_types     = (
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"),
            block_out_channels = (128, 256, 512, 512),
            layers_per_block   = 2,
            norm_num_groups    = 32,
            act_fn             = "silu",
            )
        
        self.use_conv_encoder = use_conv_encoder
        self.dynamic_dm = dynamic_dm

        if not dynamic_dm:
            self.merging_ratio = merging_ratio
            self.signal_length = ori_feature_size[0]*ori_feature_size[1]*self.n_channels
            self.pruned_dim = int(ori_feature_size[0]*ori_feature_size[1]/4*merging_ratio)
            self.seed_length = self.signal_length - self.n_channels*4*self.pruned_dim
            self.encoder = nn.Linear(int(self.seed_length), int(self.seed_length))
        if use_conv_encoder and conv_config is None:
            self.encoder = nn.Conv2d(self.n_channels, self.n_channels, stride=1, kernel_size=1, padding=0)
        elif use_conv_encoder and conv_config is not None:
            self.encoder = nn.Conv2d(self.n_channels, self.n_channels, stride=1, kernel_size=conv_config[0], padding=conv_config[1])

    def feature_merger(self, latent_code, num_pruning_neuron):
        args = self.args
        original_h, original_w = self.ori_size
        original_tokens = original_h * original_w # 原始token数目 = o_h * o_w
        downsample = int(math.ceil(math.sqrt(original_tokens // latent_code.shape[1]))) # 目前已下采样的级别
        w = int(math.ceil(original_w / downsample)) # 下采样的宽
        h = int(math.ceil(original_h / downsample)) # 下采样的高
        m, u = merge.bipartite_soft_matching_random2d(latent_code, w, h, args.sx, args.sy, num_pruning_neuron, no_rand=False, generator=generator)
        return m, u

    def forward(self, latents, SNR_dB = 0, is_train=False, merging_ratio=0.5):
        # ================== 通道数*4, 长/2， 宽/2 ========================
        if self.use_conv_encoder:
            latents = self.encoder(latents)
        n_dim = latents.shape[-1]*latents.shape[-2]/4
        if not self.dynamic_dm:
            merging_ratio = self.merging_ratio
        num_pruning_neuron = int(merging_ratio *n_dim)
        latent_code = latents.permute(0,2,3,1).reshape((latents.shape[0],latents.shape[2]*latents.shape[3],latents.shape[1]))  
        latent_code = latent_code.reshape((latents.shape[0],int(latents.shape[2]*latents.shape[3]/4),int(latents.shape[1]*4))) # [n_batch, w*h/4, n_c*4]
        m, u = self.feature_merger(latent_code, num_pruning_neuron)
        merged_latents = m(latent_code)

        transmit_signal = merged_latents.reshape((merged_latents.shape[0],merged_latents.shape[1]*merged_latents.shape[2]))

        channel_noise = AWGN(transmit_signal, SNR_dB, is_train)
        noised_latents = transmit_signal + channel_noise
        noised_latents = noised_latents.reshape((merged_latents.shape))
        unmerged_latents = u(noised_latents)
        unmerged_latents = unmerged_latents.reshape((latents.shape[0],latents.shape[2]*latents.shape[3],latents.shape[1]))
        unmerged_latents = unmerged_latents.reshape((latents.shape[0],latents.shape[2],latents.shape[3],latents.shape[1])).permute(0,3,1,2)

        x = self.post_quant_conv(unmerged_latents)
        y = self.decoder(x)
        return y, channel_noise
    
    def get_info_dim(self):
        return (self.seed_length+self.pruned_dim)