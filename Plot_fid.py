
import numpy as np
import torchvision
import os, sys
import matplotlib.pyplot as plt
import numpy as np


import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)

project_dir = os.path.abspath(os.path.dirname(__file__))


def plot_FID_score(SNR_list, FID_list, save_path=project_dir+"/result/"):
    """
    color: 
        mediumseagreen yellowgreen
        royalblue deepskyblue
        cyan
        firebrick
    """
    fig = plt.figure()
    p1, = plt.plot(SNR_list, FID_list[:,0],'-.*',color="deepskyblue",markersize=8, linewidth=2.5, clip_on=False)
    p2, = plt.plot(SNR_list, FID_list[:,1],'-->',color="mediumseagreen",markerfacecolor='white', markersize=8, linewidth=2.5, clip_on=False)
    p3, = plt.plot(SNR_list,  FID_list[:,2],'-o',color="firebrick",markerfacecolor='white', markersize=8, linewidth=2.5, clip_on=False)
    plt.ylim(0, 300)
    plt.xlim(SNR_list[0], SNR_list[-1])
    plt.legend((p1,p2,p3,),
           (
            r"Centralized Generation", 
            r"MEG-DT", 
            r"E2E-MEG-DT",
            )
           )
    plt.xlabel("SNR (dB)")
    plt.ylabel("FID score")
    fig.savefig(save_path+"/MEG_FID_SNR.png", bbox_inches="tight")
    fig.savefig(save_path+"/MEG_FID_SNR.pdf", bbox_inches="tight")

def Calculate_FID(imgs_dist_true,imgs_dist):
    """
    Args:
        img_dist_true: perfect img tensors in int8
        imgs_dist: generated img tensors in int8
    Returns:
        FID score
    """

    fid.update(imgs_dist_true, real=True)
    fid.update(imgs_dist, real=False)
    fid_score = fid.compute()
    fid.reset()

    return fid_score

def Load_Imgs(img_files):
    imgs = []
    for p in img_files:
        imgs.append(torchvision.io.read_image(p))
    return torch.stack(imgs, 0)

def main(SNR_list, save_img_path=project_dir+'/result/test/generated_images/'):
    FID_results = []
    for snr in SNR_list:
        img_files = [
            f"prompt_{'0_0'}_SNR_{snr}_step_12_cfg_2_5.png", 
            f"prompt_{'0_1'}_SNR_{snr}_step_12_cfg_2_5.png",
            f"prompt_{'1_0'}_SNR_{snr}_step_12_cfg_2_5.png",
            f"prompt_{'1_1'}_SNR_{snr}_step_12_cfg_2_5.png",
            f"prompt_{'2_0'}_SNR_{snr}_step_12_cfg_2_5.png",
            f"prompt_{'2_1'}_SNR_{snr}_step_12_cfg_2_5.png"
        ]

        files_perfect = [os.path.join(save_img_path, "Perfect_"+x) for x in img_files]
        imgs_perfect = Load_Imgs(files_perfect)

        files_CG = [os.path.join(save_img_path, "CG_"+x) for x in img_files]
        imgs_CG = Load_Imgs(files_CG)
        FID_CG = Calculate_FID(imgs_perfect, imgs_CG)

        files_MEG = [os.path.join(save_img_path, "MEG_"+x) for x in img_files]
        imgs_MEG = Load_Imgs(files_MEG)
        FID_MEG = Calculate_FID(imgs_perfect, imgs_MEG)

        files_E2E_MEG = [os.path.join(save_img_path, "E2E_MEG_"+x) for x in img_files]
        imgs_E2E_MEG = Load_Imgs(files_E2E_MEG)
        FID_E2E_MEG = Calculate_FID(imgs_perfect, imgs_E2E_MEG)

        FID_results.append([FID_CG, FID_MEG, FID_E2E_MEG])

    FID_results = np.array(FID_results)
    print(FID_results)
    plot_FID_score(SNR_list, FID_results)

if __name__ == "__main__":
    SNR_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20] # in dB
    main(SNR_list)
