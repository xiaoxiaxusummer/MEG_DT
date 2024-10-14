
import numpy as np
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)

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

def main(SNR_list, save_img_path='/homes/xx623/demo/JSCC/'):
    FID_results = []
    for snr in SNR_list:
        img_files = [
            f"prompt_{'0_0'}_SNR_{snr}_step_12_cfg_2_5.png", 
            f"prompt_{'0_1'}_SNR_{snr}_step_12_cfg_2_5.png"
        ]

        files_perfect = [os.path.join(save_img_path, "Perfect_"+x) for x in img_files]
        imgs_perfect = Load_Imgs(files_perfect)

        files_CG = [os.path.join(save_img_path, "CG_"+x) for x in img_files]
        imgs_CG = Load_Imgs(files_CG)
        FID_CG = Calculate_FID(imgs_perfect, imgs_CG)

        files_split = [os.path.join(save_img_path, "Split_"+x) for x in img_files]
        imgs_split = Load_Imgs(files_split)
        FID_split = Calculate_FID(imgs_perfect, imgs_split)

        files_JSCC = [os.path.join(save_img_path, "JM_"+x) for x in img_files]
        imgs_JSCC = Load_Imgs(files_JSCC)
        FID_JSCC = Calculate_FID(imgs_perfect, imgs_JSCC)

        FID_results.append([FID_CG, FID_split, FID_JSCC])

    FID_results = np.array(FID_results)
    print(FID_results)
    from JSCC_plot import plot_FID_score
    plot_FID_score(SNR_list, FID_results)

if __name__ == "__main__":
    SNR_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20] # in dB
    main(SNR_list)
