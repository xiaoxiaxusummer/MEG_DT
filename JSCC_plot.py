from dataclasses import asdict
import torch
import os, sys
project_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
print(project_dir)
sys.path.append(project_dir)

import matplotlib.pyplot as plt
import numpy as np

def plot_FID_score(SNR_list, FID_list, path=project_dir+"/result/"):
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
    fig.savefig(path+"JSCC_FID_SNR.png", bbox_inches="tight")
    fig.savefig(path+"JSCC_FID_SNR.pdf", bbox_inches="tight")

