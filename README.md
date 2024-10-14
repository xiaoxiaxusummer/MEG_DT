### This is official code implementation of paper "Mobile Edge Generation-Enabled Digital Twin: Architecture Design and Research Opportunities", which has been accepted by IEEE Communications Magzine. [[Preprint]](https://arxiv.org/abs/2407.02804)

# Environment
```
python=3.10
torch==2.4.1, cuda-toolkit==12.1.1, torchmetrics==1.4.3
transformers==4.45.1, xformers==0.0.28.post1
datasets==3.0.1, diffusers==0.30.3
```

# :smile: Reproduce our results
* :fire: To reproduce results in our paper, download our **model checkpoints** from [Google Drive](https://drive.google.com/drive/folders/1JJbdBb5xl5XXSGgHPQjn9mSnwNpzJmZB?usp=sharing) and put them under the project folder (i.e., [`assets/...`](./assets/))
* :hourglass_flowing_sand: Run [`test_MEG.py`](test_MEG.py)
* :star: Generated samples can be found in [`result/test/generated_images/`](./result/test/generated_images/), and the plot of FID scores vs. different SNR values will be found in [`result/JSCC_FID_SNR.pdf`](./result/JSCC_FID_SNR.pdf))
* You may test differet prompts by configuring `prompt_list` in [`test_MEG.py`](test_MEG.py) (see `line 155`), but this will create diverse results

# Model training of MEG
### :fire: Our training scripts
* For **MEG-DT**, customized decoder can be trained by our script [`train_split.py`](./train_split.py) 
* For **E2E-MEG-DT**, an advanced decoder for **transmitted feature compression (i.e., feature merging)** can be trained by our script [`train_JSCC_merging.py`](./train_JSCC_merging.py)

### :dart: Training Dataset
For MEG model training, we utilize a filtered [`laion-coco dataset`](https://huggingface.co/datasets/laion/laion-coco). The dataset can be downloaded following [Huggingface's guidelines](https://huggingface.co/docs/datasets/quickstart).
> **Reference Latents/Image Generation**: After downloading the laion-coco dataset, first generate the reference images by SDXL, and store the genrated latents and image samples in fields "latents" and "diffusion", respectively (which will be utilized for training under different SNR values).

:blue_heart: [**#TODO**] We will make our training datasets online available soon.

# Citation
If you find the code useful for your research, please consider citing
> X. Xu, R. Zhong, X. Mu, Y. Liu, and K. Huang, ``Mobile Edge Generation-Enabled Digital Twin: Architecture Design and Research Opportunities'', IEEE Communications Magazine, accepted, Oct. 2024.
