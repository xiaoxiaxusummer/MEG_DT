import os, sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

SAVE_DIR = os.path.join(project_dir, 'assets/laion-coco-aesthetic/')
from datasets import load_dataset, load_from_disk, concatenate_datasets


"""
Add your training dataset loader here. Check README.md for instructions of dataset preparation. 
[Note] Our training dataset is not available online currently. We'll make it open sourced soon. 
"""
def get_dataset_loader(type="laion-coco-high-resolution", downloading=False, save_dir=SAVE_DIR, 
                       fetch_pipe_results=True, multi_diffusion_step = False, split=True, small_dataset=False):
    if type=="laion-coco-high-resolution":
        dataset = None
        if fetch_pipe_results and multi_diffusion_step:
            num_files = 40 if small_dataset else 70
            for i in range(0, num_files):
                shard = load_from_disk(os.path.join(save_dir,f"ref_parquet_final/{i:03d}.parquet"))
                shard = shard.with_format("numpy", columns=["diffusion", "latents", "image"])
                dataset = shard if dataset is None else concatenate_datasets([dataset, shard])
        elif fetch_pipe_results:
            for i in range(0, 70):
                shard = load_from_disk(os.path.join(save_dir,f"ref_parquet_float32/{i:03d}.parquet"))
                shard = shard.with_format("numpy", columns=["diffusion", "latents", "image"])
                dataset = shard if dataset is None else concatenate_datasets([dataset, shard])
        else:
            for i in range(0, 6):
                shard = load_from_disk(os.path.join(save_dir,f"parquet/{i:03d}.parquet"))
                dataset = shard if dataset is None else concatenate_datasets([dataset, shard])
        if split:
            dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=666)
            return dataset["train"], dataset["test"]
        else:
            return dataset.shuffle(seed=666)
    else:
        AssertionError("No available dataset found")


