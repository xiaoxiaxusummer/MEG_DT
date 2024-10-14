# ========================== configure training and optimizer ========================
from dataclasses import dataclass
@dataclass
class TrainingConfig:
    image_size = 1024  # the generated image resolution
    batch_size = 1 # number of samples for each batch
    num_epochs = 600
    train_size = 20 # number of samples for each epoch
    test_size = 8
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 400
    save_image_epochs = 10
    save_model_epochs = 25
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = ''  # the model saving path
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0