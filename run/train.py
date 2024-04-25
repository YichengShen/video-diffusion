from contextlib import nullcontext
from functools import partial

import wandb
from accelerate import Accelerator
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

import src.models.unet as my_unet
from src.models.ddpm import FrameDiffusion
from src.utils.config_utils import load_config
from src.utils.data_loader import DataLoader


def main():
    cfg = load_config('run/config.yaml')

    exp_name = cfg['method_name'] + '-' + cfg['experiment_name']

    if cfg['device'] == "cpu":
        accelerator = Accelerator(device_placement=False, cpu=True)
    else:
        accelerator = Accelerator()

    data_loader = DataLoader(cfg)

    model = my_unet.create_unet(in_channels=cfg['num_frames'] + 1, out_channels=1)
    diffuser = FrameDiffusion(
        cfg=cfg,
        data_loader=data_loader,
        model=model,
        accelerator=accelerator)

    # diffuser.load(cfg['infer']['trained_weights'])


    optimizer = partial(Adam, eps=1e-5)
    scheduler = partial(OneCycleLR, max_lr=cfg['learning_rate'], total_steps=cfg['train_steps'])

    diffuser.prepare(optimizer, scheduler)

    with wandb.init(project=exp_name, group=cfg['wandb']['group'], tags=cfg['wandb']['tags'],
                    config=cfg) if cfg['wandb']['use_wandb'] else nullcontext():
        diffuser.fit()

        diffuser.save_model(exp_name)


if __name__ == "__main__":
    main()
