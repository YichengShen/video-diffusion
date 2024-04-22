from contextlib import nullcontext

import torch
import wandb
from accelerate import Accelerator

import src.models.unet as my_unet
from src.models.ddpm import FrameDiffusion
from src.utils.config_utils import load_config
from src.utils.data_loader import DataLoader


def to_image(img):
    return wandb.Image(torch.cat(img.split(1), dim=-1).cpu().numpy())


def create_predictions_table(previous_frames, next_frames):
    previous_frames_imgs = [to_image(s) for s in previous_frames]
    predictions = [to_image(s) for s in next_frames]

    table = wandb.Table(columns=["inputs", "preds"])
    for pf, pred in zip(previous_frames_imgs, predictions):
        table.add_data(pf, pred)
    return table


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

    diffuser.load(cfg['infer']['trained_weights'])

    previous_frames, next_frames = data_loader.get_batch(batch_size=4)
    model = diffuser.ema_model
    next_frames = diffuser.sample_more(model, previous_frames, n=cfg['infer']['num_frames_to_infer'])
    t = create_predictions_table(previous_frames, next_frames)

    with wandb.init(project=exp_name, group="preds", config=cfg) if cfg['wandb']['use_wandb'] else nullcontext():
        wandb.log({"ema_preds_table": t})


if __name__ == "__main__":
    main()
