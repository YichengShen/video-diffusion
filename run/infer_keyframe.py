import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import wandb
from accelerate import Accelerator

import src.models.unet as my_unet
from src.models.ddpm import FrameDiffusion
from src.utils.config_utils import load_config
from src.utils.data_loader import DataLoader
from src.utils.data_loader_keyframe import KeyFrameDataLoader


def to_image(img):
    return wandb.Image(torch.cat(img.split(1), dim=-1).cpu().numpy())


def create_predictions_table(previous_frames, next_frames):
    previous_frames_imgs = [to_image(s) for s in previous_frames]
    predictions = [to_image(s) for s in next_frames]

    table = wandb.Table(columns=["inputs", "preds"])
    for pf, pred in zip(previous_frames_imgs, predictions):
        table.add_data(pf, pred)
    return table


def to_saved_video_file(frames, file_path='video.mp4', fps=10):
    frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for frame in frames:
        color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
        out.write(color_frame)
    out.release()
    return file_path


def log_saved_video_file(pf, pred):
    video_frames = torch.cat([pf, pred], dim=0)
    video_frames = video_frames.unsqueeze(1)
    video_frames_np = video_frames.cpu().numpy().transpose(0, 2, 3, 1)
    video_path = to_saved_video_file(video_frames_np)
    return wandb.Video(video_path)


def main():
    cfg = load_config('run/config.yaml')

    exp_name = cfg['method_name'] + '-' + cfg['experiment_name'] + '-keyframe'

    with wandb.init(project=exp_name, group="preds", config=cfg) if cfg['wandb']['use_wandb'] else nullcontext():
        if cfg['device'] == "cpu":
            accelerator = Accelerator(device_placement=False, cpu=True)
        else:
            accelerator = Accelerator()

        data_loader = DataLoader(cfg)
        keyframe_data_loader = KeyFrameDataLoader(cfg)

        model = my_unet.create_unet(in_channels=cfg['num_frames'] + 1, out_channels=1)
        keyframe_model = my_unet.create_unet(in_channels=cfg['num_frames'] + 1, out_channels=1)

        diffuser = FrameDiffusion(
            cfg=cfg,
            data_loader=data_loader,
            model=model,
            accelerator=accelerator)
        diffuser.load(cfg['infer']['trained_weights'])

        keyframe_diffuser = FrameDiffusion(
            cfg=cfg,
            data_loader=data_loader,
            model=model,
            accelerator=accelerator)
        keyframe_diffuser.load(cfg['keyframe_model']['trained_weights'])

        batch_files = [os.path.join(cfg['dataset_dir'], f) for f in os.listdir(cfg['dataset_dir']) if f.endswith('.pt')]
        full_video = torch.load(batch_files[0])[1].to(cfg['device'])

        model = keyframe_diffuser.ema_model
        for i in range(0, full_video.size(0), cfg['keyframe_model']['num_frames_to_skip_per_keyframe']):
            previous_frame = full_video[i].unsqueeze(0)
            keyframe = keyframe_diffuser.sample_more(model, previous_frame, n=1)
            keyframe = keyframe.squeeze().unsqueeze(2)
            keyframe = keyframe.detach().cpu().numpy()
            wandb.log({f'keyframe_{i}': wandb.Image(keyframe)})


if __name__ == "__main__":
    main()
