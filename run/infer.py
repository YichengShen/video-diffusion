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


def to_image(img):
    return wandb.Image(torch.cat(img.split(1), dim=-1).cpu().numpy())


# def to_video(frames):
#     frames = frames.cpu().numpy()
#     frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
#     return wandb.Video(frames, fps=10, format="mp4")
#
# def log_videos(previous_frames, next_frames):
#     videos = []
#     for pf, pred in zip(previous_frames, next_frames):
#         video_frames = torch.cat([pf, pred], dim=0)
#         video_frames = video_frames.unsqueeze(1)
#         video = to_video(video_frames)
#         videos.append(video)
#     return videos


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

    previous_frames, next_frames = data_loader.get_batch(batch_size=cfg['infer']['num_videos_to_generate'])
    model = diffuser.ema_model
    next_frames = diffuser.sample_more(model, previous_frames, n=cfg['infer']['num_frames_to_infer'])
    table = create_predictions_table(previous_frames, next_frames)

    with wandb.init(project=exp_name, group="preds", config=cfg) if cfg['wandb']['use_wandb'] else nullcontext():
        wandb.log({"ema_preds_table": table})
        for i, (pf, pred) in enumerate(zip(previous_frames, next_frames)):
            video = log_saved_video_file(pf, pred)
            wandb.log({f"video_{i}": video})


if __name__ == "__main__":
    main()
