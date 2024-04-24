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


def to_saved_video_file(frames, file_path='video.mp4', fps=5):
    frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for frame in frames:
        color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
        out.write(color_frame)
    out.release()
    return file_path


def main():
    cfg = load_config('run/config.yaml')

    exp_name = cfg['method_name'] + '-' + cfg['experiment_name']

    with wandb.init(project=exp_name, group="preds", config=cfg) if cfg['wandb']['use_wandb'] else nullcontext():
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

        batch_files = [os.path.join(cfg['dataset_dir'], f) for f in os.listdir(cfg['dataset_dir']) if f.endswith('.pt')]
        full_video = torch.load(batch_files[0])[1].to(cfg['device'])

        num_frames = min(cfg['hierarchical_infer']['total_num_frames_to_infer'], full_video.size(0))
        frames_to_generate = cfg['hierarchical_infer']['num_frames_to_generate_after_each_key_frame']
        start_frame = 0
        end_frame = num_frames - frames_to_generate

        generated_frames = []

        for i in range(start_frame, end_frame, frames_to_generate):
            # The model's input shape is determined by cfg['num_frames']
            previous_frames = full_video[i:i + cfg['num_frames']]

            for _ in range(frames_to_generate):
                next_frames = diffuser.sample_more(model, previous_frames.squeeze(1).unsqueeze(0),
                                                   n=1)
                next_frames = next_frames.squeeze(0)  # Remove batch dimension for concatenation
                generated_frames.append(next_frames)

                # Update previous_frames for the next iteration
                previous_frames = torch.cat((previous_frames[1:], next_frames.unsqueeze(0)), dim=0)

        all_generated_frames = torch.stack(generated_frames).squeeze(1)

        video_path = to_saved_video_file(all_generated_frames.cpu().numpy(), file_path='video.mp4',
                                         fps=cfg['hierarchical_infer']['output_fps'])
        wandb.log({"generated_video": wandb.Video(video_path, fps=1, format="mp4")})


if __name__ == "__main__":
    main()
