import copy
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from fastprogress import progress_bar

from src.models.ema import EMA


class FrameDiffusion:
    def __init__(self, cfg, dataset, model, accelerator):
        self.cfg = cfg
        self.device = torch.device(self.cfg['device'])
        self.dataset = dataset
        self.accelerator = accelerator
        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.mse = None
        self.fp16 = False

        self.noise_steps = self.cfg['noise_scheduler']['num_diffusion_steps']
        self.beta = None
        self.alpha = None
        self.alpha_hat = None
        self.setup_noise_schedule(self.cfg['noise_scheduler']['beta_start'],
                                  self.cfg['noise_scheduler']['beta_end'])

    def setup_noise_schedule(self, beta_start, beta_end):
        """DDPM scheduler"""
        self.beta = torch.linspace(
            beta_start, beta_end, self.noise_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device, dtype=torch.long)

    def noise_images(self, x, t):
        """Add noise to images at instant t."""
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        noise = torch.randn_like(x)
        x_scaled = torch.sqrt(alpha_hat) * x
        noise_scaled = torch.sqrt(1 - alpha_hat) * noise
        noisy_image = x_scaled + noise_scaled
        return noisy_image, noise

    def get_batch(self, batch_size):
        frames = self.cfg['num_frames']
        ix = torch.randint(len(self.dataset), (batch_size,))
        b = self.dataset[ix].squeeze().to(self.device)
        return b[:, :frames], b[:, [frames]]

    @torch.inference_mode()
    def sample(self, batch_size=10, use_ema=False):
        logging.info(f"Sampling new images....")
        previous_frames, next_frames = self.get_batch(batch_size=batch_size)
        model = self.ema_model if use_ema else self.model
        return self._sample(model, previous_frames, next_frames)

    @torch.inference_mode()
    def _sample(self, model, previous_frames, next_frames):
        model.eval()
        bs = len(previous_frames)
        x = torch.randn_like(next_frames)
        for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
            t = (torch.ones(bs) * i).long().to(self.device)
            all_frames = torch.cat([previous_frames, x], dim=1)
            predicted_noise = model(all_frames, t).sample
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                         * predicted_noise) + torch.sqrt(beta) * noise

        return torch.cat([previous_frames, x], dim=1)

    def sample_more(self, model, previous_frames, n=1):
        """Generate n extra frames one by one"""
        bs, f, h, w = previous_frames.shape
        next_frames = []
        for _ in progress_bar(range(n), leave=True):
            one_frame = torch.empty(bs, 1, h, w, device=self.device)
            prediction = self._sample(model, previous_frames, one_frame)
            previous_frames = prediction[:, 1:, ...]
            next_frames.append(prediction[:, -1:, ...])  # last frame
        return torch.cat(next_frames, dim=1)

    def log_images(self):
        """Log images to wandb and save them to disk"""

        def to_image(img):
            return wandb.Image(torch.cat(img.split(1), dim=-1).cpu().numpy())

        sampled_images = self.sample(use_ema=False)
        wandb.log({"sampled_images": [to_image(img) for img in sampled_images]})
        ema_sampled_images = self.sample(use_ema=True)
        wandb.log({"ema_sampled_images": [to_image(img) for img in ema_sampled_images]})

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.scheduler.step()
        self.optimizer.step()

        # update EMA model
        self.ema.step_ema(self.ema_model, self.model)

    def train(self, train_steps):
        self.model.train()
        pbar = progress_bar(range(train_steps))
        for i in pbar:
            previous_frames, next_frames = self.get_batch(batch_size=self.cfg['train_batch_size'])
            previous_frames, next_frames = self.accelerator.prepare(previous_frames, next_frames)

            t = self.sample_timesteps(previous_frames.shape[0])  # batch size
            x_t, noise = self.noise_images(next_frames, t)

            # Stack previous frames with placeholder of next frames
            all_frames = torch.cat([previous_frames, x_t], dim=1)

            # forward
            with torch.autocast("cuda", dtype=torch.float16) if self.fp16 else nullcontext():
                predicted_noise = self.model(all_frames, t).sample
                loss = self.mse(noise, predicted_noise)

            # step
            self.train_step(loss)
            if self.cfg['wandb']['use_wandb']:
                wandb.log({"train_mse": loss.item(),
                           "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"

            if i % self.cfg['wandb']['steps_to_log_images'] == 0:
                self.log_images()

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(
            os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(
            os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, model_name, epoch=-1):
        """Save model locally and on wandb"""
        Path(os.path.join("weights", model_name)).mkdir(
            parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("weights", model_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("weights", model_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("weights", model_name, f"optim.pt"))
        if self.cfg['wandb']['save_weights_to_wandb']:
            at = wandb.Artifact(
                "weights", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
            at.add_dir(os.path.join("weights", model_name))
            wandb.log_artifact(at)

    def prepare(self, optimizer, scheduler):
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, optimizer(self.model.parameters()))
        self.scheduler = scheduler(self.optimizer)
        self.mse = nn.MSELoss(reduction="sum")
        self.ema = EMA(self.cfg['ema']['decay'])

    def fit(self):
        self.train(self.cfg['train_steps'])
