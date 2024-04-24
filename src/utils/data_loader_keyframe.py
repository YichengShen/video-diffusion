import os
import random

import torch


class KeyFrameDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg['device'])
        self.batch_files = [os.path.join(cfg['dataset_dir'], f) for f in os.listdir(cfg['dataset_dir']) if
                            f.endswith('.pt')]
        print(f"Loaded {len(self.batch_files)} batches of data.")
        self.current_batch_data = None
        self.limit_batch_size = cfg['keyframe_model']['limit_batch_size']
        self.skip_frames = cfg['keyframe_model']['num_frames_to_skip_per_keyframe']
        self.current_index = 0
        self.load_new_batch()

    def load_new_batch(self):
        selected_file = random.choice(self.batch_files)
        self.current_batch_data = torch.load(selected_file).to(self.device)

        if self.limit_batch_size is not None:
            # Overfit to a small subset
            self.current_batch_data = self.current_batch_data[:self.limit_batch_size, ...]

    def get_batch(self, batch_size):
        # Update this function to sequentially go through the data we want to overfit
        if self.current_batch_data is None:
            self.load_new_batch()

        indices = torch.randint(0, len(self.current_batch_data), (batch_size,))

        batch_samples = self.current_batch_data[indices].squeeze()
        if batch_size == 1:
            batch_samples = batch_samples.unsqueeze(0)
        batch_samples = batch_samples.to(self.device)

        frames = self.cfg['num_frames']
        print(batch_samples[:, :frames].shape)
        return batch_samples[:, :frames], batch_samples[:, [frames + self.skip_frames]]
