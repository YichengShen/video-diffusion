import os
import random

import torch


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg['device'])
        self.batch_files = [os.path.join(cfg['dataset_dir'], f) for f in os.listdir(cfg['dataset_dir']) if
                            f.endswith('.pt')]
        print(f"Loaded {len(self.batch_files)} batches of data.")

    def get_batch(self, batch_size):
        selected_file = random.choice(self.batch_files)
        batch_data = torch.load(selected_file)

        if len(batch_data) < batch_size:
            indices = torch.randint(0, len(batch_data), (batch_size,))
        else:
            indices = torch.randperm(len(batch_data))[:batch_size]

        batch_samples = batch_data[indices].squeeze().to(self.device)

        frames = self.cfg['num_frames']
        return batch_samples[:, :frames], batch_samples[:, [frames]]
