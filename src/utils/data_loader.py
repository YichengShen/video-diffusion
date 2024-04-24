import os
import random

import torch


class DataLoader:
    def __init__(self, cfg, skip_frames=5):
        self.cfg = cfg
        self.device = torch.device(self.cfg['device'])
        self.batch_files = [os.path.join(cfg['dataset_dir'], f) for f in os.listdir(cfg['dataset_dir']) if
                            f.endswith('.pt')]
        print(f"Loaded {len(self.batch_files)} batches of data.")
        self.current_batch_data = None
        self.skip_frames = skip_frames

    def load_new_batch(self):
        selected_file = random.choice(self.batch_files)
        self.current_batch_data = torch.load(selected_file)
        # overfit the same batch
        self.current_batch_data = self.current_batch_data[:self.cfg['train_batch_size'],...]

    def get_batch(self, batch_size):
        if self.current_batch_data is None:
            self.load_new_batch()

        indices = torch.randint(0, len(self.current_batch_data), (batch_size,))

        batch_samples = self.current_batch_data[indices].squeeze().to(self.device)

        frames = self.cfg['num_frames']
        return batch_samples[:, :frames], batch_samples[:, [frames + self.skip_frames]]
