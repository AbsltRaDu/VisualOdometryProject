import torch

class PoseNormalizerLie:

    def __init__(self, eps=1e-8):
        self.eps = eps

    def fit(self, data):
        self.mean_t = data[..., :3].mean(0)
        self.std_t = data[..., :3].std(0)

        self.mean_r = data[..., 3:].mean(0)
        self.std_r = data[..., 3:].std(0)

    def normalize(self, data):

        t = (data[..., :3] - self.mean_t) / (self.std_t + self.eps)
        r = (data[..., 3:] - self.mean_r) / (self.std_r + self.eps)

        return torch.cat([t, r], dim=-1)

    def denormalize(self, data):

        t = data[..., :3] * (self.std_t + self.eps) + self.mean_t
        r = data[..., 3:] * (self.std_r + self.eps) + self.mean_r

        return torch.cat([t, r], dim=-1)