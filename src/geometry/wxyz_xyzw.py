import torch

def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)

def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:], q[..., 0:1]], dim=-1)