import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Identity, Module


"""
From 'Were RNNs All We Needed?' by Feng et al., 2024.
Code adapted from https://github.com/lucidrains/minGRU-pytorch/blob/main/minGRU_pytorch/minGRU.py.

https://arxiv.org/abs/2410.01201v1
"""

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def heinsen_associative_scan_log(log_coeffs, log_values):

    """Parallel scan based on Heinsen (2023)
    https://github.com/glassroom/heinsen_sequence"""

    a_star              = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star  = (log_values - a_star).logcumsumexp(dim = 1)
    log_h               = a_star + log_h0_plus_b_star
    return log_h.exp()


def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(nn.Module):

    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out  = default(proj_out, expansion_factor != 1.)

        self.to_h_and_z = Linear(dim, dim_inner * 2, bias = False)
        self.to_out     = Linear(dim_inner, dim, bias = False) if proj_out else Identity()

    def forward(self, x, prev_h = None, return_next_prev_h = False):
        seq_len = x.shape[1]
        h, z = self.to_h_and_z(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            h = g(h)                                                        # ~h_t = Linear_dh(x_t)              candidate hidden state
            z = z.sigmoid()                                                 # z_t  = sigmoid(Linear_dz(x_t))     update gate
            out = torch.lerp(prev_h, h, z) if exists(prev_h) else (h * z)   # (1 - z) * prev_h + z * ~h_t
        else:
            # parallel

            log_coeffs = -F.softplus(z)

            log_z       = -F.softplus(-z)
            log_tilde_h = log_g(h)
            log_values  = log_z + log_tilde_h

            if exists(prev_h):
                log_values = torch.cat((prev_h.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_h = out[:, -1:]                                           # out = [h1, h2, ..., hT] and next_prev_h = hT

        out = self.to_out(out)                                              # y_t = C*h_t = Linear_dy(h_t)

        if not return_next_prev_h:
            return out

        return out, next_prev_h