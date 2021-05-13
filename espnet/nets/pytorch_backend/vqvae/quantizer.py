import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

class VectorQuantizer(nn.Module):
    def __init__(self, z_num, z_dim, normalize=True, beta=0.01, reduction='mean'):
        super(VectorQuantizer, self).__init__()

        if normalize:
            self.target_norm = 1.0 # norm_scale * math.sqrt(z.size(2))
        else:
            self.target_norm = None

        self.embeddings = nn.Parameter( torch.randn(z_num, z_dim, requires_grad=True))

        self.embed_norm()

        self.z_num = z_num
        self.z_dim = z_dim
        self.normalize = normalize
        self.reduction = reduction
        self.quantize = True
        self.beta = beta

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self.embeddings.mul_(
                    self.target_norm / self.embeddings.norm(dim=1, keepdim=True)
                )

    def encode(self, z):
        # Flatten
        B,T,D = z.shape
        z = z.contiguous().view(-1, D)

        # Normalize
        if self.target_norm:
            z_norm = self.target_norm * z / z.norm(dim=1, keepdim=True)
            embeddings = self.target_norm * self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embeddings = self.embeddings
        # Calculate distances
        distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embeddings.pow(2), dim=1)
                    - 2 * torch.matmul(z_norm, embeddings.t()))            
        # # Quantization encode
        encoding_idx = torch.argmin(distances, dim=1)
        # Deflatten
        encoding_idx = encoding_idx.view(B, T)
        return encoding_idx


    def decode(self, z_id):
        # Flatten
        B,T = z_id.shape
        encoding_idx = z_id.flatten()
        # Normalize
        if self.target_norm:
            embeddings = self.target_norm * self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        else:
            embeddings = self.embeddings
        # Quantization decode
        z_vq = embeddings.index_select(dim=0, index=encoding_idx)
        # Deflatten
        z_vq = z_vq.view(B, T, -1)

        return z_vq


    def forward(self, z, z_lens=None):
        if not self.quantize:
            tensor_0 = torch.tensor(0.0, dtype=torch.float, device=z.device)
            return z, tensor_0, tensor_0, tensor_0

        # Flatten
        B,T,D = z.shape
        z = z.contiguous().view(-1, D)
        device = z.device

        # Normalize
        if self.target_norm:
            z_norm = self.target_norm * z / z.norm(dim=1, keepdim=True)
            self.embed_norm()
            embeddings = self.target_norm * self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embeddings = self.embeddings

        # Calculate distances
        distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embeddings.pow(2), dim=1)
                    - 2 * torch.matmul(z_norm, embeddings.t()))
            
        # Quantize
        encoding_idx = torch.argmin(distances, dim=1)
        z_vq = embeddings.index_select(dim=0, index=encoding_idx)

        vq_loss, entropy = self.calculate_losses(encoding_idx, z, z_vq, z_norm, z_lens=z_lens)
        update_detail = {'entropy':entropy.item()}

        z_vq = z_norm + (z_vq-z_norm).detach()

        # Deflatten
        z_vq = z_vq.view(B, T, D)

        # Output
        return z_vq, vq_loss, update_detail


    def calculate_losses(self, encoding_idx, z, z_vq, z_norm, z_lens=None):
        if z_lens is not None:
            z_masks = make_non_pad_mask(z_lens).to(z.device).flatten()
            encoding_idx = encoding_idx.masked_select(z_masks)
            z_masks = z_masks.unsqueeze(-1)
            z_norm = z_norm.masked_select(z_masks)
            z_vq = z_vq.masked_select(z_masks)
            z = z.masked_select(z_masks)

        encodings = torch.zeros(encoding_idx.size(0), self.z_num, device=z.device)
        encodings.scatter_(1, encoding_idx.unsqueeze(1), 1)

        avg_probs = torch.sum(encodings, dim=0) / encodings.size(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_qut_loss = F.mse_loss(z_vq, z_norm.detach(), reduction=self.reduction)
        z_enc_loss = F.mse_loss(z_vq.detach(), z_norm, reduction=self.reduction)
        if self.target_norm:
            z_enc_loss += F.mse_loss(z_norm, z, reduction=self.reduction)    # Normalization loss

        vq_loss = z_qut_loss + self.beta * z_enc_loss

        return vq_loss, perplexity

    def extra_repr(self):
        s = '{z_num}, {z_dim}'
        if self.normalize is not False:
            s += ', normalize=True'
        return s.format(**self.__dict__)


