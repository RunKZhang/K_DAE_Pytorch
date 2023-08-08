import torch
import torch.nn as nn
from collections import OrderedDict
from torch.cuda import is_available
from src.autoencoder import AutoEncoder

class K_DAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super(K_DAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # build the _k_dae in modulelist
        self._k_dae = nn.ModuleList()
        for i in range(0, self.n_clusters):
            self._k_dae.append(AutoEncoder(self.input_dim, hidden_dims.copy(), # use copy instead of directly pass
                                          self.latent_dim, 1))

    def forward(self, x):
        # output the min value obtained by l2 norm
        output_list = []
        for i in range(self.n_clusters):
            value = self._k_dae[i](x)
            value = value.unsqueeze(dim=1)
            # print(value.shape)
            output_list.append(value)
        output_tensor = torch.cat(output_list, dim=1)
        x_reshape = x.unsqueeze(1).expand(-1,10,-1)
        l2_norm = torch.norm(output_tensor-x_reshape, dim=-1)
        # print(f'l2_norm:{l2_norm}')
        # print(f'l2_norm.shape :{l2_norm.shape}')
        min_indices = torch.argmin(l2_norm, dim=-1)
        # print(f'min_indices: {min_indices.shape}')
        # print(f'min_indices: {min_indices}')
        selected_tensor = output_tensor[torch.arange(output_tensor.size(0)),min_indices]
        
        # print(f'selected_tensor: {selected_tensor.shape}')
        # print(output_tensor.shape)
        return selected_tensor, min_indices
        