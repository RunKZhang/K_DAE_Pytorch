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

    def forward(self, x, kmeans_label):
        # use two lists to get the index corresponding to cluster and output
        output_list = []
        idx_list = []
        for i in range(self.n_clusters):
            index = torch.argwhere(kmeans_label==i)
            # print(index)
            # print(index.shape)
            value = self._k_dae[i](x[index.squeeze()])
            idx_list.append(index)
            output_list.append(value)
        return idx_list, output_list
        