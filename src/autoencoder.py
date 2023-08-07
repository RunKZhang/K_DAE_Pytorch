import torch
import torch.nn as nn
from collections import OrderedDict
from torch.cuda import is_available

def train_AE(AE_model, epoch, optimizer, criterion, _dataloader):
    device = torch.device('cuda' if is_available() else 'cpu')
    log_interval = 300
        
    AE_model.train()
    for e in range(0, epoch):
            for batch_idx, (data, _) in enumerate(_dataloader):
                batch_size = data.size()[0]
                data = data.to(device).view(batch_size, -1)
                rec_X = AE_model(data)
                loss = criterion(data, rec_X)

                if batch_idx % log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx,
                                     loss.detach().cpu().numpy()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return AE_model

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = hidden_dims
        self.hidden_dims.append(latent_dim)
        self.dims_list = (hidden_dims +
                          hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim),
                        'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx])
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1]),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1])
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)
    
    

