import torch
import torch.nn as nn
from src.autoencoder import AutoEncoder, train_AE
from src.utils import pretrain_k_means
from torch.cuda import is_available
import numpy as np

class KDae(nn.Module):
    def __init__(self, input_dim, latent_dim, num_cluster, pretrain_epoch, pretrain_ae_dims, 
                 train_epoch, train_ae_dims):
        super(KDae, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_cluster = num_cluster
        self.pretrain_epoch = pretrain_epoch
        self.train_epoch = train_epoch
        self.pretrain_ae_dims = pretrain_ae_dims
        self.train_ae_dims = train_ae_dims
        self.pretrain_ae = None # the pretrain ae model
        self.pretrain_kmeans_labels = None
        self.kdae = nn.ModuleDict() # the kdae model
        self.device = torch.device('cuda' if is_available() else 'cpu')

    # used to initial the autoencoder trained on entire dataset.
    def _initial_clustering(self, _dataloader, verbose=True):
        # print(self.device)
        self.pretrain_ae = AutoEncoder(self.input_dim, self.pretrain_ae_dims,
                                       self.latent_dim, self.num_cluster).to(self.device)
        lr = 1e-4
        wd = 5e-4
        pretrain_criterion = nn.MSELoss()
        pretrain_ae_optimizer = torch.optim.Adam(self.pretrain_ae.parameters(),
                                          lr=lr,
                                          weight_decay=wd)
        if verbose:
            print('========== Start Entire Dataset pretraining ==========')

        self.pretrain_ae = train_AE(self.pretrain_ae, self.pretrain_epoch, pretrain_ae_optimizer,
                                 pretrain_criterion, _dataloader)
        self.pretrain_ae.eval()
        batch_X = []

        if verbose:
            print('========== Catch labels ==========')      
        
        # Make the entire dataset from dataloader
        for batch_idx, (data, _) in enumerate(_dataloader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.pretrain_ae(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)

        # Use KMeans to find labels
        pretrain_k_means_model = pretrain_k_means(X=batch_X, num_clusters=self.num_cluster)
        self.pretrain_kmeans_labels = pretrain_k_means_model.labels_

        return None
    
    def _create_combination_model(self):
        hidden_dims = self.train_ae_dims
        for i in range(0, self.num_cluster):
            name = 'ae_cluster_'+str(i)
            # print(hidden_dims)
            self.kdae[name] = AutoEncoder(self.input_dim, hidden_dims.copy(), # use copy instead of directly pass
                                          self.latent_dim, 1)
        
        return self.kdae
    
    @staticmethod
    def k_dae_loss(y_true, y_pred):
        return None
    
    def fit(self, x_data, y_data=None, dataset_name='temp', save_init_label=True, is_pre_train=True):
        return None
    
    def predict(self, x_data):
        return None