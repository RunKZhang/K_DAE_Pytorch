import torch
import torch.nn as nn
from src.autoencoder import AutoEncoder, train_AE
from src.utils import pretrain_k_means
from torch.cuda import is_available
import numpy as np
from torch.utils.data import Subset, DataLoader

class Train_Model(object):
    def __init__(self, input_dim, latent_dim, n_clusters, pretrain_epoch, pretrain_ae_dims,
                 train_epoch, train_ae_dims):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_cluster = n_clusters
        self.pretrain_epoch = pretrain_epoch
        self.train_epoch = train_epoch
        self.pretrain_ae_dims = pretrain_ae_dims
        self.train_ae_dims = train_ae_dims
        self.embedded_features = None
        self.device = torch.device('cuda' if is_available() else 'cpu')
        self.pretrain_kmeans_labels = None # store the kmeans labels in stage 1 

        self.is_entire_pretrain = False # Flag to check step 1: pretrain on entire dataset and kmeans
        self.is_separate_train = False # Flag to check step 2: pretrain on individual AEs

    # used to initial the autoencoder trained on entire dataset.
    def _initial_clustering(self, _dataloader, verbose=True):
        # print(self.device)
        if verbose:
            print('========== Start Entire Dataset pretraining ==========')

        self.pretrain_ae = AutoEncoder(self.input_dim, self.pretrain_ae_dims,
                                       self.latent_dim, self.num_cluster).to(self.device)
        lr = 1e-4
        wd = 5e-4
        pretrain_criterion = nn.MSELoss()
        pretrain_ae_optimizer = torch.optim.Adam(self.pretrain_ae.parameters(),
                                          lr=lr,
                                          weight_decay=wd)
        
        self.pretrain_ae = train_AE(self.pretrain_ae, self.pretrain_epoch, pretrain_ae_optimizer,
                                 pretrain_criterion, _dataloader)
        
        if verbose:
            print('========== Catch labels ==========')      

        self.pretrain_ae.eval()
        batch_X = []

        for batch_idx, (data, _) in enumerate(_dataloader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.pretrain_ae(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        # print(batch_X.shape) # (60000,10)

        # Use KMeans to find labels
        pretrain_k_means_model = pretrain_k_means(X=batch_X, num_clusters=self.num_cluster)
        self.pretrain_kmeans_labels = pretrain_k_means_model.labels_

        self.is_entire_pretrain = True
        return None
    
    def fit(self):
        # step 1: check if trained on entire dataset and obtain initial kmeans
        if self.is_entire_pretrain:
            # step 2: check if separately train each AE corresponding to cluster
            if self.is_separate_train:
                # step 3: tuning by using combine loss
                return
            else:
                return 
                # self._separate_pre_train()
                # raise warnings and errors
        else:
            return 
            # self._initial_clustering()                    
            # raise warinings and errors
        
        return None


class Train_model(nn.Module):
    def __init__(self, input_dim, latent_dim, num_cluster, pretrain_epoch, pretrain_ae_dims, 
                 train_epoch, train_ae_dims, train_dataset_np_samples, train_dataset_np_targets,
                 train_set):
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
        self.train_dataset_np_samples = train_dataset_np_samples
        self.train_dataset_np_targets = train_dataset_np_targets
        self.train_set = train_set
        # self.kdae = nn.ModuleDict() # the kdae model
        self.kdae = nn.ModuleList() # the kdae model
        self.device = torch.device('cuda' if is_available() else 'cpu')

        self.is_entire_pretrain = False
        self.is_separate_train = False

    
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
        
        batch_X = self.train_dataset_np_targets

        # Use KMeans to find labels
        pretrain_k_means_model = pretrain_k_means(X=batch_X, num_clusters=self.num_cluster)
        self.pretrain_kmeans_labels = pretrain_k_means_model.labels_

        self.is_entire_pretrain = True
        return None
    
    def _create_combination_model(self):
        hidden_dims = self.train_ae_dims
        for i in range(0, self.num_cluster):
            self.kdae.append(AutoEncoder(self.input_dim, hidden_dims.copy(), # use copy instead of directly pass
                                          self.latent_dim, 1))
        self.kdae.to(self.device)
        return self.kdae
    
    def _separate_pre_train(self):
        for i in range(0,self.num_cluster):
            # selected_indices = [idx for idx, target in enumerate(self.train_set.targets) if target == 1]
            selected_indices = [idx for idx in range(0,self.pretrain_kmeans_labels.shape[0]) if self.pretrain_kmeans_labels[idx] == 1]
            subset = Subset(self.train_set, selected_indices)
            sub_loader = DataLoader(subset, batch_size=256, shuffle=False)
            lr = 1e-4
            wd = 5e-4
            _criterion = nn.MSELoss()
            _optimizer = torch.optim.Adam(self.kdae[i].parameters(),
                                          lr=lr,
                                          weight_decay=wd)
            print(f'========== Start {i}th cluster AE pretrain==========')
            self.kdae[i] = train_AE(self.kdae[i], self.pretrain_epoch, 
                                    _optimizer, _criterion, sub_loader)

        return None
    # @staticmethod
    def k_dae_loss(y_pretrain_kmeans, y_pred):
        
        return None
    
    def fit(self):
        # step 1: check if trained on entire dataset and obtain initial kmeans
        if self.is_entire_pretrain:
            # step 2: check if separately train each AE corresponding to cluster
            if self.is_separate_train:
                # step 3: tuning by using combine loss
                return
            else:
                self._separate_pre_train()
        else:
            self._initial_clustering()                    
        
        return None
    
    def predict(self, x_data):
        return None