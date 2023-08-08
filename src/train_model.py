import torch
import torch.nn as nn
from src.autoencoder import AutoEncoder, train_AE
from src.utils import pretrain_k_means
from torch.cuda import is_available
import numpy as np
from torch.utils.data import Subset, DataLoader
from src.k_dae import K_DAE
import warnings
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

class Train_Model(object):
    def __init__(self, input_dim, latent_dim, n_clusters, pretrain_epoch, pretrain_ae_dims,
                 train_epoch, train_ae_dims, tunning_epoch, train_dataset, test_dataset):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.pretrain_epoch = pretrain_epoch
        self.train_epoch = train_epoch
        self.pretrain_ae_dims = pretrain_ae_dims
        self.train_ae_dims = train_ae_dims
        self.tunning_epoch = tunning_epoch
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.embedded_features = None
        self.device = torch.device('cuda' if is_available() else 'cpu')
        self.pretrain_kmeans_labels = None # store the kmeans labels in stage 1 
        self.pretrain_ae = None # initial AE used to do kmeans clustering
        self.k_dae = None # several AEs in the paper, each represents the centroid
        

        self.is_entire_pretrain = False # Flag to check step 1: pretrain on entire dataset and kmeans
        self.is_separate_train = False # Flag to check step 2: pretrain on individual AEs

    # used to initial the autoencoder trained on entire dataset.
    def _initial_clustering(self, verbose=True):
        # print(self.device)
        if verbose:
            print('========== Start Entire Dataset pretraining ==========')

        self.pretrain_ae = AutoEncoder(self.input_dim, self.pretrain_ae_dims,
                                       self.latent_dim, self.n_clusters).to(self.device)
        lr = 1e-4
        wd = 5e-4
        batch_size = 256
        pretrain_criterion = nn.MSELoss()
        pretrain_ae_optimizer = torch.optim.Adam(self.pretrain_ae.parameters(),
                                          lr=lr,
                                          weight_decay=wd)
        
        _dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.pretrain_ae = train_AE(self.pretrain_ae, self.pretrain_epoch, pretrain_ae_optimizer,
                                 pretrain_criterion, _dataloader)
        
        if verbose:
            print('========== Catch labels ==========')      

        self.pretrain_ae.eval()
        batch_X = []
        
        # extract the latent features from the initial AE
        batch_loader = DataLoader(self.train_dataset, batch_size=1024, shuffle=False)
        
        for batch_idx, (data, _) in enumerate(batch_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.pretrain_ae(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        print(batch_X.shape) # (60000,10)

        # Use KMeans to find labels
        pretrain_k_means_model = pretrain_k_means(X=batch_X, num_clusters=self.n_clusters)
        self.pretrain_kmeans_labels = pretrain_k_means_model.labels_

        self.is_entire_pretrain = True
        return None
    
    def _separate_pretrain(self, verbose=True):
        self.k_dae = K_DAE(self.input_dim, self.train_ae_dims, 
                           self.latent_dim, self.n_clusters).to(self.device)
        # print(self.k_dae)
        for i in range(self.n_clusters):
            indices = np.argwhere(self.pretrain_kmeans_labels==i)
            _subset_dataset = Subset(self.train_dataset, indices.squeeze()) # must use squeeze(), the indices shape is (N, 1)

            print(f'length of subset_dataset:{len(_subset_dataset)}')
            
            if verbose:
                print(f'========== Start {i}th cluster AE pretrain ==========')
            # optimization parameters initialization
            lr = 1e-4
            wd = 5e-4
            batch_size = 256
            sep_pt_criterion = nn.MSELoss()
            sep_pt_ae_optimizer = torch.optim.Adam(self.k_dae._k_dae[i].parameters(), lr=lr, weight_decay=wd)
            _dataloader = DataLoader(_subset_dataset, batch_size=batch_size, shuffle=True)
            
            self.k_dae._k_dae[i] = train_AE(self.k_dae._k_dae[i], self.train_epoch, 
                                     sep_pt_ae_optimizer, sep_pt_criterion,
                                     _dataloader)

        self.is_separate_train = True
    
    def _loss(self):
        return None
    def _tunning(self, verbose=True):
        if verbose:
            print(f'========== Tunning the parameters ==========')
        
        lr = 1e-4
        wd = 5e-4
        batch_size = 256
        tun_criterion = nn.MSELoss()
        # tun_optimizer = torch.optim.Adam(self.k_dae._k_dae.parameters(), lr=lr, weight_decay=wd)
        tun_optimizer = torch.optim.Adam(self.k_dae.parameters(), lr=lr, weight_decay=wd)
        _dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        log_interval = 300
        # self.k_dae._k_dae.train()
        
        for e in range(0, self.tunning_epoch):
            
            self.k_dae.train()
            for batch_idx, (data, _) in enumerate(_dataloader):
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                output, _ = self.k_dae(data)
                loss = tun_criterion(data, output)

                if batch_idx % log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx,
                                     loss.detach().cpu().numpy()))

                tun_optimizer.zero_grad()
                loss.backward()
                tun_optimizer.step()
            
            self.k_dae.eval()
            NMI, ARI = self.evalute()
            print('\nEpoch: {:02d} | NMI: {:.3f} | ARI: {:.3f}\n'.format(e, NMI, ARI))
        return None
    
    def evalute(self):
        batch_size = 256
        _dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        y_test = []
        y_pred = []
        for data, target in _dataloader:
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)
            _, cluster = self.k_dae(data)
            cluster = cluster.detach().cpu().numpy()
            
            # to keep dimension same
            new_cluster = cluster[:,np.newaxis]

            y_test.append(target.view(-1, 1).numpy())
            # print(f'y_test.shape:{target.view(-1, 1).numpy().shape}')
            y_pred.append(new_cluster)
            # y_pred.append(model.kmeans.update_assign(latent_X).reshape(-1, 1))

        y_test = np.vstack(y_test).reshape(-1)
        y_pred = np.vstack(y_pred).reshape(-1)
        return (normalized_mutual_info_score(y_test, y_pred),
                adjusted_rand_score(y_test, y_pred))
        
    
    def fit(self):
        # step 1: check if trained on entire dataset and obtain initial kmeans
        if self.is_entire_pretrain:
            # step 2: check if separately train each AE corresponding to cluster
            if self.is_separate_train:
                # step 3: tuning by using combine loss
                print('tuning now!')
                self._tunning()
            else:
                warnings.warn('separate pretrain should be runned before final tuning', RuntimeWarning)
                warnings.warn('Begin to run self.separate_pretrain', RuntimeWarning)
                self._separate_pretrain()
                self.fit()
                # raise warnings and errors
        else:
            warnings.warn('initial clustering should be runned first!', RuntimeWarning)
            warnings.warn('Begin to run self._initial_clustering', RuntimeWarning)
            self._initial_clustering()                    
            self.fit()
            # raise warinings and errors
        
        return None
