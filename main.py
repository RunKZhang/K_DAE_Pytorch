import numpy as np
import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.autoencoder import AutoEncoder, train_AE
from src.k_dae import KDae

if __name__ == "__main__":
    # x_train, y_train = load_data(dataset_name)
    # Load data
    transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),(0.3081,))])
    dir = '/home/runkaizhang/TOSHIBA_disk/data/MNIST'
    batch_size = 256
    train_set = datasets.MNIST(dir, train=True, download=False, transform=transforms)
    test_set = datasets.MNIST(dir, train=False, download=False, transform=transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # 10 clusters
    # n_cluster = 10
    # AE = AutoEncoder(input_dim=28 * 28,
    #                  hidden_dims=[500, 500, 2000],
    #                  latent_dim=10,
    #                  n_clusters=10)
    # print(AE)
    # trained_model = train_AE(AE, )

    KDAE = KDae(input_dim=28*28, latent_dim=10, num_cluster=10, pretrain_epoch=2, 
                pretrain_ae_dims=[500, 500, 2000], train_epoch=50, train_ae_dims=[250,100])
    model = KDAE._create_combination_model()
    print(model)
    
    # model = KDae(number_cluster=n_cluster, k_dae_epoch=40, epoch_ae=10, initial_epoch=80, dataset_name=dataset_name)
    # model.fit(x_train, y_train, dataset_name=dataset_name)
    # y_pred = model.predict(x_train)
    # cluster_performance(y_pred, y_train)