import numpy as np
import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.autoencoder import AutoEncoder, train_AE
from src.k_dae import K_DAE
from src.train_model import Train_Model

# make a numpy array dataset for the original dataset, for select appointment targets.
def _make_dataset_numpy(_dataset):
    sample_np = []
    target_np = []
    for sample, target in _dataset:
        sample_np.append(sample)
        target_np.append(target)
    sample_np = np.vstack(sample_np)
    target_np = np.vstack(target_np)

    return sample_np, target_np


if __name__ == "__main__":
    # x_train, y_train = load_data(dataset_name)
    # Load data
    transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),(0.3081,))])
    dir = '/home/runkaizhang/TOSHIBA_disk/data/MNIST'
    batch_size = 256
    train_set = datasets.MNIST(dir, train=True, download=False, transform=transforms)
    test_set = datasets.MNIST(dir, train=False, download=False, transform=transforms)

    # samples, targets = _make_dataset_numpy(train_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    train_model = Train_Model(input_dim=28*28, latent_dim=10, n_clusters=10, pretrain_epoch=1, 
                pretrain_ae_dims=[500, 500, 2000], train_epoch=1, train_ae_dims=[250,100],
                tunning_epoch = 5, train_dataset=train_set, test_dataset=test_set)
    # train_model._initial_clustering(train_set, train_loader)
    train_model.fit()
    # train_model._tunning()
    # selected_indices = [idx for idx, target in enumerate(train_set.targets) if target == 1]
    # print(len(selected_indices))
    # subset = Subset(train_set, selected_indices)
    # for idx, (data, label) in enumerate(subset):
    #     print(data.shape)        
    #     print(label)
    
    # 10 clusters
    # n_cluster = 10
    # AE = AutoEncoder(input_dim=28 * 28,
    #                  hidden_dims=[500, 500, 2000],
    #                  latent_dim=10,
    #                  n_clusters=10)
    # print(AE)
    # trained_model = train_AE(AE, )

    # KDAE = KDae(input_dim=28*28, latent_dim=10, num_cluster=10, pretrain_epoch=2, 
    #             pretrain_ae_dims=[500, 500, 2000], train_epoch=50, train_ae_dims=[250,100],
    #             train_dataset_np_samples = samples, train_dataset_np_targets = targets,
    #             train_set = train_set)
    # # KDAE._initial_clustering(train_loader)
    # model = KDAE._create_combination_model()
    # KDAE._separate_pre_train()
    # KDAE.fit()
    # print(model)
    
    test_batch, test_label = next(iter(train_loader))
    print(test_batch.shape)
    batch_size = test_batch.size()[0]
    data = test_batch.view(batch_size, -1)
    print(data)
    model = K_DAE(28*28, [500,100], 10, 10)
    # print(model._k_dae.parameters())
    # for module in model._k_dae:
    #     print(module)
    # index, value = model.forward(data,test_label)
    output_tensor = model(data)
    # print(output_tensor.shape)
    data_new = data.unsqueeze(1).expand(-1, 10, -1)
    print(output_tensor)
    # print(data)
    # print(data_new)
    # print(data-output_tensor)
    # print(output_tensor.shape)
    # print(value[0].shape)
    # model = KDae(number_cluster=n_cluster, k_dae_epoch=40, epoch_ae=10, initial_epoch=80, dataset_name=dataset_name)
    # model.fit(x_train, y_train, dataset_name=dataset_name)
    # y_pred = model.predict(x_train)
    # cluster_performance(y_pred, y_train)