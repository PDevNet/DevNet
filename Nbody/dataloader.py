import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ml_collections
from typing import Tuple
from Nbody.data.generate_dataset import generate_dataset
from Nbody.data.synthetic_sim import ChargedParticlesSim
import os


def load_data(batch_size=1, suffix=''):
    loc_train = np.load('Nbody/data/loc_train' + suffix + '.npy')
    vel_train = np.load('Nbody/data/vel_train' + suffix + '.npy')

    loc_valid = np.load('Nbody/data/loc_valid' + suffix + '.npy')
    vel_valid = np.load('Nbody/data/vel_valid' + suffix + '.npy')

    loc_test = np.load('Nbody/data/loc_test' + suffix + '.npy')
    vel_test = np.load('Nbody/data/vel_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    feat_train = torch.FloatTensor(feat_train)

    feat_valid = torch.FloatTensor(feat_valid)

    feat_test = torch.FloatTensor(feat_test)

    return feat_train, feat_valid, feat_test


class Nbody_Dataset(Dataset):
    """time series dataset class for generating rolling windows timeseries """

    def __init__(self, x, k, p):
        """[summary]

        Args:
            x (torch.tensor): has shape (N,K,T,C), N number of simulation, K is number particles, 
                            T timestep, C features size
            y ([type]): [description]
            k (int): look back window
            p (int): prediction time step
        """
        N, K, T, C = x.shape
        self.x = x.permute(0, 2, 1, 3)
        self.k = k
        self.p = p
        self.N = N
        self.T = T

    def __len__(self):
        return self.N*(self.T - self.p-self.k+1)

    def __getitem__(self, idx):
        t_idx = idx % (self.T - self.p-self.k+1)
        n_idx = torch.floor(torch.tensor(
            idx/(self.T - self.p-self.k+1))).to(torch.long)
        return self.x[n_idx, t_idx:t_idx + self.k], self.x[n_idx, t_idx + self.k + self.p-1, :, :2]


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="Nbody/data/",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    data_loc = data_root+'loc_train.npy'
    if os.path.exists(data_loc):
        pass
    else:
        sim = ChargedParticlesSim(noise_var=0.0, n_balls=5)
        print("Generating {} train simulations".format(1000))
        loc_train, vel_train, edges_train = generate_dataset(sim, 1000,
                                                             5000,
                                                             10)

        print("Generating {} validation simulations".format(500))
        loc_valid, vel_valid, edges_valid = generate_dataset(sim, 500,
                                                             5000,
                                                             10)

        print("Generating {} test simulations".format(1000))
        loc_test, vel_test, edges_test = generate_dataset(sim, 1000,
                                                          5000,
                                                          10)

        np.save('Nbody/data/loc_train' + '.npy', loc_train)
        np.save('Nbody/data/vel_train' + '.npy', vel_train)
        np.save('Nbody/data/edges_train' + '.npy', edges_train)

        np.save('Nbody/data/loc_valid' + '.npy', loc_valid)
        np.save('Nbody/data/vel_valid' + '.npy', vel_valid)
        np.save('Nbody/data/edges_valid' + '.npy', edges_valid)

        np.save('Nbody/data/loc_test' + '.npy', loc_test)
        np.save('Nbody/data/vel_test' + '.npy', vel_test)
        np.save('Nbody/data/edges_test' + '.npy', edges_test)
    feat_train, feat_valid, feat_test = load_data(config.batch_size)
    train_data = Nbody_Dataset(feat_train, config.k, config.p)
    valid_data = Nbody_Dataset(feat_valid, config.k, config.p)
    test_data = Nbody_Dataset(feat_test, config.k, config.p)

    training_loader = DataLoader(
        train_data, batch_size=config.batch_size, num_workers=num_workers)
    val_loader = DataLoader(
        valid_data, batch_size=config.batch_size, num_workers=num_workers)
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size, num_workers=num_workers)

    dataloaders = {"train": training_loader, "validation": val_loader}
    return dataloaders, test_loader
