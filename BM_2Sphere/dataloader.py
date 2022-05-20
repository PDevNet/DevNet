
import os
import pathlib
import numpy as np
import torch
from BM_2Sphere.utils import split_data, load_data, save_data

import ml_collections
from typing import Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    training_set = BM_2Sphere(
        partition="train",
    )
    val_set = BM_2Sphere(
        partition="val",
    )
    test_set = BM_2Sphere(
        partition="test",
    )

    training_loader = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloaders = {"train": training_loader, "validation": validation_loader}
    return dataloaders, test_loader


class BM_2Sphere(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):

        data_loc = pathlib.Path(
            'BM_2Sphere/data/processed_data')

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            print("Generate Brownian motions on 2Sphere")
            x_input, y = BM_on_S2_projected(20000, 500, 1, [0, 0, 1])
            train_X, val_X, test_X = x_input[:16000], x_input[16000:18000], x_input[18000:20000]
            train_y, val_y, test_y = y[:16000], y[16000:18000], y[18000:20000]

            save_data(
                data_loc,
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y
            )
        X, y = self.load_data(data_loc, partition)
        super(BM_2Sphere, self).__init__(X, y)

    @staticmethod
    def load_data(data_loc, partition):

        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == 'val':
            X = tensors["val_X"]
            y = tensors["val_y"]

        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError(
                "the set {} is not implemented.".format(set))

        return X, y


def BM_on_S2_projected(N, n_T, T, start_point):
    '''
    Simulate Brownian motions on 3d unit sphere by polar coordinate.

    inputs: 
        N               batch size, integer
        n_T             length of input sequence / partition numbers of time interval, integer
        T               time duration, positive real number
        start_point     starting point on the sphere, list of floats, length three

    output: 
        W_paths         the grounded simulated Brownian motion
        X_paths         collection of N paths with n_T time dimension (without initial point)

    dW is simulated by the by the random walk approximation with iid uniform distributions sqrt(12*dt)*U(-0.5,0.5). It is two dimensional.
    '''
    dt = T/n_T
    W_paths = []
    X_paths = []
    for i in tqdm(range(N)):
        t = 0.0
        w = 0.0
        X = []
        W = []
        p = start_point
        while t <= T:
            dW = np.random.uniform(-0.5, 0.5, size=2)*((12*dt)**0.5)
            w += dW
            c = 1.0/(p[0]**2+p[2]**2)**0.5
            e1 = np.array([c*p[2], 0, -1*c*p[0]])
            e2 = np.array([-1.0*c*p[0]*p[1], -1.0*c *
                          (p[0]**2+p[2]**2), c*p[1]*p[2]])
            v = (e1*np.cos(np.pi*(dW[0]/((12*dt)**0.5))) + e2 *
                 np.sin(np.pi*(dW[0]/((12*dt)**0.5))))*(2)**0.5*dW[1]
            p = (p+v)/(np.dot((p+v), (p+v)))**0.5
            W.append(copy.deepcopy(w))

            X.append(p)
            t += dt
        W_paths.append(W)
        X_paths.append(X)
    return torch.FloatTensor(W_paths), torch.FloatTensor(X_paths)


def visualize(X):
    x1_points = []
    x2_points = []
    x3_points = []
    for i in range(len(X)):
        x1_points.append(X[i][0])
        x2_points.append(X[i][1])
        x3_points.append(X[i][2])
    v = np.linspace(0, 2*np.pi, 100)
    u = np.linspace(0, np.pi, 100)

    x_sphere = np.outer(np.sin(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.cos(v))
    z_sphere = np.outer(np.cos(u), np.ones_like(v))
    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax2.view_init(elev=ele, azim=azm)
    #ax.dist = 7
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color="y", alpha=0.2)
    ax.plot3D(x1_points, x2_points, x3_points, 'gray')
