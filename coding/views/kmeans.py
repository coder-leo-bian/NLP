from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from . import config


def load_data():
    np.random.seed(0)
    x = np.random.randn(200, 2)
    x2 = x + (4, 0)
    x3 = x + (0, 5)
    x4 = x + (4, 5)
    data = np.vstack((x, x2, x3, x4))
    train_model(data)


def train_model(data):
    model = KMeans(init='k-means++', n_clusters=4, n_init=10)
    model.fit(data)
    pred_y = model.predict(data)
    centers = model.cluster_centers_
    print(model.get_params())
    l, r = min(data[:, 0]), max(data[:, 0])
    b, t = min(data[:, 1]), max(data[:, 1])
    grid_x = np.meshgrid(np.linspace(l, r, 500),
                         np.linspace(b, t, 500))
    flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
    flat_y = model.predict(flat_x)
    grid_y = flat_y.reshape(grid_x[0].shape)
    cm_light = mp.colors.ListedColormap(['orangered', 'deepskyblue', 'limegreen', 'whitesmoke'])
    cm_dark = mp.colors.ListedColormap(['whitesmoke', 'limegreen', 'deepskyblue',  'orangered'])
    plt.figure()
    plt.pcolormesh(grid_x[0], grid_x[1], grid_y)
    plt.scatter(data[:, 0], data[:, 1], c=pred_y, cmap=cm_dark)
    plt.scatter(centers[:, 0], centers[:, 1], c='b', marker='+', s=300)
    plt.text(-1, 0, r'{}'.format(np.round(centers[0], 3)), fontsize=12, color='k')
    plt.text(-1, 5, r'{}'.format(np.round(centers[1], 3)), fontsize=12, color='k')
    plt.text(3, 0, r'{}'.format(np.round(centers[3], 3)), fontsize=12, color='k')
    plt.text(3, 5, r'{}'.format(np.round(centers[2], 3)), fontsize=12, color='k')
    plt.savefig(config.APP_IMAGES_TXT + 'k_means.png')
    plt.show()





# load_data()

