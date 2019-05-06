from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as mp
from . import config

warnings.filterwarnings("ignore")   # UndefinedMetricWarning


def load_data():
    np.random.seed(0)
    c1 = 990
    c2 = 10
    x1 = 3*np.random.randn(c1, 2)
    x2 = 0.5*np.random.randn(c2, 2) + (4, 4)
    x = np.vstack((x1, x2))
    y = np.ones(1000)
    y[:c1] = -1
    print(x.shape)
    print()
    print(y)
    train_models(x, y)


def train_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
    models = [SVC(C=1, kernel='linear'),
              SVC(C=1, kernel='linear', class_weight={-1: 1, 1: 20}),
              SVC(C=0.8, kernel='rbf', gamma=0.1, class_weight={-1:1, 1:10}),
              SVC(C=0.8, kernel='rbf', gamma=0.1, class_weight={-1: 1, 1: 50})]
    titles = ['Linear', 'linear, weight: 50', 'RBF, c:0.8, gamma:0.1, weight:10','RBF, c:0.8, gamma:0.1, weight:50']
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    l, r = min(x_test[:, 0]), max(x_test[:, 0])
    b, t = min(x_test[:, 1]), max(x_test[:, 1])
    grid_x = np.meshgrid(np.linspace(l, r, 500), np.linspace(b, t, 500))
    flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
    print(grid_x)
    mp.figure()
    for model in models:
        model.fit(x_train, y_train)

    for i, model in enumerate(models):
        flat_y = model.predict(flat_x)
        grid_y = flat_y.reshape(grid_x[0].shape)
        mp.subplot(2, 2, i+1)
        mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap=cm_light)
        mp.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_dark, s=10)
        mp.xlabel('x')
        mp.ylabel('y')
        mp.xlim(l, r)
        mp.ylim(b, t)
        mp.title(titles[i])
    mp.suptitle('no balanced deal')
    mp.tight_layout(1.5)
    mp.savefig(config.APP_IMAGES_TXT + 'SCV_unbalance.png')
    mp.show()



# load_data()