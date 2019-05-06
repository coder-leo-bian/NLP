from sklearn import svm
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as mp
import matplotlib as mpl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from . import config
import logging
from sklearn.naive_bayes import GaussianNB



def load_file():
    data = load_iris().data[:, :2]
    data = np.array(data, dtype=float)
    target = load_iris().target
    target = np.array(target, dtype=int)
    return train_model(data, target)


def train_model(data, target):
    logging.info('开始利用svm高斯核函数训练鸢尾花数据......')
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=7)
    model = svm.SVC(kernel='rbf')  # gamma高斯函数的参数， c正则强度防止过拟合
    c = np.logspace(-2, 2, 10)
    gamma = np.logspace(-2, 2, 10)
    # c = np.linspace(0.01, 10, 100)
    # gamma = np.linspace(0.01, 20, 100)
    model = GridSearchCV(model, param_grid={'C':c, 'gamma':gamma}, cv=5)
    model.fit(x_train, y_train)
    pred_y = model.predict(x_test)
    params = model.best_params_
    params['f1_score'] = f1_score(y_test, pred_y, average=None).mean()
    params['recall_score'] = recall_score(y_test, pred_y, average=None).mean()
    params['precision_score'] = precision_score(y_test, pred_y, average=None).mean()
    params['accuracy_score'] = accuracy_score(y_test, pred_y)
    print(params)
    l, r = min(x_train[:, 0]), max(x_train[:, 0])
    b, t = min(x_train[:, 1]), max(x_train[:, 1])
    grid_x = np.meshgrid(np.linspace(l, r, 500), np.linspace(b, t, 500))

    flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
    flat_y = model.predict(flat_x)
    grid_y = flat_y.reshape(grid_x[0].shape)
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    mp.figure('iris')
    mp.title('svm-iris')
    mp.xlabel('x')
    mp.ylabel('y')
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap=cm_light)
    mp.scatter(data[:, 0], data[:, 1], c=target, cmap=cm_dark)
    mp.savefig(config.APP_IMAGES_TXT + 'SCV_rbf_iris.png')
    mp.show()
    return params



# load_file()
