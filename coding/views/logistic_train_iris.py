#coding:utf-8
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mp
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import *
from . import config


# 花萼长度，花萼宽度，花瓣长度，花瓣宽度
def deal_iris():
    data = load_iris().data
    data1 = data[:, :2]
    data2 = np.c_[data[:, 0], data[:, 2]]
    data3 = np.c_[data[:, 0], data[:, 3]]
    data4 = np.c_[data[:, 1], data[:, 2]]
    data5 = np.c_[data[:, 1], data[:, 3]]
    data6 = np.c_[data[:, 2], data[:, 3]]
    target = load_iris().target
    datas = [data1, data2, data3, data4, data5, data6]
    logistic_model(datas, target)


def logistic_model(datas, target):
    models = []
    for d in datas:
        x_train, x_test, y_train, y_test = train_test_split(d, target, test_size=0.25, random_state=5)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        models.append((model, x_test, y_test))
    drawing(models)

def drawing(models):
    # mpl.rcParams['font.sans-serif'] = [u'simHei']  # 识别中文保证不乱吗
    # myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
    # mp.rcParams['axes.unicode_minus'] = False
    res = [] # pred_y, x_text, y_test, model
    for model in models:
        pred_y = model[0].predict(model[1])
        res.append((pred_y, model[1], model[2], model[0]))

    l, r = min(res[0][1][:, 0]), max(res[0][1][:, 0])
    b, t = min(res[0][1][:, 1]), max(res[0][1][:, 1])
    grid_x1 = np.meshgrid(np.linspace(l, r, 500), np.linspace(b, t, 500))
    flat_x1 = np.c_[grid_x1[0].ravel(), grid_x1[1].ravel()]
    flat_y1 = res[0][3].predict(flat_x1)
    grid_y1 = flat_y1.reshape(grid_x1[0].shape)
    pred_y1 = res[0][0]
    x1_test = res[0][1]

    # l, r = min(res[1][1][:, 0]), max(res[1][1][:, 0])
    # b, t = min(res[1][1][:, 1]), max(res[1][1][:, 1])
    # grid_x2 = np.meshgrid(np.linspace(l, r, 1000), np.linspace(b, t, 1000))
    # flat_x2 = np.c_[grid_x2[0].ravel(), grid_x2[1].ravel()]
    # flat_y2 = res[1][3].predict(flat_x2)
    # grid_y2 = flat_y2.reshape(grid_x2[0].shape)
    # pred_y2 = res[1][0]
    # x2_test = res[1][1]
    #
    # l, r = min(res[2][1][:, 0]), max(res[2][1][:, 0])
    # b, t = min(res[2][1][:, 1]), max(res[2][1][:, 1])
    # grid_x3 = np.meshgrid(np.linspace(l, r, 1000), np.linspace(b, t, 1000))
    # flat_x3 = np.c_[grid_x3[0].ravel(), grid_x3[1].ravel()]
    # flat_y3 = res[2][3].predict(flat_x3)
    # grid_y3 = flat_y3.reshape(grid_x3[0].shape)
    # pred_y3 = res[2][0]
    # x3_test = res[2][1]
    #
    # l, r = min(res[3][1][:, 0]), max(res[3][1][:, 0])
    # b, t = min(res[3][1][:, 1]), max(res[3][1][:, 1])
    # grid_x4 = np.meshgrid(np.linspace(l, r, 1000), np.linspace(b, t, 1000))
    # flat_x4 = np.c_[grid_x4[0].ravel(), grid_x4[1].ravel()]
    # flat_y4 = res[3][3].predict(flat_x4)
    # grid_y4 = flat_y4.reshape(grid_x4[0].shape)
    # pred_y4 = res[3][0]
    # x4_test = res[3][1]
    #
    # l, r = min(res[4][1][:, 0]), max(res[4][1][:, 0])
    # b, t = min(res[4][1][:, 1]), max(res[4][1][:, 1])
    # grid_x5 = np.meshgrid(np.linspace(l, r, 1000), np.linspace(b, t, 1000))
    # flat_x5 = np.c_[grid_x5[0].ravel(), grid_x5[1].ravel()]
    # flat_y5 = res[4][3].predict(flat_x5)
    # grid_y5 = flat_y5.reshape(grid_x5[0].shape)
    # pred_y5 = res[4][0]
    # x5_test = res[4][1]
    #
    # l, r = min(res[5][1][:, 0]), max(res[5][1][:, 0])
    # b, t = min(res[5][1][:, 1]), max(res[5][1][:, 1])
    # grid_x6 = np.meshgrid(np.linspace(l, r, 1000), np.linspace(b, t, 1000))
    # flat_x6 = np.c_[grid_x6[0].ravel(), grid_x6[1].ravel()]
    # flat_y6 = res[5][3].predict(flat_x6)
    # grid_y6 = flat_y6.reshape(grid_x6[0].shape)
    # pred_y6 = res[5][0]
    # x6_test = res[5][1]

    mp.figure(1)
    mp.title(u'LogisticRegression')
    # cm_light = matplotlib.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])  # 测试分类的颜色
    # cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])  # 样本点的颜色
    # ax1 = mp.subplot(2, 3, 1)
    # ax2 = mp.subplot(2, 3, 2)
    # ax3 = mp.subplot(2, 3, 3)
    # ax4 = mp.subplot(2, 3, 4)
    # ax5 = mp.subplot(2, 3, 5)
    # ax6 = mp.subplot(2, 3, 6)

    # mp.sca(ax1)
    mp.xlabel(u'sepal_len1')
    mp.ylabel(u'sepal_width1')
    mp.xlim(grid_x1[0].min(), grid_x1[0].max())
    mp.ylim(grid_x1[1].min(), grid_x1[1].max())
    mp.pcolormesh(grid_x1[0], grid_x1[1], grid_y1, cmap='brg')
    mp.scatter(x1_test[:, 0], x1_test[:, 1], c=pred_y1, cmap='RdYlBu')

    # mp.sca(ax2)
    # mp.xlabel(u'sepal_len2')
    # mp.ylabel(u'petal_len2')
    # mp.xlim(grid_x2[0].min(), grid_x2[0].max())
    # mp.ylim(grid_x2[1].min(), grid_x2[1].max())
    # mp.pcolormesh(grid_x2[0], grid_x2[1], grid_y2, cmap='brg')
    # mp.scatter(x2_test[:, 0], x2_test[:, 1], c=pred_y2, cmap='RdYlBu')
    #
    # # sepal_len, sepal_width, petal_len, petal_width
    #
    # mp.sca(ax3)
    # mp.xlabel(u'sepal_len3')
    # mp.ylabel(u'petal_width3')
    # mp.xlim(grid_x3[0].min(), grid_x3[0].max())
    # mp.ylim(grid_x3[1].min(), grid_x3[1].max())
    # mp.pcolormesh(grid_x3[0], grid_x3[1], grid_y3, cmap='brg')
    # mp.scatter(x3_test[:, 0], x3_test[:, 1], c=pred_y3, cmap='RdYlBu')
    #
    # mp.sca(ax4)
    # mp.xlabel(u'sepal_width4')
    # mp.ylabel(u'petal_len4')
    # mp.xlim(grid_x4[0].min(), grid_x4[0].max())
    # mp.ylim(grid_x4[1].min(), grid_x4[1].max())
    # mp.pcolormesh(grid_x4[0], grid_x4[1], grid_y4, cmap='brg')
    # mp.scatter(x4_test[:, 0], x4_test[:, 1], c=pred_y4, cmap='RdYlBu')
    #
    # mp.sca(ax5)
    # mp.xlabel(u'sepal_width5')
    # mp.ylabel(u'petal_width5')
    # mp.xlim(grid_x5[0].min(), grid_x5[0].max())
    # mp.ylim(grid_x5[1].min(), grid_x5[1].max())
    # mp.pcolormesh(grid_x5[0], grid_x5[1], grid_y5, cmap='brg')
    # mp.scatter(x5_test[:, 0], x5_test[:, 1], c=pred_y5, cmap='RdYlBu')
    #
    # mp.sca(ax6)
    # mp.xlabel(u'petal_len6')
    # mp.ylabel(u'petal_width6')
    # mp.xlim(grid_x6[0].min(), grid_x6[0].max())
    # mp.ylim(grid_x6[1].min(), grid_x6[1].max())
    # mp.pcolormesh(grid_x6[0], grid_x6[1], grid_y6, cmap='brg')
    # mp.scatter(x6_test[:, 0], x6_test[:, 1], c=pred_y6, cmap='RdYlBu')

    # mp.tight_layout()
    mp.savefig(config.APP_IMAGES_TXT + 'logistic.png')
    mp.show()




