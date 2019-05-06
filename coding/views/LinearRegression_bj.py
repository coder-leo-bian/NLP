import numpy as np
import random, time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as mp
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as sm
from . import config



def x_y(a, b, up, down):
    # 获取训练数据
    x = list(np.linspace(0.5, 1.5, 100))
    y = [(a*i + b) + random.uniform(up, down) for i in x]
    xy = np.c_[np.array(x).ravel(), np.array(y).ravel()]
    ridge_score = ridge_model(xy)
    lasso_score = lasso_model(xy)
    linear_score = train_model(xy)
    return {'linear_score': linear_score['linear_score'], 'ridge_score': ridge_score['ridge_score'], 'lasso_score': lasso_score['lasso_score']}


def train_model(xy):
    # linear
    x = xy[:, 0].reshape(-1, 1)
    y = xy[:, 1]
    model = LinearRegression()
    model.fit(x, y)
    pred_y = model.predict(x)
    coef = model.coef_
    intercept = model.intercept_
    params = model.get_params()
    print(params)
    linear_r2 = sm.r2_score(y, pred_y)  # r2得分
    linear_absolute = sm.mean_absolute_error(y, pred_y) # 平均绝对值误差
    linear_squared = sm.mean_squared_error(y, pred_y) # 均方误差
    linear_median = sm.median_absolute_error(y, pred_y) # 中值绝对误差
    drawing(xy, x, pred_y)
    return {'linear_score': {'linear_r2': round(linear_r2, 5), 'linear_absolute': round(linear_absolute, 5), 'linear_squared': round(linear_squared, 5), 'linear_median': round(linear_median, 5)}}


def ridge_model(xy):
    # ridge模型
    x = xy[:, 0].reshape(-1, 1)
    y = xy[:, 1]
    model = Ridge()
    alpha_can = np.linspace(-1, 10, 30)
    model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    model.fit(x, y)
    print(model.best_params_)
    pred_y = model.predict(x)
    params = model.get_params()
    print(params)
    ridge_r2 = sm.r2_score(y, pred_y)
    ridge_absolute = sm.mean_absolute_error(y, pred_y)
    ridge_squared = sm.mean_squared_error(y, pred_y)
    ridge_median = sm.median_absolute_error(y, pred_y)
    drawing_ridge(xy, x, pred_y, model.best_params_)
    return {'ridge_score': {'ridge_r2': round(ridge_r2, 5), 'ridge_absolute': round(ridge_absolute, 5), 'ridge_squared': round(ridge_squared, 5), 'ridge_median': round(ridge_median, 5)}}


def lasso_model(xy):
    #lasso模型
    x = xy[:, 0].reshape(-1, 1)
    y = xy[:, 1]
    model = Lasso()
    alpha_can = np.linspace(-1, 10, 30)
    model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    model.fit(x, y)
    print(model.best_params_)
    pred_y = model.predict(x)
    params = model.get_params()
    print(params)
    lasso_r2 = sm.r2_score(y, pred_y)
    lasso_absolute = sm.mean_absolute_error(y, pred_y)
    lasso_squared = sm.mean_squared_error(y, pred_y)
    lasso_median = sm.median_absolute_error(y, pred_y)
    drawing_lasso(xy, x, pred_y, model.best_params_)
    return {'lasso_score': {'lasso_r2': round(lasso_r2, 5), 'lasso_absolute': round(lasso_absolute, 5), 'lasso_squared': round(lasso_squared, 5), 'lasso_median': round(lasso_median, 5)}}


def drawing(xy, x, pred_y):
    # 画图
    mp.figure('Linear')
    mp.title('x, y')
    mp.xlabel('x')
    mp.ylabel('y')
    mp.scatter(xy[:, 0], xy[:, 1], c='k', label='dot')
    sorted_indices = x.T[0].argsort()
    mp.plot(x[sorted_indices], pred_y[sorted_indices],
            c='orangered', label='Regression')
    mp.legend()
    mp.savefig(config.APP_IMAGES_TXT + 'Linear.png')
    mp.show()


def drawing_ridge(xy, x, pred_y, best_params):
    # 画图
    mp.figure('Ridge')
    mp.title('x, y')
    mp.xlabel('x')
    mp.ylabel('y')
    mp.scatter(xy[:, 0], xy[:, 1], c='k', label='dot')
    sorted_indices = x.T[0].argsort()
    mp.plot(x[sorted_indices], pred_y[sorted_indices],
            c='orangered', label='Ridge')
    mp.text(x[80], pred_y[80], r'alpha: {}'.format(best_params['alpha']), fontsize=20, color='r')
    mp.legend()
    mp.savefig(config.APP_IMAGES_TXT + 'Ridge.png')
    mp.show()



def drawing_lasso(xy, x, pred_y, best_params):
    # 画图
    mp.figure('Lasso')
    mp.title('x, y')
    mp.xlabel('x')
    mp.ylabel('y')
    mp.scatter(xy[:, 0], xy[:, 1], c='k', label='dot')
    sorted_indices = x.T[0].argsort()
    mp.plot(x[sorted_indices], pred_y[sorted_indices],
            c='orangered', label='Lasso')
    mp.text(x[80], pred_y[80], r'alpha: {}'.format(best_params['alpha']), fontsize=20, color='r')
    mp.legend()
    mp.savefig(config.APP_IMAGES_TXT + 'Lasso.png')
    mp.show()
