import scipy.misc as sm
import numpy as np
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
import matplotlib.pyplot as mp
# 量化图片

def load_image():
    image1 = sm.imread('../static/images/Linear.png', True).astype(np.uint8)  # 读取图片  astype(np.uint8)转换为numpy无符号整型
    x = image1.reshape(-1, 1) # 转换数据类型
    model = KMeans(init='k-means++', n_clusters=2, n_init=6, random_state=7)
    model.fit(x)
    y = model.labels_  # 类别标签
    print(y)
    centers = model.cluster_centers_.squeeze()  # squeeze()去掉空维度
    print(centers)
    image2 = centers[y].reshape(image1.shape)  # 类别置换
    mp.figure('Image-1', facecolor='lightgray')
    mp.title('Image-1', fontsize=20)
    mp.axis('off')  # 关闭坐标轴
    mp.imshow(image1)
    mp.figure('Image-2', facecolor='lightgray')
    mp.title('Image-2', fontsize=20)
    mp.axis('off')
    mp.imshow(image2, cmap='gray')
    mp.show()


load_image()