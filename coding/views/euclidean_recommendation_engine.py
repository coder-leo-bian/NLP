import numpy as np
import json

"""欧几里得推荐引擎"""


class EuclideanEngine(object):

    def __init__(self):
        pass

    @classmethod
    def get_movie(cls):
        # 获取json文档
        with open('../../data/maoyan/comment.json', 'r') as f:
            data = json.loads(f.read())
        return data

    @classmethod
    def compute_relation(cls, data):
        # 计算相关性
        users, scmap = list(data.keys()), []
        for user1 in users:
            scrow = []
            for user2 in users:
                movie = set()
                for movie_name in data[user2]:
                    if movie_name in data[user1]:
                        movie.add(movie_name)
                if movie:
                    x, y = [], []
                    for m in movie:
                        x.append(data[user1][m])
                        y.append(data[user2][m])
                    score = 1/(1 + (np.sqrt((np.array(x) - np.array(y)) ** 2)).sum()) # 利用欧氏距离计算
                else:
                    score = 0
                scrow.append(score)
            scmap.append(scrow)

        return users, scmap

    @classmethod
    def train_main(cls):
        # 主
        data = cls.get_movie()
        users, scmap = cls.compute_relation(data)
        print(users)
        for scrow in scmap:
            print(' '.join("{:>5.2f}".format(s) for s in scrow))

EuclideanEngine().train_main()

