from sklearn.naive_bayes import MultinomialNB
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import pickle
from . import config
jieba.load_userdict(config.APP_ADD_NEW_WORDS_TXT)
"""多项式朴素贝叶斯对复仇者联盟4的评价进行分类"""


class Comment(object):

    def __init__(self):
        pass

    @classmethod
    def deal_stop_word(cls, s):
        with open(config.APP_STOPWORDS_TXT, 'r') as f:
            words = f.read()
        return [ i for i in s if i not in words]

    @classmethod
    def jieba_split(cls):
        t, f, target = [], [], []
        with open(config.APP_MAOYAN_DATA_TXT, 'r') as fr:
            lines = fr.readlines()
        for line in lines:
            if len(line[0: -2].replace('\n', ' ')) >= 5 and int((line.replace('\n', '')[-1])) == 1 and len(t) < 500:
                t.append(' '.join(cls.deal_stop_word(jieba.lcut((line[0: -2]).replace('\n', '')))))
            if len(line[0: -2].replace('\n', ' ')) >= 5 and int((line.replace('\n', '')[-1])) == 0 and len(f) < 500:
                f.append(' '.join(cls.deal_stop_word(jieba.lcut((line[0: -2]).replace('\n', '')))))
        a = [1] * 500
        b = [0] * 500
        a.extend(b)
        target = a
        t.extend(f)
        data = t
        return data, target

    @classmethod
    def train_model(cls, tf_data, target):
        model = MultinomialNB()
        model = GridSearchCV(model, param_grid={'alpha': np.linspace(1, 10, 10)}, cv=5)
        model.fit(tf_data, target)
        return model, model.best_params_

    @classmethod
    def save_model(cls, tf, cv, model):
        with open(config.APP_FILM_CV_TXT, 'wb') as f:
            pickle.dump(cv, f)

        with open(config.APP_FILM_MODEL_TXT, 'wb') as f:
            pickle.dump(model, f)

        with open(config.APP_FILM_TF_TXT, 'wb') as f:
            pickle.dump(tf, f)

    @classmethod
    def tfidf_word(cls, data, target):
        cv = CountVectorizer()
        cv_data = cv.fit_transform(data)
        tf = TfidfTransformer(use_idf=False)
        tf_data = tf.fit_transform(cv_data)
        return tf, cv, tf_data

    @classmethod
    def train_main(cls):
        data, target = cls.jieba_split()
        tf, cv, tf_data = cls.tfidf_word(data, target)
        model, best_params = cls.train_model(tf_data, target)
        cls.save_model(tf, cv, model)
        # d = [' '.join((cls.deal_stop_word(jieba.lcut('我都看得睡着了'))))]
        # cv_d = cv.transform(d)
        # tf_d = tf.transform(cv_d)
        # pred_y = model.predict(tf_d)
        # print(pred_y)

    @classmethod
    def test_main(cls, d):
        with open(config.APP_FILM_CV_TXT, 'rb') as f:
            cv = pickle.load(f)

        with open(config.APP_FILM_MODEL_TXT, 'rb') as f:
            model = pickle.load(f)

        with open(config.APP_FILM_TF_TXT, 'rb') as f:
            tf = pickle.load(f)
        d = [' '.join((cls.deal_stop_word(jieba.lcut(d))))]
        cv_d = cv.transform(d)
        tf_d = tf.transform(cv_d)
        pred_y = model.predict(tf_d)
        return pred_y


# Comment.train_main()

# print(Comment().test_main('这是一个比较好的电影，非常值得一看'))
