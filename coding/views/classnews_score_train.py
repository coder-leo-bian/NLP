from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from . import config
import os
import jieba
import sklearn.feature_extraction.text as ft
from sklearn.model_selection import train_test_split, cross_val_score
import logging, pickle

train, classes = [], []


def stopwords(line):
    # 处理停止词
    with open(config.APP_STOPWORDS_TXT, 'r') as fr:
        words = fr.read()
    return [x for x in line if x not in words]


def load_file():
    # 加载文件
    file_names = os.listdir(config.APP_DATA_PASS_TXT)
    for file_name in file_names:
        i = 0
        with open(config.APP_DATA_PASS_TXT + file_name, 'rb') as fr:
            for line in fr.readlines():
                i += 1
                line = stopwords(jieba.lcut(line))
                line = ' '.join(line)
                train.append(line)
                classes.append(file_name.split('.')[0])
                if i == 500:
                    break
    return wordbow()


def wordbow():
    # 词袋，做tfidf
    x_train, x_test, y_train, y_test = train_test_split(train, classes, test_size=0.3, random_state=7)
    cv = ft.CountVectorizer()
    cv_x_train = cv.fit_transform(x_train)
    # print('cv_x_train: ', cv_x_train)
    tfidf = ft.TfidfTransformer(use_idf=False)
    tfidf_x_train = tfidf.fit_transform(cv_x_train)
    # print('tfidf_x_train: ', tfidf_x_train)
    return train_model(tfidf_x_train, y_train, x_test, y_test, cv, cv_x_train)


def train_model(tfidf_x_train, y_train, x_test, y_test, cv, cv_x_train):
    # 训练模型
    model = MultinomialNB()
    precision_weighted = cross_val_score(model, tfidf_x_train, y_train, cv=5, scoring='precision_weighted').mean()
    recall_weighted = cross_val_score(model, tfidf_x_train, y_train, cv=5, scoring='recall_weighted').mean()
    f1_weighted = cross_val_score(model, tfidf_x_train, y_train, cv=5, scoring='f1_weighted').mean()
    score = {
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
    # print(score)
    model.fit(tfidf_x_train, y_train)
    save_model(cv, cv_x_train, model)
    return predict_self(model, x_test, y_test, cv, score)


def predict_self(model, x_test, y_test, cv, score):
    # 测试模型
    cv_x_test = cv.transform(x_test)
    # print(cv_x_test)
    tfidf = ft.TfidfTransformer(use_idf=False)
    tfidf_x_test = tfidf.fit_transform(cv_x_test)
    # print(tfidf_x_test)
    pred_y = model.predict(tfidf_x_test)
    # print(pred_y)
    result = pred_y == y_test
    result = [r for r in result if r]

    # score: 查准率，召回率，F1得分，true_count：判断正确的数量， all_count：总数量， true_score：正确率
    return {'score': score, 'true_count': len(result), 'all_count': pred_y.size, 'true_score': (len(result)/pred_y.size)*100}


def save_model(cv, cv_x_train, model):
    with open(config.APP_CV_SCORE_TXT, 'wb') as f:
        pickle.dump(cv, f)

    with open(config.APP_CV_SCORE_TRAIN_TXT, 'wb') as f:
        pickle.dump(cv_x_train, f)

    with open(config.APP_MODEL_SCORE_TXT, 'wb') as f:
        pickle.dump(model, f)










