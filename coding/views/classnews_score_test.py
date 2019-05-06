import pickle
from . import config
from . import classnews_score_train
import jieba
import sklearn.feature_extraction.text as ft


data = []


def load_file(FILE_DIR):
    global data
    data = []
    with open(FILE_DIR, 'rb') as fr:
        lines = fr.readlines()
    for line in lines:
        line = jieba.lcut(line)
        data.append(' '.join(classnews_score_train.stopwords(line)))
    return load_model()

def load_model():
    f= open(config.APP_CV_SCORE_TXT, 'rb')
    cv = pickle.load(f)

    f =  open(config.APP_CV_SCORE_TRAIN_TXT, 'rb')
    cv_train = pickle.load(f)

    f = open(config.APP_MODEL_SCORE_TXT, 'rb')
    model = pickle.load(f)
    return result(cv, cv_train, model)

def result(cv, cv_train, model):
    cv_data = cv.transform(data)
    tfidf = ft.TfidfTransformer(use_idf=False)
    tfidf_data = tfidf.fit_transform(cv_data)
    pred_y = model.predict(tfidf_data)
    return pred_y