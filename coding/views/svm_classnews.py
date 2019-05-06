from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import jieba
import os, pickle
from . import config
from sklearn.cluster import dbscan, MeanShift
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
# CountVectorizer 获得所有词在文本中的出现频率，TfidfVectorizer不仅出现词的频率，还得到tf-idf
import logging


class ClassSvm(object):

    def __init__(self):
        pass

    @classmethod
    def load_data(self, paths):
        print(u"加载训练文本信息")
        txt = []
        for path in paths:
            with open(path, 'r') as fr:
                d = fr.readlines()
                txt.extend(d[:50])
        return txt

    @classmethod
    def deal_stop_word(self, lst):
        # 处理停止词
        print(u'处理停止词')
        with open(config.APP_STOPWORDS_TXT, 'r') as f:
            words = f.read()
        return [' '.join([i for i in l.split(' ') if i not in words]) for l in lst]

    @classmethod
    def cut_word(self, txt):
        # 分词
        print(u'结巴分词')
        cut_txt = [' '.join((jieba.lcut(t))) for t in txt]
        return cut_txt

    @classmethod
    def tfidf_deal(self, x_train):
        # 做词频，做逆文档频率矩阵
        print('做tfidf')
        cv = CountVectorizer()
        cv_x_train = cv.fit_transform(x_train)
        tf = TfidfTransformer(use_idf=False)
        data = tf.fit_transform(cv_x_train)
        return cv, data

    @classmethod
    def train_svm_model(self, data, y_train):
        # 训练svm模型
        print(u'训练svm模型')
        model = svm.SVC(kernel='rbf')
        cv_model = GridSearchCV(estimator=model, param_grid={'C': [0.5, 1, 2, 3, 4, 5], 'gamma': np.linspace(0, 5, 10)}, cv=5)
        cv_model.fit(data, y_train)
        return cv_model, cv_model.best_params_

    @classmethod
    def model_score(self, x_test, y_test, cv_model, cv):
        # 模型得分
        print(u'计算模型得分')
        cv_x_test = cv.transform(x_test)
        tf = TfidfTransformer(use_idf=False)
        tf_test = tf.fit_transform(cv_x_test)
        print(tf_test)
        pred_y = cv_model.predict(tf_test)
        f1score = f1_score(y_test, pred_y, average='macro')
        ascore = accuracy_score(y_test, pred_y)
        return f1score, ascore

    @classmethod
    def save_model(self, cv, cv_model):
        # 保存训练好的模型
        with open(config.APP_SVM_CV_TXT, 'wb') as f:
            pickle.dump(cv, f)
        with open(config.APP_SVM_MODEL_TXT, 'wb') as f:
            pickle.dump(cv_model, f)

    @classmethod
    def test_model(self):
        with open(config.APP_SVM_CV_TXT, 'rb') as f:
            cv = pickle.load(f)
        with open(config.APP_SVM_MODEL_TXT, 'rb') as f:
            model = pickle.load(f)
        return cv, model

    @classmethod
    def train_main(self):
        target = [] # y值
        for path, a, names in os.walk(config.APP_DATA_PASS_TXT, followlinks=True):
            paths = [path + name for name in names]
        for name in names:
            target.extend([name[:-4]] * 50)

        txt = self.load_data(paths) # 获取全部文本内容
        cut_txt = self.cut_word(txt) # 分词
        stop_txt = self.deal_stop_word(cut_txt) # 去除停止词

        x_train, x_test, y_train, y_test = train_test_split(stop_txt, target, test_size=0.2, random_state=7) #
        cv, data = self.tfidf_deal(x_train) # tfidf模型，逆文档频率
        cv_model, best_params = self.train_svm_model(data, y_train) # best_params最优超参数
        f1score, ascore = self.model_score(x_test, y_test, cv_model, cv)  # f1得分，准确率

        # 保存模型
        self.save_model(cv, cv_model)
        return {'f1_score': f1score, 'acc': ascore, 'best_params': best_params}

    @classmethod
    def product_main(self, txt):
        # 开始预测
        d = []
        cv, model = self.test_model()
        txt = jieba.lcut(txt)
        data = self.deal_stop_word(txt)
        data = ' '.join([d for d in data if d])
        d.append(data)
        cv_data = cv.transform(d)
        tf = TfidfTransformer(use_idf=False)
        tf_data = tf.fit_transform(cv_data)
        result = model.predict(tf_data)
        return str(result[0])


if __name__ == '__main__':
    c = ClassSvm()
    c.train_main()
    c.product_main("""首页|体育新闻ï欧足联启用新分析体系 欧洲杯数据狂魔遗憾出局2016-07-04 09:10:00ï随着比利时被威尔士淘汰，欧洲杯一夜之间送别了两位在身价榜单上位列前十的球员：阿扎尔、德布劳内。值得一提的是，在欧足联官方的金足奖数据分析体系里，截至发稿，这两位球星都在前三之列，是夺取赛事官方MVP的热门人选。德布劳内的发挥再次证明：比利时成也靠他、败也因他。欧洲杯激战至今，德布劳内是大赛的数据狂魔之一。助攻榜单上，他3次助攻,位列第3，仅次于4次助攻的拉姆塞和阿扎尔。威胁传球次数，他有23次，领先19次的帕耶和17次的厄齐尔排名第一。射门次数榜单，他21次与贝尔并列第二，仅次于C罗一人。由于本次欧洲杯，官方MVP的评选，是欧足联启用了一套全新的数据分析体系，代入数据进行演算而直接得出排名，因此直到本战之前，德布劳内都是MVP即时榜单上的第一名，只是在本战后，被贝尔超越，沦为第二。""")



