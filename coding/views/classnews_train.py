import numpy as np
from sklearn.naive_bayes  import MultinomialNB
import sklearn.feature_extraction.text as ft
import jieba
import pickle
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # 防止当前系统不支持的字符
from . import config


classes, train = [], []

class News(object):

    def __init__(self, classes=None, train=None):
        pass

    @classmethod
    def stopwords(self, line):
        # 处理停止词
        with open(config.APP_STOPWORDS_TXT, 'r') as fr:
            words = fr.read()
        return [x for x in line if x not in words]

    @classmethod
    def load_file(self):
        # 加载文件
        file_names = os.listdir(config.APP_DATA_PASS_TXT)
        for file_name in file_names:
            i = 0
            with open(config.APP_DATA_PASS_TXT + file_name, 'rb') as fr:
                for line in fr.readlines():
                    i += 1
                    line = self.stopwords(jieba.lcut(line))
                    line = ' '.join(line)
                    train.append(line)
                    classes.append(file_name.split('.')[0])
                    if i == 500:
                        break
        self.wordbow()

    @classmethod
    def wordbow(self):
        # 转换词袋模型
        cv = ft.CountVectorizer()
        cv_train = cv.fit_transform(train)
        tfidf = ft.TfidfTransformer(use_idf=False)
        tfidf_train = tfidf.fit_transform(cv_train)
        # print(cv_train)
        # print(tfidf_train)
        self.trainmodel(cv_train, tfidf_train, cv)

    @classmethod
    def trainmodel(self, cv_train, tfidf_train, cv):
        # 训练模型
        model = MultinomialNB()
        model.fit(tfidf_train, classes)
        # self.testmodel(model, tfidf_train, cv)
        self.savemodel(cv, cv_train, model)

    @classmethod
    def savemodel(self, cv, cv_train, model):
        # 保存模型
        with open(config.APP_CV_TXT, 'wb') as f:
            pickle.dump(cv, f)
        with open(config.APP_CV_TRAIN_TXT, 'wb') as f:
            pickle.dump(cv_train, f)
        with open(config.APP_MODEL_TXT, 'wb') as f:
            pickle.dump(model, f)

    @classmethod
    def testmodel(self,model, tfidf_train, cv):
        tests = []
        x_test = '首页|体育新闻欧足联启用新分析体系|欧洲杯数据狂魔遗憾出局2016-07-04 09:10:00随着比利时被威尔士淘汰，' \
                 '欧洲杯一夜之间送别了两位在身价榜单上位列前十的球员：阿扎尔、德布劳内。值得一提的是，' \
                 '在欧足联官方的金足奖数据分析体系里，截至发稿，这两位球星都在前三之列，是夺取赛事官方MVP的热门人选。' \
                 '德布劳内的发挥再次证明：比利时成也靠他、败也因他。欧洲杯激战至今，德布劳内是大赛的数据狂魔之一。助攻榜单上，' \
                 '他3次助攻,位列第3，仅次于4次助攻的拉姆塞和阿扎尔。威胁传球次数，他有23次，领先19次的帕耶和17次的厄齐尔排名第一。' \
                 '射门次数榜单，他21次与贝尔并列第二，仅次于C罗一人。由于本次欧洲杯，官方MVP的评选，是欧足联启用了一套全新的数据分析体系，' \
                 '代入数据进行演算而直接得出排名，因此直到本战之前，德布劳内都是MVP即时榜单上的第一名，只是在本战后，被贝尔超越，沦为第二'
        x_test = self.stopwords(jieba.lcut(x_test))
        x_test2 = '首页|财经中心|财经频道6月中国公路物流运价指数降幅收窄2016-07-04 19:51:00北京7月4日电 (记者 刘长忠)记者4日' \
                  '从中国物流与采购联合会获悉，6月中国公路物流运价指数为101.3点，比上月回落1.5%，但比年初回升2.8%。数据显示，进入6月，' \
                  '公路物流需求较前期小幅回升。一方面，工业物流需求保持平稳增长，其中采矿业、高耗能行业等传统行业增速虽有所回落，但原油、' \
                  '橡胶等进口量较前期明显回升；另一方面，消费品物流需求继续保持平稳较快增长，特别是农副产品、食品、纺织品等物流需求加快增长。' \
                  '分品种来看，钢材、有色金属等大宗商品物流需求趋弱；农副产品、食品、纺织品等物流需求上升较快。中国物流与采购联合会分析人士称，' \
                  '总体来看，未来整车及零担公路物流需求将延续小幅回暖的走势，运量也有望继续回升。公路物流运价指数较前期可能延续回升走势，' \
                  '回升幅度难有较大提升，预计总体将与上年同期水平基本持平。(完)'
        x_test2 = self.stopwords(jieba.lcut(x_test2))
        tests.append(' '.join(x_test))
        tests.append(' '.join(x_test2))
        new_cv_train = cv.transform(tests)
        new_tfidf = ft.TfidfTransformer(use_idf=False)
        new_tfidf_train = new_tfidf.fit_transform(new_cv_train)
        pred_test_Y = model.predict(new_tfidf_train)
        print(pred_test_Y)


# News().load_file()


