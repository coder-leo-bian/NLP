import jieba
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import sklearn.feature_extraction.text as ft # 针对文本做特征抽取
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from . import config

def stopwords(line):
    # 处理停止词
    with open(config.APP_STOPWORDS_TXT, 'r') as fr:
        words = fr.read()
    return [x for x in line if x not in words]


def load_pickle():
    with open(config.APP_CV_TXT, 'rb') as f:
        cv = pickle.load(f)
    with open(config.APP_CV_TRAIN_TXT, 'rb') as f:
        cv_train = pickle.load(f)
    with open(config.APP_MODEL_TXT, 'rb') as f:
        model = pickle.load(f)
    return cv, cv_train, model


def testmodel(test):
    cv, cv_train, model = load_pickle()
    tests = []
    x_test = stopwords(jieba.lcut(test))
    # x_test = '众所周知，中国人在世界上有着最出色的生意头脑，他们有着细腻的商业思维和精准的商业预判。' \
    #          '但有的时候却聪明反被聪明误。很多人看到一个赚钱的机会，大家都会一窝蜂似的挤进来都想分到一块“大蛋糕”。' \
    #          '而很多人又为了各自利益而不停模仿，导致最后都赚不到钱。中国的女鞋行业就是个典型例子'
    # x_test = stopwords(jieba.lcut(x_test))
    # x_test2 = '首页|财经中心|财经频道6月中国公路物流运价指数降幅收窄2016-07-04 19:51:00北京7月4日电 (记者 刘长忠)记者4日' \
    #           '从中国物流与采购联合会获悉，6月中国公路物流运价指数为101.3点，比上月回落1.5%，但比年初回升2.8%。数据显示，进入6月，' \
    #           '公路物流需求较前期小幅回升。一方面，工业物流需求保持平稳增长，其中采矿业、高耗能行业等传统行业增速虽有所回落，但原油、' \
    #           '橡胶等进口量较前期明显回升；另一方面，消费品物流需求继续保持平稳较快增长，特别是农副产品、食品、纺织品等物流需求加快增长。' \
    #           '分品种来看，钢材、有色金属等大宗商品物流需求趋弱；农副产品、食品、纺织品等物流需求上升较快。中国物流与采购联合会分析人士称，' \
    #           '总体来看，未来整车及零担公路物流需求将延续小幅回暖的走势，运量也有望继续回升。公路物流运价指数较前期可能延续回升走势，' \
    #           '回升幅度难有较大提升，预计总体将与上年同期水平基本持平。(完)'
    # x_test2 = stopwords(jieba.lcut(x_test2))
    tests.append(' '.join(x_test))
    print(tests)
    new_cv_train = cv.transform(tests)
    new_tfidf = ft.TfidfTransformer(use_idf=False)
    new_tfidf_train = new_tfidf.fit_transform(new_cv_train)
    pred_test_Y = model.predict(new_tfidf_train)
    return pred_test_Y
# x_test = '众所周知，中国人在世界上有着最出色的生意头脑，他们有着细腻的商业思维和精准的商业预判。' \
#              '但有的时候却聪明反被聪明误。很多人看到一个赚钱的机会，大家都会一窝蜂似的挤进来都想分到一块“大蛋糕”。' \
#              '而很多人又为了各自利益而不停模仿，导致最后都赚不到钱。中国的女鞋行业就是个典型例子'
# print(testmodel(x_test))