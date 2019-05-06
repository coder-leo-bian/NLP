import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_CULTURE_TXT = os.path.join(APP_ROOT, '../../data/classnews/train/culture.txt')
APP_FINANCIAL_TXT = os.path.join(APP_ROOT, '../../data/classnews/train/financial.txt')
APP_MILITARY_TXT = os.path.join(APP_ROOT, '../../data/classnews/train/military.txt')
APP_SPORTS_TXT = os.path.join(APP_ROOT, '../../data/classnews/train/sports.txt')

APP_CV_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_cv')
APP_CV_TRAIN_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_cv_train')
APP_MODEL_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_model')

APP_STOPWORDS_TXT = os.path.join(APP_ROOT, '../static/stopwords')
APP_ADD_NEW_WORDS_TXT = os.path.join(APP_ROOT, '../static/newwords')


APP_DATA_PASS_TXT = os.path.join(APP_ROOT, '../../data/classnews/train/')

APP_CV_SCORE_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_cv_score')
APP_CV_SCORE_TRAIN_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_cv_train_score')
APP_MODEL_SCORE_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_model_score')

APP_IMAGES_TXT = os.path.join(APP_ROOT, '../static/images/')

APP_SVM_CV_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_cv_svm')
APP_SVM_MODEL_TXT = os.path.join(APP_ROOT, '../../pickles/classnews-pickles/classnews_model_svm')

APP_MAOYAN_DATA_TXT = os.path.join(APP_ROOT, '../../data/maoyan/data.txt')
APP_FILM_CV_TXT = os.path.join(APP_ROOT, '../../pickles/film_comment_multinomiaNB_cv')
APP_FILM_MODEL_TXT = os.path.join(APP_ROOT, '../../pickles/film_comment_multinomiaNB_model')
APP_FILM_TF_TXT = os.path.join(APP_ROOT, '../../pickles/film_comment_multinomiaNB_tf')


