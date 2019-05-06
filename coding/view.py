from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
from flask import render_template
from flask import request, redirect, jsonify
import hashlib, json, time, os
from coding.views import classnews_test, classnews_train, classnews_score_train, \
    classnews_score_test, LinearRegression_bj, logistic_train_iris, GaussianNB_iris,\
    svm_iris, svm_rbf_unbalance, kmeans, svm_classnews, film_comment
import logging, json


basedir = os.path.abspath(os.path.dirname(__file__))
FILE_DIR = basedir + '/static/uploads/'


@app.route('/login/', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    # res = classnews_score_train.load_file()
    # print(res)
    return render_template('index.html')


@app.route('/html.html/')
def html():
    # 页面1
    return render_template('html.html')

@app.route('/html2.html/')
def html2():
    # 页面2
    return render_template('html2.html')

@app.route('/html3.html/')
def html3():
    # 页面3
    return render_template('html3.html')

@app.route('/html4.html/')
def html4():
    # 页面4
    return render_template('html4.html')

@app.route('/html5.html/')
def html5():
    # 页面5
    return render_template('html5.html')

@app.route('/login/register')
def register():
    return render_template('register.html')

@app.route('/submit/', methods=['GET', 'POST'])
def submit():
    # 确认发送
    title = request.form.get('news_content')
    print(title)
    value = classnews_test.testmodel(title)
    return value[0]

@app.route('/againtrain/', methods=['GET', 'POST'])
def againtrain():
    # 页面3
    classnews_train.News.load_file()
    return render_template('html.html')

@app.route('/submit2/', methods=['GET', 'POST'])
def submit2():
    # 确认发送
    file = request.files.get('news_file')
    f = file.filename
    file.save(os.path.join(FILE_DIR, f))
    res = classnews_score_test.load_file(FILE_DIR+f)
    return '预测分类结果：{}'.format(res)

@app.route('/againtrain2/', methods=['GET', 'POST'])
def againtrain2():
    # 页面3
    res = classnews_score_train.load_file()
    return render_template('html2.html', **res)


@app.route('/submit3/', methods=['GET', 'POST'])
def submit3():
    # 确认发送
    a_value = request.form.get('a_value') # 系数
    b_intercept = request.form.get('b_intercept') # 截距
    up_error = request.form.get('up_error') # 扰动上限
    down_error = request.form.get('down_erroe') # 扰动下限
    if a_value and b_intercept and  up_error and down_error:
        scores = LinearRegression_bj.x_y(float(a_value), float(b_intercept), int(up_error), int(down_error))
    else:
        return '值不全'
    return json.dumps(scores)


@app.route('/submit4/', methods=['GET', 'POST'])
def submit4():
    # 逻辑回归和高斯朴素贝叶斯对鸢尾花数据的处理
    logistic_train_iris.deal_iris()
    GaussianNB_iris.deal_iris()
    return 'success'


@app.route('/updatapage4/', methods=['GET', 'POST'])
def updatapage_submit4():
    # 刷新
    return render_template('html4.html')


@app.route('/updatapage3/', methods=['GET', 'POST'])
def updatapage_submit3():
    # 刷新
    return render_template('html3.html')


@app.route('/submit5/', methods=['GET', 'POST'])
def submit5():
    # svm对鸢尾花数据处理
    return json.dumps(svm_iris.load_file())


@app.route('/updatapage5/', methods=['GET', 'POST'])
def updatapage_submit5():
    # 刷新
    return render_template('html5.html')


@app.route('/submit6/', methods=['GET', 'POST'])
def submit6():
    # svm 对不平衡样本的处理
    return json.dumps(svm_rbf_unbalance.load_data())


@app.route('/updatapage6/', methods=['GET', 'POST'])
def updatapage_submit6():
    # 刷新
    return render_template('html6.html')

@app.route('/html6.html/')
def html6():
    # 页面6
    return render_template('html6.html')

@app.route('/submit7/', methods=['GET', 'POST'])
def submit7():
    # kmeans
    return json.dumps(kmeans.load_data())


@app.route('/updatapage7/', methods=['GET', 'POST'])
def updatapage_submit7():
    # 刷新
    return render_template('html7.html')

@app.route('/html7.html/')
def html7():
    # 页面7
    return render_template('html7.html')


@app.route('/submit8/', methods=['GET', 'POST'])
def submit8():
    # svm_classnews
    txt = request.form.get('news_content')
    c = svm_classnews.ClassSvm()
    res = c.product_main(txt)
    return res


@app.route('/againtrain8/', methods=['GET', 'POST'])
def againtrain8():
    # 页面3
    c = svm_classnews.ClassSvm()
    result = c.train_main()
    return json.dumps(result)


@app.route('/updatapage8/', methods=['GET', 'POST'])
def updatapage_submit8():
    # 刷新
    return render_template('html8.html')


@app.route('/html8.html/')
def html8():
    # 页面8
    return render_template('html8.html')


@app.route('/submit9/', methods=['GET', 'POST'])
def submit9():
    # film_comment 对电影评价进行分类
    txt = request.form.get('news_content')
    if not txt:
        return '值不存在'
    res = film_comment.Comment().test_main(txt)
    if res[0] == 0:
        res = "dislike"
    if res[0] == 1:
        res = 'like'
    return res


@app.route('/againtrain9/', methods=['GET', 'POST'])
def againtrain9():
    # 页面9
    c = film_comment.Comment().train_main()
    return render_template('html9.html')


@app.route('/updatapage9/', methods=['GET', 'POST'])
def updatapage_submit9():
    # 刷新
    return render_template('html9.html')


@app.route('/html9.html/')
def html9():
    # 页面9
    return render_template('html9.html')



if __name__ == "__main__":
    app.run()

