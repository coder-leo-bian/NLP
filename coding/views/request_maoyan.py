import requests
import xlwt, json, xlrd
"""爬复仇者联盟4前10天的影评"""



def get_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36',
        'Cookie':'_lxsdk_cuid=1661ae34b39c8-0c8c15380422c-2711639-144000-1661ae34b39c8; v=3; __utma=17099173.21012536.1539393286.1539393286.1539393286.1; __utmc=17099173; __utmz=17099173.1539393286.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utmb=17099173.2.9.1539393288643; uuid_n_v=v1; iuuid=B7EC5090CE8511E89675C9EEEAD92C526D3F6AA2E9344F5091F4218FA118AAEF; webp=true; ci=55%2C%E5%8D%97%E4%BA%AC; __mta=207950381.1538049396050.1539393287057.1539393443480.10; _lx_utm=utm_source%3Dgoogle%26utm_medium%3Dorganic; _lxsdk=B7EC5090CE8511E89675C9EEEAD92C526D3F6AA2E9344F5091F4218FA118AAEF; _lxsdk_s=1666af306c5-403-5fd-037%7C%7C45',
        'Host':'m.maoyan.com',
        'Referer':'http://m.maoyan.com/movie/248172/comments?_v_=yes',
        'X-Requested-With':'superagent'
    }
    return requests.get(url,headers = headers)


def get_comment():
    comment = [None] * 7
    file = 'Project_Gutenberg_comment.xls'
    # 创建一个Excel
    book = xlwt.Workbook()
    # 在其中创建一个名为'复仇者联盟4'的sheet
    sheet1 = book.add_sheet('复仇者联盟4', cell_overwrite_ok=True)
    # 设置需要保存的信息栏
    row0 = ['nickName', 'startTime', 'gender', 'userLevel', 'cityName', 'score', 'content']
    # 将首行信息写入
    for l in range(len(row0)):
        sheet1.write(0, l, row0[l])
    d = 1
    # 对时间进行循环，因为电影上映的时间是9月30号，我们选择从爬取的评论是从10月1号到10月15号（也就是今天）
    for day in range(25, 31):
        print('正在爬取月{}日的评论'.format(day))
        # 每天只能爬取前1000条数据，每页15条，因此67次循环之后，便不再有数据可以爬取了
        for i in range(67):
            print('正在下载第{}页评论'.format(i + 1))
            r = get_page('http://m.maoyan.com/mmdb/comments/movie/248172.json?_v_=yes&offset=' + str(
                i * 15) + '&startTime=2019-04-{}%2023%3A59%3A59'.format(day))
            # 判断网页状态码，正常则对数据进行爬取，否则直接退出循环
            if r.status_code == 200:
                try:
                    soup = json.loads(r.text)['cmts']
                    j = 0
                    # 保存数据
                    for cmt in soup:
                        j += 1
                        try:
                            comment[0] = cmt['nickName']
                            comment[1] = cmt['startTime']
                            if cmt.get('gender'):
                                comment[2] = cmt['gender']
                            else:
                                comment[2] = None
                            comment[3] = cmt['userLevel']
                            comment[4] = cmt['cityName']
                            comment[5] = cmt['score']
                            comment[6] = cmt['content']

                        except:
                            break
                        # 写入数据
                        for k in range(len(comment)):
                            sheet1.write((d - 1) * 1005 + (i * 15 + j), k, comment[k])
                except:
                    break
            else:
                break
        d += 1

    d = 8
    # 对时间进行循环，因为电影上映的时间是9月30号，我们选择从爬取的评论是从10月1号到10月15号（也就是今天）
    for day in range(1, 6):
        print('正在爬取月{}日的评论'.format(day))
        # 每天只能爬取前1000条数据，每页15条，因此67次循环之后，便不再有数据可以爬取了
        for i in range(67):
            print('正在下载第{}页评论'.format(i + 1))
            r = get_page('http://m.maoyan.com/mmdb/comments/movie/248172.json?_v_=yes&offset=' + str(
                i * 15) + '&startTime=2019-05-{}%2023%3A59%3A59'.format(day))
            # 判断网页状态码，正常则对数据进行爬取，否则直接退出循环
            if r.status_code == 200:
                try:
                    soup = json.loads(r.text)['cmts']
                    j = 0
                    # 保存数据
                    for cmt in soup:
                        j += 1
                        try:
                            comment[0] = cmt['nickName']
                            comment[1] = cmt['startTime']
                            if cmt.get('gender'):
                                comment[2] = cmt['gender']
                            else:
                                comment[2] = None
                            comment[3] = cmt['userLevel']
                            comment[4] = cmt['cityName']
                            comment[5] = cmt['score']
                            comment[6] = cmt['content']

                        except:
                            break
                        # 写入数据
                        for k in range(len(comment)):
                            sheet1.write((d - 1) * 1005 + (i * 15 + j), k, comment[k])
                except:
                    break
            else:
                break
        d += 1
    # 保存文件
    book.save(file)

# get_comment()


def split_comments():
    data = xlrd.open_workbook('./Project_Gutenberg_comment.xls')
    a = data.sheet_by_index(0)
    tables = data.sheet_by_name('复仇者联盟4')
    n, m = 0, 0
    for i in range(1, tables.nrows):
        for x in range(1, tables.ncols):
            v = tables.row(i)[5].value
            if v:
                if v > 4.0:
                    n += 1
                    with open('./data.txt', 'a') as f:
                        f.write((((tables.row(i)[6].value).replace('\n', '')).replace(' ', '') + ' ' + '1' + '\n'))
                    break
                elif v < 1.5:
                    m += 1
                    with open('./data.txt', 'a') as f:
                        f.write((((tables.row(i)[6].value).replace('\n', '')).replace(' ', '') + ' ' + '0' + '\n'))
                    break
                else:
                    break
    print(n, m)


# split_comments()
