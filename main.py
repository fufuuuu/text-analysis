from traceback import print_tb
import pandas as pd
from pandas import DataFrame
import jieba as jieba
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from matplotlib import pyplot as plot
import numpy as np


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

# 标注数据集
def make_label(df):
    df["sentiment"] = df["total"].apply(lambda x: 1 if x > 4 else 0)

# 通过句法规则词典
def has_direct_words(text):
    with open('directWords.txt', encoding='utf-8') as f:
        direct_words = f.read()
    direct_word_list = direct_words.replace('\n', '   ').split()
    for item in direct_word_list:
        if item in text:
            print(item, text)
        return item in text


def get_custom_stopword(stop_word_file):
    with open(stop_word_file, encoding='utf-8') as f:
        stop_word = f.read()
    stop_word_list = stop_word.replace('\n', '   ').split()
    custom_stopword = [i for i in stop_word_list]
    return custom_stopword

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    # 读取文件并配置标签
    data = pd.read_csv("comments.csv", encoding='utf8')
    make_label(data)
    # 获取评论列以及标签列
    X = data[["comment"]]
    Y = data.sentiment
    # 对评论列内容分词且创造训练集以及测试集
    X["cuted_comment"] = X.comment.apply(chinese_word_cut)
    i = 0.1
    plot_x = range(10, 90, 1)
    plot_y = []
    while i < 0.90 :
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=i, random_state=1)
        # 去除停用词、过于普通和过于特殊的词
        stopwords = get_custom_stopword("StopWords.txt")
        vect = CountVectorizer(max_df=0.9,
                            min_df=0,
                            token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                            stop_words=frozenset(stopwords))
        term_matrix = DataFrame(vect.fit_transform(X_train.cuted_comment).toarray(), columns=vect.get_feature_names())
        # 导入朴素贝叶斯函数，创建分类模型
        nb = MultinomialNB()
        pipe = make_pipeline(vect, nb)
        # 将未特征向量化的数据导入，提高模型准确率
        cross_val_score(pipe, X_train.cuted_comment, y_train, cv=5, scoring='accuracy').mean()
        pipe.fit(X_train.cuted_comment, y_train)
        # 通过词典处理
        # for i in X_test.cuted_comment:
        #     if has_direct_words(i):

        y_pred = pipe.predict(X_test.cuted_comment)
        # 比较预测结果与测试结果获得准确率
        print('预测结果准确率为：', metrics.accuracy_score(y_test, y_pred) * 100, '%')
        plot_y.append(metrics.accuracy_score(y_test, y_pred))
        i +=  0.01
    plot.plot(plot_x, plot_y)
    #设置x，y轴相关信息，设置主题
    plot.xlabel("Ratio of train sets/%")
    plot.ylabel("Test set Accuracy")
    plot.title("Ratio of train sets-Test set Accuracy")

    plot.show()