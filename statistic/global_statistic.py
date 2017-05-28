# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import constants


# 统计转化时间信息
def conversion_gap():
    train_df = pd.read_csv(constants.cus_train_path)
    positive = train_df.loc[train_df["label"] == 1]
    gap_df = positive['conversionTime'] - positive['clickTime']
    print "mean: " + str(gap_df.mean()/float(60))
    print "0% :" + str(gap_df.quantile(0)/float(60))
    print "10% :" + str(gap_df.quantile(0.1)/float(60))
    print "30% :" + str(gap_df.quantile(0.3)/float(60))
    print "50% :" + str(gap_df.quantile(0.5)/float(60))
    print "70% :" + str(gap_df.quantile(0.7)/float(60))
    print "80% :" + str(gap_df.quantile(0.8) / float(60))
    print "85% :" + str(gap_df.quantile(0.85) / float(60))
    print "90% :" + str(gap_df.quantile(0.9)/float(60))
    print "90% :" + str(gap_df.quantile(0.95) / float(60))
    print "100% :" + str(gap_df.quantile(1)/float(60))


# 绘制conversion图像
def conversion_graph():
    train_df = pd.read_csv(constants.cus_train_path)
    x = []
    y1_conv = []
    y2_click = []
    for i in range(17, 31):
        rbound = (i+1) * 1440
        lbound = i * 1440
        click_df = train_df[(train_df['clickTime'] < rbound)
                          & (train_df['clickTime'] >= lbound)]
        conv_df = click_df[click_df['label'] == 1]
        # print tmp_df.head(5)
        x.append(i)
        y2_click.append(len(conv_df)/float(len(click_df)))
    plt.figure(figsize=(18, 5))
    plt.xlabel("date")
    plt.ylabel("conversion")
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.plot(x, y2_click, color="blue", label="conversion")
    # plt.plot(x, y2_click, color="red", label="click")
    # plt.show()
    img_path = constants.project_path + '/img/' + "conversion" + '.png'
    plt.savefig(img_path)

'''
alpha = [mean*(1-mean)/var - 1] * mean
beta = [mean*(1-mean)/var - 1] * (1-mean)

global: 0.025
134.673655423
5084.73200722

connection 0： 0.001
9.1154384919
2085.26585064

1: 0.03
225.507154404
7246.05062482

2: 0.008
22.4405243699
2527.52635529

3: 0.009
0.888821701803
90.0274010898
'''


def compute_alpha_beta():
    # file_path = constants.project_path+"/dataset/custom/split_online/b1/train_with_ad_info.csv"
    train_df = pd.read_csv(constants.clean_train_path)
    conn_df = train_df[train_df.connectionType == 3]
    y = []
    for i in range(17, 30):
        rbound = (i+1) * 1440
        lbound = i * 1440
        click_df = conn_df[(conn_df['clickTime'] < rbound)
                          & (conn_df['clickTime'] >= lbound)]
        conv_df = click_df[click_df['label'] == 1]
        # print tmp_df.head(5)
        y.append(len(conv_df)/float(len(click_df)))
    y_df = pd.DataFrame(y)
    mean = float(y_df.mean().values[0])
    var = float(y_df.var().values[0])
    print (mean*(1-mean)/var - 1) * mean
    print (mean*(1-mean)/var - 1) * (1-mean)


def app_cvr_connection(appID):
    file_path = constants.project_path + "/dataset/custom/split_online/b1/train_with_ad_info.csv"
    train_df = pd.read_csv(file_path)
    print train_df[train_df.appID == appID]['connectionType'].value_counts()




def app_cvr_graph():
    file_path = constants.project_path+"/dataset/custom/split_online/b1/train_with_ad_info.csv"
    # file_path = constants.cus_train_path
    train_df = pd.read_csv(file_path)
    x = []
    y = []
    for i in range(17, 31):
        rbound = (i + 1) * 1440
        lbound = i * 1440
        day_df = train_df[(train_df['clickTime'] < rbound)
                          & (train_df['clickTime'] >= lbound)]
        x.append(i)
        y.append(len(day_df[day_df['appID'] == 14])/float(len(day_df)))
    plt.figure(figsize=(18, 5))
    plt.xlabel("date")
    plt.ylabel("connection 1 rate")
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.plot(x, y, color="blue", label="connection 1 rate")
    # plt.plot(x, y2_click, color="red", label="click")
    plt.show()
    # img_path = constants.project_path + '/img/' + "connection 1 rate" + '.png'
    # plt.savefig(img_path)

if __name__ == "__main__":
    # app_cvr_graph()
    # compute_alpha_beta()
    app_cvr_connection(391)