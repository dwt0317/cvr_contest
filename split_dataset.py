# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

train_path = os.getcwd() + "/dataset/raw/train.csv"
test_path = os.getcwd() + "/dataset/raw/test.csv"
cus_train_path = os.getcwd()+"/dataset/custom/train_t.csv"
cus_test_path = os.getcwd()+"/dataset/custom/test_t.csv"

file_header = 'label,clickTime,conversionTime,creativeID,userID,positionID,connectionType,telecomsOperator'

# 统计训练集中正负样例的比例
def count_positive():
    train_df = pd.read_csv(train_path)
    # print train_df.head(20)
    count = train_df['label'].value_counts()
    print count
    print count[1] / float(count[0] + count[1])


# 将时间转化为以分钟累加的数字
def time2num(num):
    numint = int(num)
    minute = numint % 100
    numint /= 100
    hour = numint % 100
    numint /= 100
    date = numint % 100
    total_min = minute + 60 * hour + 24 * 60 * date
    return total_min


# 替换数据时间表示，便于加减操作
def convert_data_time(file_path, to_path, offset):
    df = pd.read_csv(file_path)
    # 使用0替换na
    df = df.fillna(-1)
    print df.head(5)
    narray = df.as_matrix().astype(int)
    print narray[:5]
    for row in narray:
        row[offset+1] = time2num(int(row[offset + 1]))
        row[offset+2] = time2num(int(row[offset + 2]))
    with open(to_path, 'w') as f:
        np.savetxt(f, narray, fmt='%d', delimiter=',', header=file_header)


# 统计转化时间信息
def conversion_gap():
    train_df = pd.read_csv(cus_train_path)
    positive = train_df.loc[train_df["label"] == 1]
    gap_df = positive['conversionTime'] - positive['clickTime']
    print "mean: " + str(gap_df.mean())
    print "0% :" + str(gap_df.quantile(0)/float(60))
    print "10% :" + str(gap_df.quantile(0.1)/float(60))
    print "30% :" + str(gap_df.quantile(0.3)/float(60))
    print "50% :" + str(gap_df.quantile(0.5)/float(60))
    print "70% :" + str(gap_df.quantile(0.7)/float(60))
    print "90% :" + str(gap_df.quantile(0.9)/float(60))
    print "100% :" + str(gap_df.quantile(1)/float(60))


# 绘制conversion图像
def conversion_graph():
    train_df = pd.read_csv(cus_train_path)
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
        y1_conv.append(len(conv_df))
    plt.figure(figsize=(18, 5))
    plt.xlabel("date")
    plt.ylabel("conversion")
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.plot(x, y1_conv, color="blue", label="conversion")
    # plt.plot(x, y2_click, color="red", label="click")
    # plt.show()
    img_path = 'img/' + "conversion" + '.png'
    plt.savefig(img_path)


# 舍弃部分异常日期的数据
def abandon_data(total_df, abandon_list):
    for i in abandon_list:
        total_df = total_df[(total_df['clickTime'] >= (i+1)*1440)
                            | (total_df['clickTime'] < i*1440)]
    return total_df


# 切分数据集
def split_dataset(train_percent, valid_percent, to_path):
    total_df = pd.read_csv(cus_train_path)
    print total_df.shape
    abandon_list = [19, 30]
    total_df = abandon_data(total_df, abandon_list)

    # shuffle原数据集
    random_df = total_df.sample(frac=1).reset_index(drop=True)
    print random_df.shape

    n = len(random_df)
    train_bound = int(n * train_percent)
    train_df = random_df.ix[:train_bound, :]
    train_df.to_csv(path_or_buf=to_path+"local_train.csv", index=False)
    print train_df.shape
    del train_df

    valid_bound = train_bound + int(n * valid_percent)
    print valid_bound
    valid_df = random_df.ix[train_bound:valid_bound, :]
    valid_df.to_csv(path_or_buf=to_path+"local_valid.csv", index=False)
    print valid_df.shape
    del valid_df

    test_df = random_df.ix[valid_bound:, :]
    test_df.to_csv(path_or_buf=to_path+"local_test.csv", index=False)
    print test_df.shape
    del test_df


if __name__ == '__main__':
    split_dataset(0.8, 0.1, os.getcwd()+"/dataset/custom/")
    # conversion_graph()
    # conversion_gap()
    # convert_data_time(train_path, os.getcwd()+"/dataset/custom/train_t.csv", 0)
    # convert_data_time(test_path, os.getcwd()+"/dataset/custom/test_t.csv", 1)
