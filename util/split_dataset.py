# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import constants


# 统计训练集中正负样例的比例
def count_positive():
    train_df = pd.read_csv(constants.raw_train_path)
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
    # 使用-1替换na
    df = df.fillna(-1)
    print df.head(10)
    narray = df.as_matrix().astype(int)
    print narray[:5]
    for row in narray:
        row[offset+1] = time2num(int(row[offset + 1]))
        if int(row[offset + 2]) != -1:
            row[offset+2] = time2num(int(row[offset + 2]))
    file_header = ','.join(df.columns.values.tolist())
    with open(to_path, 'w') as f:
        np.savetxt(f, narray, fmt='%d', delimiter=',', header=file_header)


# 舍弃部分异常日期的数据
def abandon_data(total_df, abandon_list):
    for i in abandon_list:
        total_df = total_df[(total_df['clickTime'] >= (i+1)*1440)
                            | (total_df['clickTime'] < i*1440)]
    return total_df


# 丢弃第30天最后3个小时的负样本, 使得改天的转化率=平均值
def abandon_thirty(cus_train_path):
    total_df = pd.read_csv(cus_train_path)
    front = total_df[(total_df['clickTime'] < 30*1440)]
    partial_thirty = total_df[(total_df['clickTime'] < (30+1)*1440-180)
                              & (total_df['clickTime'] >= 30*1440)]
    remain_positive = total_df[(total_df['clickTime'] >= (30+1)*1440-180) &
                               (total_df['clickTime'] < (30 + 1) * 1440) &
                               (total_df['label'] == 1)]
    clean_thrity = partial_thirty.append(remain_positive)
    print len(clean_thrity[clean_thrity['label'] == 1]) / float(len(clean_thrity))
    clean_total = front.append(partial_thirty).append(remain_positive)
    clean_total.to_csv(constants.project_path + "/dataset/custom/train_clean.csv", index=False)


# 按照时序进行交叉分割
def split_by_date_kfold(start_date, to_dir):
    total_df = pd.read_csv(constants.clean_train_path)
    parital_df = total_df[total_df['clickTime'] >= start_date*1440]
    del total_df
    index = 0
    for train_start in range(start_date, 25, 1):
        test_start = train_start + 7
        train_df = parital_df[(parital_df['clickTime'] >= train_start*1440) &
                              (parital_df['clickTime'] < test_start*1440)]
        test_df = parital_df[(parital_df['clickTime'] >= test_start * 1440) &
                             (parital_df['clickTime'] < (test_start+1) * 1440)]
        train_df.to_csv(to_dir + "train_x_" + str(index), index=False)
        test_df.to_csv(to_dir + "test_x_" + str(index), index=False)
        index += 1


# 按日期进行分割
def split_by_date(train_date_bound, to_dir):
    total_df = pd.read_csv(constants.clean_train_path)
    train_df = total_df[(total_df['clickTime'] < (train_date_bound+1)*1440)]
    test_df = total_df[(total_df['clickTime'] >= (train_date_bound+1)*1440)]
    train_df.to_csv(to_dir + "train_x.csv", index=False)
    print train_df.shape
    test_df.to_csv(to_dir + "test_x.csv", index=False)
    print test_df.shape


# 随机按量切分数据集
def random_split_dataset(train_percent, valid_percent, to_path):
    total_df = pd.read_csv(constants.cus_train_path)
    print total_df.shape
    # total_df = abandon_data(total_df, abandon_list)
    # shuffle原数据集
    # random_df = total_df.sample(frac=1).reset_index(drop=True)
    # print random_df.shape
    # random_df = total_df
    n = len(total_df)
    train_bound = int(n * train_percent)
    train_df = total_df.ix[:train_bound, :]
    train_df.to_csv(path_or_buf=to_path+"train_split_2.csv", index=False)
    print train_df.shape
    del train_df

    valid_bound = train_bound + int(n * valid_percent)
    print valid_bound
    valid_df = total_df.ix[train_bound:valid_bound, :]
    valid_df.to_csv(path_or_buf=to_path+"valid_split_2.csv", index=False)
    print valid_df.shape
    del valid_df

    test_df = total_df.ix[valid_bound:, :]
    test_df.to_csv(path_or_buf=to_path+"test_split_2.csv", index=False)
    print test_df.shape
    del test_df


# 在train文件中merge user信息
def merge_by_user(train_file, to_path):
    total_df = pd.read_csv(train_file)
    user_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "user.csv")
    merge_df = pd.merge(left=total_df, right=user_df, how='left', on=['userID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging user finished."


# 在train文件中merge ad信息
def merge_by_ad(train_file, to_path):
    total_df = pd.read_csv(train_file)
    ad_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "ad.csv")
    merge_df = pd.merge(left=total_df, right=ad_df, how='left', on=['creativeID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging ad finished."


# 在train文件中merge pos信息
def merge_by_pos(train_file, to_path):
    total_df = pd.read_csv(train_file)
    pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    merge_df = pd.merge(left=total_df, right=pos_df, how='left', on=['positionID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging pos finished."


if __name__ == '__main__':
    dir_path = constants.project_path + "/dataset/custom/split_5/"
    split_by_date_kfold(20, dir_path)
    # pass
    # split_dataset(0.8, 0.1, os.getcwd()+"/dataset/custom/split_3/")
    # abandon_thirty(constants.cus_train_path)
    # split_by_date(28, constants.project_path + "/dataset/custom/split_4/")
    # merge_by_ad(constants.cus_test_path, dir_path+"test_with_ad_info.csv")
    # merge_by_pos(constants.clean_train_path,
    #              constants.project_path + "/dataset/custom/split_4/train_with_pos_info.csv")
    # merge_by_user(constants.clean_train_path,
    #               constants.project_path + "/dataset/custom/split_4/train_with_user_info.csv")
    # conversion_gap()
    # convert_data_time(constants.raw_train_path, constants.project_path + "/dataset/custom/train_re-time.csv", 0)
    # convert_data_time(constants.raw_test_path, constants.project_path  + "/dataset/custom/test_re-time.csv", 1)
