# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import constants
import random
import csv

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
        # if int(row[offset + 2]) != -1:
        #     row[offset+2] = time2num(int(row[offset + 2]))
    file_header = ','.join(df.columns.values.tolist())
    with open(to_path, 'w') as f:
        np.savetxt(f, narray, fmt='%d', delimiter=',', header=file_header)


# 舍弃部分异常日期的数据
def abandon_data(total_df, abandon_list):
    for i in abandon_list:
        total_df = total_df[(total_df['clickTime'] >= (i+1)*1440)
                            | (total_df['clickTime'] < i*1440)]
    return total_df


# 丢弃第30天最后1个小时的负样本
def abandon_thirty(cus_train_path):
    total_df = pd.read_csv(cus_train_path)
    front = total_df[(total_df['clickTime'] < 300000)]
    partial_thirty = total_df[(total_df['clickTime'] < 302300)
                              & (total_df['clickTime'] >= 300000)]
    remain_positive = total_df[(total_df['clickTime'] >= 302300) &
                               (total_df['clickTime'] < 310000) &
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


# bootstrap采样线上的train数据
def bootstrap_online_train(start_date, to_dir):
    total_df = pd.read_csv(constants.clean_train_path)
    partial_df = total_df[(total_df['clickTime'] >= start_date*1440)
                          & (total_df['clickTime'] < 30*1440)]
    data_array = partial_df.as_matrix()
    header = list(partial_df.columns.values)
    for i in xrange(10):
        random.seed()
        train_file = to_dir + "train_x_" + str(i)
        sample_index = []
        n = len(data_array)
        for j in xrange(n):
            sample_index.append(random.randint(0, n-1))
        sample_index.sort()
        with open(train_file, 'w') as f:
            f.write(','.join(header))
            f.write('\n')
            for j in sample_index:
                f.write(','.join(str(x) for x in data_array[j]))
                f.write('\n')
        print str(i) + " finished."


# 按日期进行分割
def split_by_date_online(left_bound, right_bound, to_dir):
    total_df = pd.read_csv(constants.raw_train_path)
    train_df = total_df[(total_df['clickTime'] >= left_bound)
                        & (total_df['clickTime'] < right_bound)]
    # test_df = total_df[(total_df['clickTime'] > right_bound*1440)]
    train_df.to_csv(to_dir + "train_second_week.csv", index=False)
    print train_df.shape
    # test_df.to_csv(to_dir + "test_x.csv", index=False)
    # print test_df.shape


# 随机按量切分数据集
def random_split_dataset(train_percent, to_path):
    total_df = pd.read_csv(constants.clean_train_path)
    print total_df.shape
    # total_df = abandon_data(total_df, abandon_list)
    # shuffle原数据集

    for i in xrange(5):
        random_df = total_df.sample(frac=1).reset_index(drop=True)
        # print random_df.shape
        # random_df = total_df
        n = len(random_df)
        train_bound = int(n * train_percent)
        train_df = random_df.ix[:train_bound, :]
        train_df.to_csv(path_or_buf=to_path+"train_x_"+str(i)+".csv", index=False)
        print train_df.shape
        del train_df

        # valid_bound = train_bound + int(n * valid_percent)
        # print valid_bound
        # valid_df = random_df.ix[train_bound:valid_bound, :]
        # valid_df.to_csv(path_or_buf=to_path+"valid_split_2.csv", index=False)
        # print valid_df.shape
        # del valid_df

        test_df = random_df.ix[train_bound:, :]
        test_df.to_csv(path_or_buf=to_path+"test_x_"+str(i)+".csv", index=False)
        print test_df.shape
        del test_df


# 在train文件中merge user信息
def merge_by_user(train_file, to_path):
    total_df = pd.read_csv(train_file)
    user_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "user.csv")
    merge_df = pd.merge(left=total_df, right=user_df, how='left', on=['userID'])

    # pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    # merge_df = pd.merge(left=merge_df, right=pos_df, how='left', on=['positionID'])

    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging user finished."


# 在train文件中merge ad信息
def merge_by_ad(train_file, to_path):
    total_df = pd.read_csv(train_file)
    ad_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "ad.csv")

    merge_df = pd.merge(left=total_df, right=ad_df, how='left', on=['creativeID'])

    # pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    # merge_df = pd.merge(left=merge_df, right=pos_df, how='left', on=['positionID'])

    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging ad finished."


# 在train文件中merge pos信息
def merge_by_pos(train_file, to_path):
    total_df = pd.read_csv(train_file)
    pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    merge_df = pd.merge(left=total_df, right=pos_df, how='left', on=['positionID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging pos finished."


def merge_by_category(train_file, to_path):
    total_df = pd.read_csv(train_file)
    cate_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "app_categories.csv")
    merge_df = pd.merge(left=total_df, right=cate_df, how='left', on=['appID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)


def merge_with_all_data(to_dir):
    merge_by_ad(constants.clean_train_path, to_dir+"train_with_ad_info.csv")
    # merge_by_pos(constants.clean_train_path,
    #              to_dir+"train_with_pos_info.csv")
    # merge_by_user(constants.clean_train_path,
    #               to_dir+"train_with_user_info.csv")



if __name__ == '__main__':
    dir_path = constants.project_path + "/dataset/custom/"
    # bootstrap_online_train(23, dir_path)
    # split_by_date_kfold(20, dir_path)
    # pass
    abandon_thirty(constants.raw_train_path)
    split_by_date_online(240000, 310000, dir_path)
    # merge_with_all_data(constants.project_path + "/dataset/custom/")
    # conversion_gap()
    # convert_data_time(constants.raw_train_path, constants.project_path + "/dataset/custom/train_re-time.csv", 0)
    # convert_data_time(constants.raw_test_path, constants.project_path +"/dataset/custom/test_re-time.csv", 1)
    # install_merge_by_app(constants.project_path + "/dataset/raw/" + "user_installedapps.csv",
    #                      constants.project_path + "/dataset/raw/" + "user_installedapps_with_category.csv")
    # pass
    merge_by_ad(constants.clean_train_path, constants.project_path+"/dataset/custom/train_with_ad_info.csv")
    merge_by_user(constants.clean_train_path, constants.project_path + "/dataset/custom/train_with_user_info.csv")
    merge_by_pos(constants.clean_train_path, constants.project_path + "/dataset/custom/train_with_pos_info.csv")
    # merge_by_category(constants.project_path+"/dataset/custom/train_with_ad_pos_user_re.csv",
    #               constants.project_path + "/dataset/custom/train_with_ad_pos_user_re2.csv")
    random_split_dataset(0.85, constants.project_path+"/dataset/custom/split_6/")
