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
    total_df['instance'] = total_df.index
    partial_df = total_df[(total_df['clickTime'] >= start_date*10000)
                          & (total_df['clickTime'] < 310000)]
    partial_df.fillna(-1, inplace=True)
    data_array = partial_df.as_matrix().astype(int)
    header = list(partial_df.columns.values)
    for i in xrange(5):
        random.seed()
        train_file = to_dir + "train_x_" + str(i)
        sample_index = []
        n = len(data_array)
        for j in xrange(n):
            sample_index.append(random.randint(0, n-1))
        sample_index.sort()
        # np.savetxt(to_dir + "/index/train_x_" + str(i)+'_idx', sample_index, fmt="%s")
        with open(train_file, 'w') as f:
            f.write(','.join(header))
            f.write('\n')
            for j in sample_index:
                f.write(','.join(str(x) for x in data_array[j]))
                f.write('\n')
        print str(i) + " finished."


# 按日期进行分割
def split_by_date(left_bound, right_bound, to_file):
    total_df = pd.read_csv(constants.raw_train_path)
    train_df = total_df[(total_df['clickTime'] >= left_bound*1000000)
                        & (total_df['clickTime'] < right_bound*1000000)]
    train_df.to_csv(to_file, index=False)
    print train_df.shape
    # test_df.to_csv(to_dir + "test_x.csv", index=False)
    # print test_df.shape


# 随机按量切分数据集
def random_split_dataset(train_file, train_percent, to_path):
    total_df = pd.read_csv(train_file)
    print total_df.shape
    # total_df = abandon_data(total_df, abandon_list)
    # shuffle原数据集

    for i in range(2, 3):
        random_df = total_df.sample(frac=1)
        random_df['instanceID'] = random_df.index
        random_df.reset_index(drop=True, inplace=True)
        # print random_df.head()
        # print random_df.shape
        # random_df = total_df
        n = len(random_df)
        print n
        train_bound = int(n * train_percent)
        train_df = random_df.ix[:train_bound, :]
        # index = np.asarray(train_df['instance'].tolist())
        # np.savetxt(to_path + "train_x_" + str(i)+'_idx', index, fmt="%s")
        train_df.reset_index(drop=True, inplace=True)
        train_df.to_csv(path_or_buf=to_path + "train_x_" + str(i), index=False)
        print train_df.shape
        del train_df

        # valid_bound = train_bound + int(n * valid_percent)
        # print valid_bound
        # valid_df = random_df.ix[train_bound:valid_bound, :]
        # valid_df.to_csv(path_or_buf=to_path+"valid_split_2.csv", index=False)
        # print valid_df.shape
        # del valid_df

        test_df = random_df.ix[train_bound:, :]
        # index = np.asarray(test_df['old_index'].tolist())
        # np.savetxt(to_path + "test_x_" + str(i)+'_idx', index, fmt="%s")
        test_df.reset_index(drop=True, inplace=True)
        test_df.to_csv(path_or_buf=to_path + "test_x_" + str(i), index=False)
        print test_df.shape
        del test_df
        del random_df


# 重新映射age
def map_age(age):
    age = int(age)
    age_bucket = [7, 12, 16, 21, 25, 32, 40, 48, 55, 65]
    if age != 0:
        for i in xrange(len(age_bucket)):
            if age < age_bucket[i]:
                return i+1
        if age > age_bucket[9]:
            return 11
    return age


# 重新映射hometown
def map_place_func(x):
    if x != 0:
        return x / 100
    return x


# 在train文件中merge user信息
def merge_by_user(train_file, to_path):
    total_df = pd.read_csv(train_file)
    user_df = pd.read_csv(constants.raw_path + "/user.csv")
    merge_df = pd.merge(left=total_df, right=user_df, how='left', on=['userID'])

    # pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    # merge_df = pd.merge(left=merge_df, right=pos_df, how='left', on=['positionID'])
    merge_df['age'] = merge_df['age'].apply(lambda x: map_age(x))
    merge_df['hometown'] = merge_df['hometown'].apply(lambda x: map_place_func(x))
    merge_df['residence'] = merge_df['residence'].apply(lambda x: map_place_func(x))
    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging user finished."


# 在train文件中merge ad信息
def merge_by_ad(train_file, to_path):
    total_df = pd.read_csv(train_file)
    ad_df = pd.read_csv(constants.raw_path + "/ad.csv")

    merge_df = pd.merge(left=total_df, right=ad_df, how='left', on=['creativeID'])

    # pos_df = pd.read_csv(constants.project_path + "/dataset/raw/" + "position.csv")
    # merge_df = pd.merge(left=merge_df, right=pos_df, how='left', on=['positionID'])

    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging ad finished."


# 在train文件中merge pos信息
def merge_by_pos(train_file, to_path):
    total_df = pd.read_csv(train_file)
    pos_df = pd.read_csv(constants.raw_path + "/position.csv")
    merge_df = pd.merge(left=total_df, right=pos_df, how='left', on=['positionID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)
    print "Merging pos finished."


def merge_by_category(train_file, to_path):
    total_df = pd.read_csv(train_file)
    cate_df = pd.read_csv(constants.raw_path + "/app_categories.csv")
    merge_df = pd.merge(left=total_df, right=cate_df, how='left', on=['appID'])
    merge_df.to_csv(path_or_buf=to_path, index=False)


def merge_with_all_data(train_file, day, to_dir):
    merge_tmp = to_dir+"/merge_tmp"
    merge_by_ad(train_file, merge_tmp+"_1")
    merge_by_pos(merge_tmp+"_1",
                 merge_tmp + "_2")
    merge_by_category(merge_tmp + "_2", merge_tmp + "_3")
    merge_by_user(merge_tmp + "_3",
                  to_dir+"/train_with_all_"+str(day)+".csv")
    # all_info = pd.read_csv(to_dir+"train_with_all_info.csv")


# 合并分开的数据集
def concat_data(dir_path, l_bound, r_bound, to_path):
    to_file = open(to_path, 'w')
    df = None
    count = 0
    for train_file in os.listdir(dir_path):
        print train_file
        if os.path.isdir(dir_path+train_file):
            continue
        day = int(train_file.split('_')[3].split('.')[0])
        if day < l_bound or day > r_bound:
            continue

        with open(dir_path+train_file, 'r') as f:
            if count == 0:
                to_file.write(f.readline())
            else:
                f.readline()
            for line in f:
                to_file.write(line)
            count += 1
    to_file.close()


if __name__ == '__main__':
    dir_path = constants.project_path + "/dataset/custom/"
    # bootstrap_online_train(23, dir_path)
    # split_by_date_kfold(20, dir_path)
    # abandon_thirty(constants.raw_train_path)

    # conversion_gap()
    # convert_data_time(constants.raw_train_path, constants.project_path + "/dataset/custom/train_re-time.csv", 0)
    # convert_data_time(constants.raw_test_path, constants.project_path +"/dataset/custom/test_re-time.csv", 1)
    # install_merge_by_app(constants.project_path + "/dataset/raw/" + "user_installedapps.csv",
    #                      constants.project_path + "/dataset/raw/" + "user_installedapps_with_category.csv")
    # merge_by_ad(constants.custom_path+'/for_predict/train.csv', constants.project_path+"/dataset/custom/for_predict/train_with_ad_info.csv")
    # merge_by_user(constants.raw_test_path,
    #               constants.custom_path + '/test_with_user_info.csv')
    # merge_by_pos(constants.custom_path+'/for_predict/train.csv',
    # constants.project_path + "/dataset/custom/train_with_pos_info.csv")
    # merge_with_all_data(constants.custom_path+'/for_predict/train.csv', 0, constants.custom_path+'/for_predict')


    # merge_by_category(constants.custom_path+'/clean_id/train.csv',
    #               constants.project_path + "/dataset/custom/clean_id/train_with_ad_pos_user_re2.csv")
    # for day in range(17, 31):
    #     train_file = constants.custom_path+'/split_by_day/train_'+str(day)+'.csv'
    #     merge_by_user(train_file, constants.custom_path+'/split_by_day/with_user/train_with_user_'+str(day)+'.csv')
    #     print str(day) + ' finished.'
    # train_file = constants.custom_path + '/split_by_day/tmp/train_30_clean_3.csv'
    # merge_by_ad(train_file, constants.custom_path + '/split_by_day/with_ad/train_with_ad_30.csv')
    # random_split_dataset(0.85, constants.project_path+"/dataset/custom/split_6/")
    # bootstrap_online_train(24, dir_path+"split_online/")
    # headers = ['label','clickTime','conversionTime','creativeID','userID','positionID','connectionType','telecomsOperator','age','gender','education','marriageStatus','haveBaby','hometown','residence']
    # concat_data(constants.custom_path+'/split_by_day/with_pos/', 17, 30,
    #             constants.custom_path+'/for_train/train_with_pos_info.csv')
    # split_by_date(23, 27, constants.custom_path+'/for_train/train.csv')
    # split_by_date(27, 31, constants.custom_path + '/for_predict/train.csv')
    random_split_dataset(constants.custom_path+'/for_predict/train_with_user_info.csv', 0.8, constants.custom_path+'/split_0/')
