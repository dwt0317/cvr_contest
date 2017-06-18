# -*- coding:utf-8 -*-
import pandas as pd
import constants
import random


# 全局sample
def total_samplilng(raw_file, to_dir):
    train_df = pd.read_csv(raw_file)
    sample_df = train_df.sample(frac=0.5, random_state=4)
    sample_df.to_csv(to_dir+'/sample_train_05.csv', index=False)


# 对样本进行负采样
def negative_down_sampling(src_dir_path, rate):
    for i in range(0, 5):
        train_src_file = src_dir_path + "train_x_" + str(i)
        train_df = pd.read_csv(train_src_file)
        negative_df = train_df[train_df.label == 0]
        indexes = list(negative_df.index)

        n = len(train_df)
        k = len(indexes)
        train_df.fillna(-1, inplace=True)
        print n, k
        random_list = []
        for m in xrange(int(n * rate)):
            random_list.append(random.randint(0, k - 1))
        print len(random_list)

        positive_list = list(train_df[train_df.label == 1].index)
        random_list.extend(positive_list)
        random_list.sort()
        print random_list[:10]

        train_np = train_df.as_matrix().astype(int)
        print train_np[:10, :]

        sample_f = open(src_dir_path + 'sample/' + "train_x_" + str(i) + '_sample', 'w')
        sample_f.write(','.join(list(train_df.columns.values)))
        sample_f.write('\n')
        for idx in random_list:
            sample_f.write(','.join(str(x) for x in train_np[idx]))
            sample_f.write('\n')
        sample_f.close()


# bootstrap采样线上的train数据
def bootstrap_online_train(start_date, raw_file, to_dir):
    total_df = pd.read_csv(raw_file)
    total_df['instance'] = total_df.index
    partial_df = total_df
    partial_df.fillna(-1, inplace=True)
    data_array = partial_df.as_matrix().astype(int)
    header = list(partial_df.columns.values)
    print "Reading data finished."
    for i in xrange(5):
        random.seed()
        train_file = to_dir + "train_x_" + str(i)
        n = len(data_array)
        sample_index = [0] * n
        for j in xrange(n):
            sample_index[j] = random.randint(0, n-1)
        sample_index.sort()
        # np.savetxt(to_dir + "/index/train_x_" + str(i)+'_idx', sample_index, fmt="%s")
        with open(train_file, 'w') as f:
            f.write(','.join(header))
            f.write('\n')
            for j in sample_index:
                f.write(','.join(str(x) for x in data_array[j]))
                f.write('\n')
        print str(i) + " finished."


if __name__ == '__main__':
    # bootstrap_online_train(27, constants.custom_path+'/for_predict/train_with_user_info.csv', constants.custom_path+'/split_online/')
    negative_down_sampling(constants.project_path+"/dataset/custom/split_0/", 0.025)
    # total_samplilng(constants.raw_path, constants.custom_path)