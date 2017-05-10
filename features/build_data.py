# -*- coding:utf-8 -*-
import constants
import pandas as pd
import io


# 构造label
def build_y(from_path, to_path):
    df = pd.read_csv(from_path)
    y = df['label']
    print y.head(10)
    y.to_csv(path=to_path, index=False, header=False)


def build_x():
    pass

if __name__ == '__main__':
    build_y(constants.local_train_path, constants.project_path + "/dataset/custom/local_train_y")
    build_y(constants.local_valid_path, constants.project_path + "/dataset/custom/local_valid_y")
    build_y(constants.local_test_path, constants.project_path + "/dataset/custom/local_test_y")