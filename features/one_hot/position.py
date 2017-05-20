# -*- coding:utf-8 -*-
from util import constants
import pandas as pd


def pos_statistic():
    total_df = pd.read_csv(constants.cus_train_path)
    print len(total_df['positionID'].unique())


# site * 3, type * 6
def build_position():
    f = open(constants.project_path + "/dataset/raw/" + "position.csv")
    position = {}
    f.readline()
    offset = 0
    for line in f:
        fields = line.strip().split(',')
        features = {}
        offset = 0

        # site
        features[offset+int(fields[1])] = 1
        offset += 3
        # type
        features[offset+int(fields[2])] = 1
        offset += 6
        position[int(fields[0])] = features

    print "Buliding position finished."
    return position, offset


if __name__ == '__main__':
    pos_statistic()
    # pos = build_position()
    # for key in pos.keys()[:5]:
    #     print pos[key]
