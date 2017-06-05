# -*- coding:utf-8 -*-
from util import constants
import pandas as pd
from util import utils
from util.utils import list2dict

def pos_statistic():
    total_df = pd.read_csv(constants.cus_train_path)
    print len(total_df['positionID'].unique())


# positionID * 1, site * 3, type * 6
def build_position(has_sparse=False):
    f = open(constants.project_path + "/dataset/raw/" + "position.csv")
    position = {}
    f.readline()
    # postion_df = pd.read_csv(constants.project_path + "/dataset/raw/position.csv")
    # positionID_set = list2dict(postion_df['positionID'].unique())

    stat = pd.read_csv(constants.clean_train_path)
    posdf = stat['positionID'].value_counts()
    del stat
    poslist = []
    for i, row in posdf.iteritems():
        if int(row) > 1600:
            poslist.append(i)
    positionID_set = utils.list2dict(poslist)
    print "pos id:" + str(len(positionID_set))
    del poslist
    offset = 0

    for line in f:
        fields = line.strip().split(',')
        features = []
        offset = 0

        # site
        features.append(offset+int(fields[1]))
        offset += 3
        # type
        features.append(offset + int(fields[2]))
        offset += 6

        # if has_sparse:
        # positionID
        if int(fields[0]) in positionID_set:
            features.append(offset + positionID_set[int(fields[0])])
        else:
            features.append(offset)
        offset += len(positionID_set) + 1

        position[int(fields[0])] = features

    print "Buliding position finished."
    return position, offset


if __name__ == '__main__':
    pos_statistic()
    # pos = build_position()
    # for key in pos.keys()[:5]:
    #     print pos[key]
