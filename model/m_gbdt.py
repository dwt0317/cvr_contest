# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics   #Additional scklearn functions
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier
import cPickle as pickle
import constants
import xgboost as xgb
import pandas as pd
import os
import Utils
import time
import datetime


# 训练GBDT模型，并保存叶子结点特征
def train_model():
    train_x_file = constants.project_path + "/dataset/x_y/split_4/b1/train_x.fm"
    train_y_file = constants.project_path + "/dataset/x_y/split_4/b1/train_y"
    test_x_file = constants.project_path + "/dataset/x_y/split_4/b1/test_x.fm"
    test_y_file = constants.project_path + "/dataset/x_y/split_4/b1/test_y"
    valid_x_file = constants.project_path + "/dataset/x_y/split_4/b1/test_x.fm"
    test_y = np.loadtxt(test_y_file, dtype=int)
    # svmlight格式自带label
    train_data = load_svmlight_file(train_x_file)

    rounds = 150
    grid = False
    if grid:

        classifier = XGBClassifier(learning_rate=0.1, n_estimators=rounds, gamma=4, subsample=0.8, max_depth=8,
                                   min_child_weight=3,
                                   objective='binary:logistic', nthread=4)
        param_test1 = {
            'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
        }
        # 'colsample_bytree': [0.8, 0.9, 1],
        # 'scale_pos_weight': [0.5, 0.7, 1]
        gsearch = GridSearchCV(estimator=classifier, param_grid=param_test1, scoring='neg_log_loss', n_jobs=6, verbose=2)
        gsearch.fit(train_data[0], train_data[1])
        print gsearch.best_params_, gsearch.best_score_

    if not grid:
        begin = datetime.datetime.now()
        dir_path = constants.project_path + "/dataset/x_y/split_online/b2/"
        logloss = 0
        auc = 0
        online = False
        params = {"objective": 'binary:logistic',
                  "booster": "gbtree",
                  'reg_alpha': 0.01,
                  'eval_metric': 'logloss',
                  "eta": 0.04,
                  "max_depth": 8,
                  'silent': 0,
                  'min_child_weight': 3,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'gamma': 4,
                  'nthread': 20,
                  }
        if online:
            prob_test = np.zeros(338489)

        for i in xrange(4):
            train_x_file = dir_path + "train_x_onehot_" + str(i) + ".fm"
            if online:
                test_x_file = dir_path + "test_x_onehot.fm"
            else:
                test_x_file = dir_path + "test_x_onehot_" + str(i) + ".fm"

            train_x, train_y = load_svmlight_file(train_x_file)
            test_x, test_y = load_svmlight_file(test_x_file)

            train_set = xgb.DMatrix(train_x)
            print "train done"
            validation_set = xgb.DMatrix(test_x)
            print "test done"
            watchlist = [(train_set, 'train'), (validation_set, 'eval')]
            print "Training model..."
            xgb_model = xgb.train(params, train_set, rounds, watchlist, verbose_eval=True)
            train_pred = xgb_model.predict(xgb.DMatrix(test_x_file))

            if online:
                prob_test += train_pred
                print pd.DataFrame(prob_test).mean()
            else:
                prob_test = train_pred
                print pd.DataFrame(prob_test).mean()
            if not online:
                auc_local = metrics.roc_auc_score(test_y, prob_test)
                logloss_local = metrics.log_loss(test_y, prob_test)
                print str(i) + ": " + str(auc_local) + " " + str(logloss_local)
                logloss += logloss_local
                auc += auc_local

        if online:
            with open(constants.project_path+"/model/model_file/gbdt_re_200.out", 'w') as f:
                np.savetxt(f, train_pred, fmt="%s")
        else:
            end = datetime.datetime.now()
            rcd = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n'
            rcd += "lr k-fold:" + '\n'
            # rcd += "score: " + str(score) + '\n'
            rcd += "auc_test: " + str(float(auc)/4) + '\n'
            rcd += "logloss: " + str(logloss/4) + '\n'
            rcd += "time: " + str(end - begin) + '\n' + '\n'

            log_file = open(constants.result_path, "a")
            log_file.write(rcd)
            log_file.close()



        # pickle.dump(xgb_model, open(os.getcwd()+"/gbdt_model", "wb"))
        # print "dump model finished"
        # cus_train_file = constants.project_path + "/dataset/x_y/cus_train_x_no_id"
        # cus_test_file = constants.project_path + "/dataset/x_y/cus_test_x_no_id"
        #
        # test_ind = xgb_model.predict(xgb.DMatrix(cus_test_file), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
        # train_ind = xgb_model.predict(xgb.DMatrix(cus_train_file), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
        #
        # pickle.dump(train_ind, open(constants.project_path + "/dataset/feature/cus_train_2.idx", "wb"))
        # pickle.dump(test_ind, open(constants.project_path + "/dataset/feature/cus_test_2.idx", "wb"))


# 将叶子结点记录转化为onehot特征
def onehot_feature(from_path, to_path):
    print "load_data"
    onehot = []
    print "transform"
    gbdt_feature = pickle.load(open(from_path, "rb"))

    for line in gbdt_feature:
        temp_onehot = []
        i = 0
        for item in line:
            temp_onehot.append(int(item) + i*20 - 1)
            i += 1
        onehot.append(temp_onehot)
    pickle.dump(onehot, open(to_path, "wb"))


if __name__ == '__main__':
    train_model()
    # onehot_feature(constants.project_path + "/dataset/feature/cus_train_2.idx",
    #                constants.project_path + "/dataset/feature/cus_train_2.onehot")
    # onehot_feature(constants.project_path + "/dataset/feature/cus_test_2.idx",
    #                constants.project_path + "/dataset/feature/cus_test_2.onehot")


