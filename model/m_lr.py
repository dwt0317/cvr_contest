# -*- coding:utf-8 -*-
import datetime
import numpy as np
from util import constants
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
import time
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def lr():
    dir_path = constants.project_path + "/dataset/x_y/split_online/b7/"
    train_x_file = dir_path + "train_x_onehot_" + str(0) + ".fm"
    test_x_file = dir_path + "test_x_onehot.fm"

    begin = datetime.datetime.now()

    train_x, train_y = load_svmlight_file(train_x_file)
    test_x, test_y = load_svmlight_file(test_x_file)
    print train_x.shape, test_x.shape
    print "Loading data completed."
    print "Read time: " + str(datetime.datetime.now() - begin)
    classifier = LogisticRegression()

    grid = False
    if grid:
        param_grid = {'C': [0.5, 1, 1.5]}
        grid = GridSearchCV(estimator=classifier, scoring='neg_log_loss', param_grid=param_grid, n_jobs=3)
        grid.fit(train_x, train_y)
        print "Training completed."
        print grid.cv_results_
        print grid.best_estimator_

    if not grid:
        # traditional k-fold
        split = 8
        if split != 0:
            skf = StratifiedKFold(n_splits=split)
            prob_test = np.zeros(338489)
            for train_index, test_index in skf.split(train_x, train_y):
                # print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
                fold_x = train_x[train_index]
                fold_y = train_y[train_index]
                classifier.fit(fold_x, fold_y)
                # y_pred = classifier.predict(test_x)
                # score = metrics.accuracy_score(test_y, y_pred)
                # prob_train = classifier.predict_proba(training_x)[:, 1]
                # proba得到两行, 一行错的一行对的,对的是点击的概率，错的是不点的概率
                prob_test += classifier.predict_proba(test_x)[:, 1]
                print pd.DataFrame(prob_test).mean()
        else:
            classifier.fit(train_x, train_y)
            prob_test = classifier.predict_proba(test_x)[:, 1]

        prob_test /= split
        # prob_test = np.around(prob_test, 6)
        np.savetxt(constants.project_path + "/out/lr_8-fold_all-data.out", prob_test, fmt="%s")
        # auc_test = metrics.roc_auc_score(test_y, prob_test)
        # logloss = metrics.log_loss(test_y, prob_test)
        # end = datetime.datetime.now()
        # rcd = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n'
        # rcd += "lr onehot:" + '\n'
        # # rcd += "score: " + str(score) + '\n'
        # rcd += "auc_test: " + str(auc_test) + '\n'
        # rcd += "logloss: " + str(logloss) + '\n'
        # rcd += "time: " + str(end - begin) + '\n' + '\n'
        # print rcd
        # log_file = open(constants.result_path, "a")
        # log_file.write(rcd)
        # log_file.close()


def time_k_fold_lr():
    begin = datetime.datetime.now()
    dir_path = constants.project_path + "/dataset/x_y/split_5/b10/"
    logloss = 0
    auc = 0
    online = False
    if online:
        prob_test = np.zeros(338489)
    for i in range(1, 2):
        train_x_file = dir_path + "train_x_onehot_" + str(i) + ".gbdt"

        if online:
            test_x_file = dir_path + "test_x_onehot.gbdt"
        else:
            test_x_file = dir_path + "test_x_onehot_" + str(i) + ".gbdt"

        train_x, train_y = load_svmlight_file(train_x_file)
        test_x, test_y = load_svmlight_file(test_x_file)
        # print "Loading data completed."
        print "Read time: " + str(datetime.datetime.now() - begin)
        classifier = LogisticRegression(max_iter=30, penalty='l2')
        classifier.fit(train_x, train_y)
        if online:
            prob_test += classifier.predict_proba(test_x)[:, 1]
            print pd.DataFrame(prob_test).mean()
        else:
            prob_test = classifier.predict_proba(test_x)[:, 1]
            print pd.DataFrame(prob_test).mean()
        if not online:
            auc_local = metrics.roc_auc_score(test_y, prob_test)
            logloss_local = metrics.log_loss(test_y, prob_test)
            print str(i) + ": " + str(auc_local) + " " + str(logloss_local)
            logloss += logloss_local
            auc += auc_local

            np.savetxt(constants.project_path + "/dataset/custom/out.test", prob_test, fmt="%s")

    if online:
        prob_test /= 10
        np.savetxt(constants.project_path + "/out/lr_favorite.out", prob_test, fmt="%s")

    if not online:
        end = datetime.datetime.now()
        rcd = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n'
        rcd += "lr k-fold app_cvr 30:" + '\n'
        # rcd += "score: " + str(score) + '\n'
        rcd += "auc_test: " + str(float(auc)/4) + '\n'
        rcd += "logloss: " + str(logloss/4) + '\n'
        rcd += "time: " + str(end - begin) + '\n' + '\n'

        log_file = open(constants.result_path, "a")
        log_file.write(rcd)
        log_file.close()


def calibrate():
    dir_path = constants.project_path + "/dataset/x_y/split_5/b9/"
    test_x_file = test_x_file = dir_path + "test_x_onehot_0" + ".fm"
    test_x, test_y = load_svmlight_file(test_x_file)
    prob_test = np.loadtxt(constants.project_path + "/dataset/custom/out.test")
    test_df = pd.read_csv(test_x_file)
    conn_df = test_df[test_df.connectionType == 0].index

    # 0:tmp[i] = max(0, tmp[i]-0.002)



def sci2float():
    prob_test = np.loadtxt(constants.project_path + "/out/lr_k-fold_boot.out")
    prob_test = np.around(prob_test, 6)
    np.savetxt(constants.project_path + "/out/lr_k-fold_boots.out", prob_test, fmt="%s")

if __name__ == '__main__':
    # lr()
    time_k_fold_lr()
    # sci2float()