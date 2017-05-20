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


def lr():
    train_x_file = constants.project_path + "/dataset/x_y/split_online/b1/train_x_onehot.fm"
    test_x_file = constants.project_path + "/dataset/x_y/split_online/b1/test_x_onehot.fm"

    begin = datetime.datetime.now()

    train_x, train_y = load_svmlight_file(train_x_file)
    test_x, test_y = load_svmlight_file(test_x_file)
    print train_x.shape, test_x.shape
    print "Loading data completed."
    print "Read time: " + str(datetime.datetime.now() - begin)
    classifier = LogisticRegression(solver='sag', random_state=8, max_iter=120)

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
        split = 5
        if split != 0:
            skf = StratifiedKFold(n_splits=split)
            prob_test = np.zeros(len(test_y))
            for train_index, test_index in skf.split(train_x, train_y):
                # print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
                fold_x = train_x[train_index]
                fold_y = train_y[train_index]
                classifier.fit(fold_x, fold_y)
                y_pred = classifier.predict(test_x)
                # score = metrics.accuracy_score(test_y, y_pred)
                # prob_train = classifier.predict_proba(training_x)[:, 1]
                # proba得到两行, 一行错的一行对的,对的是点击的概率，错的是不点的概率
                prob_test += classifier.predict_proba(test_x)[:, 1]
        else:
            classifier.fit(train_x, train_y)
            prob_test = classifier.predict_proba(test_x)[:, 1]

        prob_test /= split
        prob_test = round(prob_test, 5)
        np.savetxt(constants.project_path+"/out/lr.out", prob_test)
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
    dir_path = constants.project_path + "/dataset/x_y/split_5/"
    logloss = 0
    auc = 0
    for i in xrange(4):
        train_x_file = dir_path + "train_x_onehot_" + str(i) + ".fm"
        test_x_file = dir_path + "test_x_onehot_" + str(i) + ".fm"
        train_x, train_y = load_svmlight_file(train_x_file)
        test_x, test_y = load_svmlight_file(test_x_file)
        print train_x.shape, test_x.shape
        print "Loading data completed."
        print "Read time: " + str(datetime.datetime.now() - begin)
        classifier = LogisticRegression(solver='sag', random_state=8, max_iter=120)
        classifier.fit(train_x, train_y)
        prob_test = classifier.predict_proba(test_x)[:, 1]

        auc_local = metrics.roc_auc_score(test_y, prob_test)
        logloss_local = metrics.log_loss(test_y, prob_test)
        print auc_local, logloss_local
        logloss += logloss_local
        auc += auc_local

    # prob_test /= 4
    # prob_test = round(prob_test, 5)
    # np.savetxt(constants.project_path + "/out/lr_k-fold.out", prob_test)

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

if __name__ == '__main__':
    time_k_fold_lr()
