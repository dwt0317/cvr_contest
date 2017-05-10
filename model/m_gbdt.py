# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics   #Additional scklearn functions
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier
import cPickle as pickle
import constants
import xgboost as xgb
import os

train_x_file = constants.project_path + "/dataset/x_y/local_train_x"
train_y_file = constants.project_path + "/dataset/x_y/local_train_y"
test_x_file = constants.project_path + "/dataset/x_y/local_test_x"
test_y_file = constants.project_path + "/dataset/x_y/local_test_y"
validation_x_file = constants.project_path + "/dataset/x_y/local_valid_y"

# 训练GBDT模型，并保存叶子结点特征
def train_model():
    test_y = np.loadtxt(test_y_file, dtype=int)
    # svmlight格式自带label
    train_data = load_svmlight_file(train_x_file)

    rounds = 30
    classifier = XGBClassifier(learning_rate=0.1, n_estimators=rounds, max_depth=3,
                               min_child_weight=1, gamma=0, subsample=0.8,
                               objective='binary:logistic', nthread=2)

    grid = False
    if grid:
        param_test1 = {
            'max_depth': range(3, 5, 2),
            'min_child_weight': range(1, 6, 3)
        }
        gsearch = GridSearchCV(estimator=classifier, param_grid=param_test1, scoring='roc_auc', n_jobs=2)
        gsearch.fit(train_data[0].toarray(), train_data[1])
        print gsearch.best_params_, gsearch.best_score_

    if not grid:
        train_set = xgb.DMatrix(train_x_file)
        print "train done"
        validation_set = xgb.DMatrix(test_x_file)
        print "test done"
        watchlist = [(train_set, 'train'), (validation_set, 'eval')]
        params = {"objective": 'binary:logistic',
                  "booster": "gbtree",
                  'eval_metric': 'error',
                  "eta": 0.1,
                  "max_depth": 3,
                  'silent': 0,
                  'min_child_weight': 1,
                  'subsample': 0.8,
                  'gamma': 0,
                  'early_stopping_rounds': 10,
                  'nthread': 2,
                  'max_leaf_nodes': 20
                  }
        print "Training model..."
        xgb_model = xgb.train(params, train_set, rounds, watchlist, verbose_eval=True)
        train_pred = xgb_model.predict(xgb.DMatrix(test_x_file))
        print train_pred
        auc_test = metrics.roc_auc_score(test_y, train_pred)
        print auc_test
        pickle.dump(xgb_model, open(os.getcwd()+"/gbdt_model", "wb"))
        print "dump model finished"
        test_ind = xgb_model.predict(xgb.DMatrix(test_x_file), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
        train_ind = xgb_model.predict(xgb.DMatrix(train_x_file), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
        pickle.dump(train_ind, open(constants.project_path + "/dataset/feature/train.idx", "wb"))
        pickle.dump(test_ind, open(constants.project_path + "/dataset/feature/test.idx", "wb"))


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
    onehot_feature(constants.project_path + "/dataset/feature/train.idx",
                   constants.project_path + "/dataset/feature/train.onehot")
    onehot_feature(constants.project_path + "/dataset/feature/test.idx",
                   constants.project_path + "/dataset/feature/test.onehot")


