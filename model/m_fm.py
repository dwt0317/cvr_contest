import pywFM
import numpy as np
import constants
import datetime
from sklearn import metrics
import Utils
import cPickle as pickle


train_x_file = constants.project_path + "/dataset/x_y/local_train_x_gbdt"
train_y_file = constants.project_path + "/dataset/x_y/local_train_y"
test_x_file = constants.project_path + "/dataset/x_y/local_test_x"
test_y_file = constants.project_path + "/dataset/x_y/local_test_y"


# build fm interaction vectors
def build_fm_interaction():
    begin = datetime.datetime.now()
    # test_y = np.loadtxt(open(test_y_file), dtype=int)
    fm = pywFM.FM(task='classification', num_iter=100, learning_method='mcmc', temp_path=constants.project_path+"/model/tmp/")

    model = fm.run(None, None, None, None, train_path=train_x_file, test_path=test_x_file,
                   model_path=constants.project_path + "/model/model_file/fm_model",
                   out_path=constants.project_path + "/model/model_file/fm_gbdt_100.out"
                   )
    end = datetime.datetime.now()

    prob_test = model.predictions
    # auc_test = metrics.roc_auc_score(test_y, prob_test)
    # logloss = Utils.logloss(test_y, prob_test)
    # print auc_test, logloss


    # log_file = open(constants.result_path, "a")
    # log_file.write("fm: onehot + gbdt + 100 iters:" + '\n')
    # log_file.write("auc_test: " + str(auc_test) + '\n')
    # log_file.write("logloss: " + str(logloss) + '\n')
    # log_file.write("time: " + str(end - begin) + '\n' + '\n')
    # log_file.close()

    print model.pairwise_interactions.shape
    # with open(constants.dir_path + "sample\\features\\fm_features\\interactions.fm_sparse_id.np", 'w') as f:
    #     for line in model.pairwise_interactions:
    #         np.savetxt(f, line, fmt='%.4f')
    # with open(constants.dir_path + "sample\\features\\fm_features\\prediction.fm.np", 'w') as f:
    #     for line in model.predictions:
    #         np.savetxt(f, line, fmt='%.4f')


def predict():
    pass

if __name__ == '__main__':
    build_fm_interaction()
