import datetime

from util import constants
import numpy as np
import pywFM
from sklearn import metrics

from util import utils

data_dir_path = constants.project_path + "/dataset/x_y/split_5/b2/"

train_x_file = data_dir_path + "train_x_onehot.fm"
train_y_file = data_dir_path + "train_y"
test_x_file = data_dir_path + "test_x_onehot.fm"
test_y_file = data_dir_path + "test_y"


# build fm interaction vectors
def build_fm_interaction():
    begin = datetime.datetime.now()
    test_y = np.loadtxt(open(test_y_file), dtype=int)
    fm = pywFM.FM(task='classification', init_stdev=0.1, k2=8,
                  learning_method='mcmc', temp_path=constants.project_path + "/model/tmp/",
                  num_iter=80)

    model = fm.run(None, None, None, None, train_path=train_x_file, test_path=test_x_file,
                   model_path=constants.project_path + "/model/model_file/fm_model",
                   out_path=constants.project_path + "/out/fm_pos_id_no_number.out"
                   )
    end = datetime.datetime.now()

    prob_test = model.predictions
    auc_test = metrics.roc_auc_score(test_y, prob_test)
    logloss = utils.logloss(test_y, prob_test)
    print auc_test, logloss

    log_file = open(constants.result_path, "a")
    log_file.write("fm: onehot k8 80:" + '\n')
    log_file.write("auc_test: " + str(auc_test) + '\n')
    log_file.write("logloss: " + str(logloss) + '\n')
    log_file.write("time: " + str(end - begin) + '\n' + '\n')
    log_file.close()

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
