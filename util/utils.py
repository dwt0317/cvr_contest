import io
import random

import numpy as np
import scipy as sp

import constants


def list2dict(mylist):
    myDict = {}
    i = 0
    for v in mylist:
        myDict[v] = i
        i += 1
    return myDict


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


# deprecated
def revise_submission(from_path, to_path):

    pred = []
    with open(from_path) as f:
        instance = 1
        for line in f:
            r_p = float(line)
            if r_p < 0.002:
                r_p = random.uniform(0, 0.0001)
            pred.append(r_p)
            # rcd = str(instance) + "," + line.strip()
            # to_file.write(unicode(rcd))
            # to_file.write(unicode('\n'))
            instance += 1
    test_y = np.loadtxt(open(constants.project_path + "/dataset/x_y/split_1/test_y"), dtype=int)
    print logloss(test_y, pred)


def build_submission(from_path, to_path):
    to_file = io.open(to_path, 'w', newline='\n')
    to_file.write(unicode("ffm pos_no-id k16 i25 0.0003"))
    to_file.write(unicode('\n'))

    with open(from_path) as f:
        instance = 1
        for line in f:
            rcd = str(instance) + "," + line.strip()
            to_file.write(unicode(rcd))
            to_file.write(unicode('\n'))
            instance += 1
    to_file.close()


if __name__ == "__main__":
    # revise_submission(constants.project_path+"/out/fm_pos_id_no_number.out",
    #                   constants.project_path + "/submission/submission.csv")
    build_submission(constants.project_path + "/out/ffm_pos_no-id.out",
                     constants.project_path + "/submission/submission.csv")
