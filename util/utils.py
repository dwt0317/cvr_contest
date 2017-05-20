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


if __name__ == "__main__":
    pass
    # revise_submission(constants.project_path+"/out/fm_pos_id_no_number.out",
    #                   constants.project_path + "/submission/submission.csv")
    # build_submission(constants.project_path + "/out/gbdt_new_all-cvr_230.out",
    #                  constants.project_path + "/submission/submission.csv")
