import scipy as sp
import io
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


def build_submission(from_path, to_path):
    to_file = io.open(to_path, 'w', newline='\n')
    to_file.write(unicode("ffm"))
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
    build_submission(constants.project_path+"/model/model_file/ffm.out",
                     constants.project_path + "/submission/submission.csv")
