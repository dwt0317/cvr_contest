import constants
import cPickle as pickle
train_path = constants.project_path + "/dataset/feature/train.onehot"
test_path = constants.project_path + "/dataset/feature/test.onehot"


def build_gbdt(dataset):
    path = ""
    if dataset == "train":
        path = train_path
    elif dataset == "test":
        path = test_path
    gbdt_feature = pickle.load(open(path, "rb"))
    print "Load GBDT features finished."
    return gbdt_feature


if __name__ == '__main__':
    build_gbdt("test")