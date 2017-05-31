from features import build_data
from model import m_lr
from model import m_gbdt

build_data.build_x()
# m_lr.time_k_fold_lr()
m_gbdt.train_model()