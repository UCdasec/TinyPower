import numpy as np
from utils import loadData
attack_window = [1800, 2800]
method = 0
test_num = 4000


whole_pack = np.load("/home/uc_sec/Documents/lhp/PowerPruning/cnn/test_results/unmasked_xmega_cnn/ranking_dir_cross_dev/ranking_raw_data.npz")
# traces, plain_text, key = loadData.load_data_base(whole_pack, attack_window, method, train_num=test_num)
print(whole_pack['y'][4500:5000])

