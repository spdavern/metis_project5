import numpy as np
import pandas as pd
from scipy import io

def load_data():
    """Function that loads the training and test data as provided by Mayr, et.al.
    There are 12,060 training compounds and 647 test compounds.  Sets have 801
    chemical features and 833 sparsely populated structural features.
    -----
    inputs: none
    returns: x_tr, y_tr, x_te, y_te
        x_tr, x_te are numpy arrays
        t_tr, y_te are pandas dataframes
    """
    raw_data = './data/raw/tox21/'
    y_tr = pd.read_csv(raw_data+'tox21_labels_train.csv.gz', index_col=0, compression="gzip")
    y_te = pd.read_csv(raw_data+'tox21_labels_test.csv.gz', index_col=0, compression="gzip")
    # There are 801 "dense features" that represent chemical descriptors, such as:
    # molecular weight, solubility or surface area, etc.
    x_tr_dense = pd.read_csv(raw_data+'tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
    x_te_dense = pd.read_csv(raw_data+'tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
    # There are 272,776 "sparse features" that represent chemical substructures:
    # (ECFP10, DFS6, DFS8)
    x_tr_sparse = io.mmread(raw_data+'tox21_sparse_train.mtx.gz').tocsc()
    x_te_sparse = io.mmread(raw_data+'tox21_sparse_test.mtx.gz').tocsc()
    # This code filters out the very sparse features:
    sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
    x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
    x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])
    # The resulting x's have 1,644 features.
    return x_tr, y_tr, x_te, y_te

    