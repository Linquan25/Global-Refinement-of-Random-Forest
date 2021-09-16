import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import unique
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

from RefinedRandomForest import RefinedRandomForest

plt.style.use('ggplot')

def ind2onehot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


def mae(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] == 0:
        return 0
    return np.abs(y_true - y_pred).mean()

def aar(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    true_age_groups = np.clip(y_pred // 10, 0, 7)
    mae_score = mae(y_true, y_pred)
    
    # MAE per age group
    sigmas = []
    maes = []
    for i in range(8):
        idx = true_age_groups == i
        mae_age_group = mae(y_true[idx], y_pred[idx])
        maes.append(mae_age_group)
        sigmas.append((mae_age_group - mae_score) ** 2)

    sigma = np.sqrt(np.array(sigmas).mean())
    
    aar_score = max(0, 7 - mae_score) + max(0, 3 - sigma)
    
    return aar_score, mae_score, sigma, sigmas, maes


if __name__ == '__main__':

    df = pd.read_csv('../../training_caip_contest.csv', 
                    header=None)
    

    x = np.load('../../../../mvl5/Images2/DeepFake/GTA/features_pred_resnext_aar.npy')
    # x = np.load('data/features_pred_resnext_mse.npy')
    y = np.array(df.iloc[:, 1])
    
    train_index = np.loadtxt('../train_index.txt', delimiter=',').astype('int')
    test_index = np.loadtxt('../test_index.txt', delimiter=',').astype('int')
    
    x_train = x[train_index]
    y_train = y[train_index]
    y_train_group = np.clip(y_train // 10, 0, 7)
    
    x_test = x[test_index]
    y_test = y[test_index]
    y_test_group = np.clip(y_test // 10, 0, 7)

    n_estimators = 100
    min_samples_leaf = 5
    max_features = 128
    random_state = 1
    
    final_aar = []
    final_MAE = []
    leaves = []
    
    ####### refine rfc #########
    # rrfc = RefinedRandomForest(rfc, C = 0.01, n_prunings = 1)
    # for i in range(35):
    #     rrfc.fit(x_train, y_train_group)
    # dump(rrfc, f'data/tlrf_rrfc_n_prunings_35.joblib')
    rrfc = joblib.load('data/tlrf_rrfc_n_prunings_35.joblib')
    y_train_prob = rrfc.predict_proba(x_train)
    x_train_cat = np.concatenate([x_train, y_train_prob], axis=1)
    outc = rrfc.predict_proba(x_test)
    x_test_cat = np.concatenate([x_test, outc], axis=1)
    
    rfr = joblib.load('data/tlrf_resnext_rfr_100_5_128_1.joblib')
    out = np.clip(rfr.predict(x_test_cat).round(), 1, 81)
    AAR, MAE, *_, sigmas, maes = aar(y_test, out)
    ae = np.abs(y_test - out)
    final_aar.append(AAR)
    final_MAE.append(ae.mean())
    print(f'rf MAE on validation: {ae.mean()}')
    print(f'rf AAR on validation: {AAR}')
    
    ####### refine rfr #########
    t0 = time.time()
    rrfr = RefinedRandomForest(rfr, C = 0.01, n_prunings = 1)
    leaves.append(sum(rrfr.n_leaves_))
    print(f'Time it took for refinement of rfr: {time.time() - t0:.3f} s.')
    for i in range(8):
        t0 = time.time()
        print('Loop:  ', i)
        rrfr.fit(x_train_cat, y_train)
        #dump(rrfr, f'data/tlrf_rrfr_n_prunings_1.joblib')
        out = np.clip(rrfr.predict_proba(x_test_cat).round(), 1, 81)
        AAR, MAE, *_, sigmas, maes = aar(y_test, out)
        ae = np.abs(y_test - out)
        final_aar.append(AAR)
        final_MAE.append(ae.mean())
        leaves.append(sum(rrfr.n_leaves_))
        print(f'Time it took for pruning of rfr: {time.time() - t0:.3f} s.')
    
    #dump(rrfr, f'data/tlrf_rrfr_n_prunings_8.joblib')
    #######  visualization  #########
    nprunings = np.arange(len(final_aar))
    fig, ax = plt.subplots(3, figsize=(14,11))
    ax[0].plot(nprunings, final_aar, '-o')
    ax[0].set_title('Test set AAR')
    ax[0].grid()
    ax[1].plot(nprunings, final_MAE, '-o')
    ax[1].set_title('Test set MAE')
    ax[1].grid()
    ax[2].plot(nprunings, np.array(leaves)/1000, '-o')
    ax[2].set_title('Number of leaves, thousands')
    ax[2].set_xlabel('Number of prunings')
    ax[2].grid()
    #fig.savefig('RegressionRefinement40.png')
    
    print('final_aar:', final_aar)
    print('final_MAE:', final_MAE)
    print('final_MAE:', final_MAE)