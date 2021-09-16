import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
    
    rfc = joblib.load('data/tlrf_resnext_rfc_100_5_128_1.joblib')
    
    classify_accuracy = []
    weighted_f1 = []
    final_aar = []
    final_MAE = []
    leaves = []
    
    y_train_prob = rfc.predict_proba(x_train)
    x_train_cat = np.concatenate([x_train, y_train_prob], axis=1)
    outc = rfc.predict_proba(x_test)
    y_pred_group = outc.argmax(axis=1)
    #print(f"Classifier accuracy: {accuracy_score(y_test_group, y_pred_group):.3f}")
    #print(f"Report: \n {classification_report(y_test_group, y_pred_group)}")
    
    classify_accuracy.append(accuracy_score(y_test_group, y_pred_group))
    r = classification_report(y_test_group, y_pred_group).split()
    weighted_f1.append(r[57])
    
    rfr = joblib.load('data/tlrf_resnext_rfr_100_5_128_1.joblib')
    x_test_cat = np.concatenate([x_test, outc], axis=1)
    out = np.clip(rfr.predict(x_test_cat).round(), 1, 81)
    AAR, MAE, *_, sigmas, maes = aar(y_test, out)
    ae = np.abs(y_test - out)
    final_aar.append(AAR)
    final_MAE.append(ae.mean())
    
    ####### refine rfc #########
    t0 = time.time()
    rrfc = RefinedRandomForest(rfc, C = 0.01, n_prunings = 1)
    print(f'Time it took for refinement of rfc: {time.time() - t0:.3f} ms.')
    leaves.append(sum(rrfc.n_leaves_))
    for i in range(100):
        t0 = time.time()
        rrfc.fit(x_train, y_train_group)
        # dump(rrfc, f'data/tlrf_rrfc_n_prunings_1.joblib')
        
        # rrfc = joblib.load('data/tlrf_rrfc_n_prunings_1.joblib')
        outc = rrfc.predict_proba(x_test)
        y_pred_group = outc.argmax(axis=1)
        #print(f"After refined Classifier accuracy: {accuracy_score(y_test_group, y_pred_group):.3f}")
        #print(f"Report: \n {classification_report(y_test_group, y_pred_group)}")
        classify_accuracy.append(accuracy_score(y_test_group, y_pred_group))
        r = classification_report(y_test_group, y_pred_group).split()
        weighted_f1.append(r[57])
        leaves.append(sum(rrfc.n_leaves_))
        
        x_test_cat = np.concatenate([x_test, outc], axis=1)
        out = np.clip(rfr.predict(x_test_cat).round(), 1, 81)
        AAR, MAE, *_, sigmas, maes = aar(y_test, out)
        ae = np.abs(y_test - out)
        final_aar.append(AAR)
        final_MAE.append(ae.mean())
        print(f'Time it took for pruning of rfc: {time.time() - t0:.3f} ms.')
    ####### refine rfr #########
    # rfr = joblib.load('data/tlrf_resnext_rfr_100_5_128_1.joblib')
    # t0 = time.time()
    # rrfr = RefinedRandomForest(rfr, C = 0.01, n_prunings = 1)
    # rrfr.fit(x_train, y_train)
    # dump(rrfr, f'data/tlrf_rrfr_n_prunings_1.joblib')
    # print(f'Time it took for refinement of rfr: {time.time() - t0:.3f} ms.')
    # out = rrfr.predict_proba(x_test).argmax(axis=1)
    # print(f'rrf MAE on validation: {np.abs(out - y_test).mean()}')
    # ARR, *_ = aar(y_test, out)
    # print(f'rrf AAR on validation: {ARR}')
    
    
    print('classify_accuracy: ', classify_accuracy, '\n',
    'weighted_f1', weighted_f1, '\n',
    'final_aar', final_aar, '\n',
    'final_MAE', final_MAE, '\n')
    nprunings = np.arange(len(classify_accuracy))
    fig, ax = plt.subplots(5, figsize=(14,11))
    ax[0].plot(nprunings, classify_accuracy, '-o')
    ax[0].set_title('Test set classify_accuracy')
    ax[0].grid()
    ax[1].plot(nprunings, weighted_f1, '-o')
    ax[1].set_title('Test set weighted_f1 score')
    ax[1].grid()
    ax[2].plot(nprunings, final_aar, '-o')
    ax[2].set_title('Test set AAR')
    ax[2].grid()
    ax[3].plot(nprunings, final_MAE, '-o')
    ax[3].set_title('Test set MAE')
    ax[3].grid()
    ax[4].plot(nprunings, np.array(leaves)/1000, '-o')
    ax[4].set_title('Number of leaves, thousands')
    ax[4].set_xlabel('Number of prunings')
    ax[4].grid()
    #fig.savefig('ClassificationRefinement.png')