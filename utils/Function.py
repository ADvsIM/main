import os
import torch
import numpy as np 
from datetime import datetime
from sklearn import metrics
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

def save_model(model, model_path , *args):

    combined_name = "_".join(map(str, args))
    print(combined_name)
    model_name = f'MLP_{combined_name}.pt'

    # torch.save(model,  model_path + model_name)

    # print(f'MLP_{combined_name} 모델이 무사히 저장되었습니다.')

def evaluation_metric(y_true,  y_pred_proba):
    warnings.filterwarnings("ignore")

    y_true = y_true.cpu().detach().numpy()
    y_pred_proba = y_pred_proba.cpu().detach().numpy()

    # acc = metrics.accuracy_score(y_true,y_pred)*100
    auc = metrics.roc_auc_score(y_true, y_pred_proba)
    prauc = average_precision_score(y_true, y_pred_proba)
    # f1_score = metrics.precision_recall_fscore_support(y_true,y_pred,zero_division=1)[2].mean()
    
    warnings.resetwarnings()
    return { 'auc':auc, 'prauc':prauc}

def result_save(dataset, AUC_list, PRAUC_list, save_path, save_name_list):
    # Check the anomaly ratio of dataset.
    n_data = dataset.train_data.shape[0] 
    n_abnormal = sum(dataset.train_y == 1 )
    n_normal = n_data - n_abnormal
    IR = (n_normal/n_abnormal).item()

    # Make a result dataframe
    colnames = ['dataset', 'loss', 'Oversampling', 'n_data', 'n_major', 'n_minor', 'IR', 'AUC_mean', 'PRAUC_mean', 'AUC_mean(sd)', 'PRAUC_mean(sd)']

    AUC_mean = "{:.1f}({:.1f})".format(100*(np.array(AUC_list).mean()), 100*(np.array(AUC_list).std()))
    print(AUC_mean)
    PRAUC_mean = "{:.1f}({:.1f})".format(100*(np.array(PRAUC_list).mean()), 100*(np.array(PRAUC_list).std()))

    new_line = {'dataset': save_name_list['data_name'], 'loss': save_name_list['loss'], 'Oversampling':save_name_list['sampling'], 'n_data':n_data, 'n_major':n_normal.item(), 'n_minor':save_name_list['number'], 'IR':IR, 'AUC_mean':100*(np.array(AUC_list).mean()), 'PRAUC_mean': 100*(np.array(PRAUC_list).mean()), 'AUC_mean(sd)':AUC_mean, 'PRAUC_mean(sd)':PRAUC_mean} 
    
    for i, seed in enumerate(save_name_list['seed']):
        new_line['AUC({})'.format(seed)] = AUC_list[i]
        new_line['PRAUC({})'.format(seed)] = PRAUC_list[i]

    for seed in save_name_list['seed']:
        colnames.append('AUC({})'.format(seed))
        colnames.append('PRAUC({})'.format(seed))
    
    if save_name_list['loss'] == 'focal':
        colnames.append('alpha')
        colnames.append('gamma')

        new_line['alpha'] = save_name_list['alpha']
        new_line['gamma'] = save_name_list['gamma']

    elif save_name_list['loss'] == 'class-balanced':
        colnames.append('beta')
        new_line['beta'] = save_name_list['beta']

        if save_name_list['cbloss'] == 'focal':
            colnames.append('alpha')
            colnames.append('gamma')

            new_line['alpha'] = save_name_list['alpha']
            new_line['gamma'] = save_name_list['gamma']

    df_results = pd.DataFrame(columns = colnames)
    df_results = df_results.append(new_line, ignore_index = True)

    now = datetime.now().strftime("%Y-%m-%d")
    save_path += f'{now}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_path + '{}_{}_{}_minor_{}.csv'.format(save_name_list['data_name'], save_name_list['loss'], save_name_list['sampling'], save_name_list['number'] )

    if os.path.isfile(save_name ):
        df_results.to_csv(save_name, mode='a', index=False, header=False)
    else:
        df_results.to_csv(save_name, mode='w', index=False, header=True)
    
    print(save_name , '으로 저장되었습니다.')
