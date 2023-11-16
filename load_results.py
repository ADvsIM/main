import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import os 
from functools import reduce

#### 6/20
### Supervised(valid)
os.chdir('/home/jjlee/results_server2/AD3')
methods_list_s = ['LR', 'SVM', 'RF', 'LGB', 'XGB', 'MLP']
results_list_s = ['Supervised_{}(10)(valid)_results.csv'.format(method) for method in methods_list_s]

methods_list_u = ["KNN", "PCA", "IForest", "LOF", "OCSVM"]
results_list_u = ['Unsupervised_{}(10)(valid)_results.csv'.format(method) for method in methods_list_u]

methods_list_s = ["IC_"+name for name in methods_list_s]
methods_list_u = ["AD_"+name for name in methods_list_u]

### IC vs UAD 
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_1_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()

### PRAUC 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_1_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

#### 0.001, 0.002, 0.005, 0.01
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        print(x)
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_2_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
#### 0.001, 0.002, 0.005, 0.01
### PRAUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )               
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_2_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    


# col_list = ['dataset', 'n_major', 'n_minor', '1/IR'] 
# for methods in methods_list_s:
#     col_list.append(methods)
# for methods in methods_list_u:
#     col_list.append(methods)
# pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_major', 'n_minor', '1/IR', 'AUC_mean(sd)']] for results_s in results_list_s]
# pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_major', 'n_minor', '1/IR', 'AUC_mean(sd)']] for results_u in results_list_u]
# pd_list = pd_list_s + pd_list_u

# results = reduce(lambda  left,right: pd.merge(left,right,on=["dataset", "n_major","n_minor", "1/IR"], how='left'), pd_list)
# results.columns = col_list
# results

#### latex table
### AUC

dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)
pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_s in results_list_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

#### latex table
### PRAUC
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)
pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_s in results_list_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

### 6/10
#### OS
methods_list_s = ['LR', 'SVM', 'RF', 'LGB', 'XGB', 'MLP']
os_methods_list_s = ['ROS', 'SMOTE', 'ADASYN', "BorderlineSMOTE"]
results_list_os_s = [['Supervised(OS)({})_{}(10)(valid)_results.csv'.format(os, method) for os in os_methods_list_s ] for method in methods_list_s]

tmp_os_list = results_list_os_s[0]
tmp_dfs = [pd.read_csv(results_s) for results_s in tmp_os_list]

df_results = pd.DataFrame(columns = tmp_dfs[0].columns)
for index in range(len(tmp_dfs)):
    j = np.nanargmax(np.array([df['AUC_mean'].iloc[index] for df in tmp_dfs]))  # 각 DataFrame에서 해당 행의 최대값 비교
    new_line = tmp_dfs[j].iloc[index].to_dict()
    df_results = df_results.append(new_line, ignore_index = True)
df_results

best_results_list_os_s = []
for tmp_os_list in results_list_os_s:
    tmp_dfs = [pd.read_csv(results_s) for results_s in tmp_os_list]
    best_results = pd.DataFrame(columns = tmp_dfs[0].columns)
    for index in range(len(tmp_dfs[0])):
        try:
            j = np.nanargmax(np.array([df['AUC_mean'].iloc[index] for df in tmp_dfs]))  # 각 DataFrame에서 해당 행의 최대값 비교
        except Exception:
            j = 0
        new_line = tmp_dfs[j].iloc[index].to_dict()
        best_results = best_results.append(new_line, ignore_index = True)
    
    best_results_list_os_s.append(best_results)

### IC vs UAD 
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC(OS)_AD_3_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()

### PRAUC 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_3_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

#### 0.001, 0.002, 0.005, 0.01
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_4_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
#### 0.001, 0.002, 0.005, 0.01
### PRAUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )               
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0620/IC_AD_4_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

### 일단은 Oversampling 방법없이 작성
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods+"(OS)")
for methods in methods_list_u:
    col_list.append(methods)

pd_list_s = [ results_os_s[['dataset', '1/IR', 'AUC_mean(sd)']] for results_os_s in best_results_list_os_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "1/IR"], how='left'), pd_list)
n_data = results['n_data'].values
results = results.loc[:,[True, True, True, True, True, True, True, True, False, True, False, True, False, True, False, True, False, True]]
results.columns = col_list

results['n_data'] = n_data
new_col_list = ['dataset', 'n_data','1/IR']
for methods in methods_list_s:
    new_col_list.append(methods+"(OS)")
for methods in methods_list_u:
    new_col_list.append(methods)
results = results[new_col_list]

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[new_col_list]
results
print(results.to_latex(index=False)) 

#### latex table
### PRAUC
### 일단은 Oversampling 방법없이 작성
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)

pd_list_s = [ results_os_s[['dataset', '1/IR', 'PRAUC_mean(sd)']] for results_os_s in best_results_list_os_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "1/IR"], how='left'), pd_list)
n_data = results['n_data'].values
results = results.loc[:,[True, True, True, True, True, True, True, True, False, True, False, True, False, True, False, True, False, True]]
results.columns = col_list

results['n_data'] = n_data
new_col_list = ['dataset', 'n_data','1/IR']
for methods in methods_list_s:
    new_col_list.append(methods)
for methods in methods_list_u:
    new_col_list.append(methods)
results = results[new_col_list]

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[new_col_list]
print(results.to_latex(index=False)) 

### Loss comparison

#### float
### AUC
os.chdir('/home/jjlee/results_server2/AD3')
methods_list_loss = ["Binary", "Binary_contrastive", "DeepSAD", "DeepSVDD_logistic"]
results_list_loss = ['{}(10)(test)_float_results.csv'.format(method) for method in methods_list_loss]
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_loss )
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_5_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
### PRAUC 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_loss )
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_5_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

### latex
###AUC
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', '1/IR'] 
for methods in methods_list_loss:
    col_list.append(methods)
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

###PRAUC
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

#### int
os.chdir('/home/jjlee/results_server2/AD3')
methods_list_loss = ["Binary", "Binary_contrastive", "DeepSAD", "DeepSVDD_logistic"]
results_list_loss = ['{}(10)(test)_int_results.csv'.format(method) for method in methods_list_loss]
for dataset in ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['n_minor']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        if methods_loss == "Binary":
            plt.plot(x, y, 'v-', label = methods_loss )
        else: 
            plt.plot(x, y, 'o-', label = methods_loss )

    plt.xticks([1, 2, 5, 10])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_6_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
### PRAUC 
for dataset in ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['n_minor']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        if methods_loss == "Binary":
            plt.plot(x, y, 'v-', label = methods_loss )
        else: 
            plt.plot(x, y, 'o-', label = methods_loss )
    plt.xticks([1, 2, 5, 10])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_6_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

### latex
###AUC
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', 'n_minor', '1/IR'] 
for methods in methods_list_loss:
    col_list.append(methods)
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', 'n_minor', '1/IR', 'AUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data", 'n_minor',"1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

###PRAUC
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', 'n_minor', '1/IR', 'PRAUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data", 'n_minor',"1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 


#### 6/8
### Supervised
os.chdir('/home/jjlee/results_server2/AD3')
methods_list_s = ['LR', 'SVM', 'RF', 'LGB', 'XGB', 'MLP']
results_list_s = ['Supervised_{}(10)(test)_results.csv'.format(method) for method in methods_list_s]

methods_list_u = ["KNN", "PCA", "IForest", "LOF", "OCSVM"]
results_list_u = ['Unsupervised_{}(10)(test)_results.csv'.format(method) for method in methods_list_u]

methods_list_s = ["IC_"+name for name in methods_list_s]
methods_list_u = ["AD_"+name for name in methods_list_u]

### IC vs UAD 
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0608/IC_AD_1_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()

### PRAUC 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0608/IC_AD_1_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

#### 0.001, 0.002, 0.005, 0.01
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        print(x)
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0608/IC_AD_2_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
#### 0.001, 0.002, 0.005, 0.01
### PRAUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )               
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0608/IC_AD_2_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    


# col_list = ['dataset', 'n_major', 'n_minor', '1/IR'] 
# for methods in methods_list_s:
#     col_list.append(methods)
# for methods in methods_list_u:
#     col_list.append(methods)
# pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_major', 'n_minor', '1/IR', 'AUC_mean(sd)']] for results_s in results_list_s]
# pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_major', 'n_minor', '1/IR', 'AUC_mean(sd)']] for results_u in results_list_u]
# pd_list = pd_list_s + pd_list_u

# results = reduce(lambda  left,right: pd.merge(left,right,on=["dataset", "n_major","n_minor", "1/IR"], how='left'), pd_list)
# results.columns = col_list
# results

#### latex table
### AUC

dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)
pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_s in results_list_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

#### latex table
### PRAUC
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)
pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_s in results_list_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

### 6/10
#### OS
os.chdir('/home/jjlee/results_server2/AD3')
os.listdir()
methods_list_s = ['LR', 'SVM', 'RF', 'LGB', 'XGB', 'MLP']
os_methods_list_s = ['ROS', 'SMOTE', 'ADASYN', "BorderlineSMOTE"]
results_list_os_s = [['Supervised(OS)({})_{}(10)(test)_results.csv'.format(os, method) for os in os_methods_list_s ] for method in methods_list_s]

tmp_os_list = results_list_os_s[0]
tmp_dfs = [pd.read_csv(results_s) for results_s in tmp_os_list]

df_results = pd.DataFrame(columns = tmp_dfs[0].columns)
for index in range(len(tmp_dfs)):
    j = np.nanargmax(np.array([df['AUC_mean'].iloc[index] for df in tmp_dfs]))  # 각 DataFrame에서 해당 행의 최대값 비교
    new_line = tmp_dfs[j].iloc[index].to_dict()
    df_results = df_results.append(new_line, ignore_index = True)
df_results

best_results_list_os_s = []
for tmp_os_list in results_list_os_s:
    tmp_dfs = [pd.read_csv(results_s) for results_s in tmp_os_list]
    best_results = pd.DataFrame(columns = tmp_dfs[0].columns)
    for index in range(len(tmp_dfs[0])):
        try:
            j = np.nanargmax(np.array([df['AUC_mean'].iloc[index] for df in tmp_dfs]))  # 각 DataFrame에서 해당 행의 최대값 비교
        except Exception:
            j = 0
        new_line = tmp_dfs[j].iloc[index].to_dict()
        best_results = best_results.append(new_line, ignore_index = True)
    
    best_results_list_os_s.append(best_results)

### IC vs UAD 
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0610/IC(OS)_AD_3_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()

### PRAUC 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0610/IC_AD_3_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

#### 0.001, 0.002, 0.005, 0.01
### AUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0610/IC_AD_4_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
#### 0.001, 0.002, 0.005, 0.01
### PRAUC
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results in zip(methods_list_s, best_results_list_os_s):
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )               
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0610/IC_AD_4_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

### 일단은 Oversampling 방법없이 작성
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods+"(OS)")
for methods in methods_list_u:
    col_list.append(methods)

pd_list_s = [ results_os_s[['dataset', '1/IR', 'AUC_mean(sd)']] for results_os_s in best_results_list_os_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "1/IR"], how='left'), pd_list)
n_data = results['n_data'].values
results = results.loc[:,[True, True, True, True, True, True, True, True, False, True, False, True, False, True, False, True, False, True]]
results.columns = col_list

results['n_data'] = n_data
new_col_list = ['dataset', 'n_data','1/IR']
for methods in methods_list_s:
    new_col_list.append(methods+"(OS)")
for methods in methods_list_u:
    new_col_list.append(methods)
results = results[new_col_list]

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[new_col_list]
results
print(results.to_latex(index=False)) 

#### latex table
### PRAUC
### 일단은 Oversampling 방법없이 작성
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', '1/IR'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)

pd_list_s = [ results_os_s[['dataset', '1/IR', 'PRAUC_mean(sd)']] for results_os_s in best_results_list_os_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "1/IR"], how='left'), pd_list)
n_data = results['n_data'].values
results = results.loc[:,[True, True, True, True, True, True, True, True, False, True, False, True, False, True, False, True, False, True]]
results.columns = col_list

results['n_data'] = n_data
new_col_list = ['dataset', 'n_data','1/IR']
for methods in methods_list_s:
    new_col_list.append(methods)
for methods in methods_list_u:
    new_col_list.append(methods)
results = results[new_col_list]

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[new_col_list]
print(results.to_latex(index=False)) 

#### 6/14
### Loss comparison

#### float
### AUC
os.chdir('/home/jjlee/results_server2/AD3')
methods_list_loss = ["Binary", "Binary_contrastive", "DeepSAD", "DeepSVDD_logistic"]
results_list_loss = ['{}(10)(test)_float_results.csv'.format(method) for method in methods_list_loss]
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_loss )
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_5_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
### PRAUC 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['1/IR']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_loss )
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_5_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

### latex
###AUC
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', '1/IR'] 
for methods in methods_list_loss:
    col_list.append(methods)
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', '1/IR', 'AUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

###PRAUC
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', '1/IR', 'PRAUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data","1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

#### int
os.chdir('/home/jjlee/results_server2/AD3')
methods_list_loss = ["Binary", "Binary_contrastive", "DeepSAD", "DeepSVDD_logistic"]
results_list_loss = ['{}(10)(test)_int_results.csv'.format(method) for method in methods_list_loss]
for dataset in ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['n_minor']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        if methods_loss == "Binary":
            plt.plot(x, y, 'v-', label = methods_loss )
        else: 
            plt.plot(x, y, 'o-', label = methods_loss )

    plt.xticks([1, 2, 5, 10])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_6_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
### PRAUC 
for dataset in ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']:    
    plt.plot(figsize=(10, 8))
    for methods_loss, results_loss in zip(methods_list_loss, results_list_loss):
        results = pd.read_csv(results_loss)
        x = results[results.dataset == dataset]['n_minor']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys)/100 for ys in auc])
        if methods_loss == "Binary":
            plt.plot(x, y, 'v-', label = methods_loss )
        else: 
            plt.plot(x, y, 'o-', label = methods_loss )
    plt.xticks([1, 2, 5, 10])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures_0614/IC_AD_6_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

### latex
###AUC
dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Hepatitis', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'wine', 'Waveform', 'WBC', 'WDBC', 'WPBC', 'Wilt','yeast']

col_list = ['dataset', 'n_data', 'n_minor', '1/IR'] 
for methods in methods_list_loss:
    col_list.append(methods)
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', 'n_minor', '1/IR', 'AUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data", 'n_minor',"1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

###PRAUC
pd_list = [ pd.read_csv(results_loss)[['dataset', 'n_data', 'n_minor', '1/IR', 'PRAUC_mean(sd)']] for results_loss in results_list_loss]
results = reduce(lambda  left, right: pd.merge(left,right,on=["dataset", "n_data", 'n_minor',"1/IR"], how='left'), pd_list)
results.columns = col_list

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', '1/IR'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 



### Supervised
os.chdir('/home/jjlee/results_server2/AD3')

results_list_s = [ name for name in os.listdir() if ('Supervised' in name) & ('OS' not in name)]
methods_list_s = [ name.split("_")[1].split("(")[0] for name in results_list_s]

results_list_u = [ name for name in os.listdir() if 'Unsupervised' in name]
methods_list_u = [ name.split("_")[1].split("(")[0] for name in results_list_u]

methods_list_s = ["IC_"+name for name in methods_list_s]
methods_list_s[5] = "IC_MLP"
methods_list_u = ["AD_"+name for name in methods_list_u]

### IC vs UAD 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC_AD_1_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
### IC vs UAD
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC_AD_1_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

##################################

os.chdir('/home/jjlee/results_server2/AD3')

results_list_s = [ name for name in os.listdir() if 'Supervised' in name]
methods_list_s = [ name.split("_")[1].split("(")[0] for name in results_list_s]

results_list_u = [ name for name in os.listdir() if 'Unsupervised' in name]
methods_list_u = [ name.split("_")[1].split("(")[0] for name in results_list_u]

methods_list_s = ["IC_"+name for name in methods_list_s]
methods_list_s[5] = "IC_MLP"
methods_list_u = ["AD_"+name for name in methods_list_u]

### IC vs UAD 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['p']
        print(x)
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC_AD_2_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
### IC vs UAD
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )               
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC_AD_2_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

#######
from functools import reduce

col_list = ['dataset', 'n_data', 'p'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)
pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_data', 'p', 'AUC_mean(sd)']] for results_s in results_list_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', 'p', 'AUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u

results = reduce(lambda  left,right: pd.merge(left,right,on=["dataset", "n_data", "n_minor", "p"], how='left'), pd_list)
results.columns = col_list
results

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', 'p'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 

### Supervised_OS
os.chdir('/home/jjlee/results_server2/AD3')
os.listdir()

results_list_s_os = [ name for name in os.listdir() if 'Supervised(OS)' in name]
methods_list_s_os = [ name.split("_")[1].split("(")[0] + "(OS)" for name in results_list_s_os]

results_list_u = [ name for name in os.listdir() if 'Unsupervised' in name]
methods_list_u = [ name.split("_")[1].split("(")[0] for name in results_list_u]

methods_list_s_os = ["IC_"+name for name in methods_list_s_os]
methods_list_s[5] = "IC_MLP(OS)"
methods_list_u = ["AD_"+name for name in methods_list_u]


results
### IC(OS) vs UAD 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s_os, results_s_os in zip(methods_list_s_os, results_list_s_os):
        results = pd.read_csv(results_s_os)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s_os )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC(OS)_AD_1_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
### IC vs UAD
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s_os, results_s_os in zip(methods_list_s_os, results_list_s_os):
        results = pd.read_csv(results_s_os)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'v-', label = methods_s_os )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC(OS)_AD_1_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

##################################

os.chdir('/home/jjlee/results_server2/AD3')

results_list_s = [ name for name in os.listdir() if 'Supervised' in name]
methods_list_s = [ name.split("_")[1].split("(")[0] for name in results_list_s]

results_list_u = [ name for name in os.listdir() if 'Unsupervised' in name]
methods_list_u = [ name.split("_")[1].split("(")[0] for name in results_list_u]

methods_list_s = ["IC_"+name for name in methods_list_s]
methods_list_s[5] = "IC_MLP"
methods_list_u = ["AD_"+name for name in methods_list_u]

### IC vs UAD 
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['p']
        print(x)
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )        
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC_AD_2_{}_AUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()
    
### IC vs UAD
for dataset in ['SpamBase', 'satellite', 'Pima', 'landsat', 'fault', 'yeast', 'Cardiotocography', 'breastw']:    
    plt.plot(figsize=(10, 8))
    for methods_s, results_s in zip(methods_list_s, results_list_s):
        results = pd.read_csv(results_s)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'v-', label = methods_s )
    for methods_u, results_u in zip(methods_list_u, results_list_u):
        results = pd.read_csv(results_u)
        x = results[results.dataset == dataset]['p']
        auc = results[results.dataset == dataset]['PRAUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x[:4], y[:4], 'o-', label = methods_u )               
    plt.xticks([0.001, 0.002, 0.005, 0.01])
    plt.yticks()
    plt.legend(loc='lower left', bbox_to_anchor=(1.0,0.0))
    plt.title(dataset)
    plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures1/IC_AD_2_{}_PRAUC.png'.format(dataset), bbox_inches = 'tight')
    plt.clf()    

#######
from functools import reduce
pd.read_csv(results_list_s[0]).columns

col_list = ['dataset', 'n_data', 'p'] 
for methods in methods_list_s:
    col_list.append(methods)
for methods in methods_list_u:
    col_list.append(methods)
pd_list_s = [ pd.read_csv(results_s)[['dataset', 'n_data', 'p', 'AUC_mean(sd)']] for results_s in results_list_s]
pd_list_u = [ pd.read_csv(results_u)[['dataset', 'n_data', 'p', 'AUC_mean(sd)']] for results_u in results_list_u]
pd_list = pd_list_s + pd_list_u

results = reduce(lambda  left,right: pd.merge(left,right,on=["dataset", "n_data", "n_minor", "p"], how='left'), pd_list)
results.columns = col_list
results

results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', 'p'])
results = results[col_list]
results.loc[results['n_data'] == "error"]['n_data'] = "-"
print(results.to_latex(index=False)) 




results_0 = pd.read_csv(results_list[0])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]
results_1 = pd.read_csv(results_list[1])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]
results_2 = pd.read_csv(results_list[2])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]
results_3 = pd.read_csv(results_list[3])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]

results = pd.merge(results_0, results_1, on=["dataset", "n", "o", "p"], how = "left")
results = pd.merge(results, results_2, on=["dataset", "n", "o", "p"], how = "left")
results = pd.merge(results, results_3, on=["dataset", "n", "o", "p"], how = "left")
results

results.columns = col_list
results = results[col_list]
results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', 'o'])
results = results[col_list]
print(results.to_latex(index=False)) 

####

results_list = ['/home/jjlee/results_server2/AD3/Binary(5)_v_results.csv',
 '/home/jjlee/results_server2/AD3/DeepSAD(5)_results.csv',
 '/home/jjlee/results_server2/AD3/DeepSVDD_contrastive(5)_results.csv',
 '/home/jjlee/results_server2/AD3/DeepSVDD_logistic(5)_results.csv']

method_list = ['Binary', 'DeepSAD', 'C_Logistic', 'C_contrastive']
color_mark_list = ['rv-', 'bs-', 'gD-', 'm^-']

dataset_names = ['annthyroid', 'arrhythmia', 'breastw', 'cardio', 'Cardiotocography', 'fault', 'glass', 'Ionosphere', 'landsat',
                 'Lymphography', 'musk', 'PageBlocks', 'Pima','satellite', 'satimage-2',  'SpamBase', 'speech', 'Stamps', 'thyroid',
                 'vertebral', 'vowels', 'WBC', 'WDBC', 'WPBC', 'Waveform', 'Wilt','yeast']

for dataset in dataset_names:    
    plt.plot(figsize=(10, 8))
    for i, method, colormark in zip(range(4), method_list, color_mark_list):
        results = pd.read_csv(results_list[i])
        x = results[results.dataset == dataset]['o']
        auc = results[results.dataset == dataset]['AUC_mean']
        y = np.array([float(ys.split('(')[0])/100 for ys in auc])
        plt.plot(x, y, colormark, label = method )
        plt.xticks([1,2,5,10])
        
        plt.yticks()
        plt.legend()
        plt.title(dataset)
        plt.tight_layout()
#     plt.show()
    plt.savefig('/home/jjlee/results_server2/AD3/figures/{}.png'.format(dataset),  bbox_inches = 'tight')
    plt.clf()
    

# fig, ax = plt.subplots(3,2, figsize = (12,6))
# ax[0,0] = 

# for i, dataset in enumerate(['annthyroid', 'cardio', 'glass', 'satimage-2', 'thyroid', 'WPBC']):

#     for i, method, colormark in zip([0,0],[], method_list, color_mark_list):
#         results = pd.read_csv(results_list[i])
#         x = results[results.dataset == dataset]['o']
#         auc = results[results.dataset == dataset]['AUC_mean']
#         y = np.array([float(ys.split('(')[0])/100 for ys in auc])
#         plt.plot(x, y, colormark, label = method )
#         plt.xticks([1,2,5,10])
#         plt.yticks()
#         plt.legend()
#         plt.title(dataset)
#         plt.tight_layout()
# #     plt.show()
#     plt.savefig('/home/jjlee/results_server2/AD3/figures/{}.png'.format(dataset), bbox_inches = 'tight')
#     plt.clf()


col_list = ['dataset', 'n', 'o', 'p'] 
for methods in method_list:
    col_list.append(methods)

results_0 = pd.read_csv(results_list[0])[['dataset', 'n', 'o', 'p', 'AUC_mean']]
results_1 = pd.read_csv(results_list[1])[['dataset', 'n', 'o', 'p', 'AUC_mean']]
results_2 = pd.read_csv(results_list[2])[['dataset', 'n', 'o', 'p', 'AUC_mean']]
results_3 = pd.read_csv(results_list[3])[['dataset', 'n', 'o', 'p', 'AUC_mean']]

results = pd.merge(results_0, results_1, on=["dataset", "n", "o", "p"], how = "left")
results = pd.merge(results, results_2, on=["dataset", "n", "o", "p"], how = "left")
results = pd.merge(results, results_3, on=["dataset", "n", "o", "p"], how = "left")
results

results.columns = col_list
results = results[col_list]
results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', 'o'])
results = results[col_list]
print(results.to_latex(index=False)) 

results_0 = pd.read_csv(results_list[0])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]
results_1 = pd.read_csv(results_list[1])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]
results_2 = pd.read_csv(results_list[2])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]
results_3 = pd.read_csv(results_list[3])[['dataset', 'n', 'o', 'p', 'PRAUC_mean']]

results = pd.merge(results_0, results_1, on=["dataset", "n", "o", "p"], how = "left")
results = pd.merge(results, results_2, on=["dataset", "n", "o", "p"], how = "left")
results = pd.merge(results, results_3, on=["dataset", "n", "o", "p"], how = "left")
results

results.columns = col_list
results = results[col_list]
results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
results = results.sort_values(by = ['sort_dataset', 'o'])
results = results[col_list]
print(results.to_latex(index=False)) 

# col_list = ['dataset', 'n', 'o', 'p'] 
# for methods in method_list:
#     col_list.append(methods+"(AUC)")
#     col_list.append(methods+"(PRAUC)")

# results_0 = pd.read_csv(results_list[0])[['dataset', 'n', 'o', 'p', 'AUC_mean', 'PRAUC_mean']]
# results_1 = pd.read_csv(results_list[1])[['dataset', 'n', 'o', 'p', 'AUC_mean', 'PRAUC_mean']]
# results_2 = pd.read_csv(results_list[2])[['dataset', 'n', 'o', 'p', 'AUC_mean', 'PRAUC_mean']]
# results_3 = pd.read_csv(results_list[3])[['dataset', 'n', 'o', 'p', 'AUC_mean', 'PRAUC_mean']]

# results = pd.merge(results_0, results_1, on=["dataset", "n", "o", "p"], how = "left")
# results = pd.merge(results, results_2, on=["dataset", "n", "o", "p"], how = "left")
# results = pd.merge(results, results_3, on=["dataset", "n", "o", "p"], how = "left")
# results

# results.columns = col_list
# results = results[col_list]
# results['sort_dataset'] = [np.where(np.array(dataset_names) == x)[0][0] for x in results['dataset'].values]
# results = results.sort_values(by = ['sort_dataset', 'o'])
# results = results[col_list]

# col_list = ['dataset', 'n', 'o', 'p'] 
# for methods in method_list:
#     col_list.append(methods+"(AUC)")
# for methods in method_list:
#     col_list.append(methods+"(PRAUC)")
# results = results[col_list]
# print(results.to_latex(index=False)) 

%%
