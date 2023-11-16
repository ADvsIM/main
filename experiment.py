import argparse
import os
import torch
import numpy as np
import pandas as pd
import utils.dataset_load as dl
import pdb
import random
from model.mlp import MLPClassifier
import utils.Function as Ft

def numbering(dataset, model, sampling, number_name, seed, loss_name):

    data_list = {0 : '1_ALOI', 
                    1 : '2_annthyroid', 
                    2 : '3_backdoor', 
                    3 : '4_breastw', 
                    4 : '5_campaign', 
                    5 : '6_cardio', 
                    6 : '7_Cardiotocography', 
                    7 : '8_celeba', 
                    8 : '9_census', 
                    9 : '10_cover', 
                    10 : '11_donors',
                    11 : '12_fault',
                    12 : '13_fraud',
                    13 : '14_glass',
                    14 : '15_Hepatitis',
                    15 : '16_http',
                    16 : '17_InternetAds',
                    17 : '18_lonosphere',
                    18 : '19_landsat',
                    19 : '20_letter',
                    20 : '21_Lymphography',
                    21 : '22_magic.gamma',
                    22 : '23_mammography',
                    23 : '24_mnist',
                    24 : '25_musk',
                    25 : '26_optdigits',
                    26 : '27_PageBlocks',
                    27 : '28_pendigits',
                    28 : '29_Pima',
                    29 : '30_satellite',
                    30 : '31_satimage-2',
                    31 : '32_shuttle',
                    32 : '33_skin',
                    33 : '34_smtp',
                    34 : '35_SpamBase',
                    35 : '36_speech',
                    36 : '37_Stamps',
                    37 : '38_thyroid',
                    38 : '39_vertebral',
                    39 : '40_vowels',
                    40 : '41_Waveform',
                    41 : '42_WBC',
                    42 : '43_WDBC',
                    43 : '44_Wilt',
                    44 : '45_wine',
                    45 : '46_WPBC',
                    46 : '47_yeast' }
    dataset = data_list[dataset]

    model_list = {0 : 'mlp', 
                    1 : 'adaboost', 
                    2 : 'lgbm',
                    3 : 'xgboost', 
                    4 : 'deepsad', 
                    5 : 'devnet', 
                    6 : 'feawad', 
                    7 : 'prenet' ,
                    8 : 'repen'}
    model_name = model_list[model]

    sampling_list = {0 : 'none', 
                    1 : 'smote', 
                    2 : 'borderline-smote',
                    3 : 'adasyn', 
                    4 : 'over-random', 
                    5 : 'tomeklinks', 
                    6 : 'enn',
                    7 : 'down-random' }
    sampling_name = sampling_list[sampling]

    number_list = {
        0: 'none',
        1: 1,
        2: 3,
        3: 5,
        4: 7,
        5: 9,
        6: 11,
        7: 13,
        8: 15,
        9: 20,
        10: 40,
        11: 100}
    number_name = number_list[number_name]

    seed_list = {
        0: [1,2],
        1: [1,3,5,7,9,11,13,15,16],
        2: [0,5,10,15,20,25,30,35,40]
    }
    seed = seed_list[seed]


    loss_list = {0 : 'mfe', 
                    1 : 'msfe', 
                    2 : 'focal',
                    3 : 'class-balanced' }
    loss_name = loss_list[loss_name]

    return dataset, model_name, sampling_name, number_name, seed,loss_name

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-d','--data_name', default = 0, type = int, 
                        help = 'anomaly dataset number')
    
    parser.add_argument('-m','--model', default = 0, type=int, 
                        help="What algorithm would you like to use")

    parser.add_argument('-s','--sampling', default = 0, type=int, 
                        help="What sampling method would you like to use")

    parser.add_argument('-l','--loss', default = 0, type=int, 
                        help="What loss would you like to use")
    
    parser.add_argument('-mi','--minor_number', default = 0, type=int, 
                        help="What gpu would you like to use")

    parser.add_argument('-se','--seed', default = 0, type=int, 
                        help="What gpu would you like to use")

    parser.add_argument('-g','--gpu', default = 0, type=int, 
                        help="What gpu would you like to use")

    parser.add_argument('-e','--epochs', default = 150, type=int, 
                        help="How epochs would you like to use")
    
    parser.add_argument('-b','--batch_size', default = 128, type=int, 
                        help="How batch size would you like to use")
    
    parser.add_argument('-t','--test', default = False, type=str2bool, 
                        help="How batch size would you like to use")
    
    parser.add_argument('--lr', default = 0.001, type=float, 
                        help="How batch size would you like to use")
    
    parser.add_argument('-ga','--gamma', default = 10, type=float, 
                        help="If you select focal loss 1")
    
    parser.add_argument('-al','--alpha', default = 0.01, type=float, 
                        help="If you select focal loss 2")

    parser.add_argument('-be','--beta', default = 0.5, type=float, 
                        help="If you select class balanced loss 1")
    
    parser.add_argument('-l2','--cbloss', default = 'squared', type=str, 
                        help="If you select class balanced loss 2")

    parser.add_argument('--file_path', default='/data/home/ppleeqq/table_data/Classical/', type=str, 
                        help='dataset path')

    parser.add_argument('--model_path', default='/home/ppleeqq/IMvsAD/log/model_parameter/', type=str, 
                        help='dataset path')    

    parser.add_argument('--save_path', default='/home/ppleeqq/IMvsAD/save_result/', type=str, 
                        help='dataset path')                 

    return parser.parse_known_args()

def main():

    # pdb.set_trace()
    args,_ = _parse_args()

    #Dataset preparation
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    validation = 1 - args.test
    data_name, model_name, sampling, number, seed_list, loss_name = numbering(args.data_name, args.model, args.sampling, args.minor_number, args.seed, args.loss)
    save_name_list = {'data_name' :data_name , 'model': model_name, 'sampling': sampling, 'number' :number , 'seed': seed_list, 'loss': loss_name, 'gamma': args.gamma, 'alpha': args.alpha, 'beta': args.beta, 'cbloss': args.cbloss}
    print('device: args.{}, dataset name : {}, model: {}, sampling: {}, minor_number : {}, loss: {}'.format(device, data_name, model_name, sampling, number, loss_name))

    AUC_list = []
    PRAUC_list = []

    for seed in seed_list:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        dataset = dl.CustomDataset(data_name, args.file_path, number = number, seed = seed)

        if model_name == 'mlp':
            model = MLPClassifier(loss_name = loss_name, 
                                    lr = args.lr,
                                    gamma = args.gamma, 
                                    alpha =args.alpha, 
                                    beta = args.beta, 
                                    loss_type=args.cbloss, 
                                    device = device)
                                    
            model.fit(dataset, epoch = args.epochs, batch_size = args.batch_size)
            Ft.save_model(model, args.model_path, data_name, model_name, sampling, loss_name)
            
            if validation:
                test_data = dataset.val_data
                y_true = dataset.val_y
            elif args.test:
                test_data = dataset.test_data
                y_true = dataset.test_y

        
        # y_predict = model.predict(test_data)
        y_predict_proba = model.predict_pro(test_data)
        result = Ft.evaluation_metric(y_true, y_predict_proba)

        print('auc: ', result['auc'])
        print('prauc: ', result['prauc'])
        AUC_list.append(result['auc'])
        PRAUC_list.append(result['prauc'])


    Ft.result_save(dataset, AUC_list, PRAUC_list, args.save_path, save_name_list)
    print('Finish!')
        

            
if __name__ == '__main__':
    main()