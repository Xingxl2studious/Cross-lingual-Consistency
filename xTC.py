import os
import json
import numpy as np
from tqdm import tqdm
import evaluate
from itertools import combinations
from scipy.stats import spearmanr, kendalltau

chrf = evaluate.load("chrf")

model_files_list = [
 'gpt-35', 
 
 'baichuan2-7b', 
 'baichuan2-13b', 
 
 'bloomz-560m', 
 'bloomz-1b', 
 'bloomz-3b', 
 'bloomz-7b', 
 
 'llama2-7b',
 'llama2-13b', 
 
 'mistral-7b',
 'mixtral-8x7b',
 ]

domains_list = ['timeliness']
language_list = ['zh', 'en', 'fr', 'es', 'ru', 'ja', 'it', 'de', 'pt', 'ko', 'el', 'nl']


def non_repeating_combinations(lst):
    return list(combinations(lst, 2))

def compute_cosine_similarity(array1, array2):
    cosine_similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))
    return cosine_similarity

def corrcoef(p, q):
    p = np.array(p)
    q = np.array(q)
    # p = p + epsilon  # 在分子中添加平滑因子
    # q = q + epsilon  # 在分母中添加平滑因子
    pearson_corr = np.corrcoef(p, q)[0, 1]
    spearman_corr, _ = spearmanr(p, q, axis=0)
    kendall_corr, _ = kendalltau(p, q)
    return pearson_corr, spearman_corr, kendall_corr

for model_file in tqdm(model_files_list):
    print('='*20)
    print(f'{model_file}') 
    print('-'*20)
    total_chrf_collect = {lang:[] for lang in language_list}
    
    for domain in domains_list:
        domain_chrf_collect = {lang:[] for lang in language_list}
        print(f'\t {domain}')
        with open(f'result2/{model_file}/result_{domain}.json', 'r') as f:
            data = json.load(f)
            print('\t Num: ', len(data))

        for idx in range(len(data)):
            for lang in language_list:
                ans = data[idx][lang]['answer']
                grdTruth_list = data[idx][lang]['groundTruth']
                score_list = []
                for grdTruth in grdTruth_list:
                    temp_score = chrf.compute(predictions=[ans], references=[grdTruth], lowercase=True)['score']
                    score_list.append(temp_score)
                totaly_score = max(score_list) / (score_list.index(max(score_list)) + 1)
                
                domain_chrf_collect[lang].append(totaly_score)


        # domain corr
        spearman_corr_collect = []
        comb = non_repeating_combinations(language_list)
        for pair in comb:
            lang1, lang2 = pair
            chrf_score1 = domain_chrf_collect[lang1]
            chrf_score2 = domain_chrf_collect[lang2]
            _, spearman_corr, _ = corrcoef(chrf_score1, chrf_score2)
            if np.isnan(spearman_corr):
                print(f'\t{lang1} {lang2} pair spearman corr is Nan')
                spearman_corr_collect.append(0.0)
            else:
                spearman_corr_collect.append(spearman_corr)
        spearman_corr_score = sum(spearman_corr_collect) / len(spearman_corr_collect)
        print('\t Spearman Corr: ', spearman_corr_score) 
        print('-'*20)
        
        for lang in language_list:
            total_chrf_collect[lang] = total_chrf_collect[lang] + domain_chrf_collect[lang] 
        
    
    # totaly corr
    
    spearman_corr_collect = []
    comb = non_repeating_combinations(language_list)
    for pair in comb:
        lang1, lang2 = pair
        chrf_score1 = total_chrf_collect[lang1]
        chrf_score2 = total_chrf_collect[lang2]
        _, spearman_corr, _ = corrcoef(chrf_score1, chrf_score2)
        spearman_corr_collect.append(spearman_corr)
    spearman_corr_score = sum(spearman_corr_collect) / len(spearman_corr_collect)
    print('\t Totally')
    print('\t Num: ', len(total_chrf_collect[lang1]))
    print('\t Spearman Corr: ', spearman_corr_score)
    

# nohup python xTC.py  > result_log/xTC.log 2>&1 &