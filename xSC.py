import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from itertools import combinations


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


domains_list = ['geo', 'history', 'literature', 'sport_club', 'science', 'movie']
language_list = ['zh', 'en', 'fr', 'es', 'ru', 'ja', 'it', 'de', 'pt', 'ko', 'el', 'nl']

model = SentenceTransformer('LaBSE')



def non_repeating_combinations(lst):
    return list(combinations(lst, 2))

def compute_cosine_similarity(array1, array2):
    cosine_similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))
    return cosine_similarity


for model_file in tqdm(model_files_list):
    print('='*20)
    print(f'{model_file}') 
    print('-'*20)
    total_similarity_collect = []
    
    for domain in domains_list:
        print(f'\t {domain}')
        with open(f'result2/{model_file}/result_{domain}.json', 'r') as f:
            data = json.load(f)
            # print(len(data))
            print('\t Num: ', len(data))
        all_result = []
        for idx in range(len(data)):
            for lang in language_list:
                all_result.append(data[idx][lang]['answer'])
                
        # encode
        embeddings = model.encode(all_result)

        # get_result
        for idx in range(len(data)):
            for lang in language_list:
                data[idx][lang]['embedding'] = embeddings[idx * len(language_list) + language_list.index(lang)]
        
        # combination
        similarity_collect = []
        for idx in range(len(data)):
            comb = non_repeating_combinations(language_list)
            similarity_list = []
            for pair in comb:
                lang1, lang2 = pair
                embedding1 = data[idx][lang1]['embedding']
                embedding2 = data[idx][lang2]['embedding']
                similarity = compute_cosine_similarity(embedding1, embedding2)
                similarity_list.append(similarity)
            similarity_avg = sum(similarity_list) / len(similarity_list)
            similarity_collect.append(similarity_avg)
        similarity_score = sum(similarity_collect) / len(similarity_collect)
        print('\t LabSE Cosine Similarity Score: ', similarity_score) 
        print('-'*20)
        
        total_similarity_collect.extend(similarity_collect)


    print('\t Totally')
    print('\t Num: ', len(total_similarity_collect))
    print('\t LabSE Cosine Similarity Score: ', sum(total_similarity_collect) / len(total_similarity_collect))

