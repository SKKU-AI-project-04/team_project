# import torch
# import torch.nn as nn
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi

from tqdm import tqdm
import os

class BM25():
    def __init__(self, datasets, model_config):
        super().__init__()
        self.cid2content = datasets.cid2content
        self.k = 1.2
        self.b = 0.75
        self.mecab = Mecab("/usr/local/lib/mecab/dic/mecab-ko-dic")        

        tokenized_corpus = [self.mecab_tokenizer(doc) for doc in tqdm(datasets.cid2content.values())]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k, b = self.b)
        
        self.Word2DocMatrix = None
        
    def mecab_tokenizer(self, sent):
        return self.mecab.morphs(sent)

        
        
    def pre_calculate_vec(self):
        pass
    
    
    def train_model(self, train_samples, valid_samples):
        print("> BM25 do not train.")    
        pass
    
    
    def valid_model(self):
        pass
    
    def test_model(self, test_qid):
        pass
    
    def Ranking(self, q_id, candidate_collection_ids, Data_class, topn= -1):
        
        # Make samples
        # Q_C_pairs = []
        # questions = q_id[0]
        # for c_id in candidate_collection_ids:
        #     corpus = Data_class.cid2content[c_id]
        #     Q_C_pairs.append([questions, corpus])
        
        ## Predict Score
        
        tokenized_query = self.mecab_tokenizer(q_id[0])
        scores = self.bm25.get_scores(tokenized_query)

        # 함께 정렬
        sorted_lists = sorted(zip(scores, candidate_collection_ids), key=lambda x: x[0], reverse=True)
        # 정렬된 결과를 다시 풀어냄        
        sorted_score, sorted_candidate = zip(*sorted_lists)

        # print(sorted_score[:10])  # [4, 3, 2, 1]
        # print(sorted_candidate[:10])
        
        if topn > 0 :
            return sorted_candidate[:topn], sorted_score[:topn]
        else:
            return sorted_candidate, sorted_score
        
        
    # def predict(self, Q_C_pairs, topn=-1):
    #     scores = []
        
    #     train_dataloader = DataLoader(Q_C_pairs, batch_size = 4, shuffle=False, collate_fn = self.ranking_collate_fn)
        
    #     pbar = tqdm(train_dataloader, desc=f"Ranking ... ", dynamic_ncols=True)
    #     with torch.no_grad():
    #         for idx, features in enumerate(pbar):
    #             model_predictions = self.model(**features, return_dict=True)
    #             # print("model_predictions", model_predictions)
                
    #             scores = self.activation(model_predictions.logits)
                
    #     return scores