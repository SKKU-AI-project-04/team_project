# import torch
# import torch.nn as nn
from konlpy.tag import Mecab

class BM25():
    def __init__(self, datasets):
        super(self).__init__()
        self.cid2content = datasets.cid2content
        self.k = 0
        self.b = 0
        self.avgdl = 0
        self.stemmer = Mecab()        
        self.Word2DocMatrix = None
        
    def cal_Word2Doc_matrix(cid2content, stemmer):
        token_list = []
        cid = cid2content.keys()
        words = set()
        for content in cid2content.values():
            token_list.append(stemmer.morphs(content))

        for tokens in token_list:
            words.update(tokens)        
        
        for tokens in token_list:
            pass

        
        
    def pre_calculate_vec(self):
        pass
    
    
    def train_model(self):
        
        pass
    
    
    def valid_model(self):
        pass
    
    def test_model(self, test_qid):
        pass