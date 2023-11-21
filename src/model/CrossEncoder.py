import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel , AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification

from collections import defaultdict

from tqdm import tqdm

# from dataset.dataloader import Data_collection

class CrossEncoder(nn.Module):
    def __init__(self, datasets, model_config):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_config = model_config
        self.bert_model_name = model_config['bert_model_name']
        
        self.config = AutoConfig.from_pretrained( self.bert_model_name)
        self.config.num_labels = 1

        self.model = AutoModel.from_pretrained( self.bert_model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        
        self.activation = nn.Identity()
        self.activation_train = nn.Identity()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.max_grad_norm = model_config['max_grad_norm']
        self.reg = float(model_config['reg'])
        self.lr = float(model_config['lr'])
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        self.to(self.device)
        
    def pre_calculate_vec(self):
        print("> CrossEncoder cannot pre calculate ")
        pass
    
    
    def train_model(self, train_samples, valid_samples = None):
        print(">> Train Model Start")
        num_epoch = self.model_config['num_epoch'] 
        batch_size = self.model_config['train_batch_size']
        
        for epoch in range(1, num_epoch+1):
            print(f"epoch-{epoch}")
            
            ## Load train data
            train_dataloader = DataLoader(train_samples, batch_size = batch_size, shuffle=True, collate_fn = self.collate_fn)
            ## 
            self.to(self.device)
            
            total_loss = 0
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} Iteration", dynamic_ncols=True)
            for idx, (features, labels) in enumerate(pbar):
                # print(features, '\n',labels)
                ## output = model predict
                model_predictions = self.model(**features, return_dict=True)
                # print("model_predictions", model_predictions)
                
                logits = self.activation_train(model_predictions.logits)
                logits = logits.view(-1)
                
                ## calculate loss
                loss = self.BCEWithLogitsLoss(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                
                ## Set Tqdm info
                total_loss = total_loss + loss.item()
                pbar.set_postfix(loss=loss.item(), avg_loss = total_loss/(idx+1)),
            pbar.close()
            
        
    
    
    def valid_model(self):
        pass
    
    def test_model(self):
        pass
    
    
    
    
    def Ranking(self, q_id, candidate_collection_ids, Data_class, topn= -1):
        
        # Make samples
        Q_C_pairs = []
        questions = q_id[0]
        for c_id in candidate_collection_ids:
            corpus = Data_class.cid2content[c_id]
            Q_C_pairs.append([questions, corpus])
        
        ## Predict Score
        scores = self.predict(Q_C_pairs)

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
    
    
    def predict(self, Q_C_pairs, topn=-1):
        scores = []
        
        train_dataloader = DataLoader(Q_C_pairs, batch_size = 4, shuffle=False, collate_fn = self.ranking_collate_fn)
        
        pbar = tqdm(train_dataloader, desc=f"Ranking ... ", dynamic_ncols=True)
        with torch.no_grad():
            for idx, features in enumerate(pbar):
                model_predictions = self.model(**features, return_dict=True)
                # print("model_predictions", model_predictions)
                
                scores = self.activation(model_predictions.logits)
                
        return scores
    
    
    
    def collate_fn(self, batch, in_batch=True):
        # batch는 DataLoader에서 반환하는 미니배치 리스트
        
        q2posd = defaultdict(list)
        for example in batch:
            Q, D, L = example[0], example[1], example[2]
            for d, l in zip(D, L):
                if l==1:
                    q2posd[Q].append(d)
        query = [example[0] for example in batch]
        docs = [doc for example in batch for doc in example[1]]

        batch_texts = []
        batch_labels = []
        
        ## in-Batch-neg
        if in_batch is True:
            for q in query:
                for d in docs:
                    # print(f"Q:{q}, D:{d}, {q2posd.get(q)}, {1 if q2posd.get(q) == d else 0}")
                    batch_texts.append([q,d])
                    batch_labels.append(1 if d in q2posd.get(q) else 0)
        ## only random-neg 
        else:
            for example in batch:
                Q, D, L = example[0], example[1], example[2]
                for d, l in zip(D, L):
                    batch_texts.append([Q,d])
                    batch_labels.append(l)
        
        encoded_input = self.tokenizer(batch_texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        batch_labels = torch.tensor([float(label) for label in batch_labels], dtype=torch.float32)
        encoded_input = encoded_input.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return encoded_input, batch_labels
    
    
    
    def ranking_collate_fn(self, batch):
        # batch는 DataLoader에서 반환하는 미니배치 리스트
        
        encoded_input = self.tokenizer(batch, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        
        encoded_input = encoded_input.to(self.device)
        
        return encoded_input