import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel , AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification

from collections import defaultdict

from tqdm import tqdm

import os

# from dataset.dataloader import Dataset



class DPR(nn.Module):
    def __init__(self, datasets, model_config):
        super().__init__()
        self.trained_epoch = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_config = model_config
        self.bert_model_name = model_config['bert_model_name']
        
        self.config = AutoConfig.from_pretrained(self.bert_model_name)
        self.config.num_labels = 1

        self.model = AutoModel.from_pretrained(self.bert_model_name, config=self.config)
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
        pass
    
    
    def train_model(self, train_samples, valid_samples = None):
        print(">> Train Model Start")
        print("model_config:")
        num_epoch = self.model_config['num_epoch'] 
        batch_size = self.model_config['train_batch_size']
        valid_batch_size = self.model_config['train_batch_size']
        best_scores = 0
        
        for epoch in range(1, num_epoch+1):
            print(f"epoch-{epoch+self.trained_epoch}")
            ################################
            ######## TRAIN MODEL ###########
            ################################
            
            ## Load train data
            train_dataloader = DataLoader(train_samples, batch_size = batch_size, shuffle=True, collate_fn = self.collate_fn)
            ## 
            self.to(self.device)
            
            total_loss = 0
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+self.trained_epoch} Iteration", dynamic_ncols=True)
            for idx, (Q_features, C_features, labels) in enumerate(pbar):
                # print(features, '\n',labels)
                ## output = model predict
                Q_model_ouput = self.model(**Q_features, return_dict=True).pooler_output
                C_model_ouput = self.model(**C_features, return_dict=True).pooler_output
                # print("Q_model_ouput:", Q_model_ouput.shape)
                
                # print("model_predictions", model_predictions)
                
                logits = self.activation_train(torch.sum(Q_model_ouput * C_model_ouput, dim=1))
                logits = logits.view(-1)
                
                # print("logits:", logits)
                # print("labels:", labels)
                # print("Q_model:", Q_model_ouput.shape)
                # print("C_model:", C_model_ouput.shape)
                ## calculate loss
                loss = self.BCEWithLogitsLoss(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                
                ## Set Tqdm info
                total_loss = total_loss + loss.item()
                pbar.set_postfix(loss=loss.item(), avg_loss = total_loss/(idx+1)),
            pbar.close()
            
            ################################
            ######## VALID MODEL ###########
            ################################
            
            if valid_samples is None: continue
            
            valid_dataloader = DataLoader(valid_samples, batch_size = 1, shuffle=False, collate_fn=lambda batch: self.collate_fn(batch, in_batch=False))
            pbar_valid = tqdm(valid_dataloader, desc=f"Epoch {epoch+self.trained_epoch} VALIDATION", dynamic_ncols=True)
            
            total_recall3 = 0
            total_mrr3 = 0
            for idx, (Q_features, C_features, labels) in enumerate(pbar_valid):
                with torch.no_grad():
                    # print(features, '\n',labels)
                    ## output = model predict
                    Q_model_ouput = self.model(**Q_features, return_dict=True).pooler_output
                    C_model_ouput = self.model(**C_features, return_dict=True).pooler_output
                    # print("Q_model_ouput:", Q_model_ouput.shape)
                    
                    # print("model_predictions", model_predictions)
                    
                    logits = self.activation_train(torch.sum(Q_model_ouput * C_model_ouput, dim=1))
                    
                    logits = logits.view(-1)
                    
                    ## calculate loss
                    # logits
                    # labels
                    # print("logit : ", logits)
                    # print("lable : ", labels)
                    sorted_pairs = sorted(zip(logits.tolist(), labels.tolist()), reverse=True)
                    
                    
                    # print("sorted_pairs:", sorted_pairs)
                    
                    mrr_score = 0
                    recall_score = 0
                    for i, (score, lable) in enumerate(sorted_pairs):
                        ### TODO : CONSIDER MULTI HOP POSITIVE 
                        if lable == 1.0:
                            # print("label==1.0:", i)
                            mrr_score = 1/(i+1)
                            total_mrr3 += mrr_score
                            if i+1 <3:
                                recall_score = 1
                                total_recall3 += recall_score
                    ## Set Tqdm info
                    pbar_valid.set_postfix(mrr3=mrr_score, total_mrr3 = total_mrr3/(idx+1), recall3=recall_score, total_recall3 = total_recall3/(idx+1))
                    
                        
            pbar_valid.close()
            
            if best_scores < total_mrr3/(idx+1):
                print(">> BEST MODEL")
                self.model_save(epoch + self.trained_epoch)
            
    
    
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
        # print("sorted_score[:100]", len(sorted_score[:100]), sorted_score[:100])

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
            for idx, (Q_features, C_features) in enumerate(pbar):
                Q_model_ouput = self.model(**Q_features, return_dict=True).pooler_output
                C_model_ouput = self.model(**C_features, return_dict=True).pooler_output
                # print("model_predictions", model_predictions)
                s = self.activation_train(torch.sum(Q_model_ouput * C_model_ouput, dim=1)).tolist()
                scores.extend(s)
                
        return scores
    
    
    
    def collate_fn(self, batch, in_batch=True):
        # batch는 DataLoader에서 반환하는 미니배치 리스트
        # print("collate_fn:", in_batch)
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
        # print("batch_texts",batch_texts)
        # print("batch_texts",type(batch_texts))
        # print(len(batch_texts))
        batch_question = [t[0] for t in batch_texts]
        batch_collection = [t[1] for t in batch_texts]
        
        encoded_Q_input = self.tokenizer(batch_question, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        encoded_C_input = self.tokenizer(batch_collection, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        # print("encoded_Q_input:", encoded_Q_input.input_ids.shape)
        # encoded_input = self.tokenizer(batch_texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        
        batch_labels = torch.tensor([float(label) for label in batch_labels], dtype=torch.float32)
        encoded_Q_input = encoded_Q_input.to(self.device)
        encoded_C_input = encoded_C_input.to(self.device)
        batch_labels = batch_labels.to(self.device)
        # print(batch)
        # print("collate_fn label:", batch_labels)
        return encoded_Q_input, encoded_C_input, batch_labels
    
    
    
    def ranking_collate_fn(self, batch):
        # batch는 DataLoader에서 반환하는 미니배치 리스트
        # print("batch",batch)
        batch_question = [t[0] for t in batch]
        batch_collection = [t[1] for t in batch]
        encoded_Q_input = self.tokenizer(batch_question, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        encoded_C_input = self.tokenizer(batch_collection, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        
        encoded_Q_input = encoded_Q_input.to(self.device)
        encoded_C_input = encoded_C_input.to(self.device)
        
        return encoded_Q_input, encoded_C_input
    
    
    
    
    
    
        
        
        
        
        
        
    def model_save(self, epoch):
        print("model save")
        dir_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(dir_path,"checkpoints", f'{epoch}_{self.__class__.__name__}_best_model.p')
        self.to('cuda')
        torch.save(self.state_dict(), checkpoint_path)
        self.to(self.device)
        
    def model_load(self, epoch):
        print("model load")
        self.trained_epoch = epoch
        dir_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(dir_path,"checkpoints", f'{epoch}_{self.__class__.__name__}_best_model.p')
        self.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.to(self.device)
    