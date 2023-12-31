import os
import csv
from collections import defaultdict
import random
from tqdm import tqdm
import torch
import pickle

class Data_collection():
    def __init__(self, data_path, config):
        seed = 42
        random.seed(seed)
        
        print("[Data Class init]")
        self.data_path = data_path
        self.cid2content = defaultdict()
        self.qid2question = defaultdict()
        self.qid2cids = defaultdict(list)
        self.qids = set()
        self.train_qids = None
        self.test_qids = None
        self.valid_qids = None
        self.pickle_neg_cand = None
        self.num = config['test_neg_num'] 
        self.train_num = config['train_neg_num'] 
        self.valid_num = config['valid_neg_num']
        
        
        # make cid2content
        with open(os.path.join(data_path,"collection.tsv"), 'r', encoding='utf-8') as tsvfile:
            tsv_reader = csv.reader(tsvfile, delimiter='\t')
            for row in tqdm(tsv_reader):
                cid, content = row[0].split("||")
                self.cid2content[cid] = content
        # make cid2content
        with open(os.path.join(data_path,"questions.tsv"), 'r', encoding='utf-8') as tsvfile:
            tsv_reader = csv.reader(tsvfile, delimiter='\t')
            for row in tqdm(tsv_reader):
                qid, question = row
                self.qid2question[qid] = question
        # make cid2content: {qid:[cid1. cid2]}
        with open(os.path.join(data_path,"qrels.tsv"), 'r', encoding='utf-8') as tsvfile:
            tsv_reader = csv.reader(tsvfile, delimiter='\t')
            for row in tqdm(tsv_reader):
                qid, cid = row
                cid = cid.split("||")
                self.qid2cids[qid] = cid
                
        if os.path.isfile(os.path.join(data_path,"neg_bm25_cand.pkl")):
            with open(os.path.join(data_path,"neg_bm25_cand.pkl"), 'rb') as pf:
                self.pickle_neg_cand = pickle.load(pf)
        
        if os.path.isfile(os.path.join(data_path,"train_qid.pkl")):
            with open(os.path.join(data_path, 'train_qid.pkl'), 'rb') as pf:
                self.train_qids = pickle.load(pf)
            with open(os.path.join(data_path, 'test_qid.pkl'), 'rb') as pf:
                self.test_qids = pickle.load(pf)
            with open(os.path.join(data_path, 'valid_qid.pkl'), 'rb') as pf:
                self.valid_qids = pickle.load(pf)
            
        else:            
            # print("true")
            self.qids = set([qid for qid in self.qid2question.keys()])
            
            a, b = int(0.7*len(self.qids)), int(0.1*len(self.qids))
            qid_list = list(self.qids)
            
            random.shuffle(qid_list)
            
            self.train_qids = qid_list[:a]
            self.test_qids = qid_list[a:a+b]
            self.valid_qids = qid_list[a+b:]
            with open(os.path.join(data_path, 'train_qid.pkl'), 'wb') as pf:
                pickle.dump(self.train_qids, pf)
            with open(os.path.join(data_path, 'test_qid.pkl'), 'wb') as pf:
                pickle.dump(self.test_qids, pf)
            with open(os.path.join(data_path, 'valid_qid.pkl'), 'wb') as pf:
                pickle.dump(self.valid_qids, pf)
                
                
            ###########
            # train/valid/test samples
            
        print("> train_qids",len(self.train_qids))
        print("> valid_qids",len(self.valid_qids))
        print("> test_qids", len(self.test_qids))
        
        self.make_samples_qids(self.num, valid_num = 50)
        
        
    def make_samples_qids(self, num=4, valid_num = 8):
        train_samples = []
        valid_samples = []
        test_samples = []
        
        if os.path.isfile(os.path.join(self.data_path,"train_samples.pkl")):
            with open(os.path.join(self.data_path, 'train_samples.pkl'), 'rb') as pf:
                train_samples = pickle.load(pf)
            with open(os.path.join(self.data_path, 'test_samples.pkl'), 'rb') as pf:
                test_samples = pickle.load(pf)
            with open(os.path.join(self.data_path, 'valid_samples.pkl'), 'rb') as pf:
                valid_samples = pickle.load(pf)
                
            self.train_samples = train_samples
            self.valid_samples = valid_samples
            self.test_samples = test_samples
            
            return train_samples, valid_samples, test_samples
        
        cid_keys = self.cid2content.keys()
        for qid in tqdm(self.train_qids, desc="Making train_samples"):
            # random_negs = cid_keys - set(self.qid2cids[qid])
            # random_negs = cid_keys
            random_negs = random.sample(list(cid_keys), num+3)
            random_negs = list(set(random_negs) - set(self.qid2cids[qid]))
            pos = self.qid2cids[qid]
            Q = self.qid2question[qid]
            pos_D = [self.cid2content[p] for p in pos]
            neg_D = [self.cid2content[n] for n in random_negs]
            labels = [1]*len(pos_D) + [0]*len(neg_D)
            
            train_samples.append([Q, (pos_D+neg_D)[:num], labels[:num]])
        
        for qid in tqdm(self.valid_qids, desc="Making valid_samples"):
            # random_negs = cid_keys - set(self.qid2cids[qid])
            # random_negs = random.sample(list(random_negs), num)
            random_negs = random.sample(list(cid_keys), num+valid_num)
            random_negs = list(set(random_negs) - set(self.qid2cids[qid]))
            pos = self.qid2cids[qid]
            Q = self.qid2question[qid]
            pos_D = [self.cid2content[p] for p in pos]
            neg_D = [self.cid2content[n] for n in random_negs]
            labels = [1]*len(pos_D) + [0]*len(neg_D)
            
            valid_samples.append([Q, (pos_D+neg_D)[:valid_num], labels[:valid_num]])
        
        for qid in tqdm(self.test_qids, desc="Making test_samples"):
            # random_negs = cid_keys - set(self.qid2cids[qid])
            # random_negs = random.sample(list(random_negs), num)
            Q = self.qid2question[qid]
            ans_cids = self.qid2cids[qid]
            
            test_samples.append([Q, ans_cids])
            
        
        
        with open(os.path.join(self.data_path, 'train_samples.pkl'), 'wb') as pf:
            pickle.dump(train_samples, pf)
        with open(os.path.join(self.data_path, 'test_samples.pkl'), 'wb') as pf:
            pickle.dump(test_samples, pf)
        with open(os.path.join(self.data_path, 'valid_samples.pkl'), 'wb') as pf:
            pickle.dump(valid_samples, pf)
        
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.test_samples = test_samples
        
        return train_samples, valid_samples, test_samples
    
    def make_train_samples_qids(self, num=4, train_neg_cand_type="random"):
        train_samples = []
        
        cid_keys = self.cid2content.keys()
        
        for qid in tqdm(self.train_qids, desc="Making train_samples"):
            # random_negs = cid_keys - set(self.qid2cids[qid])
            # random_negs = cid_keys
            if train_neg_cand_type == 'random':
                random_negs = random.sample(list(cid_keys), num+3)
                random_negs = list(set(random_negs) - set(self.qid2cids[qid]))
                pos = self.qid2cids[qid]
                Q = self.qid2question[qid]
                pos_D = [self.cid2content[p] for p in pos]
                neg_D = [self.cid2content[n] for n in random_negs]
                labels = [1]*len(pos_D) + [0]*len(neg_D)
                
                train_samples.append([Q, (pos_D+neg_D)[:num], labels[:num]])
                
            elif train_neg_cand_type == 'bm25':
                random_negs = random.sample(self.pickle_neg_cand[qid], num+3)
                random_negs = list(set(random_negs) - set(self.qid2cids[qid]))
                pos = self.qid2cids[qid]
                Q = self.qid2question[qid]
                pos_D = [self.cid2content[p] for p in pos]
                neg_D = [self.cid2content[n] for n in random_negs]
                labels = [1]*len(pos_D) + [0]*len(neg_D)
                
                train_samples.append([Q, (pos_D+neg_D)[:num], labels[:num]])
        
        with open(os.path.join(self.data_path, 'train_samples.pkl'), 'wb') as pf:
            pickle.dump(train_samples, pf)
            
        self.train_samples = train_samples

        return train_samples
    
    
    