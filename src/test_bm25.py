# from model.BM25 import BM25
# from model.CrossEncoder import CrossEncoder
# from model.DPR import DPR
import model
from dataset.dataloader import Data_collection
import os
import yaml
import time
from tqdm import tqdm
with open('config/main_config_DPR.yml') as config_file:
    config_data =list(yaml.safe_load_all(config_file))[0]
with open('config/model_config.yml') as config_file:
    model_config_data = list(yaml.safe_load_all(config_file))
    
# print(type(model_config_data))
# print(type(config_data))
model_config = {config['model_name']:config for config in model_config_data}

Data = Data_collection('../data', config_data)

# Model = CrossEncoder(Data,  model_config[config_data['first_model']])
F_MODEL_CLASS = getattr(model, config_data['first_model'])
First_Model = F_MODEL_CLASS(Data,  model_config[config_data['first_model']])

S_MODEL_CLASS = getattr(model, config_data['second_model'])
Second_Model = S_MODEL_CLASS(Data,  model_config[config_data['second_model']])
# print(data.test_qids)


print(">> test len:",len(Data.test_qids))
# print(Data.valid_samples[1])

# First_Model.train_model(Data.train_samples, valid_samples = Data.valid_samples[:1000])

# Model.test_model(Data.test_samples[:100])
Second_Model.model_load(epoch = 10)
## First Ranking
candidate_collection_ids = list(Data.cid2content.keys())

recall_list = [0,0,0,0,0]
recall_num = [1,5,10,20,100]
test_num = 100
start = time.time()
for idx in tqdm(range(0,test_num), desc = 'RANKING'):
    q_label = Data.test_samples[idx][1]
    
    sorted_candidate, _ = First_Model.Ranking(Data.test_samples[idx], candidate_collection_ids, Data, topn= 1000)
    # sorted_candidate, _ = Second_Model.Ranking(Data.test_samples[idx], sorted_candidate, Data, topn= 100)
    # sorted_candidate, _ = Second_Model.Ranking(Data.test_samples[idx], candidate_collection_ids, Data, topn= 100)
    # print("sorted_cadidate,",sorted_candidate)
    # print("q_label,",q_label)
    
    for n in range(len(recall_num)):
        if q_label[0] in list(sorted_candidate)[:recall_num[n]]:
            recall_list[n] +=1
print(f"ranking time: {time.time()-start}")
for n in range(len(recall_num)):    
    print(f"recall@{recall_num[n]}\t{recall_list[n]/test_num}")
# sorted_candidate, _ = First_Model.Ranking(Data.test_samples[1], candidate_collection_ids, Data, topn= 100)
