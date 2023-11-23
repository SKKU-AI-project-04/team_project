# from model.BM25 import BM25
# from model.CrossEncoder import CrossEncoder
# from model.DPR import DPR
import model
from dataset.dataloader import Data_collection
import os
import yaml

with open('config/main_config_CrossEncoder.yml') as config_file:
    config_data =list(yaml.safe_load_all(config_file))[0]
with open('config/model_config.yml') as config_file:
    model_config_data = list(yaml.safe_load_all(config_file))
    
# print(type(model_config_data))
# print(type(config_data))
model_config = {config['model_name']:config for config in model_config_data}

Data = Data_collection('../data', config_data)

# Model_Load
# F_MODEL_CLASS = getattr(model, config_data['first_model'])
# First_Model = F_MODEL_CLASS(Data,  model_config[config_data['first_model']])

S_MODEL_CLASS = getattr(model, config_data['second_model'])
Model = S_MODEL_CLASS(Data,  model_config[config_data['second_model']])
# print(data.test_qids)


print(">> test len:",len(Data.test_qids))
# print(Data.valid_samples[1])

Model.train_model(Data.train_samples, valid_samples = Data.valid_samples[:1000])

# Model.test_model(Data.test_samples[:100])
# Model.model_load(epoch = 1)
## First Ranking
# candidate_collection_ids = list(Data.cid2content.keys())

# sorted_candidate, _ = First_Model.Ranking(Data.test_samples[1], candidate_collection_ids, Data, topn= 1000)
# sorted_candidate, _ = First_Model.Ranking(Data.test_samples[1], candidate_collection_ids, Data, topn= 100)
