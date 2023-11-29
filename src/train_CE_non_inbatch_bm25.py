import model
from dataset.dataloader import Data_collection
import os
import yaml
import time
from tqdm import tqdm

with open('config/train_config_CE.yml') as config_file:
    config_list = list(yaml.safe_load_all(config_file))
    config_data = config_list[0]
    model_config_data = config_list[1:]
    

model_config = {config['model_name']:config for config in model_config_data}

print("[config_data]", config_data)
print("[model_config]", model_config)

Data = Data_collection('../data', config_data)

# F_MODEL_CLASS = getattr(model, config_data['first_model'])
# First_Model = F_MODEL_CLASS(Data,  model_config[config_data['first_model']])

S_MODEL_CLASS = getattr(model, config_data['second_model'])
Second_Model = S_MODEL_CLASS(Data,  model_config[config_data['second_model']])

print(">> test len:",len(Data.test_qids))
Second_Model.train_model(Data.train_samples, valid_samples = Data.valid_samples[:1000])



Second_Model.eval()
# Second_Model.pre_calculate_vec(Data)
Second_Model.pre_calculate_vec_from_pkl(Data)

candidate_collection_ids = list(Data.cid2content.keys())


recall_list = [0,0,0,0,0]
recall_num = [1,5,10,20,100]
test_num = len(Data.test_samples)
start = time.time()
for idx in tqdm(range(0,test_num), desc = 'RANKING'):
    q_label = Data.test_samples[idx][1]
    sorted_candidate, _ = Second_Model.FAST_Ranking(Data.test_samples[10], candidate_collection_ids, Data, topn= 1000)
    
    for n in range(len(recall_num)):
        if q_label[0] in list(sorted_candidate)[:recall_num[n]]:
            recall_list[n] +=1
print(f"ranking time: {time.time()-start}")
for n in range(len(recall_num)):    
    print(f"recall@{recall_num[n]}\t{recall_list[n]/test_num}")