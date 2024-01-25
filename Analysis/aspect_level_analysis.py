#%% rank for specific criterion
import re
def get_rank(score, rank_range):
    if len(rank_range) == 2:
        if score < rank_range[0]:
            return "low"
        elif score < rank_range[1]:
            return "mod"
        else:
            return "high"
    else:
        if score < rank_range[0]:
            return "low"
        else:
            return "high"
def get_ranked_df(df, criterion,rank_range):
    df[f"{criterion}-rank"] = df[criterion].apply(lambda x: get_rank(x, rank_range))
    # return df  
#%%
import pandas as pd
criterion_alias = {
    "coherence": ["coherence", "Maintains Context"],
    "consistency": ["consistency", "fact", "informativeness", "Uses Knowledge"],
    "fluency": ["fluency", "naturalness", "Natural"],
}

dataset_criterion_map = {
    "BAGEL": ["naturalness"],
    "webnlg": ["fluency"],
    "SFHOT": ["informativeness","naturalness"],
    "SFRES": ["informativeness","naturalness"],
    "Newsroom": ["coherence", "fluency"],
    "SummEval": ["coherence", "fluency", "consistency"],
    "QAGS_CNN": ["fact"],
    "QAGS_XSUM": ["fact"],
    "usr_persona_mean": ["Natural", "Maintains Context"],
    "usr_topic_mean": ["Natural", "Maintains Context"],
}

def get_criterion_alias(criterion):
    for criterion_type in criterion_alias:
        for k in criterion_alias[criterion_type]:
            if re.match(k, criterion):
                return criterion_type

dataset_rank_range_map = {
    "usr_persona_mean": [2],
    "usr_topic_mean": [2],
    "BAGEL": [3,5],
    "webnlg": [2,3],
    "SFHOT": [3,5],
    "SFRES": [3,5],
    "Newsroom": [3,4],
    "SummEval": [3,4],
    "QAGS_CNN": [1],
    "QAGS_XSUM": [1],
}



#%%
from scipy import stats
import pandas as pd
import os
import json

def get_ks_result(file_path, criterion_list, rank_range):
    ks_result = {}
    df = pd.read_csv(file_path)
    for criterion in criterion_list:
        get_ranked_df(df, criterion,rank_range=rank_range)
    rank = ["low", "mod", "high"]
    df_rank = {}
    
    for criterion in criterion_list:
        df_rank[criterion] = {}
        for r in rank:
            df_rank[criterion][r] = df[df[f"{criterion}-rank"] == r]
            print(criterion,r, len(df_rank[criterion][r]))
            
    for criterion in criterion_list:
        criterion_alias = get_criterion_alias(criterion)
        ks_result[criterion_alias] = {}
        for metric in df.columns:
            metric_alias = metric
            if "_" not in metric:
                continue
            ks_result[criterion_alias][f"{metric_alias}-lohi"] = stats.ks_2samp(df_rank[criterion]["high"][metric], df_rank[criterion]["low"][metric])[0]
            if len(rank_range) == 2:
                ks_result[criterion_alias][f"{metric_alias}-lomod"] = stats.ks_2samp(df_rank[criterion]["mod"][metric], df_rank[criterion]["low"][metric])[0]
                ks_result[criterion_alias][f"{metric_alias}-himod"] = stats.ks_2samp(df_rank[criterion]["high"][metric], df_rank[criterion]["mod"][metric])[0]
    return ks_result

root = "../result/origin"
result = {}
for file in os.listdir(root):
    # if "usr" in file.lower():
    #     continue    
    dataset_name = file.split(".")[0]
    print(dataset_name)
    file_path = os.path.join(root, file)
    criterion_list = dataset_criterion_map[dataset_name]
    result[dataset_name] = get_ks_result(file_path, criterion_list,
                                         rank_range=dataset_rank_range_map[dataset_name])
    json.dump(result, open("ks_score/ks_result.json", "w"), indent=4)
