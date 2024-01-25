#%%
import Scorers
import json
import os
import pandas as pd
import re
# load config
#%%
import config
import importlib
importlib.reload(config)
config = config.config
importlib.reload(Scorers)
from Scorers import *

# util function
def get_scorer(metric_name,config):
    if "bart" in metric_name.lower():
        scorer = BARTScoreScorer(config=config)
    elif "unieval" in metric_name.lower():
        scorer = UniEvalScorer(config=config)
    elif "bertscore" in metric_name.lower():
        scorer = BERTScoreScorer(config=config)
    elif "rouge" in metric_name.lower():
        scorer = RougeScorer(config=config)
    elif "meteor" in metric_name.lower():
        scorer = MeteorScorer(config=config)
    elif "bleu" in metric_name.lower():
        scorer = BleuScorer(config=config)
    elif "gpt" in metric_name.lower():
        scorer = GPTScoreScorer(config=config)
    elif "moverscore" in metric_name.lower():
        scorer = MoverScoreScorer(config=config)
    return scorer

# metric config
metric_config = config["metrics_config"]
metric_list = config["metric_list"]
# dataset config
dataset_config = config["datasets"]
dataset_root = dataset_config["root"]
# file list
dataset_file_list = dataset_config.get("file_list",[])
# save root
save_root = config["save_root"]
save_perturb_root = config["save_perturb_root"]

if dataset_file_list==[]:
    dataset_name_regex = dataset_config.get("dataset_name_regex",r".*")
    for file_name in os.listdir(dataset_root):
        if file_name.endswith(".csv"):
            if re.match(dataset_name_regex,file_name.lower()):
                dataset_file_list.append(file_name)

#%%
results = []
for metric in metric_list:
    print(metric, metric_config[metric])
    scorer = get_scorer(metric, metric_config[metric])
    for dataset_file in dataset_file_list:
        print(dataset_file)
        file_path = os.path.join(dataset_root,dataset_file)
        # load df
        df = pd.read_csv(file_path)
        source = df["src"].to_list()
        candidate = df["sys"].to_list()
        reference = df.get("ref", None)
        references = df.get("refs", None)
        
        if references is None:
            if reference is not None:
                reference = reference.to_list()
                references = [[ref] for ref in reference]
        else:
            references = references.to_list()
            references = [eval(ref) for ref in references]
        
        result = scorer.score(candidates=candidate, sources=source, references=references)
        df_result = pd.DataFrame()
        df_result["id"] = df["id"]

        for c in df.columns:
            if c not in ["id","src","sys","ref","refs"]:
                if "_" not in c:
                    df_result[c] = df[c]
        
        if "perturb" in file_path:
            save_path = save_perturb_root + '/' + dataset_file
        else:
            save_path = save_root + '/' + dataset_file

        if os.path.exists(save_path):
            df_old = pd.read_csv(save_path)
            for key in df_old.columns:
                if key not in result.keys():
                    df_result[key] = df_old[key]

        for key in result.keys():
            df_result[key] = result[key]

        df_result.to_csv(save_path,index=False)
        print("save to {}".format(save_path))


# %% 
