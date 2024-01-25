#%% calculate average score for each model in SummEval, Newsroom
# index_metric_map for summeval
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
# %% 
dataset_list = ["SummEval", "Newsroom"]
for dataset_name in dataset_list:
    print(dataset_name)
    file = f"{dataset_name}.csv"
    file_path = os.path.join("../result/origin", file)
    df_dataset = pd.read_csv(file_path)
    
    index_set = set()
    for id in df_dataset["id"]:
        index_set.add("_".join(id.split('_')[1:]))
    criteria_set = set()
    for criterion in df_dataset.columns:
        if criterion == "id" or "_" in criterion:
            continue
        criteria_set.add(criterion)
    
    result = []
    for index in index_set:
        index_result = {
            "model": index,
            }
        df_index = df_dataset[df_dataset["id"].str.endswith(index)]
        score_sum = 0
        cnt = 0
        # calculate avg human eval score
        for criterion in criteria_set:
            cnt += 1
            index_result[criterion] = df_index[criterion].mean()
            score_sum += index_result[criterion]
        index_result["avg"] = score_sum / cnt
        
        # calculate correlation
        for metric in df_index.columns:
            if "_" not in metric:
                continue
            for criterion in criteria_set:
                criterion_score = df_index[criterion].to_list()
                if "ref_hypo" in metric:
                    metric_score = df_index[metric].apply(lambda x: eval(x)[0]).tolist()
                else:
                    metric_score = df_index[metric].tolist()

                index_result[f"{criterion}_{metric}_spearman"] = spearmanr(criterion_score, metric_score)[0]
                index_result[f"{criterion}_{metric}_pearson"] = pearsonr(criterion_score, metric_score)[0]
                index_result[f"{criterion}_{metric}_kendall"] = kendalltau(criterion_score, metric_score)[0]
        result.append(index_result)
    
    df_result = pd.DataFrame(result)
    df_result.to_csv(f"system/{file}", index=False)

# %%
import os
import pandas as pd
dir_root= "system/"
for file in os.listdir(dir_root):
    result_meta = []
    if file.endswith("meta.csv"):
        continue
    
    df_dataset = pd.read_csv(os.path.join(dir_root, file))
    criteria_set = set()
    for criterion in df_dataset.columns:
        if criterion in ["model", "id"] or "_" in criterion:
            continue
        criteria_set.add(criterion)
    
    for criterion in criteria_set:
        criterion_result ={
            "criterion": criterion,
        }
        criterion_score = df_dataset[criterion].to_list()
        for metric in df_dataset.columns:
            if "_" not in metric:
                continue
            metric_score = df_dataset[metric].tolist()
            if metric.endswith("spearman"):
                criterion_result[metric] = spearmanr(criterion_score, metric_score)[0]
            elif metric.endswith("pearson"):
                criterion_result[metric] = pearsonr(criterion_score, metric_score)[0]
            elif metric.endswith("kendall"):
                criterion_result[metric] = kendalltau(criterion_score, metric_score)[0]
        result_meta.append(criterion_result)

    df_result_meta = pd.DataFrame(result_meta)
    df_result_meta.to_csv(f'system/{file.replace(".csv","_meta.csv")}', index=False)



# %%
