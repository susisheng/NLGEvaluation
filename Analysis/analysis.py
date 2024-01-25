#%%
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import numpy as np

#%%
def get_correlation_by_csv(csv_path, func, metric1=None, metric2=None):
    df = pd.read_csv(csv_path)
    if metric1 is None:
        metric1 = df.columns.tolist()[1:]
    if metric2 is None:
        metric2 = df.columns.tolist()[1:]
    return func(df, metric1, metric2)

#%%
def get_correlation_by_df(df, metric1, metric2):
    if isinstance(metric1, str):
        metric1 = [metric1]
    if isinstance(metric2, str):
        metric2 = [metric2]
    result = []
    for m1 in metric1:
        if "_" in m1 or "ref_hypo" in m1:
            continue
        for m2 in metric2:
            if "_" not in m2:
                continue
            m1_scores = df[m1].tolist()

            if "ref_hypo" in m2:
                m2_scores = df[m2].apply(lambda x: eval(x)[0]).tolist()
            else:
                m2_scores = df[m2].tolist()
            pearson, _ = pearsonr(m1_scores, m2_scores)
            spearman, _ = spearmanr(m1_scores, m2_scores)
            kendall, _ = kendalltau(m1_scores, m2_scores)
            result.append(['%s_%s'%(m1,m2) ,pearson, spearman, kendall])
    headers = ['metrics',"pearson",'spearman',"kendalltau"]
    df = pd.DataFrame(result, columns=headers)
    return df

def perturb_detection(df_perturb, df_origin, criterion="coherence", threshold=3):
    df_merged = df_perturb[df_origin[criterion]>threshold].merge(df_origin, on="id",suffixes=("-perturb", "-origin"))
    print(df_merged.columns)
    result = []
    for c in df_origin.columns: 
        if "_" not in c:
            continue
        if "ref_hypo" in c:
            df_merged[c+"-perturb"] = df_merged[c+"-perturb"].apply(lambda x: eval(x)[0])
            df_merged[c+"-origin"] = df_merged[c+"-origin"].apply(lambda x: eval(x)[0])
        acc = sum(df_merged[c+"-perturb"]<df_merged[c+"-origin"]) / len(df_merged)
        result.append([c, acc])
    return pd.DataFrame(result, columns=["metrics", "acc"])

def ref_num_analysis(df):
    pass

#%% caculate correlation
root = "../result/origin/"
for file in os.listdir(root):
    df = get_correlation_by_csv(os.path.join(root + file), get_correlation_by_df)
    df.to_csv("correlation/" + file, index=False)

# %% calculate perturb detection
root = "../result/perturb/"
for dataset_name in ["Newsroom", "SummEval"]:
    for file in os.listdir(root):
        if dataset_name not in file:
            continue
        print(file)
        df_perturb = pd.read_csv(os.path.join(root, file))
        df_origin = pd.read_csv(f"../result/origin/{dataset_name}.csv")
        df = perturb_detection(df_perturb, df_origin)
        df.to_csv("detection/" + file, index=False)



# %% generate table
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
    "usr_persona_mean": ["Uses Knowledge", "Natural", "Maintains Context"],
    "usr_topic_mean": ["Uses Knowledge", "Natural", "Maintains Context"],
}

metrics = {
    "reference-free":{
        "GPTScore": "gptscore_gpt2-large_src_hypo",
        "BARTScore": "bartscore_facebook/bart-large-cnn_src_hypo",
        "UniEval": "unieval_summarization_",
        "UniEval_all": "unieval_summarization_overall",
        "UniEval-fact": "unieval_fact",
        "MoverScore": "moverscore_default_ref_hypo"
        
    },
    "reference-based":{
        "BERTScore-F": "bertscore_microsoft/deberta-large-mnli_layer_default_ref_hypo_F",
        "BERTScore-P": "bertscore_microsoft/deberta-large-mnli_layer_default_ref_hypo_P",
        "Rouge-1-F": "rouge-1-f_ref_hypo",
        "Rouge-2-F": "rouge-2-f_ref_hypo",
        "Rouge-L": "rouge-l-p_ref_hypo",
        "Meteor": "meteor",
        "Bleu": "bleu",
    }
}
correlation_type = "pearson"
header = ["criterion", "dataset"] + [metric for metric_type in metrics for metric in metrics[metric_type]]
result = []
for dataset in dataset_criterion_map:
    for criterion in dataset_criterion_map[dataset]:
        alias = [k for k in criterion_alias if criterion in criterion_alias[k]][0]
        df = pd.read_csv(f"correlation/{dataset}.csv")
        row = [alias, dataset]
        for metric_type in metrics:
            for metric in metrics[metric_type]:
                pattern = f"{criterion}.*{metrics[metric_type][metric]}.*"
                if "unieval_summarization" in pattern and "overall" not in pattern:
                    pattern += f"{alias}_src_hypo" 
                    # print(dataset, criterion, metric, pattern)
                score = df[df['metrics'].str.match(pattern)][correlation_type].tolist()
                if len(score) != 1:
                    print(pattern, df[df['metrics'].str.match(pattern)]["metrics"])
                row.append(score[0])
        result.append(row)
pd.DataFrame(result, columns=header).to_csv(f"table_{correlation_type}.csv", index=False)
