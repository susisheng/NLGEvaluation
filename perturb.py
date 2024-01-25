#%%
import random
import spacy
import pandas as pd
import os

random.seed(666)
nlp = spacy.load("en_core_web_sm")

#%% perturb function
def sentence_reorder(sent):
    text_split = [i.text for i in nlp(sent).sents]
    if len(text_split) == 1:
        return ""
    origin_text_split = text_split.copy()
    while text_split == origin_text_split:
        random.shuffle(text_split)
    return " ".join(text_split)

def subject_verb_dis(sent):
    cases = {'was':'were', 
            'were':'was', 
            'is':'are',
            'are':'is', 
            'has':'have',
            'have':'has',
            'does':'do',
            'do':'does'}
    sentence =''
    doc = nlp(sent)
    for i in doc:
        if i.pos_ =="AUX":
            try:
                w = cases[i.text]
            except:
                w =i.text
            sentence  = sentence + w + ' '
        else:
            sentence = sentence + i.text + ' '
    return sentence.strip()

#%%
perturb_function_map = {
    "fluency": [subject_verb_dis],
    "coherence": [sentence_reorder]
}
data_root = "data/processed/"
file_name = "SummEval.csv"

for criterion in perturb_function_map.keys():
    df_perturb = pd.read_csv(os.path.join(data_root, file_name))
    save_root = f"data/perturb/"
    for perturb_func in perturb_function_map[criterion]:
        df_perturb["sys"] = df_perturb["sys"].apply(perturb_func)
        perturb_file_name = file_name.replace(".csv", f"_{criterion}_{perturb_func.__name__}.csv")
        df_perturb[df_perturb["sys"] != ""].to_csv(os.path.join(save_root, perturb_file_name), index=False)


# %%
