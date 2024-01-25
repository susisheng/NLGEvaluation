#%%
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
from pyemd import emd
from torch import nn
from math import log
from itertools import chain
from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

import os
import sys
import requests
import zipfile
from typing import List

USERHOME = os.path.expanduser("~")
MOVERSCORE_DIR = os.environ.get('MOVERSCORE', os.path.join(USERHOME, '.moverscore'))

MNLI_BERT = 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip'
output_dir = os.path.join(MOVERSCORE_DIR)


def download_MNLI_BERT(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('-' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

tarball = os.path.join(output_dir, os.path.basename(MNLI_BERT))
rawdir = os.path.join(output_dir, 'raw')

if not os.path.exists(tarball):    
    print("Downloading %s to %s" %(MNLI_BERT, tarball))
    download_MNLI_BERT(MNLI_BERT, tarball)
    
    if tarball.endswith('.zip'):                 
        z = zipfile.ZipFile(tarball, 'r')
#        z.printdir()
        z.extractall(output_dir)
        z.close()
        

#output_dir = "./uncased_L-12_H-768_A-12/mnli/" 

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        return encoded_layers, pooled_output

class MoverScorer:
    def __init__(self, config):
        
        if config["checkpoint"] == "default":
            self.model = BertForSequenceClassification.from_pretrained(output_dir, 3)
            self.tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
        else:
            self.model = AutoModel.from_pretrained(config.get("checkpoint", "destilbert-base-uncased"))
            self.tokenizer = AutoTokenizer.from_pretrained(config.get("checkpoint", "destilbert-base-uncased"))
        self.device = config.get("device", "cuda")
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer.model_max_length = 512
        self.batch_size = config.get("batch_size", 128)
    
    def truncate(self, tokens):
        if len(tokens) > self.tokenizer.model_max_length - 2:
            tokens = tokens[0:(self.tokenizer.model_max_length - 2)]
        return tokens

    def process(self, a):
        a = ["[CLS]"]+self.truncate(self.tokenizer.tokenize(a))+["[SEP]"]
        a = self.tokenizer.convert_tokens_to_ids(a)
        return set(a)
    
    @staticmethod
    def padding(arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask

    def get_idf_dict(self, arr, nthreads=4):
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(self.process)

        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

        idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
        idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def bert_encode(self, x, attention_mask):
        self.model.eval()
        x_seg = torch.zeros_like(x, dtype=torch.long)
        with torch.no_grad():
            x_encoded_layers, pooled_output = self.model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True)
        return x_encoded_layers

    def collate_idf(self, arr, tokenize, numericalize, idf_dict, pad="[PAD]"):
        tokens = [["[CLS]"] + self.truncate(tokenize(a))+["[SEP]"] for a in arr]
        arr = [numericalize(a) for a in tokens]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]

        pad_token = numericalize([pad])[0]

        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding(idf_weights, pad_token, dtype=torch.float)

        padded = padded.to(device=self.device)
        mask = mask.to(device=self.device)
        lens = lens.to(device=self.device)
        return padded, padded_idf, lens, mask, tokens

    def get_bert_embedding(self, all_sens, idf_dict):
        batch_size = self.batch_size
        padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens,
                                                        self.tokenizer.tokenize, self.tokenizer.convert_tokens_to_ids,
                                                        idf_dict)

        if batch_size == -1: batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode( padded_sens[i:i+batch_size],
                                            attention_mask=mask[i:i+batch_size])
                batch_embedding = torch.stack(batch_embedding)
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    # plus_mask = lambda x, m: x + (1.0 - m).unsqueeze(-1) * 1e30
    # minus_mask = lambda x, m: x - (1.0 - m).unsqueeze(-1) * 1e30
    # mul_mask = lambda x, m: x * m.unsqueeze(-1)
    # masked_reduce_min = lambda x, m: torch.min(plus_mask(x, m), dim=1, out=None)
    # masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1, out=None)
    # masked_reduce_mean = lambda x, m: mul_mask(x, m).sum(1) / (m.sum(1, keepdim=True) + 1e-10)
    # masked_reduce_geomean = lambda x, m: np.exp(mul_mask(np.log(x), m).sum(1) / (m.sum(1, keepdim=True) + 1e-10))
    # idf_reduce_mean = lambda x, m: mul_mask(x, m).sum(1)
    # idf_reduce_max = lambda x, m, idf: torch.max(mul_mask(minus_mask(x, m), idf), dim=1, out=None)
    # idf_reduce_min = lambda x, m, idf: torch.min(mul_mask(plus_mask(x, m), idf), dim=1, out=None)

    @staticmethod
    def pairwise_distances(x, y=None):
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        y_t = torch.transpose(y, 0, 1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)    
        return torch.clamp(dist, 0.0, np.inf)

    @staticmethod
    def slide_window(a, w = 3, o = 2):
        if a.size - w + 1 <= 0:
            w = a.size
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
        return view.copy().tolist()

    @staticmethod
    def _safe_divide(numerator, denominator):
        return numerator / (denominator+0.00001)
        
    def load_ngram(self, ids, embedding, idf, n, o):
        new_a = []        
        new_idf = []

        slide_wins = self.slide_window(np.array(ids), w=n, o=o)
        for slide_win in slide_wins:               
            new_idf.append(idf[slide_win].sum().item())
            scale = self._safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(self.device)
            tmp =  (scale * embedding[slide_win]).sum(0)    
            new_a.append(tmp)
        new_a = torch.stack(new_a, 0).to(self.device)
        return new_a, new_idf

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords = True):
        batch_size=self.batch_size
        preds = []
        for batch_start in tqdm(range(0, len(refs), batch_size)):
            batch_refs = refs[batch_start:batch_start+batch_size]
            batch_hyps = hyps[batch_start:batch_start+batch_size]
            
            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs,idf_dict_ref)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, idf_dict_hyp)
            
            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1)) 
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        
            ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)
            
            ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_min,_ = torch.min(hyp_embedding[-5:], dim=0, out=None)
            
            ref_embedding_avg = ref_embedding[-5:].mean(0)
            hyp_embedding_avg = hyp_embedding[-5:].mean(0)
            
            ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
            hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

            for i in range(len(ref_tokens)):   
                if remove_subwords:
                    ref_ids = [k for k, w in enumerate(ref_tokens[i]) if w not in set(string.punctuation)and '##' not in w and w not in stop_words]
                    hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if w not in set(string.punctuation)and '##' not in w and w not in stop_words]
                else:
                    ref_ids = [k for k, w in enumerate(ref_tokens[i]) if w not in set(string.punctuation) and w not in stop_words]
                    hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if w not in set(string.punctuation) and w not in stop_words]                

                ref_embedding_i, ref_idf_i = self.load_ngram(ref_ids, ref_embedding[i], ref_idf[i], n_gram, 1)
                hyp_embedding_i, hyp_idf_i = self.load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], n_gram, 1)
                
                raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001) 
                
                distance_matrix = self.pairwise_distances(raw, raw)

                c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                
                c1[:len(ref_idf_i)] = ref_idf_i
                c2[-len(hyp_idf_i):] = hyp_idf_i
                
                c1 = self._safe_divide(c1, np.sum(c1))
                c2 = self._safe_divide(c2, np.sum(c2))
                score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
                preds.append(score)
        return preds

    def sentence_score(self, hypothesis: List[str], references: List[str]):
        assert len(hypothesis) == len(references)
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        
        scores = self.word_mover_score(references, hypothesis, idf_dict_ref,        idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
        
        return scores

# # %%
# if __name__ == "__main__":
# refs = ["Winger dean cox says he will have to remain patient as he searches for a new club after leaving league two side leyton orient by mutual consent . The 29-year-old terminated his contract with the o 's after the transfer window closed , and can not join another efl side until january . `` i would n't say i 'm in a predicament , but i have never been in this position before , '' cox told bbc radio london . `` it is not a nice thing for a footballer . I 'm not able to do my job . '' The former brighton man continued : `` i am going to have to sit it out again for four months before i can kick a ball in the league again . `` i 'll try to make the best of it . It is hard to train on your own and keep yourself motivated but it is something which has got to be done . '' Cox left orient on 1 september after turning down a move to league one northampton town . Having spent six years with the o 's , scoring 59 times in 275 appearances , cox said he was `` an emotional wreck '' on his departure from brisbane road . `` i did n't really want to leave but , circumstances being what they were , i felt like i had no choice , '' he said . `` we had come to our conclusion that we were going to go our separate ways . I ca n't really elaborate on it for legal reasons . `` it is a club i will always love . When i finish playing i want to be a manager and if i can go back there and manage one day that would be great . '' Cox , who has only just recovered from a long-term knee injury , is aiming to agree a contract with an efl club which will commence in january before seeking a short-term deal with a non-league side to keep up his match fitness . `` i was just getting back in the groove , '' he said . `` if i can get something sorted sooner rather than later league-wise , then great . `` hopefully the clubs i speak to will understand my situation . I 'm not too proud to play in lower divisions as i need to play . `` come january , i need to be ready to kick on . '' Cox has already held initial negotiations with league two side crawley town . `` it interests me because they are local to where i am , '' he said . `` it ticks the boxes and i used to play with the captain jimmy smith at orient . The manager [ dermot drummy ] wants attractive attacking football , which is great for me because that is the way i like to play . `` by no means is it a done deal . We have had talks and we 'll see how that goes . '' You can hear an interview with dean cox on bbc radio london 's saturday sport show , which begins at 13:00 bst ."]
# cand = ['Former leyton orient striker dean cox says he will have to wait four months to play in the english football league .']
# scorer = MoverScorer({"device": "cpu"})
# scores = scorer.sentence_score(cand, refs)
# print(scores)
# #%%
# idf_dict_hyp = defaultdict(lambda: 1.)
# idf_dict_ref = defaultdict(lambda: 1.)
# #%%
# scorer.get_bert_embedding(refs, idf_dict_hyp)
# #%%
# self = scorer
# all_sens = refs
# padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens,
#                                                         self.tokenizer.tokenize, self.tokenizer.convert_tokens_to_ids,
#                                                         idf_dict_ref)
# # %%
# refs = ['The dog bit the man.', 'It was not unexpected.', 'The man had bitten the dog.'] 
# cand = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
# scorer = MoverScorer({"device": "cpu"})
# # scores = scorer.sentence_score(cand, refs)
# # print(scores)

# score = scorer.sentence_score(cand, refs)
# %%
