#%% 
import numpy as np
from collections import defaultdict
from metrics.BARTScore.bart_score import BARTScorer
from metrics.BERTScore.scorer import BERTScorer
from metrics.UniEval.evaluator import get_evaluator
from metrics.GPTScore.opt_score import OPTScorer
from metrics.MoverScore.moverscore import MoverScorer
from tqdm import tqdm
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
#%%
class Scorer:
    def process_ref_hypo(self,references, candidates):
        assert isinstance(references, list) 
        if isinstance(references[0], str):
            references = [[ref] for ref in references]
        ref_group_boundaries = []
        ori_cands, ori_refs = candidates, references
        candidates, references = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            candidates += [cand] * len(ref_group)
            references += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)
        return references, candidates, ref_group_boundaries

#%%
class BARTScoreScorer(Scorer):
    def __init__(self,config) -> None:
        from mosestokenizer import MosesDetokenizer
        self.detokenizer = MosesDetokenizer('en')
        self.device = config["device"]
        self.checkpoint = config.get("checkpoint", None)
        self.verbose = config.get("verbose", False)
        self.variant = None
        self.score_type = config.get("score_type", "src_hypo")

        if self.checkpoint is None:
            self.variant = config.get("variant", None)
            if self.variant is not None:
                if 'cnn' in self.variant or 'para' in self.variant:
                    self.checkpoint = "facebook/bart-large-cnn"
                else:
                    self.checkpoint = "facebook/bart-large"

        if self.checkpoint is None:
            self.checkpoint = "facebook/bart-large"

        scorer = BARTScorer(device=self.device, checkpoint=self.checkpoint, verbose=self.verbose)
        self.scorer = scorer
        self.config = config
        self.prefix = config.get("prefix", "")

    def detokenize(self, text: str):
        words = text.split(" ")
        return self.detokenizer(words)

    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, ref_agg_type="max", task=""):

        assert isinstance(candidates[0], str)

        candidates = [self.detokenize(cand) for cand in candidates]
        
        if self.score_type == "src_hypo":
            assert isinstance(sources, list) and isinstance(sources[0], str)
            sources = [self.detokenize(self.prefix+src) for src in sources] if sources else None
            scores = self.scorer.score(sources, candidates, batch_size=batch_size)
        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
            # detokenize
            references = [self.detokenize(ref) for ref in references]
            raw_scores = self.scorer.score(references, candidates, batch_size=batch_size)
            scores = []
            for i, (start, end) in enumerate(ref_group_boundaries):
                scores.append(raw_scores[start:end])

        score_name = "bartscore_"+self.config["checkpoint"] + "_" + self.score_type
        if self.variant:
            score_name += "_" + self.variant
        if self.prefix:
            score_name += "_prefix_" + self.prefix
        
        return {score_name: scores}

class BERTScoreScorer(Scorer):
    def __init__(self,config) -> None:
        self.checkpoint = config.get("checkpoint", '')
        if self.checkpoint == '' or self.checkpoint is None:
            self.lang = "en"
        else:
            self.lang = None
        self.device = config["device"]
        self.all_layers = config.get("all_layers", False)
        self.score_type = config.get("score_type", "src_hypo")
        self.batch_size = config.get("batch_size", 32)
        self.idf = config.get("idf", False)
        self.scorer = BERTScorer(
            model_type=self.checkpoint,
            all_layers=self.all_layers,
            lang=self.lang,
            device=self.device,
            idf=self.idf,
        )
        self.config = config
    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, ref_agg_type="max", task=""):
        
        # P, R, F: (layer_num, len(candidates))
        if self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
            raw_score = self.scorer.score(
                candidates, references, batch_size=self.batch_size,verbose=True
            )
            P, R, F = [], [], []
            if self.all_layers:
                for layer in range(len(raw_score[0])):
                    P.append([])
                    R.append([])
                    F.append([])
                    for i, (start, end) in enumerate(ref_group_boundaries):
                        P[layer].append(raw_score[0][layer][start:end].tolist())
                        R[layer].append(raw_score[1][layer][start:end].tolist())
                        F[layer].append(raw_score[2][layer][start:end].tolist())
            else:
                for i, (start, end) in enumerate(ref_group_boundaries):
                    P.append(raw_score[0][start:end].tolist())
                    R.append(raw_score[1][start:end].tolist())
                    F.append(raw_score[2][start:end].tolist())

                
                
        elif self.score_type == "src_hypo":
            P, R, F = self.scorer.score(
                candidates, sources, batch_size=self.batch_size,verbose=True
            )
        elif self.score_type == "multiref_hypo":
            references = [' '.join(refs) for refs in references]            
            P, R, F = self.scorer.score(
                candidates, references, batch_size=self.batch_size,verbose=True
            )
        
        result = {}
        if self.all_layers:
            for layer in range(len(P)):
                score_name = f"bertscore_{self.scorer.model_type}_layer_{layer}_{self.score_type}_%s"
                if self.idf:
                    score_name += "_idf"
                result[score_name%"P"] = P[layer]
                result[score_name%"R"] = R[layer]
                result[score_name%"F"] = F[layer]

        else:
            score_name = f"bertscore_{self.scorer.model_type}_layer_default_{self.score_type}_%s"
            if self.idf:
                score_name += "_idf"
            result[score_name%"P"] = P
            result[score_name%"R"] = R
            result[score_name%"F"] = F
            
        return result

class UniEvalScorer(Scorer):
    def __init__(self, config) -> None:
        self.config = config
        self.device = config["device"]
        self.task = config["variant"]
        self.scorer = get_evaluator(self.task,device=self.device)
        self.score_type = config.get("score_type", "src_hypo")
        if self.task == "summarization":
            self.aspect = ["coherence", "fluency", "consistency"]
        elif self.task == "dialogue":
            self.aspect = ['naturalness', 'coherence']
        elif self.task == "fact":
            self.aspect = ["consistency"]
        else:
            self.aspect = ["naturalness"]
    def convert_to_json(self, output_list, src_list=None, ref_list=None, context_list=None, \
            scores=None, doc_id=None, system_id=None):
        """
            Convert the data into the json format.

            output_list: a list of model output
            src_list: source input for different NLG tasks. For example, source document for summarization 
                    and dialogue history for dialogue response generation
            ref_list: human-annotated groundtruth
            context_list: the context needed to evaluate several specific dimension. For example,
                        additional factual information when evaluating engagingness and groundedness in dialogues
            scores: human scores for evaluating the model output. They can be used to calculate the correlation
                    between evaluators and human judgements. The scores should be stored in a dictionary. For example,
                    {'fluency': 2.0, 'coherence': 3.0} could be the human score for a sample.
            doc_id: the index of the input source. It can be used to calculate summary-level correlation for summarzation
            system_id: the index of the generation system. It can be used to calculate system-level correlation.
        """
        json_data = []
        for i in range(len(output_list)):
            cur = {}
            cur['system_output'] = output_list[i]
            if src_list is not None:
                cur['source'] = src_list[i]
            if ref_list is not None:
                cur['reference'] = ref_list[i]
            if context_list is not None:
                cur['context'] = context_list[i]
            if scores is not None:
                cur['scores'] = scores[i]
            if doc_id is not None:
                cur['doc_id'] = doc_id[i]
            if system_id is not None:
                cur['system_id'] = system_id[i]
            json_data.append(cur)
        return json_data
    
    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, score_type="src_hypo", ref_agg_type="max", task=""):
        scores = []
        if self.score_type == "src_hypo":
            data = self.convert_to_json(output_list=candidates, src_list=sources) 
        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
            data = self.convert_to_json(output_list=candidates, src_list=references)
        if self.task == "fact":
            scores = self.scorer.evaluate(data, print_result=False)
        else:
            scores = self.scorer.evaluate(data, print_result=False, dims=["coherence", "consistency", "fluency"], overall=True)
        result = {}
        score_name = "unieval_"+self.config["variant"] + "_%s" + "_" + self.score_type
        
        aspect_range = scores[0].keys()
        for aspect in aspect_range:
            if self.score_type == "ref_hypo":
                result[score_name%aspect] = []
                for i, (start, end) in enumerate(ref_group_boundaries):
                    result[score_name%aspect].append(scores[start:end][aspect])
            else:
                result[score_name%aspect] = [s[aspect] for s in scores]
        return result


class BleuScorer(Scorer):
    def __init__(self, config) -> None:
        self.config = config
        self.score_type = config.get("score_type", "ref_hypo")
    
    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, score_type="src_hypo", ref_agg_type="max", task=""):
        # cumulative BLEU scores
        if self.score_type == "src_hypo":
            references = sources
        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
        # detokenize
        references = [word_tokenize(ref) for ref in references]
        candidates = [word_tokenize(cand) for cand in candidates]
        
        raw_scores = [sentence_bleu(ref, cand) for ref, cand in tqdm(zip(references, candidates))]
        scores = []
        for i, (start, end) in enumerate(ref_group_boundaries):
            scores.append(raw_scores[start:end])
        
        score_name = "bleu" + "_" + self.score_type
        return {score_name: scores}

class MeteorScorer(Scorer):
    def __init__(self, config) -> None:
        self.config = config
        self.score_type = config.get("score_type", "ref_hypo")
    
    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, score_type="src_hypo", ref_agg_type="max", task=""):
        if self.score_type == "src_hypo":
            references = sources
        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
        references = [word_tokenize(ref) for ref in references]
        candidates = [word_tokenize(cand) for cand in candidates]
        raw_scores = [meteor_score.meteor_score([ref], cand) for ref, cand in tqdm(zip(references, candidates))]
        scores = []
        for i, (start, end) in enumerate(ref_group_boundaries):
            scores.append(raw_scores[start:end])
        
        score_name = "meteor" + "_" + self.score_type
        return {score_name: scores}

class RougeScorer(Scorer):
    def __init__(self, config) -> None:
        self.config = config
        self.score_type = config.get("score_type", "ref_hypo")
    
    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, score_type="src_hypo", ref_agg_type="max", task=""):
        # cumulative BLEU scores
        if self.score_type == "src_hypo":
            references = sources
        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
        rouge = Rouge()
        raw_scores = [rouge.get_scores(cand, ref) for ref, cand in tqdm(zip(references, candidates))]
        scores = []
        
        for i, (start, end) in enumerate(ref_group_boundaries):
            scores.append(raw_scores[start:end])
        result = {}
        for rouge_type in ["rouge-1","rouge-2","rouge-l"]:
            for value_type in ["p","r","f"]:
                score_name = f"{rouge_type}-{value_type}_{self.score_type}"
                result[score_name] = [[s[0][rouge_type][value_type] for s in s_group]for s_group in scores]
        return result
    
# %%
class GPTScoreScorer(Scorer):
    """ Support GPT3-based (davinci, curie, babbage, ada), OPT-based, GPT2-based, FLAN-T5-based (19 models) """
    def __init__(self, config):
        self.checkpoint = config["checkpoint"]
        self.device = config["device"]
        self.scorer = OPTScorer(device=self.device, checkpoint=self.checkpoint)
        self.score_type = config["score_type"]

    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, score_type="src_hypo", ref_agg_type="max", task=""):
        if self.score_type == "src_hypo":
            assert isinstance(sources, list) and isinstance(sources[0], str)
            # score(self, srcs, tgts, prompt_text, batch_size):
            scores = self.scorer.score(srcs=sources, tgts=candidates, prompt_text="")
        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
            scores = []
            raw_scores = scores = self.scorer.score(srcs=references, tgts=candidates, prompt_text="")
            for i, (start, end) in enumerate(ref_group_boundaries):
                scores.append(raw_scores[start:end])
        score_name = "gptscore_"+self.checkpoint + "_" + self.score_type
        return {score_name: scores}

#%%
class MoverScoreScorer(Scorer):
    def __init__(self,config):
        self.scorer = MoverScorer(config)
        # 如果将来要用v2再改
        self.checkpoint = config["checkpoint"]
        self.score_type = config["score_type"]
    
    def score(self, candidates, sources=None, references=None, aspects=None,
              batch_size=8, score_type="src_hypo", ref_agg_type="max", task=""):
        if self.score_type == "src_hypo":
            assert isinstance(sources, list) and isinstance(sources[0], str)
            scores = self.scorer.sentence_score(hypothesis=candidates, references=sources)

        elif self.score_type == "ref_hypo":
            references, candidates, ref_group_boundaries = self.process_ref_hypo(references, candidates)
            scores = []
            raw_scores = self.scorer.sentence_score(hypothesis=candidates, references=references)
            for i, (start, end) in enumerate(ref_group_boundaries):
                scores.append(raw_scores[start:end])

        score_name = "moverscore_"+self.checkpoint + "_" + self.score_type
        return {score_name: scores}