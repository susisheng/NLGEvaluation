device = "cuda:2"
config = {
    "metrics_config": {
        "bartscore":{
            "variant": "cnn",
            "checkpoint": "facebook/bart-large-cnn",
            "score_type": "src_hypo",
            "device": device,
            "verbose": True,
            "prefix": ""
        },
        "bertscore":{
            "checkpoint": "microsoft/deberta-large-mnli",
            "score_type": "ref_hypo",
            "device": device,
            "all_layers": False,
            "idf": False
        },
        "unieval":{
            "variant": "summarization",
            "device": device,
            "score_type": "src_hypo"
        },
        "rouge":{
            "score_type": "ref_hypo"
        },
        "meteor":{
            "score_type": "ref_hypo"
        },
        "bleu":{
            "score_type": "ref_hypo"
        },
        "ctrleval":{
            "checkpoint": "google/pegasus-large",
            "device": device
        },
        "gptscore":{
            "device": device,
            "checkpoint": "gpt2-large",
            "score_type": "src_hypo"
        },
        "moverscore":{
            "device": device,
            "score_type": "ref_hypo",
            "batch_size": 256,
            "checkpoint": "default"
        },
        "frugalscore":{
            "checkpoint":"moussaKam/frugalscore_medium_roberta_bert-score",
            "device": device,
            "score_type": "ref_hypo"
        }
    },
    "metric_list": [
        "gptscore", "moverscore", "bartscore", "unieval", "bertscore", "rouge","meteor", "bleu"
                    ],
    "datasets": {
        # "root": "data/perturb",
        "root": "data/processed",
        "dataset_name_regex": r".*\.csv$",
        "file_list": []
    },
    "save_root": "result/origin",
    "save_perturb_root": "result/perturb"
}