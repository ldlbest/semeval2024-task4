import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pathlib
import numpy as np
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import (
   EvalPrediction,
   AutoTokenizer,
   TrainingArguments, 
   Trainer,
   AutoModelForSequenceClassification,
   EarlyStoppingCallback
)
from subtask_1_2a import G,_h_fbeta_score,_h_recall_score,_h_precision_score
from sklearn_hierarchical_classification.metrics import h_fbeta_score, h_recall_score, h_precision_score, \
    fill_ancestors, multi_labeled
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,precision_score,recall_score

metric_name = "/home/lidailin/metrics/accuracy"
pretrained_lm=None
all_label=set()
label2id={}
id2label={}
data_dir_prefix="dataset"
models_dir_prefix="pretrained_models"
result_dir_prefix="result"
final_f1=0
final_result={}


def set_dataset(train_data_file,val_data_file):
    encoded_dataset=load_dataset("json",data_files={"train":train_data_file,
                                        "validation":val_data_file
                                        })
    for item in encoded_dataset["train"]:
        for lb in item["labels"]:
            all_label.add(lb)
            


    global id2label,label2id
    id2label = {idx:label for idx, label in enumerate(all_label)}
    label2id = {label:idx for idx, label in enumerate(all_label)}

    return encoded_dataset,all_label,id2label,label2id


def set_train_arguments(**kwargs):
    global pretrained_lm
    pretrained_lm=os.path.join(models_dir_prefix,kwargs["pretrained_lm"])
    del kwargs["pretrained_lm"]
    train_data_file=os.path.join(data_dir_prefix,kwargs["train_dataset"])
    del kwargs["train_dataset"]
    val_data_file=os.path.join(data_dir_prefix,kwargs["val_dataset"])
    del kwargs["val_dataset"]

    encoded_dataset,all_label,id2label,label2id=set_dataset(train_data_file=train_data_file,val_data_file=val_data_file)
    #tokenizer_id="./my_models/roberta-finetuned-sem_eval-englishs_bc16_2/checkpoint-3066"
    #model_id="/home/lidailin/bert_semeval/my_models/roberta-finetuned-sem_eval-englishs_bc16_2/checkpoint-3066"
    #tokenizer_id="/home/lidailin/bert_semeval/output/semeval-roberta-mlm-final"
    #model_id="/home/lidailin/bert_semeval/output/semeval-roberta-mlm-final"4096
    #tokenizer_id="/home/lidailin/bert_semeval/output/semeval-roberta-base-mlm-final-all-data"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_lm)

    def preprocess_data(examples):
        text = examples["text"] 
        #labels=examples["labels"]#1000个拼在一起，每个都是['id', 'link', 'text', 'labels']
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=256)
        #print(len(encoding["input_ids"]))#把1000个input_ids拼在一起

        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(all_label)))
        # fill numpy array
        for id,label in enumerate(examples["labels"]):
            for lb in label:
                if lb in all_label:
                    idx=label2id[lb]#得到对应的编号
                    labels_matrix[id, idx] = 1

        encoding["labels"] = labels_matrix.tolist()
  
        return encoding
    
    encoded_dataset=encoded_dataset.map(preprocess_data,batched=True,remove_columns=['id','text'])
    encoded_dataset.set_format("torch")
    args = TrainingArguments(
        **kwargs
    )
    return encoded_dataset,all_label,id2label,label2id,args


def hierarchical_compute_metric(predictions, labels,threshold=0.25,):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    predictions = np.zeros(probs.shape)##1000*20
    predictions[np.where(probs >= threshold)] = 1
    predicted_labels=[]
    goleden_labels=[]
    for line in predictions:
        label=[id2label[idx] for idx, label in enumerate(line) if label == 1.0]
        predicted_labels.append(label)
    for lb in labels:
        goleden_labels.append([id2label[idx] for idx, label in enumerate(lb) if label == 1.0])

    # return as dictionary
    with multi_labeled(goleden_labels, predicted_labels, G) as (gold_, pred_, graph_):
        print(_h_precision_score(gold_, pred_,graph_), _h_recall_score(gold_, pred_,graph_), _h_fbeta_score(gold_, pred_,graph_))
    metrics = {'f1': _h_fbeta_score(gold_, pred_,graph_),
               'h_recall': _h_recall_score(gold_, pred_,graph_),
               'h_precision': _h_precision_score(gold_, pred_,graph_)}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = hierarchical_compute_metric(
        predictions=preds, 
        labels=p.label_ids)
    ####metrics = {'f1','h_recall','h_precision'}
    f1=result['f1']
    if f1> final_f1:
        global final_result
        final_result=result
    return result


def model_init(**kwargs):###改成闭包
    def closure_model_init():

        return AutoModelForSequenceClassification.from_pretrained(kwargs["pretrained_lm"], 
                                                                problem_type="multi_label_classification", 
                                                                num_labels=len(all_label),
                                                                id2label=id2label,
                                                                label2id=label2id)
    return closure_model_init

def model_intit():
    return AutoModelForSequenceClassification.from_pretrained(pretrained_lm, 
                                                                problem_type="multi_label_classification", 
                                                                num_labels=len(all_label),
                                                                id2label=id2label,
                                                                label2id=label2id)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate",3e-5, 3e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.01),
        "num_train_epochs": trial.suggest_int("num_train_epochs",7,7),
        "seed": trial.suggest_int("seed",42,42),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16]),
    }


def set_trainer(**kwargs):
    encoded_dataset,all_label,id2label,label2id,args=set_train_arguments(**kwargs)
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=13)],
    )
    return trainer

def train(**kwargs):
    final_model_dir=kwargs["final_model"]
    reslut_dir=os.path.join(result_dir_prefix,kwargs["result_dir"])+".txt"
    del kwargs["final_model"]
    del kwargs["result_dir"]
    trainer=set_trainer(**kwargs)
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=1,
        hp_space=hp_space
    )
    print("*************************************")
    print(" Best run %s" % str(best_trial))
    print("*************************************")
    trainer.evaluate()
    trainer.save_model(final_model_dir)
    
    with pathlib.Path(reslut_dir).open("w") as f:
        json.dump(final_result,f)
