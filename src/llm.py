"""This module uses LLMs to perform checkworthiness detection using Huggingface models
through the transformers library"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    Pipeline,
)
from claimbuster_utils import load_claimbuster_dataset
from tqdm.auto import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig
import enum
import torch
import pandas as pd
from typing import List, Tuple, Union
import os
import json
import re
import numpy as np
from dataset_utils import ProgressDataset
from result_analysis import flatten_classification_report

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

DEFAULT_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'k_proj', 'down_proj', 'o_proj', 'q_proj', 'v_proj', 'gate_proj']
)


class HuggingFaceModel(enum.Enum):
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_INSTRUCT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"

class PromptType(enum.Enum):

    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "CoT"

class ICLUsage(enum.Enum):

    ZERO_SHOT = "zeroshot"
    FEW_SHOT = "fewshot"


class ThresholdOptimizer(BaseEstimator, TransformerMixin):
    """Optimizing the threshold value (0-100) to separate check-worthy
    from non check-worthy claims"""

    def __init__(self, label_column="Verdict"):
        self.threshold = None
        self.label_column = label_column

    def fit(self, x: pd.DataFrame, y: pd.Series):

        y_gold = x[self.label_column].values

        reports = []
        for threshold in range(1, 100):
            y_pred = x["score"].map(lambda x: 1 if x >= threshold else 0).values
            report = classification_report(y_gold, y_pred, output_dict=True)
            report["threshold"] = threshold
            reports.append(report)
        self.threshold = max(
            reports, key=lambda report: report["macro avg"]["f1-score"]
        )["threshold"]
        print(f"{self.threshold=}")

    def predict(self, x: pd.DataFrame):
        predictions = x["score"].map(lambda x: 1 if x >= self.threshold else 0)
        return predictions


def run_llm_cross_validation(
    data: pd.DataFrame,
    crossval_folder: str,
    n_splits=4,
    label_column="Verdict",
    save_folder=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run cross validation. Assumes that the predictions have already been
    generated trough the llm jupyter notebook"""
    reports = []
    predictions = pd.DataFrame(index=pd.Series(name=data.index.name))
    for i in range(n_splits):
        train = pd.read_json(f"{crossval_folder}/train_{i}.json")
        test = pd.read_json(f"{crossval_folder}/test_{i}.json")
        optimizer = ThresholdOptimizer(label_column=label_column)
        train_data = data[data.index.isin(train[data.index.name].values)]
        optimizer.fit(train_data, None)
        test_data = data[data.index.isin(test[data.index.name].values)]
        preds = optimizer.predict(test_data)
        for index, pred in enumerate(preds):
            predictions.loc[test[data.index.name][index], "prediction"] = pred
        report = flatten_classification_report(
            classification_report(test_data[label_column], preds, output_dict=True)
        )
        reports.append(report)
        print(predictions.head(10))
        print(test_data.head(10))
    result = pd.DataFrame(reports)
    result.loc["Average"] = result.mean()
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        result.to_csv(f"{save_folder}/crossval.csv")
        predictions.to_csv(f"{save_folder}/predictions.csv")
    return result, predictions

    
def generate_llm_predictions(
    data: pd.DataFrame,
    prompts: List[str],
    pipe: Pipeline,
    save_path: str,
    batch_size=128,
    label_column="Verdict",
    text_column="Text",
    
):
    """Generate a set of predictions using an LLM"""
    prompts_data = ProgressDataset(prompts)
    dataset_with_scores = data.copy()
    dict_matcher = re.compile(r"{.*}")
    score_matcher = re.compile(r"([Ss]core[^\d]*)(\d+)")
    non_check_worthy_matcher = re.compile(
        r"(non-checkworthy)|(not check-worthy)|(non check-worthy)"
    )
    responses = pipe(prompts_data, batch_size=batch_size)
    for index, result in enumerate(tqdm(responses, total=len(prompts))):
        response = result[0]["generated_text"].replace("\n", "")
        dataset_with_scores.loc[data.index[index], "raw_response"] = response 
        dataset_index = data.index[index]
        try:
            parsed_json = json.loads(dict_matcher.search(response).group(0))
            dataset_with_scores.loc[dataset_index, "score"] = parsed_json["score"]
            dataset_with_scores.loc[dataset_index, "reasoning"] = parsed_json["reasoning"]
        except (json.decoder.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
            score = score_matcher.search(response)
            if score is not None:
                score = score[2]
            else:
                score = 0.0 if non_check_worthy_matcher.search(response) else np.nan
            dataset_with_scores.loc[dataset_index, "score"] = score
            dataset_with_scores.loc[dataset_index, "reasoning"] = None
            continue
    columns =  [label_column, "score", text_column, "reasoning", "raw_response"]
    dataset_with_scores = dataset_with_scores[columns]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset_with_scores.to_csv(save_path, index=True)
    return dataset_with_scores

def load_huggingface_model(
    model_id: HuggingFaceModel,
    bnb_config=BNB_CONFIG,
    max_new_tokens=256,
    return_full_text=False,
    return_pipeline=True,
    lora_path=None,
) -> Union[Pipeline, AutoModelForCausalLM]:
    """Load a Huggingface LLM model as a pipeline. Note that this has only been tested
    with models from Mistral, so it might not work with other models. If the parameter
    return_pipeline is set to True, then a pipeline containing a model and tokenizer is
    returned. Otherwise, only the model is retutned"""
    model_class = (
        AutoModelForCausalLM
        if lora_path is None
        else AutoPeftModelForCausalLM
    )
    model = model_class.from_pretrained(
        lora_path if lora_path is not None else model_id.value,
        quantization_config=bnb_config,
        device_map={"": 0},
        config=DEFAULT_LORA_CONFIG if not isinstance(model_id, HuggingFaceModel) else None,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if not return_pipeline:
        return model

    tokenizer = AutoTokenizer.from_pretrained(model_id.value)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"{max_new_tokens=}")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=return_full_text,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipe


def main():

    device = "cuda"  # the device to load the model onto

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )

    with open("prompts/ClaimBuster/standard/zero-shot.txt", "r") as f:
        instruction = f.read()

    data = load_claimbuster_dataset("data/ClaimBuster/datasets")
    texts = data["Text"]
    prompts = [f"{instruction} '''{text}'''" for text in texts]
    print(prompts.__getitem__(0))

    # dataset_with_scores = data.copy()
    # for index, result in enumerate(tqdm(pipe(prompts, batch_size=256), total=len(prompts))):
    #     parsed_json = json.loads(result.replace("\n", ""))["generated_text"]
    #     dataset_with_scores.loc[index, "score"] = parsed_json["score"]
    #     dataset_with_scores.loc[index, "reasoning"] = parsed_json["reasoning"]
    # dataset_with_scores.to_csv("results/ClaimBuster/zeroshot.csv", index=True)

    # generated_ids = []
    # for text in texts:
    #     prompt = f"{instruction} '''{text}'''"
    #     with torch.no_grad():
    #         generated_ids += pipe(prompt)
    # print(generated_ids)

    # messages = [
    #     {"role": "user", "content": "What is your favourite condiment?"},
    #     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    #     {"role": "user", "content": "Do you have mayonnaise recipes?"}
    # ]


if __name__ == "__main__":
    main()
