"""This module uses LLMs to perform checkworthiness detection using Huggingface models
through the transformers library"""

from src.checkthat_utils import load_check_that_dataset
from src.claimbuster_utils import load_claimbuster_dataset
from src.dataset_utils import ProgressDataset
from src.result_analysis import flatten_classification_report
from src.dataset_utils import CustomDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    Pipeline,
)
from tqdm.auto import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig
import enum
import torch
import pandas as pd
from typing import Dict, Iterable, List, Tuple, Union
import os
import json
import re
import numpy as np
import argparse
import timeit
import scipy.stats as st

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

DICT_MATCHER = re.compile(r"{.*}")
SCORE_MATCHER = re.compile(r"([Ss]core[^\d]*)(\d+)")
NON_CHECKWORTHY_MATCHER = re.compile(
    r"(non-checkworthy)|(not check-worthy)|(non check-worthy)"
)

class HuggingFaceModel(enum.Enum):
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_INSTRUCT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"

class PromptType(enum.Enum):

    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "CoT"
    LORA = "lora"

class ICLUsage(enum.Enum):

    ZERO_SHOT = "zeroshot"
    FEW_SHOT = "fewshot"

class Experiment(enum.Enum):

    CHECKWORTHINESS = "CW"
    INFERENCE_TIME = "IT"
    FINE_TUNING = "FT"
    TRUTH_FULNESS = "TF"


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
        report["threshold"] = optimizer.threshold
        reports.append(report)
    result = pd.DataFrame(reports)
    result = add_average_row(result)
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        result.to_csv(f"{save_folder}/crossval.csv")
        predictions.to_csv(f"{save_folder}/predictions.csv")
    return result, predictions

def calculate_confidence_interval_error(values: Iterable[float|int], confidence=0.95) -> float:
    if (st.sem(values) == 0):
        return 0
    confidence_interval: Tuple[float, float] = st.t.interval(
        confidence, len(values)-1, loc=np.mean(values), scale=st.sem(values)
    )
    return confidence_interval[1] - np.mean(values)

def add_average_row(df: pd.DataFrame, use_confidence_intervals=True) -> pd.DataFrame:
    """Adds an average row to a pandas dataframe. This assumes that all cells contain numbers."""
    if not use_confidence_intervals:
        df.loc["Average"] = df.mean()
        return df
    df.loc["Average"] = ['' for _ in df.columns]
    for column in df.columns:
        values = df.loc[df.index != "Average", column]
        error = calculate_confidence_interval_error(values)
        average_value = values.mean()
        df.loc["Average", column] = f"{average_value:.4f} ± {error:.4f}"
    return df

def _output_to_pred(
        raw_response: str, 
        dict_matcher: re.Pattern, 
        score_matcher: re.Pattern,
        non_check_worthy_matcher: re.Pattern
    ) -> Dict[str, str | int]:
    """Maps from a raw response to a prediction."""
    try:
        prediction = json.loads(dict_matcher.search(raw_response).group(0))
        score = prediction["score"]
        reasoning = prediction["reasoning"]
    except (json.decoder.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
        score = score_matcher.search(raw_response)
        reasoning = None
        if score is not None:
            score = score[2]
        else:
            score = 0.0 if non_check_worthy_matcher.search(raw_response) else np.nan
    return {"score": score, "reasoning": reasoning}


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
    
    responses = pipe(prompts_data, batch_size=batch_size)
    for index, result in enumerate(tqdm(responses, total=len(prompts))):
        response = result[0]["generated_text"].replace("\n", "")
        dataset_with_scores.loc[data.index[index], "raw_response"] = response 
        dataset_index = data.index[index]
        prediction = _output_to_pred(response, DICT_MATCHER, SCORE_MATCHER, NON_CHECKWORTHY_MATCHER)
        dataset_with_scores.loc[dataset_index, "score"] = prediction["score"]
        dataset_with_scores.loc[dataset_index, "reasoning"] = prediction["reasoning"]
    columns =  [label_column, "score", text_column, "reasoning", "raw_response"]
    dataset_with_scores = dataset_with_scores[columns]
    if save_path is not None:
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
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=return_full_text,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipe

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate check-worthiness detection predicitons with Open Source LLMs"
    )
    parser.add_argument(
        "--experiment",
        default=Experiment.CHECKWORTHINESS.value,
        choices=[Experiment.CHECKWORTHINESS.value, Experiment.INFERENCE_TIME.value]
    )
    parser.add_argument(
        "--dataset",
        default= CustomDataset.CLAIMBUSTER.value,
        choices=[CustomDataset.CLAIMBUSTER.value, CustomDataset.CHECK_THAT.value],
    )
    parser.add_argument(
        "--prompt-type",
        default=PromptType.STANDARD.value,
        choices=[PromptType.STANDARD.value, PromptType.CHAIN_OF_THOUGHT.value]
    )
    parser.add_argument(
        "--icl_usage",
        default=ICLUsage.ZERO_SHOT.value,
        choices=[ICLUsage.ZERO_SHOT.value, ICLUsage.FEW_SHOT.value]
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        choices=range(1, 257),
        type=int
    )
    parser.add_argument(
        "--max-new-tokens",
        default=64,
        choices=range(1, 1025),
        type=int
    )
    parser.add_argument(
        "--model-id",
        default=HuggingFaceModel.MISTRAL_7B_INSTRUCT.value,
        choices=[
            HuggingFaceModel.MISTRAL_7B_INSTRUCT.value, 
            HuggingFaceModel.MIXTRAL_INSTRUCT.value, 
            HuggingFaceModel.LLAMA2_7B_CHAT.value
        ]
    )
    return parser.parse_args()


def main():
    """Generates LLM predictions (experiment E1-E3) or performs an inference time
    evaluation (experiment E6) for one specific LLM configuration"""
    args = parse_arguments()
    dataset_name = args.dataset


    if dataset_name == CustomDataset.CLAIMBUSTER.value:
        dataset = load_claimbuster_dataset("data/ClaimBuster/datasets")
        label_column = "Verdict"
        text_column = "Text"
    else:
        dataset = load_check_that_dataset("data/CheckThat")
        label_column = "check_worthiness"
        text_column = "tweet_text"

    instruction_path = os.path.join(
        "prompts",
        args.dataset,
        args.prompt_type,
        args.icl_usage,
        "instruction.txt"
    )
    assert os.path.exists(instruction_path), "No prompt exist for the given configuration."
    with open(instruction_path, "r") as f:
        instruction = f.read().replace("\n", "")
    print(instruction)

    print("Loading model...")
    model_id = [value for value in HuggingFaceModel if value.value == args.model_id][0]
    pipe = load_huggingface_model(
        model_id=model_id, 
        max_new_tokens=args.max_new_tokens
    )
    if args.experiment == Experiment.INFERENCE_TIME.value:
        dataset = dataset.sample(100)
    prompts = [ f"[INST]{instruction} '''{text}'''[/INST]" for text in dataset[text_column]]

    if args.experiment == Experiment.CHECKWORTHINESS.value:
        model_name = [value.name for value in HuggingFaceModel if value.value == args.model_id][0]
        save_path = os.path.join(
            "../results",
            dataset_name,
            model_name,
            args.prompt_type,
            args.icl_usage,
            "generated_scores.csv"
        )
        print("Generating predictions...")
        generate_llm_predictions(
            data=dataset,
            prompts=prompts, 
            pipe=pipe,
            batch_size=args.batch_size,
            label_column = label_column,
            text_column=text_column,
            save_path=save_path
        )
    else:
        print("Starting inference time experiment")
        def generate_predictions():
            output = pipe(prompts, batch_size=args.batch_size)
            preds = []
            for output in output:
                output = output[0]["generated_text"].replace("\n", "")
                pred = _output_to_pred(
                    output,
                    DICT_MATCHER,
                    SCORE_MATCHER,
                    NON_CHECKWORTHY_MATCHER
                )
                preds.append(pred)
            return preds
        print(f"Total inference time: {timeit.timeit(generate_predictions, number=10)}")


if __name__ == "__main__":
    main()
