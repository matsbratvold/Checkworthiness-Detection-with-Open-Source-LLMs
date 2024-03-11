"""This module uses LLMs to perform checkworthiness detection using Huggingface models
through the transformers library"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, Pipeline
from claimbuster_utils import load_claimbuster_dataset
from tqdm.auto import tqdm
import enum
import torch
import pandas as pd
from typing import Union

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

class HuggingFaceModel(enum.Enum):
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_INSTRUCT = "mistralai/Mixtral-8x7B-Instruct-v0.1"

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
        self.threshold = max(reports, key=lambda report: report["macro avg"]["f1-score"])["threshold"]
    
    def predict(self, x: pd.DataFrame):
        predictions = x["score"].map(lambda x: 1 if x >= self.threshold else 0)
        return predictions

def run_llm_cross_validation(
    data: pd.DataFrame,
    n_splits=4,
    label_column="Verdict",
) -> pd.DataFrame:
    """Run cross validation. Assumes that the predictions have already been 
    generated trough the llm jupyter notebook"""
    result = pd.DataFrame(cross_validate(
        ThresholdOptimizer(label_column=label_column),
        X=data,
        y=data[label_column],
        cv=StratifiedKFold(n_splits=n_splits),
        scoring=["precision_macro", "recall_macro", "f1_macro", "accuracy"],
    ))
    result.loc["Average"] = result.mean()
    return result

def load_huggingface_model(
    model_id: HuggingFaceModel, 
    bnb_config=BNB_CONFIG,
    max_new_tokens=256,
    return_full_text=False,
    return_pipeline=True,
) -> Union[Pipeline, AutoModelForCausalLM]: 
    """Load a Huggingface LLM model as a pipeline. Note that this has only been tested
    with models from Mistral, so it might not work with other models. If the parameter 
    return_pipeline is set to True, then a pipeline containing a model and tokenizer is
    returned. Otherwise, only the model is retutned"""
    model = AutoModelForCausalLM.from_pretrained(
        model_id.value,
        quantization_config = bnb_config,
        device_map={"":0},
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
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe




def main():

    device = "cuda" # the device to load the model onto

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config = bnb_config,
        device_map={"": 0}
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
        pad_token_id=tokenizer.eos_token_id
    )

    with open("prompts/ClaimBuster/standard/zero-shot.txt", "r") as f:
        instruction = f.read()
    
    data = load_claimbuster_dataset("data/ClaimBuster_Datasets/datasets")
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