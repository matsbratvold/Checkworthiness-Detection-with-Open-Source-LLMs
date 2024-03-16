"""In this module, LORA is used to fine-tune LLMs from HuggingFace. It is based on the implemenation from 
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from transformers import TrainingArguments, Pipeline
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from llm import load_huggingface_model, HuggingFaceModel, BNB_CONFIG
from checkthat_utils import load_check_that_dataset
from claimbuster_utils import load_claimbuster_dataset
from dataset_utils import convert_to_lora_dataset, Dataset
import re
from result_analysis import flatten_classification_report
from tqdm.auto import tqdm

DEFAULT_TRAINING_ARGS = TrainingArguments(
    output_dir="checkpoints/",
    max_steps=1000,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_steps=1000,
    logging_steps=1,
    fp16=True,
    gradient_checkpointing=False,
    seed=0,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=4,
)

DEFAULT_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'k_proj', 'down_proj', 'o_proj', 'q_proj', 'v_proj', 'gate_proj']
)

class FineTunedLLM(BaseEstimator, TransformerMixin):
    """Fine tuned LLM using LORA"""

    def __init__(self, pipe: Pipeline, run_name: str):
        self.pipe = pipe
        self.run_name = run_name

    def fit(self, x: pd.DataFrame, y: pd.Series):
        run_training(
            pipe=self.pipe,
            train_data=x,
            run_name=self.run_name
        )

    def predict(self, x: pd.DataFrame):
        pass
    

def run_training(
    pipe: Pipeline, 
    run_name: str,
    train_data: pd.DataFrame,
    args: TrainingArguments = DEFAULT_TRAINING_ARGS, 
):
    """Run model training using LORA"""

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'k_proj', 'down_proj', 'o_proj', 'q_proj', 'v_proj', 'gate_proj']
    )

    training_args = TrainingArguments(
        output_dir=f"models/{run_name}/{args.output_dir}",
        dataloader_drop_last=True,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        weight_decay=args.weight_decay,
        run_name=run_name,
        report_to="wandb",
        optim=args.optim,
        ddp_find_unused_parameters=False,
    )

    train_dataset = Dataset.from_pandas(train_data)

    model, tokenizer = pipe.model, pipe.tokenizer
    trainer = SFTTrainer(
        model=get_peft_model(prepare_model_for_kbit_training(model), lora_config),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=lora_config,
        packing=True,
        dataset_text_field="text"
    )

    trainer.train()
    trainer.model.save_pretrained(f"models/{run_name}/final_checkpoint")

def output_to_pred(output, regex_finder):
    text = output[0]["generated_text"]
    pred = regex_finder.search(text)
    if pred:
        return int(pred[0])
    return -1




def main():
    dataset = Dataset.CLAIMBUSTER
    already_finetuned = False
    folder="data/ClaimBuster_Datasets" if dataset == Dataset.CLAIMBUSTER else "data/CheckThat2021Task1a"
    with open(f"prompts/{dataset.value}/standard/zero-shot-lora.txt") as f:
        instruction = f.read().replace("\n", " ").strip()
    label_column = "Verdict" if dataset == Dataset.CLAIMBUSTER else "check_worthiness"
    text_column = "Text" if dataset == Dataset.CLAIMBUSTER else "tweet_text"
    model_id = HuggingFaceModel.MISTRAL_7B_INSTRUCT
    
    # Run cross validation
    reports = []
    predictions = pd.DataFrame()
    for i in range(4):
        lora_path = f"models/{dataset.value}/crossval{i}/final_checkpoint"
        pipe = load_huggingface_model(
            model_id,
            lora_path=f"models/checkthat_crossval{i}/final_checkpoint/" if already_finetuned else None
        )
        if not already_finetuned:
            print(f"Starting training on train set for fold {i}")
            train = pd.read_json(f"{folder}/crossval/train_{i}.json")
            train_data = convert_to_lora_dataset(
                train, 
                label_column=label_column, 
                text_column=text_column, 
                instruction=instruction
            )
            run_training(pipe=pipe, run_name=f"claimbuster_crossval{i}", train_data=train_data) 
        print(f"Starting inference on test set for fold {i}")
        test = pd.read_json(f"{folder}/crossval/test_{i}.json")
        prompts = [f"{instruction} '''{text}'''" for text in test[text_column]]
        outputs = pipe(prompts, batch_size=4)
        pred_finder = re.compile("0|1")
        preds = []
        for output in tqdm(outputs):
            preds.append(output_to_pred(output, pred_finder))
        predictions.loc[test.index, "prediction"] = preds
        report = flatten_classification_report(
            classification_report(test[label_column], preds, output_dict=True)
        )
        reports.append(report)
    predictions.to_csv(f"results/{dataset.value}/{model_id.name}/lora/predictions.csv")
    result = pd.DataFrame(reports)
    result.loc["Average"] = result.mean()
    result.to_csv(f"results/{dataset.value}/{model_id.name}/lora/crossval.csv")
    

if __name__ == "__main__":
    main()

