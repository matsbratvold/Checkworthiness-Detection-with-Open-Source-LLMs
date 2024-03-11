"""In this module, LORA is used to fine-tune LLMs from HuggingFace. It is based on the implemenation from 
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py"""

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import TrainingArguments, Pipeline
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from llm import load_huggingface_model, HuggingFaceModel, BNB_CONFIG
from checkthat_utils import load_check_that_dataset
from dataset_utils import convert_to_lora_dataset
import re
import torch

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
    trainer.model.save_pretrained(f"models/{run_name}/final_chekpoint/")

def output_to_pred(output, regex_finder):
    text = output[0]["generated_text"]
    pred = regex_finder.search(text)
    if pred:
        return int(pred[0])
    return -1


def main():
    pipe = load_huggingface_model(HuggingFaceModel.MISTRAL_7B_INSTRUCT)
    data = load_check_that_dataset("data/CheckThat2021Task1a")
    with open("prompts/CheckThat/standard/zero-shot-lora.txt") as f:
        instruction = f.read().replace("\n", " ").strip()
    lora_data = convert_to_lora_dataset(
        data, 
        label_column="check_worthiness", 
        text_column="tweet_text", 
        instruction=instruction
    )
    train, test = train_test_split(lora_data, train_size=0.75, random_state=0, stratify=lora_data["output"])
    # run_training(pipe=pipe, run_name="checkthat_test", train_data=train)
    # Merge model with trained LORA weights
    lora_name = "models/checkthat_test/final_chekpoint"
    test_data = data[data["tweet_text"].isin(test["input"].str.join(""))]
    print(f"{test_data.shape=} {test.shape=}")
    prompts = [f"{instruction} '''{text}'''" for text in test_data["tweet_text"]]
    pred_finder = re.compile("0|1")
    for i in range(2):
        if i == 1:
            pipe.model = AutoPeftModelForCausalLM.from_pretrained(lora_name, device_map={"":0}, quantization_config=BNB_CONFIG)
        outputs = pipe(prompts, batch_size=4)
        preds = [output_to_pred(output, pred_finder) for output in outputs]
        print(preds)
        print(classification_report(test_data["check_worthiness"], preds))



if __name__ == "__main__":
    main()

