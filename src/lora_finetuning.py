"""In this module, LORA is used to fine-tune LLMs from HuggingFace. It is based on the implemenation from 
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py"""

from peft import LoraConfig
from sklearn.model_selection import train_test_split
from transformers import Pipeline, TrainingArguments
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from llm import load_huggingface_model, HuggingFaceModel
from checkthat_utils import load_check_that_dataset
from dataset_utils import convert_to_lora_dataset

DEFAULT_TRAINING_ARGS = TrainingArguments(
    output_dir="./checkpoints",
    max_steps=1000,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=1,
    fp16=False,
    bf16=False,
    gradient_checkpointing=False,
    seed=0,
    local_rank=0,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

def run_training(
    pipe: Pipeline, 
    run_name: str,
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    args: TrainingArguments = DEFAULT_TRAINING_ARGS, 
):
    """Run model training using LORA"""

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=f"models/{run_name}/{args.output_dir}",
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=run_name,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)

    model, tokenizer = pipe.model, pipe.tokenizer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=lora_config,
        packing=True,
        dataset_text_field="text"
    )

    trainer.train()
    trainer.model.save_pretrained(f"models/{run_name}/final_chekpoint/")

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

    run_training(pipe=pipe, run_name="checkthat_test", train_data=train, eval_data=test)


if __name__ == "__main__":
    main()

