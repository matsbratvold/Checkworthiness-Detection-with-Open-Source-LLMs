"""In this module, LORA is used to fine-tune LLMs from HuggingFace. It is based on the implemenation from 
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py"""

from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import pandas as pd
from datasets import from_pandas

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
    model: AutoModelForCausalLM, 
    run_name: str,
    train_data: pd.Dataframe,
    eval_data: pd.Dataframe,
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

    print(training_args)

    train_dataset = from_pandas(train_dataset)
    eval_dataset = from_pandas(eval_dataset)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.model.save_pretrained(f"models/{run_name}/final_chekpoint/")

def main():
    model = None
    run_training(model=model, run_name="test", train_data=None, eval_data=None)


if __name__ == "__main__":
    main()

