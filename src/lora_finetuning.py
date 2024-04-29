"""In this module, LORA is used to fine-tune LLMs from HuggingFace. It is based on the implemenation from 
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from transformers import TrainingArguments, Pipeline
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from llm import add_average_row, load_huggingface_model, HuggingFaceModel
from dataset_utils import convert_to_lora_dataset, CustomDataset, ProgressDataset
from claimbuster_utils import load_claimbuster_dataset
from liar_utils import load_liar_dataset
from rawfc_utils import load_rawfc_dataset
import re
from result_analysis import flatten_classification_report
from tqdm.auto import tqdm
import os

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
    return 0

def run_truthfulness_experiment(
        model_id=HuggingFaceModel.MISTRAL_7B_INSTRUCT,
        test_dataset=CustomDataset.LIAR
    ):
    """Runs experiment E5 where a model is fine-tuned on the whole ClaimBuster 
    dataset and is used to assess the check-wortiness of claims from the LIAR dataset."""

    claimbuster = load_claimbuster_dataset("data/ClaimBuster/datasets")
    with open(f"prompts/ClaimBuster/standard/zeroshot/instruction-lora.txt") as f:
        instruction = f.read().replace("\n", " ").strip()
                                           
    train_data = convert_to_lora_dataset(claimbuster, instruction=instruction)
    run_name = "ClaimBuster_full"
    model_already_finetuned = os.path.exists(f"models/{run_name}") 
    lora_path = run_name if model_already_finetuned else None
    lora_path = None if model_already_finetuned else None
    pipe = load_huggingface_model(model_id, lora_path=lora_path, max_new_tokens=64)
    if not model_already_finetuned:
        print("Training model on whole ClaimBuster dataset")
        run_training(pipe, run_name, train_data, DEFAULT_TRAINING_ARGS)
    else:
        print("Model already trained")

    test_data = load_liar_dataset() if test_dataset == CustomDataset.LIAR else load_rawfc_dataset()
    text_column = "statement" if test_dataset == CustomDataset.LIAR else "claim"
    model_inputs = ProgressDataset([
        f"[INST]{instruction} '''{text}'''[/INST]" for text in test_data[text_column]
    ])
    
    print(f"Generating checkworthiness predictions on {test_dataset.name} dataset")
    pred_finder = re.compile("0|1")
    outputs = pipe(model_inputs, batch_size=32)
    for index, output in enumerate(tqdm(outputs)):
        pred = output_to_pred(output, pred_finder)
        dataset_index = test_data.index.values[index]
        test_data.loc[dataset_index, "check_worthiness"] = pred
    os.makedirs(f"results/{test_dataset.name}", exist_ok=True)
    test_data.to_csv(f"results/{test_dataset.name}/checkworthiness.csv", index=True)

def run_fine_tuning_experiment(dataset, model_id, folder):
    with open(f"prompts/{dataset.value}/standard/zeroshot/instruction-lora.txt") as f:
        instruction = f.read().replace("\n", " ").strip()
    label_column = "Verdict" if dataset == CustomDataset.CLAIMBUSTER else "check_worthiness"
    text_column = "Text" if dataset == CustomDataset.CLAIMBUSTER else "tweet_text"
    batch_size = 32
    index_col = "sentence_id" if dataset == CustomDataset.CLAIMBUSTER else "tweet_id"
    
    # Run cross validation
    reports = []
    predictions = pd.DataFrame(index=pd.Index([], name=index_col))
    for i in range(4):
        lora_path = f"models/{model_id.name}/{dataset.value}_crossval{i}/final_checkpoint"
        already_finetuned = os.path.exists(lora_path)
        pipe = load_huggingface_model(
            model_id,
            lora_path=lora_path if already_finetuned else None,
            max_new_tokens=64
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
            run_training(pipe=pipe, run_name=f"{model_id.name}/{dataset.value}_crossval{i}", train_data=train_data) 
        print(f"Starting inference on test set for fold {i}")
        test = pd.read_json(f"{folder}/crossval/test_{i}.json")
        prompts = ProgressDataset([f"[INST]{instruction} '''{text}'''[/INST]" for text in test[text_column]])
        outputs = pipe(prompts, batch_size=batch_size)
        pred_finder = re.compile("0|1")
        preds = []
        for index, output in enumerate(tqdm(outputs)):
            pred = output_to_pred(output, pred_finder)
            preds.append(pred)
            dataset_index = test[index_col].values[index]
            predictions.loc[dataset_index, "prediction"] = pred
        report = flatten_classification_report(
            classification_report(test[label_column], preds, output_dict=True)
        )
        reports.append(report)
    os.makedirs(f"results/{dataset.value}/{model_id.name}/lora", exist_ok=True)
    predictions.to_csv(f"results/{dataset.value}/{model_id.name}/lora/predictions.csv")
    result = pd.DataFrame(reports)
    result = add_average_row(result)
    result.to_csv(f"results/{dataset.value}/{model_id.name}/lora/crossval.csv")


def main():
    run_truthfulness_experiment(test_dataset=CustomDataset.RAWFC)
    # dataset = CustomDataset.CLAIMBUSTER
    # model_id = HuggingFaceModel.LLAMA2_7B_CHAT
    # folder="data/ClaimBuster" if dataset == CustomDataset.CLAIMBUSTER else "data/CheckThat"
    # run_fine_tuning_experiment(dataset, model_id, folder)


    

if __name__ == "__main__":
    main()

