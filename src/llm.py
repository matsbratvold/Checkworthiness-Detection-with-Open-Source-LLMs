"""This module uses LLMs to perform checkworthiness detection using Huggingface models
through the transformers library"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from claimbuster_utils import load_claimbuster_dataset
import torch
from tqdm.auto import tqdm
import json


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