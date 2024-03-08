"""This module contains common utils to be used for several datasets"""

import pandas as pd
import os

def convert_to_lora_dataset(
    data: pd.DataFrame, 
    instruction:str, 
    label_column="Verdict",
    text_column="Text",
    output_path: str = None
):
    """Convert the dataset to a format that can be used to fine-tune LLMs
    using the LORA technique."""
    lora_data = pd.DataFrame(
        index=data.index,
        data={
            "text": data.apply(lambda row: f"""<s>[INST] {instruction} '''{row[text_column]}''' [\INST] \\n {row[label_column]} </s>""", 1),
            "instruction": [instruction for _ in range(len(data))],
            "input": data[text_column],
            "output": data[label_column],
        },
    )
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lora_data.to_json(output_path, orient="records")
    return lora_data

def main():
    pass

if __name__ == "__main__":
    main()