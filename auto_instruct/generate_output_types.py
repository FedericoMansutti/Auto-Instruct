import os
import argparse
import json
import re
from tqdm import tqdm
from gpt3_api import make_requests as make_gpt3_requests
from templates.output_template import template_2, template_3

templates = {
    "template_2": template_2,
    "template_3": template_3
}

def process_text(text):
    # Helper function to clean and join text
    lines = re.split(r"\n", text)
    cleaned = [line.strip() for line in lines if line.strip() != ""]
    return " ".join(cleaned)

def post_process_no_clf(results):
    output_types = ""
    input_type = ""
    
    # Extract strategies/output types
        # Extract input type if it appears before strategies
    if "input:" in results.lower():
        input_match = re.search(r"input:(.+?)(?=strategies:|$)", results, re.IGNORECASE | re.DOTALL)
        if input_match:
            input_text = input_match.group(1)
            input_type = process_text(input_text)
    
    # Extract strategies/output types
    if "strategies:" in results.lower():
        output_match = re.search(r"strategies:(.+?)$", results, re.IGNORECASE | re.DOTALL)
        if output_match:
            output_text = output_match.group(1)
            output_types = process_text(output_text)
    return output_types, input_type

def post_process_clf(results):
    output_types = ""
    input_type = ""
    
    # Extract strategies/output types
        # Extract input type if it appears before strategies
    if "input:" in results.lower():
        input_match = re.search(r"input:(.+?)(?=labels:|$)", results, re.IGNORECASE | re.DOTALL)
        if input_match:
            input_text = input_match.group(1)
            input_type = process_text(input_text)
    
    # Extract strategies/output types
    if "labels:" in results.lower():
        output_match = re.search(r"labels:(.+?)$", results, re.IGNORECASE | re.DOTALL)
        if output_match:
            output_text = output_match.group(1)
            output_types = process_text(output_text)
    return output_types, input_type

def load_existing_results(output_path):
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in fin:
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing results")
    return existing_requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key for GPT analysis"
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="GPT engine"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size"
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    # load the instructions
    with open(os.path.join(args.batch_dir, "is_clf_or_not_gpt-4o_template_1.jsonl"), encoding="utf-8") as fin:
        loaded = [json.loads(line) for line in fin]
        non_classification_instructions = [line["instruction"] for line in loaded if line["is_classification"]=="No"]
        classification_instructions = [line["instruction"] for line in loaded if line["is_classification"]=="Yes"]


    output_path = os.path.join(args.batch_dir, f"output_types.jsonl")
    existing_requests = load_existing_results(output_path)

    count_removed_clf = 0
    count_removed_non_clf = 0
    for instruction in existing_requests:
        if instruction in non_classification_instructions:
            non_classification_instructions.remove(instruction)
            count_removed_non_clf += 1
        if instruction in classification_instructions:
            classification_instructions.remove(instruction)
            count_removed_clf += 1

    print(f"Removed {count_removed_non_clf} from non-classification instructions")
    print(f"Removed {count_removed_clf} from classification instructions")

    progress_bar = tqdm(total=len(non_classification_instructions), desc="Processing non-classification")

    with open(output_path, 'a', encoding='utf-8') as fout:
        for batch_idx in range(0, len(non_classification_instructions), args.batch_size):
            batch = non_classification_instructions[batch_idx: batch_idx + args.batch_size]

            
            template = templates["template_2"]
            prompts = [template.format(instruction=instruction) for instruction in batch]

            results = make_gpt3_requests(
                        engine=args.engine,
                        prompts=prompts,
                        max_tokens=1000,
                        temperature=0,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop_sequences=[],
                        n=1,
                        api_key=args.api_key)
            

            for i, result in enumerate(results):
                output_types, input = post_process_no_clf(result["response"])
                data = {
                    "instruction": batch[i],
                    "input": input,
                    "is_classification": "No",
                    "output_type": output_types
                }
                fout.write(json.dumps(data, ensure_ascii=True) + "\n")
                fout.flush()

            progress_bar.update(len(batch))

    progress_bar.close()

    progress_bar = tqdm(total=len(classification_instructions), desc="Processing classification")

    with open(output_path, 'a', encoding='utf-8') as fout:
        for batch_idx in range(0, len(classification_instructions), args.batch_size):
            batch = classification_instructions[batch_idx: batch_idx + args.batch_size]
            template = templates["template_3"]
            prompts = [template.format(instruction=instruction) for instruction in batch]

            results = make_gpt3_requests(
                        engine=args.engine,
                        prompts=prompts,
                        max_tokens=2000,
                        temperature=0,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop_sequences=[],
                        n=1,
                        api_key=args.api_key)
            

            for i, result in enumerate(results):
                labels = process_text(result["response"])
                data = {
                    "instruction": batch[i],
                    "is_classification": "Yes",
                    "labels": labels
                }
                fout.write(json.dumps(data, ensure_ascii=True) + "\n")
                fout.flush()

            progress_bar.update(len(batch))

    progress_bar.close()

