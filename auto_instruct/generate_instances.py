import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template import output_first_template_for_clf, method_first_template_for_gen


random.seed(42)


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

def process_labels(result):
    return [line.strip() for line in result.split(",")]

def process_strategies(result):
     return [line.strip() for line in result.split(".") if line.strip()]

def post_process(result):
    return result.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, "output_types.jsonl"), encoding="utf-8") as fin:
        loaded = [json.loads(line) for line in fin]
        non_classification_instructions = [line for line in loaded if line["is_classification"]=="No"]
        classification_instructions = [line for line in loaded if line["is_classification"]=="Yes"]
        labels_dict = {line["instruction"]: process_labels(line["labels"]) for line in classification_instructions}
        strategies_dict = {line["instruction"]: process_strategies(line["output_type"]) for line in non_classification_instructions}


    output_path = os.path.join(args.batch_dir, args.output_file)

    existing_requests = load_existing_results(output_path)

    count_removed_clf = 0
    count_removed_non_clf = 0
    for instruction in existing_requests:
        # Find and remove matching instructions by comparing instruction strings
        non_clf_match = next((item for item in non_classification_instructions if item["instruction"] == instruction), None)
        if non_clf_match:
            non_classification_instructions.remove(non_clf_match)
            count_removed_non_clf += 1
            
        clf_match = next((item for item in classification_instructions if item["instruction"] == instruction), None)
        if clf_match:
            classification_instructions.remove(clf_match)
            count_removed_clf += 1


    print(f"Removed {count_removed_non_clf} from non-classification instructions")
    print(f"Removed {count_removed_clf} from classification instructions")

    progress_bar = tqdm.tqdm(total=len(classification_instructions), desc="Processing classification")

    with open(output_path, 'a', encoding='utf-8') as fout:
        for batch_idx in range(0, len(classification_instructions), args.batch_size):
            batch = classification_instructions[batch_idx: batch_idx + args.batch_size]
            template = output_first_template_for_clf
            
            prompts = []
            prompt_metadata = []

            for instruction in batch:
                for label in labels_dict[instruction["instruction"]]:
                    prompts.append(template.format(instruction=instruction["instruction"], class_labels=label))
                    prompt_metadata.append({"instruction": instruction["instruction"], "label": label})
                    


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
                input = post_process(result["response"])
                data = {
                    "instruction": prompt_metadata[i]["instruction"],
                    "input": input,
                    "is_classification": "Yes",
                    "output": prompt_metadata[i]["label"]
                }
                fout.write(json.dumps(data, ensure_ascii=True) + "\n")
                fout.flush()

            progress_bar.update(len(batch))

    progress_bar.close()

    progress_bar = tqdm.tqdm(total=len(non_classification_instructions), desc="Processing non-classification")

    with open(output_path, 'a', encoding='utf-8') as fout:
        for batch_idx in range(0, len(non_classification_instructions), args.batch_size):
            batch = non_classification_instructions[batch_idx: batch_idx + args.batch_size]
            template = method_first_template_for_gen
            
            prompts = []
            prompt_metadata = []

            for instruction in batch:
                for strategy in strategies_dict[instruction["instruction"]]:
                    prompts.append(template.format(instruction=instruction["instruction"], input=instruction["input"], strategy=strategy))
                    prompt_metadata.append({"instruction": instruction["instruction"], "input": instruction["input"], "strategy": strategy})
                    

            results = make_gpt3_requests(
                        engine=args.engine,
                        prompts=prompts,
                        max_tokens=200,
                        temperature=0,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=1.5,
                        stop_sequences=[],
                        n=1,
                        api_key=args.api_key)
            

            for i, result in enumerate(results):
                output = post_process(result["response"])
                data = {
                    "instruction": prompt_metadata[i]["instruction"],
                    "input": prompt_metadata[i]["input"],
                    "is_classification": "No",
                    "strategy": prompt_metadata[i]["strategy"],
                    "output": output
                }
                fout.write(json.dumps(data, ensure_ascii=True) + "\n")
                fout.flush()

            progress_bar.update(len(batch))
