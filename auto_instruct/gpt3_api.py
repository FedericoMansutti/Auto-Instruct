import json
import tqdm
import os
import random
import openai
from openai import OpenAI
from datetime import datetime
import argparse
import time

'''
async def make_single_request(
    client, prompt, engine, target_length, temperature, top_p,
    frequency_penalty, presence_penalty, stop_sequences, retries, backoff_time
):
    retry_cnt = 0
    while retry_cnt <= retries:
        try:
            response = await asyncio.to_thread(client.chat.completions.create,
                model=engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                n=1,
                logprobs=False
            )
            return response
        except openai.OpenAIError as e:
            print(f"OpenAIError for prompt: {prompt[:50]}... Error: {e}")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
    return None

async def make_requests_async(
        engine, prompts, max_tokens, temperature, top_p,
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    if api_key is not None:
        client = OpenAI(api_key=api_key)
    
    tasks = [
        make_single_request(
            client, prompt, engine, max_tokens, temperature, top_p,
            frequency_penalty, presence_penalty, stop_sequences, retries, 30
        )
        for prompt in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
    

    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": responses[j].choices[0] if responses[j] else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)

        return results
    else:
        data = {
            "prompt": prompts,
            "response": responses[0].choices[0] if responses[j] else None,
            "created_at": str(datetime.now()),
        }
        return [data]
    

def make_requests(*args, **kwargs):
    return asyncio.run(make_requests_async(*args, **kwargs))

'''

def make_requests(
        engine, prompts, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, n=1, retries=3, api_key=None
    ):
    responses = []
    target_length = max_tokens
    if api_key is not None:
        client = OpenAI(api_key=api_key)

    retry_cnt = 0
    backoff_time = 30

    for prompt in prompts:
        while retry_cnt <= retries:
            try:
                response = client.chat.completions.create(
                model=engine,
                messages=[
                    {"role": "user", "content": prompt}
                    ],
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                n=n,
                logprobs=False)

                responses.append(response)
                break

            except openai.OpenAIError as e:
                print(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    target_length = int(target_length * 0.8)
                    print(f"Reducing target length to {target_length}, retrying...")
                else:
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                retry_cnt += 1


    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": ' '.join(choice.message.content for choice in responses[j].choices) if responses[j] else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)

        return results
    else:
        data = {
            "prompt": prompts,
            "response": ' '.join(choice.message.content for choice in responses[0].choices) if responses[0] else None,
            "created_at": str(datetime.now()),
        }
        return [data]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--best_of",
        type=int,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompts = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                    best_of=args.best_of,
                )
                for data in results:
                    fout.write(json.dumps(data) + "\n")