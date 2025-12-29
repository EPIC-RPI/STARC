import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from evaluation.starc_attention import enable_starc_attention_eval
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--starc", action="store_true", help="Enable STARC attention")
    return parser.parse_args(args)


def get_pred_single(
    model,
    tokenizer,
    json_obj,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):

    prompt = prompt_format.format(**json_obj)

    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if "chatglm3" in model_name:
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = (
            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        )

    if dataset not in [
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "lcc",
        "repobench-p",
    ]:
        prompt = build_chat(tokenizer, prompt, model_name)

    if dataset in ["qasper", "hotpotqa"]:
        q_pos = prompt.rfind("Question:")
    elif dataset in ["multifieldqa_en", "gov_report"]:
        q_pos = prompt.rfind("Now,")
    elif dataset in ["triviaqa"]:
        q_pos = prompt.rfind("Answer the question")
    elif dataset in ["narrativeqa"]:
        q_pos = prompt.rfind("Do not provide")
    else:
        q_pos = -1

    q_pos = max(len(prompt) - 100, q_pos)
    if q_pos != -1 and q_pos is not None:
        question = prompt[q_pos:]
        prompt = prompt[:q_pos]
    else:
        question = ""

    with torch.no_grad():
        input_ids_prompt = tokenizer(prompt, return_tensors="pt").to(device)
        output = model(
            input_ids=input_ids_prompt.input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values

        if question.strip():
            input_ids_question = tokenizer(question, return_tensors="pt").to(device)
            if input_ids_question.input_ids.shape[-1] > 1:
                input_ids_question.input_ids = input_ids_question.input_ids[:, 1:]

            for input_id in input_ids_question.input_ids[0]:
                output = model(
                    input_ids=input_id.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values

        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content = [pred_token_idx.item()]

        for _ in range(max_gen - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content.append(pred_token_idx.item())
            if pred_token_idx.item() == tokenizer.eos_token_id:
                break

    pred = tokenizer.decode(generated_content, skip_special_tokens=True)
    pred = post_process(pred, model_name)

    return {
        "pred": pred,
        "answers": json_obj["answers"],
        "all_classes": json_obj["all_classes"],
        "length": json_obj["length"],
    }


def build_chat(tokenizer, prompt, model_name):
    """Wrap the raw prompt into the model-specific chat format (when applicable)."""
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    """Apply model-specific post-processing to the decoded response text."""
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:
            prompt = build_chat(tokenizer, prompt, model_name)

        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        q_pos = max(len(prompt) - 100, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        if "chatglm3" in model_name:
            input = prompt.to("cuda")
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to("cuda")
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if dataset == "samsum":
            assert False
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                for input_id in q_input.input_ids[0]:
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values

                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() == tokenizer.eos_token_id:
                        break

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    """Set random seeds for reproducibility across torch/numpy/python."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    """Load the model and tokenizer, and optionally enable tuple KV cache and STARC attention."""
    if "llama" in model_name.lower() or "longchat" in model_name.lower():
        enable_tuple_kv_cache_for_llama()
    if "mistral" in model_name.lower():
        enable_tuple_kv_cache_for_mistral()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    if args.starc:

        enable_starc_attention_eval(model, args)

    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]

    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
            "musique",
            "narrativeqa",
            "qmsum"
        ]
    else:
        datasets = [args.task]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    for dataset in datasets:
        if args.e:
            data = load_dataset("zai-org/LongBench", f"{dataset}_e", split="test")

            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            if args.starc:
                out_path = f"pred_e/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("zai-org/LongBench", dataset, split="test")

            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if args.starc:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        with open(out_path, "a", encoding="utf-8") as f:
            for json_obj in tqdm(data):
                pred_info = get_pred_single(
                    model,
                    tokenizer,
                    json_obj,
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    device,
                    model_name,
                )
                json.dump(pred_info, f, ensure_ascii=False)
                f.write("\n")
