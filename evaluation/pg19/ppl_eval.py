import os
import sys
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral


device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--fixed-length", type=int)
parser.add_argument("--max_tokens", type=int, default=8192)
parser.add_argument("--tokens-step", type=int)
parser.add_argument("--length-step", type=int, default=128)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--start_idx", type=int, default=0, help="Starting index of the token")
parser.add_argument("--num_eval_tokens", type=int, default=None)
parser.add_argument("--starc", action="store_true", help="Enable STAR-C attention")
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=16)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)


def load(model_name_or_path):

    print(f"Loading model from {model_name_or_path} ...")

    lower_name = model_name_or_path.lower()
    if "llama" in lower_name or "longchat" in lower_name:
        enable_tuple_kv_cache_for_llama()
    if "mistral" in lower_name:
        enable_tuple_kv_cache_for_mistral()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()
    return model, tokenizer


args = parser.parse_args()

data = load_dataset("emozilla/pg19-test", split="test")
model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.starc:
    print("Enable STARC attention")
    from evaluation.starc_attention import enable_starc_attention_eval
    enable_starc_attention_eval(model, args)

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log_PG19.txt", "w")

num_eval_tokens = 0
for text in data["text"]:
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")

    start_idx = args.start_idx
    if start_idx >= seq_len:
        print(f"Start index {start_idx} exceeds sequence length {seq_len}.")
        break

    context_input_ids = encodings.input_ids[:, : start_idx + 1].to(device)

    # Prefill the cache using the context tokens.
    with torch.no_grad():
        outputs = model(
            context_input_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    # Evaluate token-by-token negative log-likelihood from start_idx onward.
    pbar = tqdm(range(start_idx, seq_len - 1), desc="Processing tokens")
    for idx in pbar:
        input_ids = encodings.input_ids[:, idx: idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1: idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

        nlls.append(neg_log_likelihood)
        mean_nll = torch.stack(nlls).mean()
        overall_ppl = torch.exp(mean_nll).item()

        pbar.set_description(f"nll: {neg_log_likelihood.item():.2f}, ppl: {overall_ppl:.2f}")
        print(overall_ppl, file=f, flush=True)

        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

    # Stop after processing the first eligible sequence.
    print(f"Processed a sequence with start_idx {start_idx}. Stopping further processing for this text.")
    break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl_PG19.txt", "w") as f_out:
    f_out.write(f"{ppl.item()}\n")
