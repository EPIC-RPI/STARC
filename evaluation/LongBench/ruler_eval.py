import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from evaluation.starc_attention import enable_starc_attention_eval
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="RULER evaluation script"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
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
        help="Logical model name (must appear in config/model2path.json).",
    )

    parser.add_argument(
        "--model_template_type",
        type=str,
        required=True,
        help=(
            "MODEL_TEMPLATE_TYPE used when generating RULER data "
            "(e.g., llama-3, yi, glm, qwen, phi, ...)."
        ),
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=(
            "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3"
        ),
        help=(
            "Comma-separated list of RULER datasets, e.g. "
            "'ruler/niah_single_1,ruler/qa_1,...'."
        ),
    )

    parser.add_argument(
        "--datalen",
        type=int,
        default=128 * 1024,
        help="Context length (must match the one used to generate the jsonl data).",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="ruler/data",
        help="Root dir containing MODEL_TEMPLATE_TYPE/<datalen>/<task>/validation.jsonl.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Which split to read from the jsonl files.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="If >0, only evaluate the first num_samples examples of each task.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate for each example.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (currently 1 is recommended).",
    )

    parser.add_argument(
        "--starc",
        action="store_true",
        help="Enable STARC attention.",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=None,
        help="STARC token budget; required if --starc.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="STARC chunk size; required if --starc.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ruler",
        help="Logical task name only used to form the STARC log directory name.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="ruler_pred",
        help="Directory to save per-example predictions and scores.",
    )

    return parser.parse_args(args)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path: str, model_name: str, device: torch.device):

    lower_name = model_name.lower()

    if "llama" in lower_name or "longchat" in lower_name or "vicuna" in lower_name:
        enable_tuple_kv_cache_for_llama()
    if "mistral" in lower_name:
        enable_tuple_kv_cache_for_mistral()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = model.eval()

    if args.starc:
        if args.token_budget is None or args.chunk_size is None:
            raise ValueError("--starc requires --token_budget and --chunk_size.")
        save_folder = f"./starc_logs/{args.task}-{args.token_budget}"
        os.makedirs(save_folder, exist_ok=True)
        model.config.starc_save_folder = save_folder
        enable_starc_attention_eval(model, args)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _normalize_answer(s: str) -> str:
    """Normalize text by lowercasing and removing punctuation, articles, and extra whitespace."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _postprocess_pred(s: str) -> str:
    """Strip common special tokens and surrounding whitespace."""
    return s.strip().replace("<|eot_id|>", "")


def metric_multi_number(prediction: str, ground_truth: list) -> float:
    """Compute the fraction of ground-truth numeric strings that appear in the prediction."""
    import re

    prediction = _normalize_answer(_postprocess_pred(prediction))
    prediction_list = re.findall(r"\d+", prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    if not ground_truth:
        return 0.0
    return float(len(hits)) / float(len(ground_truth))


def metric_multi_words(prediction: str, ground_truth: list) -> float:
    """Compute the fraction of ground-truth tokens that appear in the prediction."""
    import re

    prediction = prediction.lower()
    ground_truth = [gt.lower() for gt in ground_truth]
    prediction_list = re.findall(r"\b\w+\b", prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    if not ground_truth:
        return 0.0
    return float(len(hits)) / float(len(ground_truth))


def metric_needle_score(prediction: str, ground_truth: str) -> float:
    """Needle-in-a-haystack score based on normalized prefix match and token containment."""
    prediction = _normalize_answer(_postprocess_pred(prediction))
    ground_truth = _normalize_answer(ground_truth)

    if not ground_truth:
        return 0.0

    min_length = len(ground_truth)
    score = float(prediction[:min_length] == ground_truth[:min_length])

    pred_list = prediction.split()
    score = max(float(ground_truth in pred_list), score)
    return score


def metric_string_match_part(prediction: str, ground_truth: list) -> float:
    """Return 1.0 if any normalized reference answer appears as a substring in the prediction."""
    prediction = _normalize_answer(_postprocess_pred(prediction))
    norm_gts = [_normalize_answer(gt) for gt in ground_truth]
    for gt in norm_gts:
        if gt and gt in prediction:
            return 1.0
    return 0.0


def choose_metric_fn(dataset_name: str):
    """
    Select the scoring function following the dataset family:
      - multiquery/multivalue -> numeric hit rate
      - niah*                -> needle score
      - vt/cwe/fwe            -> token hit rate
      - qa_*                 -> substring match
    """
    name = dataset_name
    if "multiquery" in name or "multivalue" in name:
        return metric_multi_number
    if "niah" in name:
        return metric_needle_score
    if "vt" in name or "cwe" in name or "fwe" in name:
        return metric_multi_words
    if "qa" in name:
        return metric_string_match_part
    raise ValueError(f"Cannot choose metric for dataset_name={dataset_name}")


def load_ruler_jsonl(
    data_root: str,
    model_template_type: str,
    datalen: int,
    task: str,
    subset: str,
):
    """
    Load RULER jsonl produced by the dataset preparation pipeline:
      {data_root}/{model_template_type}/{datalen}/{task}/{subset}.jsonl
    Each line is expected to contain: { "index": int, "input": str, "outputs": [str] }.
    """
    path = Path(data_root) / model_template_type / str(datalen) / task / f"{subset}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"RULER jsonl not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token.item()]

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def evaluate_single_dataset(
    model,
    tokenizer,
    dataset_name: str,
    records: list,
    device: torch.device,
    max_new_tokens: int,
    num_samples: int,
    output_file: Path,
):
    metric_fn = choose_metric_fn(dataset_name)
    scores = []

    if num_samples > 0:
        records = records[:num_samples]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    with output_file.open("w", encoding="utf-8") as f_out:
        for rec in tqdm(records, desc=f"RULER eval: {dataset_name}"):
            idx = rec.get("index", None)
            prompt = rec["input"]
            gt_list = rec["outputs"]

            pred = generate_one(model, tokenizer, prompt, device, max_new_tokens)

            name = dataset_name
            if "multiquery" in name or "multivalue" in name:
                score = metric_fn(pred, gt_list)
            elif "niah" in name:
                score = metric_fn(pred, gt_list[0])
            else:
                score = metric_fn(pred, gt_list)

            scores.append(float(score))

            out_obj = {
                "index": idx,
                "outputs": gt_list,
                "prediction": pred,
                "score": float(score),
            }
            json.dump(out_obj, f_out, ensure_ascii=False)
            f_out.write("\n")

    scores_np = np.array(scores, dtype=np.float32)
    return float(scores_np.mean()) if len(scores_np) > 0 else 0.0, len(scores)


def main():
    global args
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model2path_path = Path("config/model2path.json")
    if not model2path_path.is_file():
        raise FileNotFoundError(
            "config/model2path.json not found; please create it as in LongBench setup."
        )
    model2path = json.loads(model2path_path.read_text())
    if args.model not in model2path:
        raise KeyError(f"Model {args.model} not found in config/model2path.json")
    model_path = model2path[args.model]

    model, tokenizer = load_model_and_tokenizer(model_path, args.model, device)

    dataset_names = [name for name in args.dataset_name.split(",") if name]
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for full_name in dataset_names:
        if not full_name.startswith("ruler/"):
            raise ValueError(f"dataset_name must start with 'ruler/': got {full_name}")
        task = full_name.split("/")[-1]

        records = load_ruler_jsonl(
            data_root=args.data_root,
            model_template_type=args.model_template_type,
            datalen=args.datalen,
            task=task,
            subset=args.subset,
        )

        out_file = Path(args.output_dir) / args.model / f"{task}_{args.datalen}.jsonl"
        avg_score, n = evaluate_single_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=full_name,
            records=records,
            device=device,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            output_file=out_file,
        )

        all_results[full_name] = {"avg_score": avg_score, "num_samples": n}
        print(
            f"[RULER] {full_name} (len={args.datalen}): "
            f"avg_score={avg_score:.4f} over {n} samples"
        )

    print("\n===== RULER summary =====")
    for name, stats in all_results.items():
        print(
            f"{name:30s}  |  acc = {stats['avg_score']:.4f}  "
            f"(n={stats['num_samples']})"
        )


if __name__ == "__main__":
    main()
