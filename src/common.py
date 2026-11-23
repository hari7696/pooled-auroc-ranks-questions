"""Shared configuration and helpers for the within-question UQ study."""
from __future__ import annotations

import json
import os
import random
import re
import string
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("WQ_OUT", ROOT / "outputs"))

MODELS = {
    # key: (hf repo, load kwargs hint)
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
}
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NLI_MODEL = "microsoft/deberta-large-mnli"

DATASETS = ["triviaqa", "nq", "popqa"]
N_QUESTIONS = 2000
N_SAMPLES = 10
MAX_NEW_TOKENS = 24
SEED = 0

SYSTEM_PROMPT = "Answer the question with a short answer only (a few words). Do not explain."
FEWSHOT = [
    ("What is the capital of Australia?", "Canberra"),
    ("Who wrote the novel Pride and Prejudice?", "Jane Austen"),
    ("In what year did the Berlin Wall fall?", "1989"),
]


def run_dir(model: str, dataset: str) -> Path:
    d = OUT / f"{model}__{dataset}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_questions(name: str, n: int = N_QUESTIONS, seed: int = SEED) -> list[dict]:
    """Return a fixed random subset of n questions as dicts {qid, question, golds}."""
    from datasets import load_dataset

    if name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        rows = [
            {"qid": r["question_id"], "question": r["question"],
             "golds": sorted(set([r["answer"]["value"]] + r["answer"]["aliases"]))}
            for r in ds
        ]
    elif name == "nq":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        rows = [{"qid": f"nq{i}", "question": r["question"].strip() + "?" if not r["question"].strip().endswith("?") else r["question"],
                 "golds": list(r["answer"])} for i, r in enumerate(ds)]
    elif name == "popqa":
        ds = load_dataset("akariasai/PopQA", split="test")
        rows = [{"qid": f"pop{r['id']}", "question": r["question"],
                 "golds": json.loads(r["possible_answers"])} for r in ds]
    else:
        raise ValueError(name)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def build_messages(question: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in FEWSHOT:
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs


def ptrue_messages(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a careful fact checker."},
        {"role": "user", "content": (
            f"Question: {question}\nProposed answer: {answer}\n"
            "Is the proposed answer correct? Reply with only True or False.")},
    ]


def judge_messages(question: str, golds: list[str], answer: str) -> list[dict]:
    refs = "; ".join(golds[:12])
    return [
        {"role": "system", "content": "You grade short answers to factual questions."},
        {"role": "user", "content": (
            f"Question: {question}\nReference answer(s): {refs}\nCandidate answer: {answer}\n\n"
            "Does the candidate answer refer to the same thing as one of the reference answers? "
            "Minor differences in wording, spelling or detail are fine; a different entity, number "
            "or date is not. An empty or evasive answer is incorrect. Reply with only Yes or No.")},
    ]


# --- answer normalisation (SQuAD / TriviaQA style) ---------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation) | {"‘", "’", "´", "`", "“", "”"}


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def alias_match(answer: str, golds: list[str]) -> bool:
    """Lenient match: normalised equality or gold contained in answer as a token span."""
    a = normalize(answer)
    if not a:
        return False
    a_pad = f" {a} "
    for g in golds:
        g = normalize(g)
        if not g:
            continue
        if a == g or f" {g} " in a_pad:
            return True
    return False


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_lm(repo: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="cuda:0")
    if "bnb-4bit" not in repo:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    model.eval()
    return tok, model
