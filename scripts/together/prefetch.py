"""Download every model and dataset into the shared HF cache (run once, by worker w0)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

sys.path.insert(0, "/work/wq/src")
from common import DATASETS, JUDGE_MODEL, MODELS, NLI_MODEL, load_questions

ST = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]
jobs = [(r, ST) for r in set(MODELS.values()) | {JUDGE_MODEL}]
jobs.append((NLI_MODEL, ["*.json", "*.txt", "pytorch_model.bin"]))
jobs.append(("sentence-transformers/all-MiniLM-L6-v2", ["*.json", "*.txt", "*.safetensors", "1_Pooling/*"]))


def dl(job):
    repo, pats = job
    for i in range(5):
        try:
            p = snapshot_download(repo, allow_patterns=pats, max_workers=16)
            print("ok", repo, p, flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print("retry", repo, e, flush=True)
    raise SystemExit(f"failed {repo}")


with ThreadPoolExecutor(len(jobs) + 3) as ex:
    futs = [ex.submit(dl, j) for j in jobs] + [ex.submit(load_questions, d) for d in DATASETS]
    for f in futs:
        f.result()
print("PREFETCH DONE", flush=True)
