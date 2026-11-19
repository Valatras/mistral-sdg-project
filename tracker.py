"""Logs every LLM call: model, prompt version, step, tokens, latency, cost and estimated footprint."""
import csv
import time
from datetime import datetime
from pathlib import Path

from ecologits.tracers.utils import llm_impacts
from langchain_core.callbacks import BaseCallbackHandler

LOG_FILE = Path("calls_log.csv")
FIELDS = ["timestamp", "model", "prompt_version", "tag", "prompt_tokens", "completion_tokens",
          "latency_s", "cost_eur", "energy_Wh", "gwp_gCO2e", "water_mL"]


def _check_header():
    # If the log was written by an older version of this module (other columns), keep it aside.
    if LOG_FILE.exists():
        with LOG_FILE.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != FIELDS:
            LOG_FILE.rename(LOG_FILE.with_name(f"calls_log_old_{datetime.now():%Y%m%d_%H%M%S}.csv"))


def _value(x):
    return (x.min + x.max) / 2 if hasattr(x, "min") else x


def log_call(model, prompt_tokens, completion_tokens, latency, prices=None, prompt_version="v0", tag=""):
    impacts = llm_impacts(provider="mistralai", model_name=model,
                          output_token_count=completion_tokens, request_latency=latency)
    price = (prices or {}).get(model)
    cost = (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6 if price else None
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model, "prompt_version": prompt_version, "tag": tag,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "latency_s": round(latency, 3), "cost_eur": cost,
        "energy_Wh": _value(impacts.energy.value) * 1000,
        "gwp_gCO2e": _value(impacts.gwp.value) * 1000,
        "water_mL": _value(impacts.wcf.value) * 1000,
    }
    _check_header()
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def tracked_chat(client, model, messages, prices=None, prompt_version="v0", tag="", **kwargs):
    """Mistral SDK: calls client.chat.complete and logs the call (notebook 2)."""
    start = time.perf_counter()
    response = client.chat.complete(model=model, messages=messages, **kwargs)
    usage = response.usage
    log_call(model, usage.prompt_tokens, usage.completion_tokens, time.perf_counter() - start,
             prices, prompt_version, tag)
    return response


class TokenTracker(BaseCallbackHandler):
    """LangChain: logs every chat model call of a chain. The step is read from tags 'step:<name>'."""

    def __init__(self, prices=None, prompt_version="v0"):
        self.prices, self.prompt_version = prices, prompt_version
        self._starts = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, **kwargs):
        model = (kwargs.get("invocation_params") or {}).get("model", "unknown")
        steps = [t[5:] for t in (tags or []) if t.startswith("step:")]
        self._starts[run_id] = (time.perf_counter(), model, steps[-1] if steps else "")

    def on_llm_end(self, response, *, run_id, **kwargs):
        start, model, step = self._starts.pop(run_id, (time.perf_counter(), "unknown", ""))
        output = response.llm_output or {}
        model = output.get("model_name", model)
        usage = output.get("token_usage") or {}
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if prompt_tokens is None:  # fall back on the message metadata
            meta = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
            prompt_tokens, completion_tokens = meta.get("input_tokens", 0), meta.get("output_tokens", 0)
        log_call(model, prompt_tokens, completion_tokens, time.perf_counter() - start,
                 self.prices, self.prompt_version, step)


def load_log():
    import pandas as pd
    log = pd.read_csv(LOG_FILE)
    # Logs written by an older version of this module had the cost in USD.
    return log.rename(columns={"cost_usd": "cost_eur"})
