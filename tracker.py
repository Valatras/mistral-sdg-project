"""Logs every LLM call: model, prompt version, tokens, latency, cost and estimated footprint."""
import csv
import time
from datetime import datetime
from pathlib import Path

from ecologits.tracers.utils import llm_impacts

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


def tracked_chat(client, model, messages, prices=None, prompt_version="v0", tag="", **kwargs):
    """Calls client.chat.complete and logs the call. Returns the response unchanged."""
    start = time.perf_counter()
    response = client.chat.complete(model=model, messages=messages, **kwargs)
    latency = time.perf_counter() - start

    usage = response.usage
    impacts = llm_impacts(provider="mistralai", model_name=model,
                          output_token_count=usage.completion_tokens, request_latency=latency)
    price = (prices or {}).get(model)
    cost = None
    if price:
        cost = (usage.prompt_tokens * price[0] + usage.completion_tokens * price[1]) / 1e6

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt_version": prompt_version,
        "tag": tag,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "latency_s": round(latency, 3),
        "cost_eur": cost,
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
    return response


def load_log():
    import pandas as pd
    log = pd.read_csv(LOG_FILE)
    # Logs written by an older version of this module had the cost in USD.
    return log.rename(columns={"cost_usd": "cost_eur"})
