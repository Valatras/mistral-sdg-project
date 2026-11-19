"""Downloads a frozen snapshot of a few World Bank indicators for the notebooks.

Usage: python make_wb_snapshot.py   (writes wb_snapshot.json next to this file)
The notebooks use the snapshot so that tool outputs, and therefore evaluations, are reproducible.
"""
import json
import urllib.request
from datetime import date
from pathlib import Path

INDICATORS = {
    "SH.H2O.SMDW.ZS": "People using safely managed drinking water services (% of population)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total final energy consumption)",
    "EN.GHG.CO2.PC.CE.AR5": "CO2 emissions per capita (t CO2e/capita, excluding LULUCF)",
    "SP.POP.TOTL": "Population, total",
}
COUNTRIES = ["BEL", "FRA", "DEU", "NLD", "SEN", "KEN", "NGA", "COD", "MAR", "IND", "BGD", "CHN", "BRA", "USA"]
YEARS = "2015:2024"


def fetch(indicator):
    url = (f"https://api.worldbank.org/v2/country/{';'.join(COUNTRIES)}/indicator/{indicator}"
           f"?format=json&date={YEARS}&per_page=1000")
    with urllib.request.urlopen(url, timeout=60) as r:
        meta, rows = json.load(r)
    return rows or []


def main():
    data = {}
    for code in INDICATORS:
        for row in fetch(code):
            if row["value"] is None:
                continue
            country = row["countryiso3code"]
            data.setdefault(country, {"name": row["country"]["value"]}).setdefault(code, {})[row["date"]] = row["value"]
    snapshot = {"source": "World Bank, World Development Indicators API (api.worldbank.org/v2)",
                "retrieved": date.today().isoformat(), "indicators": INDICATORS, "data": data}
    out = Path(__file__).with_name("wb_snapshot.json")
    out.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    print(f"{len(data)} countries written to {out}")


if __name__ == "__main__":
    main()
