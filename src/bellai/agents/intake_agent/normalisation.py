"""Post-traitement déterministe du CaseFile (aucun appel LLM)."""

import re
from datetime import date

from .schema import CaseFile, Region

# Champs indispensables par cas d'usage avant de passer aux agents spécialisés.
REQUIS: dict[str, list[str]] = {
    "licenciement": ["date_debut_contrat", "date_notification_licenciement", "categorie", "salaire_brut_mensuel"],
    "chomage": ["region_domicile", "age", "date_fin_contrat"],
    "embauche": ["region_travail"],
    "discrimination": ["region_travail"],
}

MAX_QUESTIONS = 3

QUESTIONS: dict[str, dict[str, str]] = {
    "date_debut_contrat": {
        "fr": "Depuis quelle date travaillez-vous (ou travailliez-vous) chez cet employeur ?",
        "nl": "Sinds welke datum werkt (of werkte) u bij deze werkgever?",
    },
    "date_notification_licenciement": {
        "fr": "À quelle date votre licenciement vous a-t-il été notifié ?",
        "nl": "Op welke datum werd uw ontslag u betekend?",
    },
    "categorie": {
        "fr": "Êtes-vous ouvrier ou employé ?",
        "nl": "Bent u arbeider of bediende?",
    },
    "salaire_brut_mensuel": {
        "fr": "Quel est votre salaire brut mensuel (approximatif) ?",
        "nl": "Wat is uw (geschat) bruto maandloon?",
    },
    "region_domicile": {
        "fr": "Dans quelle commune (ou code postal) habitez-vous ?",
        "nl": "In welke gemeente (of postcode) woont u?",
    },
    "region_travail": {
        "fr": "Dans quelle commune (ou code postal) se situe le lieu de travail ?",
        "nl": "In welke gemeente (of postcode) bevindt zich de werkplaats?",
    },
    "age": {
        "fr": "Quel âge avez-vous ?",
        "nl": "Hoe oud bent u?",
    },
    "date_fin_contrat": {
        "fr": "À quelle date votre contrat a-t-il pris (ou prendra-t-il) fin ?",
        "nl": "Op welke datum eindigde (of eindigt) uw contract?",
    },
}


def region_depuis_code_postal(code_postal: str | None) -> Region:
    """Déduit la région à partir d'un code postal belge."""
    if not code_postal:
        return "inconnu"
    m = re.search(r"\b(\d{4})\b", code_postal)
    if not m:
        return "inconnu"
    cp = int(m.group(1))
    if 1000 <= cp <= 1299:
        return "bruxelles"
    if 4700 <= cp <= 4799:  # communes germanophones (Eupen, Saint-Vith, ...)
        return "communaute_germanophone"
    if 1300 <= cp <= 1499 or 4000 <= cp <= 7999:
        return "wallonie"
    if 1500 <= cp <= 3999 or 8000 <= cp <= 9999:
        return "flandre"
    return "inconnu"


def normaliser(case: CaseFile, today: date | None = None) -> CaseFile:
    """Corrige et complète le CaseFile à partir de règles fixes."""
    today = today or date.today()
    data = case.model_copy(deep=True)

    # Code postal → région (prioritaire sur la déduction du LLM)
    for cp_field, region_field in (("code_postal_travail", "region_travail"),
                                   ("code_postal_domicile", "region_domicile")):
        region = region_depuis_code_postal(getattr(data, cp_field))
        if region != "inconnu":
            setattr(data, region_field, region)

    # Commission paritaire : garder uniquement le numéro (ex. "CP 200" → "200", "124.00" → "124")
    if data.commission_paritaire:
        m = re.search(r"\d{3}(?:\.\d{2})?", data.commission_paritaire)
        cp = m.group(0) if m else None
        if cp and cp.endswith(".00"):
            cp = cp[:-3]
        data.commission_paritaire = cp

    # Une date de début de contrat dans le futur est une erreur d'extraction
    if data.date_debut_contrat and data.date_debut_contrat > today:
        data.date_debut_contrat = None

    if data.age is not None and not 14 <= data.age <= 100:
        data.age = None

    return data


def _manquant(case: CaseFile, champ: str) -> bool:
    valeur = getattr(case, champ)
    return valeur is None or valeur == "inconnu"


def champs_requis_manquants(case: CaseFile) -> list[str]:
    manquants: list[str] = []
    for usage in case.cas_usage:
        for champ in REQUIS.get(usage, []):
            if champ not in manquants and _manquant(case, champ):
                manquants.append(champ)
    return manquants


def rediger_questions(manquants: list[str], langue: str) -> list[str]:
    """Questions de clarification (gabarits fixes, max. MAX_QUESTIONS). FR par défaut."""
    langue = langue if langue in ("fr", "nl") else "fr"
    return [QUESTIONS[c][langue] for c in manquants[:MAX_QUESTIONS] if c in QUESTIONS]


def anciennete_mois(case: CaseFile, today: date | None = None) -> int | None:
    """Ancienneté en mois complets, de la date de début à la notification (ou à aujourd'hui)."""
    if not case.date_debut_contrat:
        return None
    fin = case.date_notification_licenciement or today or date.today()
    mois = (fin.year - case.date_debut_contrat.year) * 12 + fin.month - case.date_debut_contrat.month
    if fin.day < case.date_debut_contrat.day:
        mois -= 1
    return max(mois, 0)
