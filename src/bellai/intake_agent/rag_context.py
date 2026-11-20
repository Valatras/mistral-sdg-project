"""Traduit le CaseFile en requêtes et filtres de métadonnées pour la recherche RAG.

Les valeurs (`themes`, `niveaux`) doivent correspondre aux métadonnées posées
à l'ingestion des documents de `fichier_rag/`.
"""

from typing import TypedDict

from .schema import CaseFile

# Région du CaseFile → niveau de norme dans le corpus
NIVEAU_PAR_REGION = {
    "wallonie": "wallonie",
    "bruxelles": "bruxelles",
    "flandre": "flandre",
    "communaute_germanophone": "communaute_germanophone",
}

# Thème → quelle région compte : lieu de travail ou domicile
#   - chômage : le service régional compétent (Forem / Actiris / VDAB / ADG) dépend du domicile
#   - embauche / discrimination : le décret applicable dépend du lieu de travail
#   - licenciement : surtout fédéral, mais le lieu de travail sert pour les règles régionales
REGION_PERTINENTE = {
    "chomage": "region_domicile",
    "embauche": "region_travail",
    "discrimination": "region_travail",
    "licenciement": "region_travail",
}


class ContexteRAG(TypedDict):
    requetes: list[str]          # textes à chercher (embeddings)
    themes: list[str]            # filtre metadata "theme"
    niveaux: list[str]           # filtre metadata "niveau" (ue, federal + régions pertinentes)
    langue: str                  # langue de réponse


def construire_contexte_rag(case: CaseFile) -> ContexteRAG:
    themes = [t for t in case.cas_usage if t != "autre"]

    # UE et fédéral s'appliquent partout ; on ajoute les régions pertinentes pour les thèmes.
    niveaux = ["ue", "federal"]
    for theme in themes or ["autre"]:
        champ = REGION_PERTINENTE.get(theme)
        region = getattr(case, champ) if champ else "inconnu"
        if region in NIVEAU_PAR_REGION:
            niveau = NIVEAU_PAR_REGION[region]
            if niveau not in niveaux:
                niveaux.append(niveau)
        elif region == "inconnu":
            # Région inconnue : ne pas exclure de documents régionaux
            for niveau in NIVEAU_PAR_REGION.values():
                if niveau not in niveaux:
                    niveaux.append(niveau)

    requetes = list(case.questions_juridiques) or list(case.faits_cles)

    return ContexteRAG(requetes=requetes, themes=themes, niveaux=niveaux, langue=case.langue)
