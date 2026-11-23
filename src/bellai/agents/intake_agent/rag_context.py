"""Prépare les requêtes RAG et les indications de thème et de région du dossier."""

from typing import TypedDict

from .schema import CaseFile

REGIONS = ("wallonie", "bruxelles", "flandre", "communaute_germanophone")

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
    themes: list[str]            # thèmes repérés dans le dossier
    niveaux: list[str]           # niveaux à considérer (ue, federal et régions pertinentes)
    langue: str                  # langue de réponse


def construire_contexte_rag(case: CaseFile) -> ContexteRAG:
    themes = [t for t in case.cas_usage if t != "autre"]

    # UE et fédéral s'appliquent partout ; on ajoute les régions pertinentes pour les thèmes.
    niveaux = dict.fromkeys(("ue", "federal"))
    for theme in themes or ["autre"]:
        champ = REGION_PERTINENTE.get(theme)
        region = getattr(case, champ) if champ else "inconnu"
        if region in REGIONS:
            niveaux[region] = None
        elif region == "inconnu":
            # Région inconnue : ne pas exclure de documents régionaux
            niveaux.update(dict.fromkeys(REGIONS))

    requetes = list(case.questions_juridiques) or list(case.faits_cles)

    return ContexteRAG(requetes=requetes, themes=themes, niveaux=list(niveaux), langue=case.langue)
