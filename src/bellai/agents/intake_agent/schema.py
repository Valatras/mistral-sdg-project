"""Schéma du dossier structuré (CaseFile) produit par l'intake agent.

C'est le contrat partagé par tous les agents de la boucle : le modifier
impacte les agents suivants.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

Region = Literal["wallonie", "bruxelles", "flandre", "communaute_germanophone", "inconnu"]
CasUsage = Literal["embauche", "licenciement", "chomage", "discrimination", "autre"]
Protection = Literal["delegue_syndical", "candidat_delegue", "grossesse", "credit_temps", "maladie"]


class CaseFile(BaseModel):
    """Faits extraits du récit de l'utilisateur. Null / 'inconnu' quand l'info n'est pas donnée."""

    langue: Literal["fr", "nl", "en", "de"] = Field("fr", description="Langue du récit")
    cas_usage: list[CasUsage] = Field(default_factory=list, description="Domaines juridiques concernés")

    # Situation d'emploi
    statut_emploi: Literal["candidat", "en_poste", "licencie", "demission", "chomeur", "inconnu"] = "inconnu"
    categorie: Literal["ouvrier", "employe", "inconnu"] = Field(
        "inconnu", description="Catégorie de travailleur (ouvrier ou employé)"
    )
    type_contrat: Literal["CDI", "CDD", "travail_nettement_defini", "interim", "inconnu"] = "inconnu"
    commission_paritaire: Optional[str] = Field(
        None, description="Numéro de commission paritaire, ex. '200' ou '124'. Null si non mentionné."
    )
    secteur: Optional[str] = None

    # Région de travail (discrimination, droit régional) ≠ région de domicile (chômage : Forem/Actiris/VDAB)
    region_travail: Region = "inconnu"
    region_domicile: Region = "inconnu"
    code_postal_travail: Optional[str] = Field(None, description="Code postal belge du lieu de travail")
    code_postal_domicile: Optional[str] = Field(None, description="Code postal belge du domicile")

    # Dates (format ISO, relatives converties à partir de la date du jour)
    date_debut_contrat: Optional[date] = None
    date_notification_licenciement: Optional[date] = None
    date_fin_contrat: Optional[date] = None

    salaire_brut_mensuel: Optional[float] = Field(None, description="Salaire brut mensuel en euros")
    age: Optional[int] = None
    protection_speciale: list[Protection] = Field(
        default_factory=list, description="Protections contre le licenciement. Liste vide si aucune."
    )
    motif_invoque: Optional[str] = Field(None, description="Motif de licenciement donné par l'employeur")
    motif_grave: Optional[bool] = Field(None, description="Licenciement pour motif grave ?")

    # Pour l'orchestrateur
    questions_juridiques: list[str] = Field(
        default_factory=list, description="Questions de l'utilisateur reformulées en termes juridiques"
    )
    faits_cles: list[str] = Field(default_factory=list)
    urgence: Literal["faible", "moyenne", "haute"] = Field(
        "moyenne", description="'haute' si un délai légal risque d'expirer"
    )
    etat_emotionnel: Optional[str] = Field(
        None, description="Pour adapter le ton uniquement, jamais pour le raisonnement juridique"
    )
