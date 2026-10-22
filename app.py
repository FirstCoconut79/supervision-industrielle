"""
Application de supervision industrielle - Maintenance prédictive
Bachelor MSIC - Python / Streamlit / Pandas / matplotlib

Lancement :  python -m streamlit run app.py
"""

from __future__ import annotations

import math
import socket
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.ticker import MaxNLocator
from plotly.subplots import make_subplots

###############################################################################
# PARTIE 1 - CONFIGURATION, CONSTANTES ET CHARGEMENT DES DONNÉES
###############################################################################

# --- Configuration de la page ----------------------------------------------
# (st.set_page_config doit être la première commande Streamlit du script)
st.set_page_config(
    page_title="Supervision industrielle",
    page_icon="🛠️",
    layout="wide",
)


# --- Constantes : fichier et noms des colonnes du dataset ------------------
CHEMIN_DONNEES = Path(__file__).parent / "data" / "train_FD001.csv"

COL_MACHINE = "unit_number"  # identifiant de la machine
COL_TEMPS = "time_cycle"     # instant de la mesure (numéro de cycle)



# --- Chargement et analyse des données -------------------------------------
@st.cache_data
def charger_donnees() -> pd.DataFrame:
    """Charge le fichier CSV dans un DataFrame pandas.

    @st.cache_data garde le résultat en mémoire : le fichier n'est lu
    qu'une seule fois, même quand l'utilisateur change de section.
    """
    return pd.read_csv(CHEMIN_DONNEES)


def identifier_colonnes(df: pd.DataFrame) -> dict:
    """Repère les différents types de colonnes du dataset."""
    reglages = [c for c in df.columns if c.startswith("op_setting")]
    capteurs = [c for c in df.columns if c.startswith("sensor")]
    # Un capteur constant (une seule valeur) ne donne aucune information :
    # inutile de le proposer à la surveillance.
    capteurs_utiles = [c for c in capteurs if df[c].nunique() > 1]
    reglages_utiles = [c for c in reglages if df[c].nunique() > 1]
    return {
        "reglages": reglages,
        "capteurs": capteurs,
        "capteurs_utiles": capteurs_utiles,
        # Toutes les variables qui varient (capteurs + réglages), pour les requêtes
        "variables_utiles": capteurs_utiles + reglages_utiles,
    }


def nb_decimales(serie: pd.Series) -> int:
    """Choisit le nombre de décimales à afficher selon l'ordre de grandeur.

    Un capteur de valeur ~500 s'affiche bien avec 3 décimales, mais un réglage
    de valeur ~0.0002 afficherait "0.000" : on adapte donc la précision.
    """
    ecart = serie.std()
    if pd.isna(ecart) or ecart == 0:
        return 3
    return min(8, max(3, 2 - math.floor(math.log10(ecart))))


# Opérations de calcul disponibles : nom affiché -> méthode pandas correspondante
OPERATIONS = {
    "Moyenne": "mean",
    "Minimum": "min",
    "Maximum": "max",
    "Médiane": "median",
    "Écart-type": "std",
}

###############################################################################
# PARTIE 2 - DICTIONNAIRE DES CAPTEURS (symbole, description, unité)
###############################################################################

# Le CSV ne contient aucune unité : elles viennent de la documentation du jeu
# de données NASA C-MAPSS. "" = grandeur sans unité (rapport).
# °R = degré Rankine, psia = livre par pouce carré, tr/min = tours par minute,
# pps/psi et lbm/s = unités de débit.
CAPTEURS_INFO = {
    "sensor_1": ("T2", "Température entrée soufflante", "°R"),
    "sensor_2": ("T24", "Température sortie compresseur BP", "°R"),
    "sensor_3": ("T30", "Température sortie compresseur HP", "°R"),
    "sensor_4": ("T50", "Température sortie turbine BP", "°R"),
    "sensor_5": ("P2", "Pression entrée soufflante", "psia"),
    "sensor_6": ("P15", "Pression totale conduit de dérivation", "psia"),
    "sensor_7": ("P30", "Pression totale sortie compresseur HP", "psia"),
    "sensor_8": ("Nf", "Vitesse physique soufflante", "tr/min"),
    "sensor_9": ("Nc", "Vitesse physique cœur", "tr/min"),
    "sensor_10": ("epr", "Rapport de pression moteur (P50/P2)", ""),
    "sensor_11": ("Ps30", "Pression statique sortie compresseur HP", "psia"),
    "sensor_12": ("phi", "Débit carburant / Ps30", "pps/psi"),
    "sensor_13": ("NRf", "Vitesse corrigée soufflante", "tr/min"),
    "sensor_14": ("NRc", "Vitesse corrigée cœur", "tr/min"),
    "sensor_15": ("BPR", "Taux de dilution", ""),
    "sensor_16": ("farB", "Rapport carburant/air chambre de combustion", ""),
    "sensor_17": ("htBleed", "Enthalpie de prélèvement", ""),
    "sensor_18": ("Nf_dmd", "Vitesse soufflante demandée", "tr/min"),
    "sensor_19": ("PCNfR_dmd", "Vitesse corrigée soufflante demandée", "tr/min"),
    "sensor_20": ("W31", "Prélèvement refroidissement turbine HP", "lbm/s"),
    "sensor_21": ("W32", "Prélèvement refroidissement turbine BP", "lbm/s"),
}


def description(variable: str) -> str:
    """Description lisible de la variable (sans l'unité)."""
    if variable in CAPTEURS_INFO:
        return CAPTEURS_INFO[variable][1]
    return "Réglage de fonctionnement"  # op_setting_1, 2, 3


def unite(variable: str) -> str:
    """Unité de la variable ("" si grandeur sans unité ou unité inconnue)."""
    return CAPTEURS_INFO[variable][2] if variable in CAPTEURS_INFO else ""


def libelle(variable: str) -> str:
    """Texte affiché dans les menus : 'sensor_2 - Température ... (°R)'."""
    u = unite(variable)
    return f"{variable} - {description(variable)}" + (f" ({u})" if u else "")


def nom_axe(variable: str) -> str:
    """Texte pour un axe de graphique : 'Température ... (°R)'."""
    u = unite(variable)
    return description(variable) + (f" ({u})" if u else "")


def avec_unite(texte: str, variable: str) -> str:
    """Ajoute l'unité après une valeur : '644.210' -> '644.210 °R'."""
    return f"{texte} {unite(variable)}".strip()


###############################################################################
# PARTIE 3 - FONCTIONS DE CALCUL ET DE DÉTECTION D'ANOMALIES
# Fonctions réutilisées par plusieurs fenêtres (Défauts, Alertes...).
###############################################################################
def filtrer_machine(df: pd.DataFrame, machine: int) -> pd.DataFrame:
    """Ne garde que les mesures de la machine choisie, triées dans le temps.

    Fonction réutilisée dans plusieurs sections (Visualisation, Indicateurs...).
    """
    return df[df[COL_MACHINE] == machine].sort_values(COL_TEMPS)


def detecter_anomalies(
    donnees: pd.DataFrame,
    variable: str,
    seuil_haut: float | None = None,
    seuil_bas: float | None = None,
    limite_variation: float | None = None,
) -> pd.DataFrame:
    """Détecte les situations anormales d'une variable pour UNE machine.

    Trois détections possibles (None = détection désactivée) :
      - seuil haut : la valeur dépasse `seuil_haut` ;
      - seuil bas : la valeur passe sous `seuil_bas` ;
      - variation brutale : l'écart entre deux cycles consécutifs dépasse
        `limite_variation` (en valeur absolue).

    `donnees` doit contenir une seule machine, triée par cycle.
    Retourne un tableau avec une ligne par anomalie : machine, cycle, type,
    valeur observée, seuil, message d'alerte.
    """
    d = nb_decimales(donnees[variable])

    def fmt(x):  # nombre + unité, ex. "644.210 °R"
        return avec_unite(f"{x:.{d}f}", variable)

    base = donnees[[COL_MACHINE, COL_TEMPS, variable]].rename(columns={variable: "valeur"})
    morceaux = []

    if seuil_haut is not None:
        s = base[base["valeur"] > seuil_haut].copy()
        s["type"] = "Seuil haut"
        s["seuil"] = seuil_haut
        s["message"] = [
            f"Valeur {fmt(v)} supérieure au seuil haut ({fmt(seuil_haut)})"
            for v in s["valeur"]
        ]
        morceaux.append(s)

    if seuil_bas is not None:
        s = base[base["valeur"] < seuil_bas].copy()
        s["type"] = "Seuil bas"
        s["seuil"] = seuil_bas
        s["message"] = [
            f"Valeur {fmt(v)} inférieure au seuil bas ({fmt(seuil_bas)})"
            for v in s["valeur"]
        ]
        morceaux.append(s)

    if limite_variation is not None:
        variation = base["valeur"].diff()          # écart avec le cycle précédent
        cycle_avant = base[COL_TEMPS].shift(1)
        masque = variation.abs() > limite_variation  # le 1er cycle (NaN) est ignoré
        s = base[masque].copy()
        s["type"] = "Variation brutale"
        s["seuil"] = limite_variation
        s["message"] = [
            f"Variation brutale de {avec_unite(f'{dv:+.{d}f}', variable)} entre les "
            f"cycles {int(c0)} et {int(c1)}"
            for dv, c0, c1 in zip(variation[masque], cycle_avant[masque], s[COL_TEMPS])
        ]
        morceaux.append(s)

    colonnes_sortie = [COL_MACHINE, COL_TEMPS, "type", "valeur", "seuil", "message"]
    if not morceaux:
        return pd.DataFrame(columns=colonnes_sortie)
    resultat = pd.concat(morceaux)[colonnes_sortie]
    return resultat.sort_values([COL_TEMPS, "type"]).reset_index(drop=True)


def afficher_tableau_anomalies(anomalies: pd.DataFrame, variable: str, d: int) -> None:
    """Affiche le détail des anomalies (cycle, type, valeur, seuil, message).

    Utilisé dans les sections Défauts et Alertes. `d` = nombre de décimales.
    """
    suffixe = f" ({unite(variable)})" if unite(variable) else ""
    detail = anomalies.drop(columns=[COL_MACHINE]).rename(
        columns={
            COL_TEMPS: "Cycle",
            "type": "Type",
            "valeur": f"Valeur{suffixe}",
            "seuil": f"Seuil{suffixe}",
            "message": "Message d'alerte",
        }
    )
    st.dataframe(
        detail,
        hide_index=True,
        width="stretch",
        column_config={
            "Cycle": st.column_config.NumberColumn(width="small"),
            "Type": st.column_config.TextColumn(width="medium"),
            f"Valeur{suffixe}": st.column_config.NumberColumn(format=f"%.{d}f", width="small"),
            f"Seuil{suffixe}": st.column_config.NumberColumn(format=f"%.{d}f", width="small"),
            "Message d'alerte": st.column_config.TextColumn(width="large"),
        },
    )


###############################################################################
# PARTIE 4 - FONCTIONS DE GRAPHIQUES
# - Plotly : courbes interactives (zoom, déplacement, valeurs au survol)
# - matplotlib : version statique (Visualisation) + histogramme
###############################################################################
COULEUR_MESURE = "#1f77b4"
COULEUR_HAUT = "#d62728"
COULEUR_BAS = "#7b4fc4"
COULEUR_VARIATION = "#ff7f0e"


def figure_temporelle(
    donnees: pd.DataFrame,
    variable: str,
    titre: str,
    anomalies: pd.DataFrame | None = None,
    seuil_haut: float | None = None,
    seuil_bas: float | None = None,
    zone_reference: tuple[int, int] | None = None,
) -> go.Figure:
    """Courbe interactive d'une variable, avec seuils, anomalies et zone de référence."""
    d = nb_decimales(donnees[variable])
    u = unite(variable)

    # Message d'alerte à ajouter dans l'infobulle des cycles en anomalie
    messages = [""] * len(donnees)
    if anomalies is not None and not anomalies.empty:
        par_cycle = anomalies.groupby(COL_TEMPS)["message"].apply("<br>".join)
        messages = [
            f"<br>⚠ {par_cycle[c]}" if c in par_cycle.index else ""
            for c in donnees[COL_TEMPS]
        ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=donnees[COL_TEMPS],
            y=donnees[variable],
            mode="lines",
            name="Mesure",
            line=dict(color=COULEUR_MESURE, width=1.6),
            customdata=messages,
            hovertemplate=(
                f"Cycle %{{x}}<br>{variable} : %{{y:.{d}f}} {u}%{{customdata}}<extra></extra>"
            ),
        )
    )
    if zone_reference is not None:
        fig.add_vrect(
            x0=zone_reference[0], x1=zone_reference[1], fillcolor="#999999",
            opacity=0.18, line_width=0, annotation_text="Référence",
            annotation_position="top left",
        )
    if seuil_haut is not None:
        fig.add_hline(
            y=seuil_haut, line=dict(color=COULEUR_HAUT, dash="dash", width=1.5),
            annotation_text="Seuil haut", annotation_position="top left",
        )
    if seuil_bas is not None:
        fig.add_hline(
            y=seuil_bas, line=dict(color=COULEUR_BAS, dash="dash", width=1.5),
            annotation_text="Seuil bas", annotation_position="bottom left",
        )
    if anomalies is not None and not anomalies.empty:
        for type_anomalie, couleur, symbole in [
            ("Seuil haut", COULEUR_HAUT, "circle"),
            ("Seuil bas", COULEUR_BAS, "circle"),
            ("Variation brutale", COULEUR_VARIATION, "diamond"),
        ]:
            a = anomalies[anomalies["type"] == type_anomalie]
            if a.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=a[COL_TEMPS], y=a["valeur"], mode="markers", name=type_anomalie,
                    marker=dict(color=couleur, size=9, symbol=symbole,
                                line=dict(color="white", width=1)),
                    customdata=a["message"],  # le message s'affiche au survol
                    hovertemplate="Cycle %{x}<br>%{customdata}<extra></extra>",
                )
            )
    fig.update_layout(
        title=titre, xaxis_title="Cycle", yaxis_title=nom_axe(variable),
        template="plotly_white", height=430, hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig


def figure_multi(donnees: pd.DataFrame, capteurs: list[str], superposer: bool) -> go.Figure:
    """Plusieurs capteurs : un graphique par capteur, ou une superposition normalisée."""
    x = donnees[COL_TEMPS]
    if superposer:
        # Normalisation min-max : chaque capteur est ramené entre 0 et 1.
        # Seule la forme de la courbe est comparée (les unités sont différentes).
        fig = go.Figure()
        for c in capteurs:
            s = donnees[c]
            amplitude = s.max() - s.min()
            norm = (s - s.min()) / amplitude if amplitude else s * 0
            fig.add_trace(
                go.Scatter(
                    x=x, y=norm, mode="lines", name=f"{c} ({CAPTEURS_INFO[c][0]})",
                    customdata=s,  # valeur réelle affichée au survol
                    hovertemplate=f"{c} : %{{customdata:.{nb_decimales(s)}f}} {unite(c)}"
                                  "<extra></extra>",
                )
            )
        fig.update_layout(yaxis_title="Valeur normalisée (0 = minimum, 1 = maximum)",
                          xaxis_title="Cycle", height=430)
    else:
        fig = make_subplots(
            rows=len(capteurs), cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=[libelle(c) for c in capteurs],
        )
        for i, c in enumerate(capteurs, start=1):
            s = donnees[c]
            fig.add_trace(
                go.Scatter(
                    x=x, y=s, mode="lines", name=c, showlegend=False,
                    line=dict(color=COULEUR_MESURE, width=1.5),
                    hovertemplate=f"{c} : %{{y:.{nb_decimales(s)}f}} {unite(c)}<extra></extra>",
                ),
                row=i, col=1,
            )
            fig.update_yaxes(title_text=unite(c) or "valeur", row=i, col=1)
        fig.update_xaxes(title_text="Cycle", row=len(capteurs), col=1)
        fig.update_layout(height=250 * len(capteurs) + 60)
    fig.update_layout(
        template="plotly_white", hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig


def figure_matplotlib(donnees: pd.DataFrame, capteurs: list[str], superposer: bool, machine: int):
    """Version statique (matplotlib) de la visualisation : 1 à 4 capteurs."""
    x = donnees[COL_TEMPS]
    if superposer and len(capteurs) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in capteurs:
            s = donnees[c]
            amplitude = s.max() - s.min()
            ax.plot(x, (s - s.min()) / amplitude if amplitude else s * 0, label=c, linewidth=1.3)
        ax.set_ylabel("Valeur normalisée (0 à 1)")
        ax.set_xlabel("Cycle")
        ax.set_title(f"Comparaison de capteurs - machine {machine}")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        fig, axes = plt.subplots(len(capteurs), 1, figsize=(10, 3 * len(capteurs)),
                                 sharex=True, squeeze=False)
        for ax, c in zip(axes[:, 0], capteurs):
            ax.plot(x, donnees[c], color=COULEUR_MESURE, linewidth=1.5)
            ax.set_ylabel(nom_axe(c), fontsize=8)
            ax.set_title(f"{c} - machine {machine}", fontsize=10)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("Cycle")
        fig.tight_layout()
    return fig


###############################################################################
# PARTIE 5 - LES 6 FENÊTRES DE L'APPLICATION
# Chaque fenêtre = un onglet du menu de gauche = une fonction page_xxx(df, colonnes).
# Le dictionnaire PAGES (partie 7) fait le lien entre le menu et ces fonctions.
#
# Pour retrouver un bloc : Ctrl+F puis « [NOM DE LA PAGE] » (ex. « [VISUALISATION] »),
# ce qui liste tous les blocs de la page, chacun avec une phrase qui dit ce qu'il fait.
###############################################################################


# =============================================================================
# =============================================================================
# ===  FENÊTRE 1/6 : ACCUEIL  🏠
# ===  Présentation de l'application, du jeu de données et des capteurs
# =============================================================================
# =============================================================================


def page_accueil(df, colonnes):
    # ▶ [ACCUEIL] Titre de la page
    st.header("🏠 Accueil")

    # ▶ [ACCUEIL] Texte de présentation de l'application
    st.write(
        "Cette application est une **interface de supervision industrielle**. "
        "Elle permet de consulter les mesures de plusieurs machines, de calculer "
        "des indicateurs et de repérer des situations anormales à partir des "
        "données capteurs : une première étape vers la **maintenance prédictive**."
    )

    # ▶ [ACCUEIL] Objectifs : une carte (encadré) par fonctionnalité
    st.subheader("Objectifs de l'application")
    fonctions = [
        ("📈 Visualisation", "Suivre l'évolution d'un ou plusieurs capteurs d'une machine, "
                             "avec zoom et choix de la période."),
        ("📊 Indicateurs", "Calculer moyenne, minimum, maximum, médiane et écart-type "
                           "sur la période choisie."),
        ("⚠️ Défauts", "Fixer des seuils et détecter les anomalies d'une machine : "
                       "valeur trop haute ou basse, variation brutale."),
        ("🚨 Alertes", "Repérer d'un coup d'œil les machines et les capteurs en alerte."),
        ("🔎 Requête utilisateur", "Interroger les données : une machine, une variable, "
                                   "une opération de calcul."),
    ]
    for debut_ligne in (0, 3):
        cartes = st.columns(3)
        for colonne, (titre, texte) in zip(cartes, fonctions[debut_ligne:debut_ligne + 3]):
            with colonne, st.container(border=True, height=135):
                st.markdown(f"**{titre}**")
                st.caption(texte)

    # ▶ [ACCUEIL] Informations générales sur le dataset
    st.subheader("Informations générales sur les données")
    st.write(
        "Les données proviennent du jeu **NASA C-MAPSS (FD001)** : des "
        "turboréacteurs suivis cycle après cycle, jusqu'à leur défaillance."
    )

    # ▶ [ACCUEIL] Chiffres clés : nombre de lignes / colonnes / machines
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", f"{len(df):,}".replace(",", " "))
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Nombre de machines", df[COL_MACHINE].nunique())

    # ▶ [ACCUEIL] Durée d'observation de chaque machine (nombre de cycles)
    cycles = df.groupby(COL_MACHINE)[COL_TEMPS].max()
    st.caption(
        f"Chaque machine est suivie pendant {cycles.min()} à {cycles.max()} cycles "
        f"(moyenne : {cycles.mean():.0f} cycles)."
    )

    # ▶ [ACCUEIL] Création des 3 sous-onglets (aperçu / structure / capteurs)
    onglet_apercu, onglet_structure, onglet_capteurs = st.tabs(
        ["Aperçu des données", "Structure du dataset", "Capteurs et unités"]
    )

    # ▶ [ACCUEIL] Sous-onglet 1 : aperçu des 10 premières lignes du dataset
    with onglet_apercu:
        st.dataframe(df.head(10), width="stretch")

    # ▶ [ACCUEIL] Sous-onglet 2 : rôle des colonnes (machine, temps, réglages, capteurs)
    with onglet_structure:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Machine**\n\n`{COL_MACHINE}`")
        c2.markdown(f"**Temps**\n\n`{COL_TEMPS}` (numéro de cycle)")
        c3.markdown(
            f"**Réglages de fonctionnement**\n\n{len(colonnes['reglages'])} colonnes "
            f"(`{colonnes['reglages'][0]}` à `{colonnes['reglages'][-1]}`)"
        )
        st.markdown(
            f"**Capteurs** : {len(colonnes['capteurs'])} colonnes "
            f"(`{colonnes['capteurs'][0]}` à `{colonnes['capteurs'][-1]}`), voir l'onglet "
            "« Capteurs et unités »."
        )

    # ▶ [ACCUEIL] Sous-onglet 3 : tableau des capteurs (symbole, description, unité)
    with onglet_capteurs:
        st.write(
            f"{len(colonnes['capteurs'])} capteurs au total, dont "
            f"**{len(colonnes['capteurs_utiles'])} utiles** (valeur qui varie). "
            "Les capteurs constants sont ignorés pour la surveillance."
        )
        tableau = pd.DataFrame(
            {
                "Colonne": colonnes["capteurs"],
                "Symbole": [CAPTEURS_INFO[c][0] for c in colonnes["capteurs"]],
                "Description": [CAPTEURS_INFO[c][1] for c in colonnes["capteurs"]],
                "Unité": [unite(c) or "-" for c in colonnes["capteurs"]],
                "Statut": [
                    "Utile" if c in colonnes["capteurs_utiles"] else "Constant"
                    for c in colonnes["capteurs"]
                ],
            }
        )
        st.dataframe(tableau, hide_index=True, width="stretch")
        st.caption(
            "Unités issues de la documentation NASA C-MAPSS : °R = degré Rankine, "
            "psia = livre par pouce carré, tr/min = tours par minute."
        )


# =============================================================================
# =============================================================================
# ===  FENÊTRE 2/6 : VISUALISATION  📈
# ===  Courbes des capteurs d'une machine (interactives ou matplotlib)
# =============================================================================
# =============================================================================


def page_visualisation(df, colonnes):
    # ▶ [VISUALISATION] Titre de la page et texte d'introduction
    st.header("📈 Visualisation")
    st.write(
        "Choisis une machine et un ou plusieurs capteurs (jusqu'à 4) pour voir leur "
        "évolution dans le temps et les comparer."
    )

    # ▶ [VISUALISATION] Choix de la machine (liste déroulante)
    col1, col2 = st.columns([1, 2])
    machine = col1.selectbox(
        "Machine", sorted(df[COL_MACHINE].unique()), help="Identifiant de la machine"
    )
    # ▶ [VISUALISATION] Sélection de PLUSIEURS capteurs (liste à choix multiples, 4 maximum)
    capteurs = col2.multiselect(
        "Capteur(s)",
        colonnes["capteurs_utiles"],
        default=[colonnes["capteurs_utiles"][0]],
        format_func=libelle,
        max_selections=4,
    )

    # ▶ [VISUALISATION] Curseur d'intervalle de cycles + filtrage des données de la machine
    donnees = filtrer_machine(df, machine)
    cycle_min, cycle_max = int(donnees[COL_TEMPS].min()), int(donnees[COL_TEMPS].max())
    debut, fin = st.slider(
        "Intervalle de cycles", cycle_min, cycle_max, (cycle_min, cycle_max)
    )
    donnees = donnees[donnees[COL_TEMPS].between(debut, fin)]

    # ▶ [VISUALISATION] Message si aucun capteur n'est sélectionné (on arrête la page ici)
    if not capteurs:
        st.info("Choisis au moins un capteur pour afficher le graphique.")
        return

    # ▶ [VISUALISATION] Interrupteurs : « Graphique interactif » et « Superposer les capteurs »
    opt1, opt2 = st.columns(2)
    interactif = opt1.toggle(
        "Graphique interactif", value=True,
        help="Zoom, déplacement et valeurs au survol (Plotly). Désactivé : image fixe matplotlib.",
    )
    superposer = False
    if len(capteurs) > 1:
        superposer = opt2.toggle(
            "Superposer les capteurs", value=False,
            help="Un seul graphique, valeurs ramenées entre 0 et 1 pour comparer les formes "
                 "des courbes malgré des unités différentes.",
        )

    # ▶ [VISUALISATION] Affichage du graphique (Plotly interactif ou matplotlib image fixe)
    if interactif:
        if len(capteurs) == 1:
            fig = figure_temporelle(donnees, capteurs[0], f"{capteurs[0]} - machine {machine}")
        else:
            fig = figure_multi(donnees, capteurs, superposer)
        st.plotly_chart(fig, theme=None, width="stretch")
    else:
        fig = figure_matplotlib(donnees, capteurs, superposer, machine)
        st.pyplot(fig)
        plt.close(fig)  # libère la mémoire de la figure

    # ▶ [VISUALISATION] Légende sous le graphique + tableau des données filtrées
    st.caption(
        f"Machine {machine} | {len(donnees)} mesures affichées (cycles {debut} à {fin})."
    )

    with st.expander("Voir les données filtrées"):
        st.dataframe(donnees[[COL_MACHINE, COL_TEMPS] + capteurs], width="stretch")


# =============================================================================
# =============================================================================
# ===  FENÊTRE 3/6 : INDICATEURS  📊
# ===  Statistiques (moyenne, min, max...) + histogramme
# =============================================================================
# =============================================================================


def page_indicateurs(df, colonnes):
    # ▶ [INDICATEURS] Titre de la page et texte d'introduction
    st.header("📊 Indicateurs")
    st.write("Statistiques calculées sur les données de la machine et de la période choisies.")

    # ▶ [INDICATEURS] Choix de la machine et de la variable (listes déroulantes)
    col1, col2 = st.columns(2)
    machine = col1.selectbox("Machine", sorted(df[COL_MACHINE].unique()))
    variable = col2.selectbox("Variable", colonnes["variables_utiles"], format_func=libelle)

    # ▶ [INDICATEURS] Curseur d'intervalle de cycles + filtrage des données
    donnees = filtrer_machine(df, machine)
    cycle_min, cycle_max = int(donnees[COL_TEMPS].min()), int(donnees[COL_TEMPS].max())
    debut, fin = st.slider(
        "Intervalle de cycles", cycle_min, cycle_max, (cycle_min, cycle_max)
    )
    # Données sélectionnées = machine choisie + intervalle de cycles choisi
    donnees = donnees[donnees[COL_TEMPS].between(debut, fin)]
    serie = donnees[variable]

    # ▶ [INDICATEURS] Cartes de chiffres : moyenne, minimum, maximum
    st.subheader(f"{variable} - machine {machine} (cycles {debut} à {fin})")
    d = nb_decimales(serie)  # précision adaptée à la variable choisie
    c1, c2, c3 = st.columns(3)
    c1.metric("Moyenne", avec_unite(f"{serie.mean():.{d}f}", variable))
    c2.metric("Minimum", avec_unite(f"{serie.min():.{d}f}", variable))
    c3.metric("Maximum", avec_unite(f"{serie.max():.{d}f}", variable))

    # ▶ [INDICATEURS] Cartes de chiffres : médiane, écart-type, dernière valeur
    c4, c5, c6 = st.columns(3)
    c4.metric("Médiane", avec_unite(f"{serie.median():.{d}f}", variable))
    c5.metric("Écart-type", avec_unite(f"{serie.std():.{d}f}", variable))
    c6.metric(
        "Dernière valeur",
        avec_unite(f"{serie.iloc[-1]:.{d}f}", variable),
        delta=avec_unite(f"{serie.iloc[-1] - serie.iloc[0]:+.{d}f}", variable) + " depuis le début",
        delta_color="off",  # une variation n'est ni "bonne" ni "mauvaise" ici
    )
    st.caption(f"Calculé sur {len(serie)} mesures.")

    # ▶ [INDICATEURS] Histogramme de la distribution des valeurs (matplotlib)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(serie, bins=25, color="#9ecae1", edgecolor="white")
    ax.axvline(serie.mean(), color="#d62728", linestyle="--", label="Moyenne")
    ax.axvline(serie.min(), color="#2ca02c", linestyle=":", label="Minimum")
    ax.axvline(serie.max(), color="#ff7f0e", linestyle=":", label="Maximum")
    ax.set_title(f"Distribution de {variable}")
    ax.set_xlabel(nom_axe(variable))
    ax.set_ylabel("Nombre de mesures")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ▶ [INDICATEURS] Tableau récapitulatif de tous les capteurs (menu déroulant)
    with st.expander("Indicateurs de tous les capteurs (même machine, même période)"):
        resume = donnees[colonnes["capteurs_utiles"]].agg(["mean", "min", "max", "std"]).T
        resume.columns = ["Moyenne", "Minimum", "Maximum", "Écart-type"]
        resume.insert(0, "Unité", [unite(c) or "-" for c in resume.index])
        st.dataframe(resume.round(3), width="stretch")


# =============================================================================
# =============================================================================
# ===  FENÊTRE 4/6 : DÉFAUTS  ⚠️
# ===  Détection d'anomalies sur une variable : seuil haut / bas / variation brutale
# =============================================================================
# =============================================================================


def page_defauts(df, colonnes):
    # ▶ [DÉFAUTS] Titre de la page et texte d'introduction
    st.header("⚠️ Défauts")
    st.write(
        "Choisis la machine, la variable à surveiller et la période, puis règle les "
        "seuils de détection."
    )

    # ▶ [DÉFAUTS] Choix de la machine et de la variable à surveiller
    col1, col2 = st.columns(2)
    machine = col1.selectbox("Machine", sorted(df[COL_MACHINE].unique()))
    variable = col2.selectbox(
        "Variable à surveiller", colonnes["capteurs_utiles"], format_func=libelle
    )

    # ▶ [DÉFAUTS] Curseur d'intervalle de cycles + filtrage des données
    donnees = filtrer_machine(df, machine)
    cycle_min, cycle_max = int(donnees[COL_TEMPS].min()), int(donnees[COL_TEMPS].max())
    debut, fin = st.slider(
        "Intervalle de cycles", cycle_min, cycle_max, (cycle_min, cycle_max)
    )
    donnees = donnees[donnees[COL_TEMPS].between(debut, fin)]
    serie = donnees[variable]

    # ▶ [DÉFAUTS] Calcul des valeurs de seuil proposées par défaut (moyenne ± 2σ...)
    d = nb_decimales(serie)
    moyenne, ecart = float(serie.mean()), float(serie.std())
    ecart_variation = float(serie.diff().std())
    if pd.isna(ecart):
        ecart = 0.0
    if pd.isna(ecart_variation):
        ecart_variation = 0.0
    defaut_haut = round(moyenne + 2 * ecart, d)
    defaut_bas = round(moyenne - 2 * ecart, d)
    defaut_variation = round(3 * ecart_variation, d)
    pas = 10.0 ** -(d - 2)  # pas des flèches +/- des champs numériques
    u = unite(variable)
    suffixe = f" ({u})" if u else ""

    # ▶ [DÉFAUTS] Réglage des seuils : cases à cocher + champs numériques (haut / bas / variation)
    st.subheader("Réglage des seuils")
    c_haut, c_bas, c_var = st.columns(3)

    actif_haut = c_haut.checkbox("Seuil haut", value=True)
    seuil_haut = c_haut.number_input(
        f"Valeur du seuil haut{suffixe}", value=defaut_haut, step=pas,
        format=f"%.{d}f", disabled=not actif_haut,
    )
    actif_bas = c_bas.checkbox("Seuil bas", value=False)
    seuil_bas = c_bas.number_input(
        f"Valeur du seuil bas{suffixe}", value=defaut_bas, step=pas,
        format=f"%.{d}f", disabled=not actif_bas,
    )
    actif_var = c_var.checkbox("Variation brutale", value=True)
    limite_var = c_var.number_input(
        f"Variation maximale entre 2 cycles{suffixe}", value=defaut_variation,
        min_value=0.0, step=pas, format=f"%.{d}f", disabled=not actif_var,
    )
    st.caption(
        "Valeurs proposées : moyenne ± 2 écarts-types pour les seuils, et 3 écarts-types "
        "des variations pour la variation brutale (calculés sur la sélection). "
        "Active le seuil bas pour les capteurs qui baissent avec l'usure "
        "(par exemple sensor_7)."
    )

    # ▶ [DÉFAUTS] Lancement de la détection d'anomalies (fonction detecter_anomalies)
    anomalies = detecter_anomalies(
        donnees,
        variable,
        seuil_haut=seuil_haut if actif_haut else None,
        seuil_bas=seuil_bas if actif_bas else None,
        limite_variation=limite_var if actif_var else None,
    )

    # ▶ [DÉFAUTS] Message « Situation normale » (vert) ou « Situation anormale » (rouge)
    st.subheader(f"Analyse - machine {machine}, {variable}")
    if not (actif_haut or actif_bas or actif_var):
        st.warning("Aucune détection activée : coche au moins un seuil.")
    elif anomalies.empty:
        st.success(
            f"✅ **Situation normale** : aucune anomalie détectée sur la machine "
            f"{machine} (cycles {debut} à {fin})."
        )
    else:
        nb_cycles = anomalies[COL_TEMPS].nunique()
        st.error(
            f"🚨 **Situation anormale** : {len(anomalies)} anomalie(s) détectée(s) sur la "
            f"machine {machine}, variable {variable} ({nb_cycles} cycle(s) concerné(s))."
        )

    # ▶ [DÉFAUTS] Cartes de chiffres : nombre d'anomalies par type
    nb_par_type = anomalies["type"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomalies détectées", len(anomalies))
    m2.metric("Seuil haut", int(nb_par_type.get("Seuil haut", 0)) if actif_haut else "-")
    m3.metric("Seuil bas", int(nb_par_type.get("Seuil bas", 0)) if actif_bas else "-")
    m4.metric(
        "Variation brutale", int(nb_par_type.get("Variation brutale", 0)) if actif_var else "-"
    )
    st.caption(
        f"Machine {machine} | variable surveillée : {variable} | "
        f"{len(serie)} mesures analysées (cycles {debut} à {fin})."
    )

    # ▶ [DÉFAUTS] Graphique avec lignes de seuil et points d'anomalie
    fig = figure_temporelle(
        donnees,
        variable,
        f"{variable} - machine {machine}",
        anomalies=anomalies,
        seuil_haut=seuil_haut if actif_haut else None,
        seuil_bas=seuil_bas if actif_bas else None,
    )
    st.plotly_chart(fig, theme=None, width="stretch")

    # ▶ [DÉFAUTS] Tableau détaillé des anomalies (cycle, type, valeur, seuil)
    if not anomalies.empty:
        st.subheader("Détail des anomalies")
        afficher_tableau_anomalies(anomalies, variable, d)


# =============================================================================
# =============================================================================
# ===  FENÊTRE 5/6 : ALERTES  🚨
# ===  2 vues : toutes les machines (flotte) et une machine (tous les capteurs)
# =============================================================================
# =============================================================================


# --- Alertes, vue 1 : toutes les machines, une variable --------------------
def alertes_flotte(df, colonnes):
    """Vue 1 : toutes les machines, une variable. Quelles machines sont en alerte ?"""
    st.write(
        "Choisis une variable : l'application contrôle **les 100 machines** avec les "
        "mêmes seuils et liste celles qui ont au moins une anomalie."
    )
    variable = st.selectbox(
        "Variable à surveiller", colonnes["capteurs_utiles"], format_func=libelle
    )

    # ▶ [ALERTES · TOUTES LES MACHINES] Calcul des seuils proposés (moyenne ± 3σ sur toute la flotte)
    serie = df[variable]
    d = nb_decimales(serie)
    moyenne, ecart = float(serie.mean()), float(serie.std())
    ecart_variation = float(df.groupby(COL_MACHINE)[variable].diff().std())
    pas = 10.0 ** -(d - 2)
    u = unite(variable)
    suffixe = f" ({u})" if u else ""

    # ▶ [ALERTES · TOUTES LES MACHINES] Réglage des seuils (cases à cocher + champs numériques)
    c_haut, c_bas, c_var = st.columns(3)
    actif_haut = c_haut.checkbox("Seuil haut", value=True)
    seuil_haut = c_haut.number_input(
        f"Valeur du seuil haut{suffixe}", value=round(moyenne + 3 * ecart, d),
        step=pas, format=f"%.{d}f", disabled=not actif_haut,
    )
    actif_bas = c_bas.checkbox("Seuil bas", value=False)
    seuil_bas = c_bas.number_input(
        f"Valeur du seuil bas{suffixe}", value=round(moyenne - 3 * ecart, d),
        step=pas, format=f"%.{d}f", disabled=not actif_bas,
    )
    actif_var = c_var.checkbox("Variation brutale", value=True)
    limite_var = c_var.number_input(
        f"Variation maximale entre 2 cycles{suffixe}",
        value=round(4 * ecart_variation, d), min_value=0.0, step=pas,
        format=f"%.{d}f", disabled=not actif_var,
    )
    st.caption(
        "Valeurs proposées : moyenne ± 3 écarts-types, et 4 écarts-types pour la "
        "variation brutale, calculés sur l'ensemble des machines. Ces seuils sont plus "
        "larges que dans Défauts : sur 100 machines, un seuil à 2 écarts-types "
        "déclencherait une alerte sur presque toutes, à cause du simple bruit des mesures."
    )

    # ▶ [ALERTES · TOUTES LES MACHINES] Avertissement si aucun seuil n'est coché
    if not (actif_haut or actif_bas or actif_var):
        st.warning("Aucune détection activée : coche au moins un seuil.")
        return

    # ▶ [ALERTES · TOUTES LES MACHINES] Boucle : contrôle des 100 machines une par une
    lignes = []
    anomalies_par_machine = {}
    for machine, donnees in df.groupby(COL_MACHINE):
        donnees = donnees.sort_values(COL_TEMPS)
        anomalies = detecter_anomalies(
            donnees,
            variable,
            seuil_haut=seuil_haut if actif_haut else None,
            seuil_bas=seuil_bas if actif_bas else None,
            limite_variation=limite_var if actif_var else None,
        )
        anomalies_par_machine[machine] = anomalies
        nb = anomalies["type"].value_counts()
        lignes.append(
            {
                "Machine": machine,
                "Statut": "🚨 Alerte" if len(anomalies) else "✅ Normal",
                "Anomalies": len(anomalies),
                "Seuil haut": int(nb.get("Seuil haut", 0)),
                "Seuil bas": int(nb.get("Seuil bas", 0)),
                "Variation brutale": int(nb.get("Variation brutale", 0)),
                "Premier cycle": (
                    int(anomalies[COL_TEMPS].min()) if len(anomalies) else None
                ),
            }
        )
    synthese = pd.DataFrame(lignes).sort_values(
        ["Anomalies", "Machine"], ascending=[False, True]
    )
    en_alerte = synthese[synthese["Anomalies"] > 0]
    nb_machines = len(synthese)

    # ▶ [ALERTES · TOUTES LES MACHINES] Message d'alerte + cartes de chiffres clés
    st.subheader(f"Résultat - {variable}")
    if en_alerte.empty:
        st.success(f"✅ **Aucune alerte** : les {nb_machines} machines sont normales.")
    else:
        st.error(
            f"🚨 **{len(en_alerte)} machine(s) en alerte** sur {nb_machines} "
            f"({len(en_alerte) / nb_machines:.0%} de la flotte)."
        )
    m1, m2, m3 = st.columns(3)
    m1.metric("Machines en alerte", len(en_alerte))
    m2.metric("Machines normales", nb_machines - len(en_alerte))
    m3.metric("Anomalies au total", int(synthese["Anomalies"].sum()))

    if en_alerte.empty:
        return

    # ▶ [ALERTES · TOUTES LES MACHINES] Graphique en barres : les 10 machines les plus touchées
    top = en_alerte.head(10).iloc[::-1]  # inversé pour avoir le n°1 en haut du graphique
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.barh([f"Machine {m}" for m in top["Machine"]], top["Anomalies"], color="#d62728")
    ax.set_title(f"Machines les plus touchées - {variable}")
    ax.set_xlabel("Nombre d'anomalies")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # graduations entières
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ▶ [ALERTES · TOUTES LES MACHINES] Tableau des machines en alerte
    st.subheader("Machines en alerte")
    afficher_normales = st.checkbox("Afficher aussi les machines normales")
    st.dataframe(
        synthese if afficher_normales else en_alerte,
        hide_index=True,
        width="stretch",
        column_config={
            "Machine": st.column_config.NumberColumn(width="small"),
            "Premier cycle": st.column_config.NumberColumn(format="%d"),
        },
    )

    # ▶ [ALERTES · TOUTES LES MACHINES] Détail des anomalies d'une machine choisie
    st.subheader("Détail d'une machine")
    machine_detail = st.selectbox(
        "Machine à examiner",
        list(en_alerte["Machine"]),
        format_func=lambda m: f"Machine {m} - {int(synthese.loc[synthese['Machine'] == m, 'Anomalies'].iloc[0])} anomalie(s)",
    )
    afficher_tableau_anomalies(anomalies_par_machine[machine_detail], variable, d)


# --- Alertes, vue 2 : une machine, tous les capteurs -----------------------
def alertes_machine(df, colonnes):
    """Vue 2 : une machine, tous les capteurs. Quels capteurs sont en alerte ?"""
    st.write(
        "Choisis une machine : l'application contrôle **ses 15 capteurs** d'un coup. "
        "Les seuils sont calculés automatiquement à partir des premiers cycles de la "
        "machine, supposés en bon état (la référence)."
    )
    # ▶ [ALERTES · UNE MACHINE] Choix de la machine, des cycles de référence et de la tolérance
    c1, c2, c3 = st.columns(3)
    machine = c1.selectbox("Machine à contrôler", sorted(df[COL_MACHINE].unique()))
    nb_ref = c2.number_input(
        "Cycles de référence (bon état)", min_value=10, max_value=100, value=30, step=5
    )
    k = c3.number_input(
        "Tolérance (en écarts-types)", min_value=1.0, max_value=6.0, value=3.0, step=0.5,
        help="Un capteur est en alerte si sa valeur sort de moyenne ± k écarts-types "
             "de la période de référence.",
    )

    # ▶ [ALERTES · UNE MACHINE] Données de la machine + période de référence (premiers cycles)
    donnees = filtrer_machine(df, machine)
    reference = donnees.head(int(nb_ref))

    # ▶ [ALERTES · UNE MACHINE] Boucle : contrôle des 15 capteurs un par un (seuils = référence ± kσ)
    lignes = []
    anomalies_par_capteur = {}
    bornes = {}
    for capteur in colonnes["capteurs_utiles"]:
        m_ref, s_ref = reference[capteur].mean(), reference[capteur].std()
        if pd.isna(s_ref) or s_ref == 0:
            continue  # référence sans variation : impossible de définir un seuil
        haut, bas = m_ref + k * s_ref, m_ref - k * s_ref
        bornes[capteur] = (haut, bas)
        anomalies = detecter_anomalies(donnees, capteur, seuil_haut=haut, seuil_bas=bas)
        anomalies_par_capteur[capteur] = anomalies
        nb = anomalies["type"].value_counts()
        ecart_max = ((donnees[capteur] - m_ref).abs() / s_ref).max()
        lignes.append(
            {
                "Capteur": libelle(capteur),
                "Statut": "🚨 Alerte" if len(anomalies) else "✅ Normal",
                "Anomalies": len(anomalies),
                "Seuil haut": int(nb.get("Seuil haut", 0)),
                "Seuil bas": int(nb.get("Seuil bas", 0)),
                "Premier cycle": (
                    int(anomalies[COL_TEMPS].min()) if len(anomalies) else None
                ),
                "Écart max (σ)": round(float(ecart_max), 1),
                "_capteur": capteur,
            }
        )
    synthese = pd.DataFrame(lignes).sort_values(
        ["Anomalies", "Capteur"], ascending=[False, True]
    )
    en_alerte = synthese[synthese["Anomalies"] > 0]

    # ▶ [ALERTES · UNE MACHINE] Message d'alerte + cartes de chiffres clés
    st.subheader(f"Résultat - machine {machine}")
    if en_alerte.empty:
        st.success(f"✅ **Aucune alerte** : les {len(synthese)} capteurs de la machine {machine} sont normaux.")
    else:
        st.error(
            f"🚨 **{len(en_alerte)} capteur(s) en alerte** sur {len(synthese)} "
            f"pour la machine {machine}."
        )
    m1, m2, m3 = st.columns(3)
    m1.metric("Capteurs en alerte", len(en_alerte))
    m2.metric("Capteurs normaux", len(synthese) - len(en_alerte))
    m3.metric("Anomalies au total", int(synthese["Anomalies"].sum()))

    # ▶ [ALERTES · UNE MACHINE] Tableau des capteurs (statut, anomalies, écart max)
    st.dataframe(
        synthese.drop(columns=["_capteur"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Capteur": st.column_config.TextColumn(width="medium"),
            "Statut": st.column_config.TextColumn(width="small"),
            "Anomalies": st.column_config.NumberColumn(width="small"),
            "Seuil haut": st.column_config.NumberColumn(width="small"),
            "Seuil bas": st.column_config.NumberColumn(width="small"),
            "Premier cycle": st.column_config.NumberColumn(format="%d", width="small"),
            "Écart max (σ)": st.column_config.NumberColumn(width="small"),
        },
    )
    st.caption(
        f"Référence : cycles 1 à {int(nb_ref)} de la machine {machine} | "
        f"seuils = moyenne ± {k:g} écart(s)-type(s) de la référence. "
        "Écart max (σ) = plus grand écart à la moyenne de référence, en écarts-types."
    )

    # ▶ [ALERTES · UNE MACHINE] Courbe d'un capteur choisi, avec seuils et zone de référence
    st.subheader("Courbe d'un capteur")
    capteurs_dispo = list(synthese["_capteur"])  # le plus touché en premier
    capteur = st.selectbox("Capteur à afficher", capteurs_dispo, format_func=libelle)
    haut, bas = bornes[capteur]
    anomalies = anomalies_par_capteur[capteur]

    fig = figure_temporelle(
        donnees,
        capteur,
        f"{capteur} - machine {machine}",
        anomalies=anomalies,
        seuil_haut=haut,
        seuil_bas=bas,
        zone_reference=(int(donnees[COL_TEMPS].min()), int(reference[COL_TEMPS].max())),
    )
    st.plotly_chart(fig, theme=None, width="stretch")


# --- Alertes : page qui regroupe les 2 vues dans deux sous-onglets ---------
def page_alertes(df, colonnes):
    # ▶ [ALERTES] Titre de la page et texte d'introduction
    st.header("🚨 Alertes")
    st.write(
        "Vue d'ensemble pour repérer rapidement ce qui ne va pas. Pour analyser "
        "une machine en détail, utilise la section **Défauts**."
    )
    # ▶ [ALERTES] Les 2 sous-onglets : chacun appelle sa fonction (alertes_flotte / alertes_machine)
    onglet_flotte, onglet_machine = st.tabs(
        ["🏭 Toutes les machines (1 variable)", "🔧 Une machine (tous les capteurs)"]
    )
    with onglet_flotte:
        alertes_flotte(df, colonnes)
    with onglet_machine:
        alertes_machine(df, colonnes)


# =============================================================================
# =============================================================================
# ===  FENÊTRE 6/6 : REQUÊTE UTILISATEUR  🔎
# ===  Requête simple : machine + variable + opération de calcul
# =============================================================================
# =============================================================================


def page_requete(df, colonnes):
    # ▶ [REQUÊTE] Titre de la page et texte d'introduction
    st.header("🔎 Requête utilisateur")
    st.write("Construis une requête simple : machine, variable et opération de calcul.")

    # ▶ [REQUÊTE] Choix de la machine, de la variable et de l'opération (3 listes déroulantes)
    col1, col2, col3 = st.columns(3)
    machine = col1.selectbox("Machine", sorted(df[COL_MACHINE].unique()))
    variable = col2.selectbox("Variable", colonnes["variables_utiles"], format_func=libelle)
    operation = col3.selectbox("Opération", list(OPERATIONS.keys()))

    # ▶ [REQUÊTE] Bouton « Exécuter » : calcul et affichage du résultat
    if st.button("Exécuter la requête", type="primary"):
        donnees = filtrer_machine(df, machine)
        # getattr(serie, "mean")() équivaut à serie.mean()
        resultat = getattr(donnees[variable], OPERATIONS[operation])()

        st.success(f"**{operation}** de `{variable}` pour la machine **{machine}**")
        d = nb_decimales(donnees[variable])
        st.metric(f"{operation} de {variable}", avec_unite(f"{resultat:.{d}f}", variable))
        st.caption(f"Calculé sur {len(donnees)} mesures (cycles {donnees[COL_TEMPS].min()} à {donnees[COL_TEMPS].max()}).")



###############################################################################
# PARTIE 6 - ACCÈS DEPUIS UN AUTRE APPAREIL (RÉSEAU LOCAL)
# L'application tourne sur ce poste (le "serveur") et peut être consultée
# depuis un autre appareil du même réseau, via un simple navigateur.
###############################################################################

def adresse_reseau() -> str | None:
    """Adresse IP de ce poste sur le réseau local (None si introuvable)."""
    try:
        # On "connecte" une socket UDP vers une adresse privée : aucune donnée
        # n'est envoyée, mais le système choisit la carte réseau (Wi-Fi...) qui
        # serait utilisée, et on lit l'adresse de ce poste sur cette carte.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None


def afficher_acces_reseau() -> None:
    """Encadré de la barre latérale : adresse à saisir sur l'autre appareil."""
    port = st.get_option("server.port")
    ip = adresse_reseau()
    with st.sidebar.expander("📡 Accès depuis un autre appareil"):
        if ip and not ip.startswith("127."):
            st.write("Sur un appareil connecté au **même réseau**, ouvre :")
            st.code(f"http://{ip}:{port}", language=None)
        else:
            st.warning("Aucune connexion réseau détectée sur ce poste.")
        # Adresse de l'appareil qui consulte la page (preuve de la communication réseau)
        try:
            client = st.context.ip_address
        except Exception:
            client = None
        if client:
            ce_poste = client in ("127.0.0.1", "::1", "localhost") or client == ip
            st.caption(
                f"Page consultée depuis : {client}" + (" (ce poste)" if ce_poste else " (appareil distant)")
            )


###############################################################################
# PARTIE 7 - STYLE, MENU DE NAVIGATION ET PROGRAMME PRINCIPAL
###############################################################################

# --- Style (CSS) ------------------------------------------------------------
STYLE = """
<style>
/* Indicateurs (st.metric) affichés comme des cartes */
[data-testid="stMetric"] {
    background-color: #F0F4F8;
    border: 1px solid #DCE4EE;
    border-radius: 10px;
    padding: 12px 16px;
}
/* Titre principal plus compact */
h1 { font-size: 2.1rem !important; }
</style>
"""


# --- Menu : nom affiché -> fenêtre ------------------------------------------
# Dictionnaire "nom affiché dans le menu" -> "fonction qui affiche la page"
PAGES = {
    "🏠 Accueil": page_accueil,
    "📈 Visualisation": page_visualisation,
    "📊 Indicateurs": page_indicateurs,
    "⚠️ Défauts": page_defauts,
    "🚨 Alertes": page_alertes,
    "🔎 Requête utilisateur": page_requete,
}


# --- Programme principal : titre, menu latéral, affichage de la fenêtre -----
def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("🛠️ Supervision industrielle")
    st.caption("Maintenance prédictive | données de turboréacteurs NASA C-MAPSS (FD001)")

    # Chargement des données (une seule fois grâce au cache)
    df = charger_donnees()
    colonnes = identifier_colonnes(df)

    # Barre latérale de navigation
    st.sidebar.title("Navigation")
    choix = st.sidebar.radio("Aller à la section :", list(PAGES.keys()))

    st.sidebar.divider()
    nb_lignes = f"{len(df):,}".replace(",", " ")  # 20631 -> "20 631"
    st.sidebar.caption(
        f"Données chargées : {nb_lignes} lignes, "
        f"{df[COL_MACHINE].nunique()} machines"
    )
    afficher_acces_reseau()

    # Affichage de la page choisie
    PAGES[choix](df, colonnes)


# Lancement de l'application
main()