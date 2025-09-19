# uv de Astral

## Introduction

Vois `uv` comme le nouveau couteau suisse du développeur Python. C'est un outil ultra-rapide et tout-en-un pour gérer tes projets Python. Il remplace à la fois pip (pour installer les paquets) et venv (pour créer les environnements virtuels), mais en beaucoup plus performant, car il tourne sous des algorithmes développés en `Rust`.

## Installation

### Windows

Ouvre `Powershell` et exécute la ligne suivante:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Mac OS & Linux

Ouvre un `terminal` et exécute la commande suivante:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Autre

Si tu as besoin de tous les détails, ou d'une façon alternative qui te convient mieux pour installer l'outil, [visite directement leur documentation officielle](https://docs.astral.sh/uv/getting-started/installation/).

## Utilisation

`uv` va être utilisé dans 3 situations bien spécifiques:

1. Initialiser ton environnement de développement

*Synchroniser l'environnement avec les dépendences demandées dans pyproject.toml:*

```sh
uv sync
```

2. Ajouter/Enlever des packages Python

*Ajouter:*

```sh
uv add <librairie 1> <librairie 2> ... <librairie n>
```

*Retirer:*
```sh
uv remove <librairie 1> <librairie 2> ... <librairie n>
```

3. Exécuter des scripts depuis notre environnement virtuel

Un exemple:

```sh
uv run canlock --name <ton nom>
```
