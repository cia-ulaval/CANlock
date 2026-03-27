**Attack Synthesis - Reference**

Ce document décrit en détail tous les fichiers, composants et commandes liés à la génération, l'inspection et la visualisation des jeux de données attaqués (spoofing, replay, suspension, masquerade) dans ce dépôt. Il servira de base pour la maintenance, l'extension et les tests futurs.

**Scope**: synthèse d'attaques CAN sur des fenêtres temporelles extraites de la base de données locale, exports en pickle/CSV, outils d'inspection et de visualisation.

**Repository Files**
- **Scripts - Orchestration et utilitaires**
  - **scripts/attack_synthesis.py**: Générateur principal. Charge une fenêtre de messages depuis la base (lecture seule), applique séquentiellement les attaques (Spoofing, Masquerade, Replay, Suspension), reconstruit métadonnées (pgn/src/length), et écrit la sortie soit en pickle (.pkl) soit en texte/CSV (.txt/.csv) avec colonne payload_hex.
    - Entrées: `--start`, `--end`, `--session-id`, `--limit`, `--out`, `--spoof-rate`, `--suspend-frac`, `--masq-prob`, `--attacker-src`, `--mode`, `--spn`.
    - Usage exemple (exécutable via `uv run`):
      ```bash
      uv run python -m scripts.attack_synthesis --limit 2000 --out data/cache/attacked_window.pkl
      uv run python -m scripts.attack_synthesis --limit 2000 --out data/cache/attacked_all_attacks.txt
      ```
  - **scripts/attack_inspect.py**: Outils d'inspection et de validation. Supporte pickle ou CSV (payload_hex) et peut:
    - afficher un aperçu lisible (`--preview`),
    - valider que seules les valeurs d'un unique SPN ont été modifiées (`--validate`).
    - Commande d'exemple:
      ```bash
      uv run python -m scripts.attack_inspect --pkl data/cache/attacked_all_attacks.txt --preview --validate --n 15
      ```
  - **scripts/attack_visualize.py**: Génère graphiques d'aperçu de payloads et séries temporelles d'un SPN.
    - Entrée: `--pkl` (pickle ou CSV), `--spn`, `--out`.
    - Exemples:
      ```bash
      uv run python -m scripts.attack_visualize --pkl data/cache/attacked_window.pkl --out data/cache/preview.png
      uv run python -m scripts.attack_visualize --pkl data/cache/attacked_window.pkl --spn 161 --out data/cache/spn_161.png
      ```

- **Attacks - implémentation (une classe / fichier)**
  - **src/canlock/attacks/AttackBase.py**: base abstraite et utilitaires bit‑level pour lire/écrire les bits d'un SPN dans une charge utile de 8 octets.
    - Fonctions clés: `set_spn_bits(payload, spn, new_raw)`, `get_spn_bits(payload, spn)`.
    - Règle de bit-indexing: conversion en chaîne binaire 64b big-endian puis remplacement par indices `bit_start:bit_start+bit_length`.
    - Fichier: [src/canlock/attacks/AttackBase.py](src/canlock/attacks/AttackBase.py)

  - **src/canlock/attacks/SpoofingAttack.py**: Usurpation ciblée.
    - Mécanique:
      - Si `target` (SPN) fourni, résout le SPN puis modifie uniquement les bits de ce SPN via `set_spn_bits`.
      - Pour SPN analogiques: convertit `raw -> phys` via scale/offset, applique bruit gaussien (sigma par défaut ~5% ou 1.0), reconvertit en raw.
      - Pour SPN digitaux: si `DefinedDigitalValues` existants, choisit une valeur définie différente si possible, sinon aléatoire dans la plage de bits.
    - Modes: `append` (par défaut) ajoute des trames spoofées; `replace` écrase la trame choisie.
    - Fichier: [src/canlock/attacks/SpoofingAttack.py](src/canlock/attacks/SpoofingAttack.py)

  - **src/canlock/attacks/ReplayAttack.py**: Re‑transmission de trames valides.
    - Mécanique: capture index/ligne(s) candidates (PGN ciblé si `target`), clone les lignes sélectionnées et ré‑insère des copies avec un décalage temporel `delay_seconds`.
    - Annotations: `attack_type = replay`, `replay_source_index`, `replay_delay_s`.
    - Fichier: [src/canlock/attacks/ReplayAttack.py](src/canlock/attacks/ReplayAttack.py)

  - **src/canlock/attacks/SuspensionAttack.py**: Suspension / Bus‑Off simulation.
    - Mécanique (implémentation actuelle):
      - Si `target` SPN résolu, sélectionne la source (LSB de CAN id) la plus fréquente pour le PGN cible (victim_src).
      - Itère chronologiquement les trames du victim_src pour le PGN: injecte des échecs successifs (payload mis à None) et incrémente un compteur TEC-like de +8 par échec.
      - Quand TEC >= 256 marque `attack_type = bus_off` et continue de supprimer les frames (simule l'isolation ECU).
      - Sinon, si pas de résolution SPN, supprime une fraction `suspend_fraction` aléatoire.
    - Fichier: [src/canlock/attacks/SuspensionAttack.py](src/canlock/attacks/SuspensionAttack.py)

  - **src/canlock/attacks/MasqueradeAttack.py**: Masquerade (usurpation + suspension combinées).
    - Mécanique:
      - Change le LSB (source) de certaines trames pour l'adresse `attacker_source` configurable.
      - Si ciblage SPN: ajuste légèrement la valeur SPN pour rester plausible (bruit gaussien réduit ou choix proche).
    - Fichier: [src/canlock/attacks/MasqueradeAttack.py](src/canlock/attacks/MasqueradeAttack.py)

- **Dépendances et utilitaires DB / Decoder**
  - **src/canlock/decoder.py**: fonctions utilitaires pour extraire PGN et SPN bits depuis l'identifiant/charge utile. Utilisé par les attacks et les scripts pour résoudre PGN/SPN.
    - Fichier: [src/canlock/decoder.py](src/canlock/decoder.py)
  - **src/canlock/db/models.py**: définitions SQLModel/ORM: `CanMessage`, `PgnDefinition`, `SpnDefinition`, `AnalogAttributes`, `DefinedDigitalValues`.
    - Fichier: [src/canlock/db/models.py](src/canlock/db/models.py)
  - **src/canlock/db/database.py**: session utilities: `get_session()`, `init_db()`.
    - Fichier: [src/canlock/db/database.py](src/canlock/db/database.py)

**Formats de sortie & Emplacements**
- Sorties usuelles: `data/cache/*.pkl` (pickle DataFrame) et `data/cache/*.txt` ou `*.csv` (CSV texte avec colonne `payload_hex`).
- Colonnes clés dans l'export CSV: `timestamp`, `can_identifier` (formaté en hex dans l'export), `pgn`, `src`, `length`, `payload_hex`, `attack_type`, (optionnel) `replay_source_index`, `replay_delay_s`.

**Flux d'exécution recommandé**
1. Lancer DB si nécessaire (docker-compose up -d) - vérifier connexion.
2. Générer un dataset attaqué (ex. SPN ciblé 161, sortie CSV lisible):
   ```bash
   uv run python -m scripts.attack_synthesis --limit 2000 --out data/cache/attacked_all_attacks.txt --spn 161
   ```
3. Inspecter et valider la contrainte SPN unique:
   ```bash
   uv run python -m scripts.attack_inspect --pkl data/cache/attacked_all_attacks.txt --preview --validate --n 20
   ```
4. Visualiser (aperçu payload / série SPN):
   ```bash
   uv run python -m scripts.attack_visualize --pkl data/cache/attacked_all_attacks.txt --spn 161 --out data/cache/spn161.png
   ```

**Validation & Tests**
- La validation `--validate` compare les frames attaquées aux originales en recherchant changements de valeur par SPN (extrait via SessionDecoder). Elle suppose que les timestamps et can_identifier des lignes attaquées correspondent aux originales pour faire la comparaison.
- Cas spéciaux:
  - Les trames Bus‑Off et les payloads supprimés sont exportées avec payload_hex vide/NULL; `attack_visualize` mappe ces trames sur une valeur négative pour les tracer.
  - Les timestamps doivent être homogènes (les exports CSV convertissent correctement les timestamps en ISO). Si vous générez des replay avec types mixtes, `attack_visualize` convertit via `pd.to_datetime`.

**Ordre d'application des attaques**

Le script `attack_synthesis.py` accepte désormais l'option `--attack-order` pour contrôler l'ordre d'exécution des attaques.

- Usage :
  ```bash
  uv run python -m scripts.attack_synthesis --limit 2000 --out data/cache/ordered_attacks.txt --attack-order suspension,masquerade,spoofing
  ```

- Valeurs valides (séparées par des virgules) : `spoofing`, `masquerade`, `replay`, `suspension`.
- Synonymes acceptés : `spoof` -> `spoofing`, `masq` -> `masquerade`, `susp` -> `suspension`.

Remarques :
- L'ordre influe fortement sur le résultat final (par ex. appliquer `suspension` avant `masquerade` favorise la mise hors réseau d'un ECU, tandis que faire `masquerade` d'abord peut réduire les effets de suspension sur certaines sources).
- Testez les ordres sur de petites fenêtres (`--limit`) pour mesurer l'impact avant d'exécuter de gros exports.
