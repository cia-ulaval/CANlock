# CANlock 🔒

## Fiche d'Identité

*   **Type de projet :** Projet Partenaire (Entreprise)
*   **Team Lead :** Loïc Baret
*   **Partenaire Académique/Industriel :** Thales
*   **Effectif recherché :** 5 membres
*   **Profils recherchés :** IA/Data Science, Cybersécurité, Systèmes embarqués

## Description du Projet

- Aidez une équipe en cybersécurité à sécuriser une flotte de véhicules terrestres en étudiant les signaux échangés entre les capteurs et les actuateurs via le bus de données CAN (Controller Area Network). L’objectif est d’assurer la sécurité globale du véhicule, et par extension, celle de ses occupants.
- Les véhicules modernes intègrent de plus en plus d’unités électroniques embarquées, comme des capteurs et des actuateurs, qui doivent échanger en permanence des données critiques pour le fonctionnement du véhicule. Ces échanges sont assurés par le bus CAN, un protocole de communication standardisé dans l’industrie automobile. Malheureusement, un attaquant qui parvient à accéder au réseau CAN d’un véhicule peut potentiellement injecter, modifier ou supprimer des messages, et ainsi altérer le comportement du véhicule — freinage, direction, affichages, etc.
- Les modèles les plus récents peuvent atteindre des taux d’erreur inférieurs à 1%, ce qui semble très performant. Mais ce faible taux est trompeur lorsqu'on considère la quantité massive de signaux échangés — souvent plusieurs centaines de milliers par minute. Un taux d’erreur de 1% dans ce contexte peut générer des milliers de faux positifs par minute, rendant la détection peu exploitable en pratique. L’objectif est de concevoir des approches de détection plus fiables et contextuellement pertinentes, capables de distinguer des anomalies réellement critiques tout en réduisant drastiquement le taux de faux positifs.

## Objectifs & Livrables

*   **Objectif Principal :** Obtenir un modèle/algorithme de détection d'anomalies plus performant pour réduire drastiquement les faux positifs sur le bus CAN.
*   **Livrables attendus :**
    *   Pipeline(s) de traitement des données et modèle(s) testé(s).
    *   Rapport technique de mi-session et de fin de session.
    *   Article de recherche ou preuve de concept exploitable par Thales (objectif secondaire).

## Timeline Prévisionnelle de la Session

| Semaine | Activité/Phase                                                                                     |
| :-----: | :------------------------------------------------------------------------------------------------- |
|  **1-2**  | **Onboarding :** Familiarisation avec l'algorithme existant, les outils et le domaine (bus CAN). |
|  **3-4**  | **Recherche de solutions :** Proposition et étude de pistes d'amélioration.                        |
|  **5-7**  | **Développement - Phase 1 :** Itérations sur le traitement des données et premier pipeline modèle. |
|   **8**   | **Présentation d'avancement #1** (Rapport mi-session)                                              |
| **9-12**  | **Développement - Phase 2 :** Itérations d'amélioration et tests des modèles.                      |
|  **13**   | **Présentation d'avancement #2 / Préparation finale**                                              |
| **14-15** | **Finalisation :** Tests finaux, rédaction du rapport technique final et documentation.            |

## Technologies & Compétences Visées

*   **Logiciels :** Python, Librairies Data Science (PyTorch/TensorFlow, Scikit-learn, Pandas), Jupyter, Git
*   **Matériels :** Machine avec GPU (accès fourni), données CAN
*   **Compétences :** Analyse de données, Machine Learning pour la cybersécurité, reverse engineering, travail sur un projet industriel réel, rédaction technique.

## Pourquoi rejoindre ce projet ?

Tu vas aimer ce projet si :
*   Tu veux travailler sur un challenge de cybersécurité concret et critique avec un leader mondial (Thales).
*   Tu es intéressé par l'IA appliquée à la détection d'anomalies et la protection des systèmes critiques.
*   Tu veux acquérir une expérience précieuse et concrète qui impressionnera sur ton CV.

## Contact & Liens Utiles
*   **Contact :** [Thales](https://www.thalesgroup.com/en)

## Ressources

Tu retrouveras les articles séléctionnés dans la revue de littérature faite par la stagiaire Thales dans le fichier [`doc/litterature.md`](./doc/litterature.md).

## Outils

Pour l'utilisation des outils, veuillez vous référer à la documentation en place suivante:

### VS Code

Il est vivement conseillé d'utiliser l'IDE de Microsoft Visual Studio Code (ou VS Code) pour ce projet. Il est simple d'utilisation, léger et avec les bonnes extensions, il te permet de faire ce que tu veux au niveau du code (Python ou autre). De plus, il va t'éviter beaucoup de problèmes au niveau de l'environnement, puisque Microsoft t'offre la possibilité d'utiliser des devcontainers, qui est tout simplement un conteneur Docker pour développer votre code.

[Lien vers les détails sur VS Code](./doc/vscode.md)

### Git

Un incontournable du développement ! Impossible de faire de la programmation sans connaitre, ni utiliser `git`. Il te permettra de versionner ton code et de revenir en arrière si jamais quelque chose tourne mal, et il te permettra également de développer des nouvelles *features* sans casser tout le code déjà en place!

Aller, [clique sur ce lien si tu veux en savoir plus sur comment nous allons fonctionner avec git](./doc/git.md)

### UV

C'est un outil magique, un tout-en-un nécessaire en python. Une fois que tu auras testé cet outil, impossible pour toi de simplement utiliser `pip` !

Alors [clique sur ce lien qui te dirige droit aux explications !](./doc/uv.md)

### Hydra & mlflow

Nous verrons ces outils un peu plus tard au courant du projet, car pour l'instant ne t'embrouilles pas à apprendre d'un coup trop d'outils !
