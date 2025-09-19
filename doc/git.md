# git

## Introduction

Quand tu travailles sur un projet, comme rédiger ton rapport sur Word, et que tu fais plein de changements, il se pourrait qu'un jour, tu supprimes accidentellement un paragraphe important et tu voudrais revenir en arrière. Et beh sans `git`, tu serais peut-être obligé de chercher une ancienne version que tu aurais sauvegardée manuellement, ou de tout réécrire, pas cool non ? C'est justement là que `git` entre en jeu.

`git` est un outil qui te permet de sauvegarder l'historique de toutes les modifications de ton projet, un peu comme une machine à remonter le temps pour ton code. À chaque fois que tu fais une modification importante (par exemple, quand tu ajoutes une nouvelle fonctionnalité ou que tu corriges une erreur), tu peux garder en mémoire l'état actuel et précédent de ton projet, cette sauvgarde s'appelle un commit.

## Installation

Rien de plus simple, rends toi sur le [site officiel et télécharge la version qui correspond à ton OS](https://git-scm.com/downloads).

## Utilisation

1. *Cloner un dépôt:*
`git clone https://github.com/cia-ulaval/CANlock.git`

2. *Ajouter des modifications pour le commit à venir:*

Tous les fichiers modifiés:
```sh
git add .
```

Seulement les fichiers séléctionnés:
```sh
git add <file 1> <file 2> ... <file n>
```

3. *Effectuer le commit, donc la sauvegarde de votre code:*

```sh
git commit -m "<important de bien mettre un message qui décrit très brièvement le commit>"
```

4. *Pousser les modifications sur le dépôt en ligne (github):"*

```sh
git push
```

5. *Si des modifications ont été faites par ton collègue:*

```sh
git pull
```

6. *Créer une branche:*

```sh
git checkout -b <nom de la branche>
```

7. *Changer de branche:*

```sh
git checkout <nom de la branche>
```

8. *Ramener un fichier d'une autre branche sur votre branche*

```sh
git checkout <nom de la branche> <file 1> <file 2> ... <file n>
```

Pour tout ce qui est des `merges` nous verrons cela en temps voulu, dès que l'occasion se présentera, pour éviter de te surcharger d'informations.

## Fonctionnement

Durant le projet, nous entretiendrons 3 branches, une pour chaque équipe:

- `team-1`
- `team-2`
- `team-3`

Chaque étudiant partira une nouvelle branche depuis la branche de son équipe et crééra ensuite une merge request ou un pull request sur la branche de la team.
Un (ou plusieurs) autres étudiants de son équipe devra(ont) faire la review de ses modifications et devra(ont) l'accépter pour procéder au merge.

Donc, tous les détails concernant les merge/pull request te sera décrites un peu plus tard dans ce même document un peu plus tard.
