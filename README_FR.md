# Call Me Maybe
*This project has been created as part of the 42 curriculum by cpietrza.*

## Description
**Call Me Maybe** est un projet pédagogique consacré à l'implémentation, à partir de zéro, de l'appel de fonctions (*function calling*) pour les grands modèles de langage (LLM). L'objectif est de construire un mécanisme fiable faisant le pont entre des requêtes en langage naturel et des appels de fonctions JSON structurés et exécutables par une machine, même avec de très petits modèles de langage (comme `Qwen/Qwen3-0.6B`, comptant environ 600 millions de paramètres).

Le projet repose sur la mise en place d'un système de **décodage contraint** (*constrained decoding*) qui guide la génération token par token pour garantir une conformité syntaxique et sémantique à 100 %, évitant ainsi les hallucinations de syntaxe JSON courantes lors d'une génération libre.

## Instructions

### Prérequis
- Python 3.10 ou version ultérieure
- Le gestionnaire de paquets et d'environnements `uv`
- Le paquet interne `llm_sdk` placé à la racine du projet

### Installation
Pour installer les dépendances du projet et synchroniser l'environnement virtuel :
```bash
make install
```
*Note : Le projet peut également être initialisé directement avec la commande standard :*
```bash
uv sync
```

### Exécution
Pour exécuter le programme principal avec les fichiers et arguments par défaut :
```bash
make run
```
Conformément au sujet, le programme peut aussi être lancé directement via `uv run` :
```bash
uv run python -m src [--functions_definition <chemin>] [--input <chemin>] [--output <chemin>]
```
Par exemple :
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

Les fichiers d'entrée et de sortie peuvent également être personnalisés via la variable `ARGS` du `Makefile` :
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json"
```

Il est également possible de modifier la définition des fonctions et le modèle utilisé :
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json --functions_definition data/input/custom_functions.json --model Qwen/Qwen3-1.7B"
```

### Fonctionnalités bonus implémentées
- **Prise en charge de plusieurs modèles LLM** : configurable via l'argument `--model` (`Qwen/Qwen3-0.6B` par défaut, `Qwen/Qwen3-1.7B` et `Qwen/Qwen3-0.6B-Base`).
- **Suite de tests complète** : création d'un script `src/test.py` exécutable avec `make test`.
- **Personnalisation des fichiers d'entrée et de sortie** : configurable en CLI ou via `ARGS`.
- **Cache et visualiseur** : intégration du cache de génération et d'une visualisation de l'espace de tokens lors du décodage.

### Toutes les commandes (Makefile)
- `make install` : installe les dépendances du projet via `uv`.
- `make run` : lance le programme par défaut (prend en charge `ARGS`).
- `make cache` : lance le programme en activant le cache.
- `make visual` : lance le programme avec le visualiseur de décodage.
- `make multi` : permet de sélectionner interactivement le modèle LLM à exécuter.
- `make test` : exécute les tests d'intégration automatisés de bout en bout (`src/test.py`).
- `make lint` : exécute les vérifications Flake8 et Mypy selon les exigences du sujet (`flake8` et `mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`).
- `make lint-strict` : exécute `flake8` et `mypy --strict`.
- `make clean` : supprime tous les fichiers de cache et temporaires (`__pycache__`, `.mypy_cache`, etc.).
- `make debug` : lance le script dans le débogueur interactif Python (`pdb`).

### Les arguments disponibles
| Argument | Description | Valeur par défaut |
|---|---|---|
| `--input` | Chemin vers le fichier JSON contenant les prompts de test. | `data/input/function_calling_tests.json` |
| `--output` | Chemin vers le fichier JSON dans lequel les résultats seront écrits. | `data/output/function_calling_results.json` |
| `--functions_definition` | Chemin vers le fichier JSON contenant les définitions des fonctions. | `data/input/functions_definition.json` |
| `--model` | Modèle LLM à utiliser. | `Qwen/Qwen3-0.6B` |
| `--multi` | Active la sélection interactive du modèle. | `False` |
| `--cache` | Active le cache. | `False` |
| `--visual` | Active le visualiseur. | `False` |

Les modèles disponibles pour `--model` sont :
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-0.6B-Base`

### Exemples avec `ARGS`
Utiliser uniquement un autre fichier d'entrée :
```bash
make run ARGS="--input data/input/custom_tests.json"
```

Utiliser un autre fichier d'entrée et de sortie :
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json"
```

Utiliser un autre modèle :
```bash
make run ARGS="--model Qwen/Qwen3-1.7B"
```

Combiner plusieurs options :
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json --functions_definition data/input/custom_functions.json --model Qwen/Qwen3-1.7B --cache"
```

## Ressources

### Références techniques
- Référence sur le *Constrained Decoding* : [Hugging Face Text Generation Strategies & LogitsProcessors](https://huggingface.co/docs/transformers/main_classes/text_generation)
- Documentation officielle de Pydantic : [Pydantic Documentation](https://docs.pydantic.dev/) pour la validation stricte des schémas.
- Spécification standard JSON (RFC 8259).

### Utilisation de l'Intelligence Artificielle
Conformément aux directives du sujet (Chapitre II & Chapitre VI), l'assistance par IA a été employée pour des tâches ciblées :
- **Fichier de test (`src/test.py`)** : L'IA a été utilisée pour aider à générer et structurer la suite de tests automatisée, notamment pour définir une variété de cas limites (nombres négatifs, flottants, chaînes complexes avec regex ou caractères spéciaux, fonctions sans arguments) et comparer les sorties au schéma attendu.
- **Documentation (`README.md`)** : L'IA a été utilisée comme assistant de rédaction pour structurer, synthétiser et formater le présent fichier README afin de s'assurer de la conformité exhaustive avec l'ensemble des exigences du barème de 42.
- *Note d'intégrité* : L'implémentation algorithmique du décodage contraint (manipulation des logits, logique d'automate grammatical et masquage direct du vocabulaire sans librairies tierces d'inférence) a été intégralement réalisée et comprise par l'étudiant.

## Explication de l'algorithme
Notre algorithme de *constrained decoding* intercepte le processus de génération des tokens du modèle, un token à la fois, en agissant directement sur la distribution de probabilité (logits) avant la sélection du token suivant :
1. **Évaluation de l'état structurel** : À chaque position de la séquence, un analyseur d'état détermine la nature de l'élément attendu dans la syntaxe JSON (ouverture d'accolade, clé, séparateur, chaîne de caractères, nombre flottant/entier, fermeture).
2. **Filtrage des tokens admissibles** : À partir de la correspondance ID-texte extraite du vocabulaire via `get_path_to_vocab_file()`, nous identifions les tokens valides autorisés par l'état syntaxique et par le typage de l'argument décrit dans `functions_definition.json`.
3. **Masquage des logits** : Tout token incompatible voit son score logit fixé à `-inf` (masquage vectorisé).
4. **Sélection du token** : Le token ayant le logit le plus élevé parmi les tokens valides restants est sélectionné (sélection gloutonne). Ce processus se répète de manière itérative jusqu'à l'obtention d'un objet JSON complet et parfaitement valide.

## Choix de conception
- **Respect strict des contraintes de dépendances** : Aucune bibliothèque de haut niveau interdite (telles que PyTorch, Transformers, Outlines ou DSPy) n'est importée dans le code source. Seuls `numpy`, `pydantic` et la bibliothèque standard Python sont mis à profit.
- **Validation avec Pydantic** : Toutes les entrées (prompts, définitions de fonctions) et sorties d'appels de fonction sont instanciées à travers des modèles Pydantic pour garantir le typage statique et la conformité au schéma.
- **Robustesse et gestion des erreurs** : L'ensemble des lectures de fichiers et des parsings JSON est enveloppé dans des blocs `try/except` avec gestionnaires de contexte (`with`) pour prévenir les fuites de descripteurs et retourner des messages d'erreur explicites en cas de fichier mal formé ou manquant.

## Analyse des performances
- **Précision syntaxique** : 100 % des sorties générées sont des objets JSON valides et conformes au format requis.
- **Exactitude fonctionnelle** : Taux d'exactitude supérieur à 90 % sur le choix de fonction et l'extraction des arguments, y compris sur le modèle léger `Qwen3-0.6B`.
- **Vitesse** : L'ensemble du jeu de tests standard s'exécute en moins de 5 minutes sur du matériel standard grâce à l'optimisation des opérations de masquage sous NumPy.

## Difficultés rencontrées
1. **Gestion des sous-mots et caractères spéciaux** : Le tokenizer BPE découpe les mots en sous-tokens incluant des préfixes d'espaces (tels que `Ġ` ou ` `). La correspondance fine entre tokens et fragments autorisés a demandé une gestion rigoureuse pour ne pas exclure des sous-tokens valides.
2. **Chaînes complexes et expressions régulières** : La génération d'arguments contenant des caractères d'échappement (ex. `\d+`) a nécessité d'étendre dynamiquement les caractères admis dans les valeurs textuelles sans briser la structure JSON.
3. **Distinction et cohérence numérique** : La validation des types numériques (entiers vs flottants, signes négatifs, gestion du point décimal unique) a nécessité un suivi d'état dédié pour éviter toute séquence mal formée (comme un double point décimal).

## Stratégie de test
- **Tests unitaires et fonctionnels (`src/test.py`)** : Exécution automatisée de tests couvrant l'ensemble des fonctions types (calculs, salutations, manipulation de chaînes, conversions) et vérification stricte des types de retour.
- **Vérification statique du code** : Conformité validée par `make lint` (`flake8` et `mypy` sans avertissement ni omission de typage).
- **Tests de résistance aux erreurs** : Vérification de la résilience du programme face à des fichiers inexistants, des arguments manquants ou des schémas malformés.

## Exemple d'utilisation
Si `data/input/function_calling_tests.json` contient :
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

Le fichier de sortie `data/output/function_calling_results.json` contiendra exactement :
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```
