# Analyse de sentiments avec Hugging Face

Projet realise dans le cadre du cours **Large Language Models**.

## 1. Objectif du projet

Ce projet etudie l'utilisation de modeles de langage modernes pour resoudre une tache d'**analyse de sentiments** sur des avis clients en francais.

La problematique choisie est la suivante :

> Peut-on classer automatiquement un avis textuel en francais comme positif ou negatif a l'aide de modeles pre-entraines et de methodes de traitement automatique du langage ?

Le cas d'usage est pertinent pour des plateformes de commerce, de streaming, de tourisme ou de service client, car il permet de suivre automatiquement la satisfaction des utilisateurs a partir de commentaires textuels.

## 2. Jeu de donnees

Le projet utilise le jeu de donnees Hugging Face **allocine**, disponible via la bibliotheque `datasets`.

Ce dataset contient des critiques de films en francais annotees avec deux classes :

- `0` : avis negatif
- `1` : avis positif

Les donnees sont separees en ensembles d'entrainement, validation et test.

## 3. Methodologie

Le projet suit les etapes suivantes :

1. Chargement du dataset Hugging Face.
2. Exploration des donnees textuelles.
3. Nettoyage simple des textes.
4. Entrainement d'une baseline classique :
   - TF-IDF
   - Regression logistique
5. Specialisation d'un modele Transformer pre-entraine.
6. Evaluation avec des metriques adaptees :
   - accuracy
   - precision
   - recall
   - F1-score
   - matrice de confusion
7. Analyse et interpretation des resultats.

## 4. Structure du depot

```text
.
|-- README.md
|-- requirements.txt
|-- app.py
|-- .gitignore
|-- src
|   |-- config.py
|   |-- data_utils.py
|   |-- explore_data.py
|   |-- train_baseline.py
|   |-- train_transformer.py
|   `-- predict.py
|-- reports
|   `-- results_template.md
|-- models
`-- outputs
```

## 5. Installation

Creer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement :

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Pour reproduire aussi l'entrainement Transformer localement :

```bash
pip install -r requirements-full.txt
```

## 6. Exploration des donnees

Pour explorer le dataset :

```bash
python src/explore_data.py
```

Ce script affiche :

- la structure du dataset ;
- le nombre d'exemples par split ;
- la distribution des labels ;
- quelques exemples d'avis ;
- des statistiques simples sur la longueur des textes.

## 7. Baseline TF-IDF + regression logistique

Pour entrainer et evaluer la baseline :

```bash
python src/train_baseline.py
```

Cette approche transforme les textes en vecteurs TF-IDF, puis entraine un classifieur de regression logistique. Elle sert de reference interpretable avant l'utilisation d'un modele de type Transformer.

Les resultats sont sauvegardes dans :

```text
outputs/baseline_metrics.json
outputs/baseline_confusion_matrix.png
models/baseline_tfidf_logreg.joblib
```

## 8. Modele Transformer

Pour specialiser un modele pre-entraine sur un sous-ensemble du dataset :

```bash
python src/train_transformer.py
```

Par defaut, le script utilise le modele :

```text
prajjwal1/bert-tiny
```

Ce choix permet de realiser une experimentation Transformer beaucoup plus legere sur une machine personnelle. Pour une experience plus ambitieuse, il est possible de remplacer ce modele par `distilbert-base-multilingual-cased`, mais l'entrainement sera nettement plus long sans GPU.

Les parametres sont volontairement limites dans `src/config.py` afin de permettre une experimentation raisonnable sur une machine personnelle. Pour une evaluation plus solide, il est possible d'augmenter :

- `TRAIN_SAMPLE_SIZE`
- `EVAL_SAMPLE_SIZE`
- `NUM_EPOCHS`

Le modele fine-tune est sauvegarde dans :

```text
models/transformer_sentiment/
```

## 9. Prediction sur de nouveaux avis

Apres entrainement du Transformer, il est possible de tester une prediction :

```bash
python src/predict.py "Ce film est vraiment excellent, les acteurs sont remarquables."
```

Exemple de sortie attendue :

```text
Texte: Ce film est vraiment excellent, les acteurs sont remarquables.
Prediction: positif
Score: 0.94
```

## 10. Rendu visuel avec Streamlit

Le projet contient aussi une interface visuelle pour presenter le travail :

```bash
streamlit run app.py
```

Cette application permet de visualiser :

- la problematique du projet ;
- la distribution des sentiments dans le dataset ;
- des statistiques sur la longueur des avis ;
- des exemples de textes ;
- les resultats des modeles apres entrainement ;
- une prediction interactive sur un nouvel avis.

## 10.1 Deploiement en ligne

L'application est prevue pour etre deployee facilement avec **Streamlit Community Cloud**.

Etapes :

1. Pousser ce projet sur un depot GitHub public.
2. Aller sur `https://share.streamlit.io/`.
3. Cliquer sur `New app`.
4. Choisir le depot GitHub du projet.
5. Indiquer le fichier principal :

```text
app.py
```

6. Lancer le deploiement.

Le fichier `runtime.txt` fixe une version Python stable pour le cloud, et le dossier `.streamlit/` contient la configuration visuelle de l'application.

Le deploiement cloud utilise `requirements.txt`, volontairement plus leger pour eviter les erreurs de compilation de dependances lourdes. Le fichier `requirements-full.txt` contient les dependances supplementaires necessaires pour relancer l'entrainement Transformer en local.

Vercel est surtout adapte aux applications JavaScript/Next.js. Pour une application Streamlit Python, Streamlit Community Cloud est plus simple et plus fiable.

## 11. Resultats attendus

Les resultats exacts dependent de la taille d'echantillon utilisee et de la machine. La baseline TF-IDF donne generalement deja de bons resultats sur cette tache, car les avis positifs et negatifs contiennent souvent des indices lexicaux explicites.

Le modele Transformer devrait mieux capturer :

- le contexte des mots ;
- certaines formulations longues ;
- des nuances semantiques ;
- des tournures moins evidentes lexicalement.

Les resultats finaux doivent etre reportes dans `reports/results_template.md` apres execution des scripts.

## 12. Limites

Ce projet presente plusieurs limites :

- le dataset porte sur des critiques de films, donc la generalisation vers d'autres domaines n'est pas garantie ;
- l'entrainement complet d'un Transformer peut etre couteux sans GPU ;
- la classification binaire ne capture pas les avis neutres ou ambigus ;
- certains textes peuvent contenir de l'ironie, difficile a detecter automatiquement.

## 13. Ameliorations possibles

Plusieurs extensions sont possibles :

- comparer plusieurs modeles Hugging Face francophones ;
- utiliser un modele deja fine-tune sur l'analyse de sentiments ;
- ajouter une classe neutre ;
- tester le modele sur un autre domaine, par exemple des avis produits ;
- deployer une petite interface avec Streamlit ou Gradio.

## 14. Reproductibilite

Pour reproduire le projet :

1. Cloner le depot.
2. Installer les dependances avec `pip install -r requirements.txt`.
3. Executer `python src/explore_data.py`.
4. Executer `python src/train_baseline.py`.
5. Executer `python src/train_transformer.py`.
6. Lancer `streamlit run app.py` pour ouvrir le rendu visuel.
7. Completer le fichier `reports/results_template.md` avec les resultats obtenus.
