# Resultats experimentaux

Ce fichier resume les observations et resultats obtenus apres execution des scripts.

## Dataset

- Dataset : `allocine`
- Tache : classification binaire de sentiment
- Labels :
  - `0` : negatif
  - `1` : positif

## Exploration

- Nombre d'exemples :
  - train : 160000
  - validation : 20000
  - test : 20000
- Repartition des classes :
  - train : 79413 avis negatifs, 80587 avis positifs
  - validation : 10204 avis negatifs, 9796 avis positifs
  - test : 10408 avis negatifs, 9592 avis positifs
- Longueur moyenne :
  - train : 91.30 mots
  - validation : 92.50 mots
  - test : 92.53 mots
- Observation : les classes sont globalement equilibrees, ce qui permet d'utiliser des metriques classiques comme accuracy, precision, recall et F1-score.

## Resultats baseline TF-IDF + regression logistique

| Modele | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.879 | 0.879 | 0.879 | 0.879 |

Commentaires :

- La baseline obtient deja de bons resultats. Cela montre que les avis positifs et negatifs contiennent souvent des indices lexicaux forts, bien captes par TF-IDF.

## Resultats Transformer

| Modele | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| BERT tiny fine-tune | 0.475 | 0.497 | 0.475 | 0.387 |

Commentaires :

- Le modele Transformer utilise une version legere (`prajjwal1/bert-tiny`) afin de rendre l'experimentation realisable sur CPU. Les resultats sont faibles, ce qui s'explique par la petite taille du modele, le faible nombre d'exemples utilises et l'unique epoque d'entrainement.

## Analyse comparative

La baseline TF-IDF + regression logistique obtient les meilleurs resultats avec un F1-score d'environ 0.879. Elle est rapide, stable et bien adaptee a cette tache car les avis positifs et negatifs contiennent souvent des mots fortement associes au sentiment.

Le modele Transformer leger obtient un F1-score d'environ 0.387. Ce resultat ne signifie pas que les Transformers sont moins pertinents, mais plutot que le modele utilise est tres petit et entraine avec peu de donnees pour rester executable sur CPU. Un modele plus grand, comme `distilbert-base-multilingual-cased`, et un entrainement plus long devraient ameliorer les performances, mais demanderaient davantage de ressources.

Cette comparaison montre l'interet d'avoir une baseline classique : elle permet de verifier qu'une methode simple peut etre tres competitive avant d'utiliser des modeles plus couteux.

## Conclusion

Ce projet montre qu'il est possible de construire un pipeline complet d'analyse de sentiments avec Hugging Face : chargement de donnees textuelles reelles, exploration, nettoyage, entrainement, evaluation et prediction. La baseline classique donne de tres bons resultats sur ce dataset. L'experimentation Transformer illustre l'utilisation de modeles pre-entraines, mais montre aussi l'importance des ressources de calcul et du choix du modele.
