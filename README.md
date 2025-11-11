# ReSUnet_Segmentation

## Présentation
Projet de segmentation d'images médicales utilisant une architecture **ReSUnet** (U-Net avec blocs résiduels). Le notebook principal inclut la préparation des données (LiTS2017, LIDC-IDRI via Kaggle), la définition du modèle, l'entraînement et l'évaluation des performances ainsi que des visualisations des masques prédits.

## Objectifs
- Implémenter et entraîner un modèle ReSUnet pour la segmentation d'organes/tumeurs.
- Comparer les résultats visuels et quantitatives (Dice, IoU) sur des échantillons de validation.

## Contenu du dépôt
```
ReSUnet_Segmentation/
├── README.md
├── requirements.txt
├── notebooks/               # notebook principal (nettoyé)
├── src/                     # scripts Python réutilisables
├── images/examples/         # images d'exemple (entrée / prédiction)
└── reports/                 # résumé des résultats
```

## Installation
1. Créer un environnement Python (venv ou conda)
```bash
python -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` sous Windows
pip install -r requirements.txt
```

2. Ajouter vos clés Kaggle si vous souhaitez télécharger les datasets :
- Placez `kaggle.json` dans le dossier `~/.kaggle/` ou utilisez l'upload dans Colab.
- Les notebooks contiennent les instructions d'utilisation de l'API Kaggle.

## Exécution
- Pour entraîner le modèle : `python src/train.py --data_dir path/to/data --epochs 50 --batch_size 8`
- Pour lancer l'inférence : `python src/inference.py --model checkpoints/model_last.h5 --input examples/input.png`

## Metrics 
- Dice coefficient, IoU, précision/recall par classe.
- Visualisation des courbes loss & metric pendant l'entraînement.


