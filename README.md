# VIVE Labeler - Multimodal Data Annotation Tool

Application professionnelle de labellisation avancée pour données multimodales combinant vidéo et tracking spatial 3D HTC VIVE.

## Caractéristiques

### Ingestion de données
- Support natif du format LeRobotDataset v3.0 (Hugging Face)
- Chargement de séquences synchronisées (vidéo + états capteurs)
- Support streaming pour datasets volumineux
- Compatible données locales et distantes (Hugging Face Hub)

### Synchronisation
- Alignement précis frame vidéo ↔ timestamps capteurs
- Interpolation temporelle adaptative
- Gestion des écarts de fréquence d'échantillonnage

### Visualisation
- Lecteur vidéo avec navigation frame par frame
- Affichage superposé des données 3D (position, orientation, trajectoires)
- Vue 3D interactive pour inspection spatiale
- Timeline avec marqueurs d'événements

### Labellisation
- Labels personnalisables multi-classes/multi-tags
- Annotation par frame ou intervalle temporel
- Export structuré compatible pipeline ML
- Format de sortie LeRobot-compatible

## Architecture

```
vive_labeler/
├── src/
│   ├── core/              # Ingestion, synchronisation, transformations 3D
│   ├── visualization/     # Lecteur vidéo, vue 3D, timeline, overlays
│   ├── labeling/          # Gestion labels, annotations, export
│   ├── ui/                # Interface PyQt6
│   └── utils/             # Configuration, helpers
└── tests/                 # Tests unitaires
```

## Installation

### Prérequis
- Python 3.10+
- PyQt6
- HuggingFace `lerobot >= 0.4.0`

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python -m vive_labeler
```

## Format de données

Compatible avec LeRobotDataset v3.0 :
- **Vidéos** : MP4 synchronisées (multi-caméras)
- **États/Actions** : Fichiers Parquet (positions, orientations, timestamps)
- **Métadonnées** : info.json, stats.json, tasks.parquet
- **Tracking VIVE** : Position (x,y,z), rotation (quaternions ou euler), matrices de transformation

## Export

Les annotations sont exportées dans un format structuré :
- JSON (structure flexible)
- CSV (tabular)
- Format LeRobot (intégration directe pipeline ML)

## Sources techniques

- [LeRobotDataset v3.0 Documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Blog: Bringing large-scale datasets to lerobot](https://huggingface.co/blog/lerobot-datasets-v3)
# annotator
# annotator
