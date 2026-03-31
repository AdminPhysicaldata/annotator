# Dataset Synthétique de Test - VIVE Labeler

## Description

Dataset synthétique généré automatiquement pour tester et valider VIVE Labeler.

**Format**: LeRobot v3.0 compatible
**Taille**: ~2 MB
**Générateur**: `scripts/create_test_dataset.py`

## Contenu

### Episodes

| Episode | Frames | Durée | Trajectoire |
|---------|--------|-------|-------------|
| 0 | 150 | 5.0s | Cercle rayon 0.5m |
| 1 | 150 | 5.0s | Cercle rayon 0.7m |

**Total**: 300 frames, 10 secondes

### Données VIVE

Chaque frame contient:

- **Position 3D** (mètres):
  - `x`: Coordonnée X (trajectoire circulaire)
  - `y`: Coordonnée Y (trajectoire circulaire)
  - `z`: Coordonnée Z (légère ondulation)

- **Rotation** (quaternions):
  - `w`, `x`, `y`, `z`: Rotation autour de l'axe Z

### Vidéo

**Résolution**: 640x480
**FPS**: 30
**Format**: MP4 (H.264)
**Contenu**: Cercle coloré suivant la même trajectoire que les données VIVE

**Effets visuels**:
- Couleur changeante (fonction de l'angle)
- Tracé de trajectoire (50 derniers points)
- Numéro de frame affiché

## Structure

```
test_dataset/
├── README.md                  # Ce fichier
├── data/
│   └── chunk_000000.parquet   # Données capteurs (300 lignes)
├── meta/
│   ├── episodes/
│   │   └── episodes.parquet   # Métadonnées (2 épisodes)
│   ├── info.json              # Schéma dataset
│   └── stats.json             # Statistiques position
└── videos/
    └── front_camera/
        ├── episode_000000.mp4 # Episode 0 (5s)
        └── episode_000001.mp4 # Episode 1 (5s)
```

## Utilisation

### Ouvrir dans VIVE Labeler

```bash
# Lancer l'application
python -m vive_labeler

# Dans l'application:
# File → Open Dataset → Sélectionner 'test_dataset'
```

### Visualisation attendue

1. **Vidéo**: Cercle coloré tournant dans le sens horaire
2. **Vue 3D**: Trajectoire circulaire dans le plan XY
3. **Timeline**: 150 frames par épisode

### Navigation

- `→`: Frame suivant
- `←`: Frame précédent
- `Espace`: Play/Pause
- Clic timeline: Aller à un frame

## Données techniques

### Trajectoire mathématique

```python
# Episode 0: radius = 0.5m
# Episode 1: radius = 0.7m

angle(t) = 4π * t / duration  # 2 rotations complètes

x(t) = radius * cos(angle(t))
y(t) = radius * sin(angle(t))
z(t) = 1.0 + 0.1 * sin(2 * angle(t))

# Rotation autour Z
quat.w = cos(angle(t) / 2)
quat.z = sin(angle(t) / 2)
quat.x = 0
quat.y = 0
```

### Statistiques position

**Episode 0** (radius 0.5m):
- X: min=-0.5, max=0.5
- Y: min=-0.5, max=0.5
- Z: min=0.9, max=1.1

**Episode 1** (radius 0.7m):
- X: min=-0.7, max=0.7
- Y: min=-0.7, max=0.7
- Z: min=0.9, max=1.1

## Régénération

Si vous souhaitez régénérer ou personnaliser ce dataset:

```bash
python scripts/create_test_dataset.py \
    --output-dir ./test_dataset \
    --episodes 2 \
    --frames 150 \
    --fps 30
```

**Options**:
- `--episodes N`: Nombre d'épisodes
- `--frames N`: Frames par épisode
- `--fps F`: Framerate vidéo

## Validation

Vérifier l'intégrité du dataset:

```bash
python scripts/simple_test.py
```

**Résultat attendu**:
```
✅ Dataset structure valid
✅ Metadata readable
✅ Sensor data readable
✅ Video files accessible
```

## Limitations

Ce dataset synthétique:
- ✅ Valide la structure LeRobot v3.0
- ✅ Teste synchronisation vidéo/capteurs
- ✅ Valide visualisation 3D
- ✅ Permet test complet de labellisation
- ❌ N'a pas le réalisme de données réelles
- ❌ Trajectoire très simple (cercle parfait)
- ❌ Pas de bruits/perturbations

Pour tester avec données réelles, utilisez:
```bash
python scripts/download_lerobot_dataset.py --dataset pusht
```

## Cas d'usage

### Test rapide (2 min)
1. Ouvrir dataset
2. Naviguer quelques frames
3. Observer synchronisation vidéo/3D

### Test labellisation (5 min)
1. Créer label "circular_motion"
2. Annoter frames 0-75 (première rotation)
3. Créer label "second_rotation"
4. Annoter frames 76-150 (deuxième rotation)
5. Exporter en JSON

### Test export (3 min)
1. Annoter quelques frames
2. Export JSON → Vérifier structure
3. Export CSV → Ouvrir dans Excel/Python
4. Export LeRobot → Vérifier Parquet

## Support

Questions ou problèmes:
- Voir [TESTING_GUIDE.md](../TESTING_GUIDE.md)
- Voir [USER_GUIDE.md](../USER_GUIDE.md)

**Généré le**: Février 2026
**Version VIVE Labeler**: 0.1.0
**Format**: LeRobot v3.0
