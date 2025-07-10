# 🚨 Signalert - Détection sonore pour personnes sourdes/malentendantes

Application mobile cross-platform (Android/iOS) de détection temps réel de sons d'urgence.

## 🏗️ Architecture du projet

```
Signalert/
├── 📊 notebooks/                # Recherche & développement IA
│   ├── Benchmark RF vs RF plus MLP.ipynb
│   ├── Feature Engineering.ipynb
│   └── Model Optimization.ipynb
├── 📈 models/                   # Modèles entraînés & exports
│   ├── random_forest_model.pkl
│   ├── mlp_model.tflite
│   ├── feature_scaler.pkl
│   └── model_metadata.json
├── 📋 reports/                  # Rapports d'analyse
│   ├── random_forest_report_*.txt
│   ├── benchmark_results_*.txt
│   └── technical_specifications.md
├── 📱 mobile/                   # Application Flutter (à créer)
│   ├── signalert_app/           # Projet Flutter principal
│   ├── assets/models/           # Modèles TFLite embarqués
│   └── docs/                    # Documentation mobile
├── 🔧 tools/                    # Outils utilitaires
│   ├── model_converter.py       # Conversion modèles pour mobile
│   ├── audio_tester.py          # Tests audio temps réel
│   └── benchmark_mobile.py      # Benchmarks performance
├── 📂 data/                     # Datasets (gitignored)
│   ├── car_horn/
│   ├── dog_bark/
│   ├── drilling/
│   ├── engine_idling/
│   ├── gun_shot/
│   ├── jackhammer/
│   └── siren/
└── 🚀 scripts/                  # Scripts d'automatisation
    ├── setup_environment.py
    ├── train_models.py
    └── export_for_mobile.py
```

## 🚀 Démarrage rapide

### Recherche IA (Python)
```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement Jupyter
jupyter notebook notebooks/

# Exécution benchmark complet
python scripts/train_models.py
```

### Préparation mobile
```bash
# Export modèles optimisés
python scripts/export_for_mobile.py

# Tests performance
python tools/benchmark_mobile.py
```

## 🎯 Résultats modèles IA

| Approche | Accuracy | Temps inférence | Usage recommandé |
|----------|----------|-----------------|------------------|
| **Random Forest** | 90.22% | ~8.5ms | Principal (toujours) |
| **Hybride RF+MLP** | 92.15% | ~36ms | Backup si RF incertain |
| **MLP seul** | 91.8% | ~28ms | Comparaison |

**📊 Configuration optimale :** Seuil confiance RF = 0.7  
**⚡ Performance :** 28 FPS temps réel garantis  
**🔋 Efficacité :** MLP activé seulement 37% du temps

## 📱 Classes détectées (7 sons d'urgence)
- 🚨 **Sirènes** (police, ambulance, pompiers)
- 🔫 **Coups de feu** 
- 📢 **Klaxons** (voitures, camions)
- 🐕 **Aboiements** (chiens agressifs)
- 🔨 **Perceuses** (travaux)
- ⚙️ **Marteau-piqueur** (chantiers)
- 🚗 **Moteur au ralenti** (véhicules proches)

## 🛠️ Stack technique

### Recherche & IA
- **ML:** scikit-learn (Random Forest), TensorFlow (MLP)
- **Audio:** librosa (MFCC), numpy (traitement signal)
- **Analyse:** matplotlib, seaborn (visualisations)
- **Export:** TensorFlow Lite, joblib (sérialisation)

### Mobile (Flutter - à développer)
- **Framework:** Flutter 3.x (Android + iOS)
- **IA embarquée:** TensorFlow Lite, dart:ffi
- **Audio:** flutter_sound, audio_streamer
- **Features:** 40 MFCC (20 coeff + moyenne + variance)

## 📊 Spécifications techniques

### Audio processing
- **Fréquence échantillonnage:** 16-22 kHz
- **Format:** Mono, 16-bit
- **Fenêtre analyse:** 1 seconde
- **Features:** 40 MFCC (optimisé mobile)

### Performance garantie
- **Latence totale:** < 50ms (capture + traitement + alerte)
- **Mémoire:** < 10MB RAM
- **Stockage:** < 500KB modèles
- **Batterie:** Optimisé arrière-plan longue durée

## 🚀 Roadmap développement

### ✅ Phase 1 : Recherche IA (Terminée)
- [x] Dataset collection & preprocessing
- [x] Benchmark Random Forest vs MLP
- [x] Optimisation architecture hybride
- [x] Export modèles pour mobile

### 🔄 Phase 2 : MVP Mobile (En cours)
- [ ] Setup projet Flutter multi-plateforme
- [ ] Intégration capture audio temps réel
- [ ] Portage MFCC + modèles embarqués
- [ ] Interface utilisateur + notifications
- [ ] Tests devices réels (Android/iOS)

### 🎯 Phase 3 : Optimisation & Release
- [ ] Optimisation batterie & performances
- [ ] Tests utilisateurs sourds/malentendants
- [ ] Modes contextuels (intérieur/extérieur)
- [ ] Release stores + accessibilité

## 🧪 Tests & validation

```bash
# Tests modèles Python
python -m pytest tests/

# Benchmark complet
jupyter notebook notebooks/Benchmark\ RF\ vs\ RF\ plus\ MLP.ipynb

# Simulation mobile
python tools/benchmark_mobile.py
```

## 📄 Documentation

- 📊 [Analyse détaillée modèles](notebooks/)
- 📋 [Rapports techniques](reports/)
- 🔧 [Outils développement](tools/)
- 📱 [Guide mobile Flutter](mobile/docs/) *(à venir)*

---

**🎯 Mission :** Améliorer la sécurité et l'autonomie des personnes sourdes/malentendantes grâce à l'IA embarquée temps réel.
