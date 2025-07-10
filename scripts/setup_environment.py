"""
SIGNALERT - SETUP ENVIRONMENT
==============================
Configure l'environnement de développement Signalert.

Usage:
    python setup_environment.py --install-deps
    python setup_environment.py --create-folders --download-models
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import requests
from datetime import datetime

class EnvironmentSetup:
    """Configuration de l'environnement Signalert"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        
    def create_folder_structure(self):
        """Crée la structure de dossiers du projet"""
        print("📁 CRÉATION STRUCTURE DE DOSSIERS")
        print("=" * 50)
        
        folders = [
            "data",
            "models",
            "reports", 
            "tools",
            "scripts",
            "mobile",
            "mobile/assets",
            "mobile/assets/models",
            "mobile/docs",
            "notebooks",
            "tests",
            "tests/unit",
            "tests/integration",
            "docs",
            "docs/images",
            "docs/api"
        ]
        
        created_folders = []
        for folder in folders:
            folder_path = self.project_root / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            created_folders.append(folder_path)
            print(f"✅ {folder}")
        
        # Créer fichiers .gitkeep pour dossiers vides
        for folder_path in created_folders:
            if not any(folder_path.iterdir()):
                gitkeep_file = folder_path / ".gitkeep"
                gitkeep_file.touch()
        
        print(f"\n📁 Structure créée: {len(created_folders)} dossiers")
        return created_folders
    
    def create_requirements_file(self):
        """Crée le fichier requirements.txt"""
        print("\n📝 CRÉATION REQUIREMENTS.TXT")
        print("=" * 40)
        
        requirements = [
            "# Signalert Requirements",
            "# Generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "# Machine Learning & IA",
            "scikit-learn>=1.3.0",
            "tensorflow>=2.13.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "",
            "# Audio Processing",
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "sounddevice>=0.4.0",
            "",
            "# Visualization",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "",
            "# Utilities",
            "joblib>=1.3.0",
            "scipy>=1.11.0",
            "tqdm>=4.65.0",
            "",
            "# Development",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "",
            "# System monitoring",
            "psutil>=5.9.0",
            "",
            "# Optional - for advanced features",
            "# flask>=2.3.0  # For web interface",
            "# fastapi>=0.100.0  # For API",
            "# streamlit>=1.25.0  # For demo app"
        ]
        
        with open(self.requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        
        print(f"✅ Requirements créé: {self.requirements_file}")
        return self.requirements_file
    
    def install_dependencies(self):
        """Installe les dépendances Python"""
        print("\n📦 INSTALLATION DÉPENDANCES")
        print("=" * 40)
        
        if not self.requirements_file.exists():
            print("⚠️ requirements.txt manquant, création...")
            self.create_requirements_file()
        
        try:
            # Mettre à jour pip
            print("🔄 Mise à jour pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Installer les dépendances
            print("🔄 Installation des packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)])
            
            print("✅ Dépendances installées avec succès!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation: {e}")
            return False
        
        return True
    
    def create_config_files(self):
        """Crée les fichiers de configuration"""
        print("\n⚙️ CRÉATION FICHIERS DE CONFIGURATION")
        print("=" * 50)
        
        configs = {}
        
        # Configuration principale
        main_config = {
            "project_name": "Signalert",
            "version": "1.0.0",
            "description": "Détection sonore temps réel pour accessibilité",
            "author": "Signalert Team",
            "created": datetime.now().isoformat(),
            "paths": {
                "data": "data/",
                "models": "models/",
                "reports": "reports/",
                "notebooks": "notebooks/"
            },
            "audio": {
                "sample_rate": 16000,
                "duration_seconds": 1.0,
                "n_mfcc": 20,
                "fmax": 4000
            },
            "models": {
                "random_forest": {
                    "name": "random_forest_model.pkl",
                    "type": "sklearn",
                    "confidence_threshold": 0.7
                },
                "mlp": {
                    "name": "mlp_model.tflite",
                    "type": "tensorflow_lite"
                }
            },
            "classes": {
                "car_horn": 0,
                "dog_bark": 1,
                "drilling": 2,
                "engine_idling": 3,
                "gun_shot": 4,
                "jackhammer": 5,
                "siren": 6
            }
        }
        
        config_path = self.project_root / "config.json"
        with open(config_path, 'w') as f:
            json.dump(main_config, f, indent=2)
        configs['main'] = config_path
        
        # .gitignore
        gitignore_content = [
            "# Signalert .gitignore",
            "",
            "# Data files",
            "data/",
            "*.wav",
            "*.mp3",
            "*.flac",
            "",
            "# Models (except metadata)",
            "models/*.pkl",
            "models/*.h5",
            "models/*.tflite",
            "!models/model_metadata.json",
            "",
            "# Python",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "",
            "# Jupyter",
            ".ipynb_checkpoints",
            "",
            "# Environment",
            ".env",
            ".venv",
            "env/",
            "venv/",
            "ENV/",
            "env.bak/",
            "venv.bak/",
            "",
            "# IDE",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "*~",
            "",
            "# OS",
            ".DS_Store",
            "Thumbs.db",
            "",
            "# Reports (except examples)",
            "reports/*.txt",
            "reports/*.json",
            "!reports/example_*",
            "",
            "# Temporary files",
            "*.tmp",
            "*.temp",
            "*.log"
        ]
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write('\n'.join(gitignore_content))
        configs['gitignore'] = gitignore_path
        
        # README technique
        readme_content = """# Signalert - Configuration Technique

## Installation

```bash
# 1. Cloner le repo
git clone [URL_REPO]
cd Signalert

# 2. Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate  # Windows

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Configurer environnement
python scripts/setup_environment.py --install-deps
```

## Structure

- `notebooks/` - Recherche & développement IA
- `models/` - Modèles entraînés
- `tools/` - Outils de développement
- `scripts/` - Scripts d'automatisation
- `mobile/` - Application Flutter
- `tests/` - Tests unitaires & intégration

## Utilisation

```bash
# Entraîner les modèles
python scripts/train_models.py

# Exporter pour mobile
python scripts/export_for_mobile.py

# Tester performance
python tools/benchmark_mobile.py --full-benchmark
```
"""
        
        readme_path = self.project_root / "SETUP.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        configs['readme'] = readme_path
        
        print(f"✅ Fichiers de configuration créés:")
        for name, path in configs.items():
            print(f"   {name}: {path}")
        
        return configs
    
    def setup_jupyter_kernel(self):
        """Configure le kernel Jupyter pour le projet"""
        print("\n🪐 CONFIGURATION JUPYTER")
        print("=" * 30)
        
        try:
            # Installer ipykernel si pas présent
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])
            
            # Créer kernel spécifique au projet
            kernel_name = "signalert"
            subprocess.check_call([
                sys.executable, "-m", "ipykernel", "install", "--user", 
                "--name", kernel_name, "--display-name", "Signalert"
            ])
            
            print(f"✅ Kernel Jupyter créé: {kernel_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur configuration Jupyter: {e}")
            return False
        
        return True
    
    def run_setup(self, install_deps=True, create_folders=True, create_configs=True):
        """Lance la configuration complète"""
        print("🚀 CONFIGURATION ENVIRONNEMENT SIGNALERT")
        print("=" * 60)
        print(f"📁 Dossier projet: {self.project_root}")
        print(f"🐍 Python: {sys.version}")
        
        success = True
        
        # 1. Créer structure
        if create_folders:
            self.create_folder_structure()
        
        # 2. Créer fichiers config
        if create_configs:
            self.create_config_files()
        
        # 3. Installer dépendances
        if install_deps:
            if not self.install_dependencies():
                success = False
        
        # 4. Configurer Jupyter
        self.setup_jupyter_kernel()
        
        # 5. Rapport final
        print("\n📋 CONFIGURATION TERMINÉE")
        print("=" * 30)
        
        if success:
            print("✅ Environnement configuré avec succès!")
            print("\n🚀 Prochaines étapes:")
            print("   1. Placer vos données dans data/")
            print("   2. Lancer Jupyter: jupyter notebook")
            print("   3. Exécuter notebooks/Benchmark*")
            print("   4. Tester: python tools/benchmark_mobile.py")
        else:
            print("❌ Configuration incomplète, vérifiez les erreurs")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Configure l'environnement Signalert")
    parser.add_argument("--install-deps", action="store_true", help="Installer les dépendances")
    parser.add_argument("--create-folders", action="store_true", help="Créer la structure de dossiers")
    parser.add_argument("--create-configs", action="store_true", help="Créer les fichiers de configuration")
    parser.add_argument("--project-root", help="Dossier racine du projet")
    parser.add_argument("--all", action="store_true", help="Configuration complète")
    
    args = parser.parse_args()
    
    # Configuration par défaut si aucun argument
    if not any([args.install_deps, args.create_folders, args.create_configs, args.all]):
        args.all = True
    
    # Créer le configurateur
    setup = EnvironmentSetup(project_root=args.project_root)
    
    if args.all:
        # Configuration complète
        setup.run_setup(install_deps=True, create_folders=True, create_configs=True)
    else:
        # Configuration sélective
        setup.run_setup(
            install_deps=args.install_deps,
            create_folders=args.create_folders,
            create_configs=args.create_configs
        )

if __name__ == "__main__":
    main()
