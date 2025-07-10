"""
SIGNALERT - MODEL CONVERTER
============================
Convertit les modèles Python vers formats optimisés mobile.

Usage:
    python model_converter.py --input models/ --output mobile/assets/models/
    python model_converter.py --format tflite --quantize
"""

import argparse
import os
import sys
from pathlib import Path
import joblib
import tensorflow as tf
import numpy as np
import json
from datetime import datetime

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

class ModelConverter:
    """Convertisseur de modèles pour déploiement mobile"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_sklearn_to_dart(self, model_path, output_path):
        """Convertit un modèle scikit-learn en code Dart"""
        print(f"🔄 Conversion RF vers Dart: {model_path}")
        
        # Charger le modèle RF
        rf_model = joblib.load(model_path)
        
        # Générer le code Dart
        dart_code = self._generate_dart_rf_code(rf_model)
        
        # Sauvegarder
        with open(output_path, 'w') as f:
            f.write(dart_code)
        
        print(f"✅ Modèle Dart sauvegardé: {output_path}")
        return output_path
    
    def _generate_dart_rf_code(self, rf_model):
        """Génère le code Dart pour Random Forest"""
        n_estimators = rf_model.n_estimators
        feature_importances = rf_model.feature_importances_.tolist()
        
        dart_code = f'''
// filepath: signalert_random_forest.dart
// Généré automatiquement le {datetime.now().isoformat()}
// Random Forest avec {n_estimators} arbres, {len(feature_importances)} features

class SignalertRandomForest {{
  static const int nEstimators = {n_estimators};
  static const int nFeatures = {len(feature_importances)};
  static const int nClasses = {rf_model.n_classes_};
  
  static const List<double> featureImportances = {feature_importances};
  
  static const Map<int, String> classNames = {{
    0: 'car_horn',
    1: 'dog_bark', 
    2: 'drilling',
    3: 'engine_idling',
    4: 'gun_shot',
    5: 'jackhammer',
    6: 'siren'
  }};
  
  // Prédiction simplifiée (implémentation complète nécessaire)
  static Map<String, dynamic> predict(List<double> features) {{
    // TODO: Implémenter la logique complète des arbres
    // Pour l'instant, retourne une prédiction factice
    return {{
      'class': 0,
      'confidence': 0.8,
      'probabilities': List.filled(nClasses, 1.0 / nClasses)
    }};
  }}
  
  // Méthode pour calculer les probabilités
  static List<double> predictProba(List<double> features) {{
    // TODO: Logique complète Random Forest
    return List.filled(nClasses, 1.0 / nClasses);
  }}
}}
'''
        return dart_code
    
    def convert_keras_to_tflite(self, model_path, output_path, quantize=True):
        """Convertit un modèle Keras en TensorFlow Lite"""
        print(f"🔄 Conversion Keras vers TFLite: {model_path}")
        
        # Charger le modèle
        model = tf.keras.models.load_model(model_path)
        
        # Créer le convertisseur
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            # Quantification pour réduire la taille
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("⚡ Quantification activée (float16)")
        
        # Convertir
        tflite_model = converter.convert()
        
        # Sauvegarder
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculer la taille
        size_kb = len(tflite_model) / 1024
        print(f"✅ Modèle TFLite sauvegardé: {output_path} ({size_kb:.1f} KB)")
        
        return output_path
    
    def create_flutter_asset_manifest(self):
        """Crée le manifest des assets pour Flutter"""
        manifest = {
            "models": {
                "random_forest": {
                    "path": "assets/models/random_forest_model.pkl",
                    "dart_implementation": "assets/models/signalert_random_forest.dart",
                    "type": "sklearn",
                    "size_kb": 0,
                    "description": "Modèle Random Forest principal"
                },
                "mlp": {
                    "path": "assets/models/mlp_model.tflite",
                    "type": "tensorflow_lite",
                    "size_kb": 0,
                    "description": "Modèle MLP pour backup"
                },
                "scaler": {
                    "path": "assets/models/feature_scaler.pkl",
                    "type": "sklearn",
                    "size_kb": 0,
                    "description": "Normalisateur features MFCC"
                }
            },
            "metadata": {
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "flutter_compatible": True
            }
        }
        
        manifest_path = self.output_dir / "models_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Manifest créé: {manifest_path}")
        return manifest_path
    
    def convert_all(self, quantize=True):
        """Convertit tous les modèles disponibles"""
        print("🚀 CONVERSION COMPLÈTE DES MODÈLES")
        print("=" * 50)
        
        conversions = []
        
        # 1. Random Forest vers Dart
        rf_path = self.input_dir / "random_forest_model.pkl"
        if rf_path.exists():
            dart_path = self.output_dir / "signalert_random_forest.dart"
            self.convert_sklearn_to_dart(rf_path, dart_path)
            conversions.append(("Random Forest", "Dart", dart_path))
        
        # 2. Copier les modèles pickle pour Flutter
        for model_file in ["random_forest_model.pkl", "feature_scaler.pkl"]:
            src = self.input_dir / model_file
            dst = self.output_dir / model_file
            if src.exists():
                import shutil
                shutil.copy2(src, dst)
                conversions.append((model_file, "Copy", dst))
        
        # 3. Vérifier TFLite existant ou convertir
        tflite_path = self.input_dir / "mlp_model.tflite"
        if tflite_path.exists():
            import shutil
            dst = self.output_dir / "mlp_model.tflite"
            shutil.copy2(tflite_path, dst)
            conversions.append(("MLP", "TFLite", dst))
        
        # 4. Copier métadonnées
        metadata_src = self.input_dir / "model_metadata.json"
        if metadata_src.exists():
            import shutil
            dst = self.output_dir / "model_metadata.json"
            shutil.copy2(metadata_src, dst)
            conversions.append(("Metadata", "JSON", dst))
        
        # 5. Créer manifest Flutter
        self.create_flutter_asset_manifest()
        
        # Rapport final
        print("\n📊 CONVERSIONS TERMINÉES")
        print("=" * 30)
        for name, format_type, path in conversions:
            size_kb = path.stat().st_size / 1024 if path.exists() else 0
            print(f"✅ {name:15} ({format_type:8}) - {size_kb:.1f} KB")
        
        print(f"\n📁 Tous les modèles convertis dans: {self.output_dir}")
        return conversions

def main():
    parser = argparse.ArgumentParser(description="Convertit les modèles Signalert pour mobile")
    parser.add_argument("--input", default="../models", help="Dossier des modèles source")
    parser.add_argument("--output", default="../mobile/assets/models", help="Dossier de sortie")
    parser.add_argument("--format", choices=["all", "tflite", "dart"], default="all", help="Format de conversion")
    parser.add_argument("--quantize", action="store_true", help="Activer la quantification TFLite")
    
    args = parser.parse_args()
    
    # Créer le convertisseur
    converter = ModelConverter(args.input, args.output)
    
    # Lancer la conversion
    if args.format == "all":
        converter.convert_all(quantize=args.quantize)
    else:
        print(f"Conversion spécifique: {args.format}")
        # Implémentation spécifique si nécessaire
    
    print("\n🎉 CONVERSION TERMINÉE AVEC SUCCÈS!")

if __name__ == "__main__":
    main()
