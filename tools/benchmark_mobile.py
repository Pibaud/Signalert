"""
SIGNALERT - MOBILE BENCHMARK
=============================
Benchmark complet pour évaluer les performances en conditions mobiles.

Usage:
    python benchmark_mobile.py --full-benchmark
    python benchmark_mobile.py --quick-test --iterations 100
"""

import argparse
import time
import numpy as np
import joblib
import json
import psutil
import platform
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

class MobileBenchmark:
    """Benchmark complet pour simulation mobile"""
    
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir)
        self.load_models()
        self.system_info = self._get_system_info()
        
    def load_models(self):
        """Charge les modèles pour benchmark"""
        print("📦 Chargement des modèles...")
        
        try:
            # Random Forest
            rf_path = self.models_dir / "random_forest_model.pkl"
            self.rf_model = joblib.load(rf_path)
            print(f"✅ Random Forest chargé: {rf_path}")
            
            # Scaler
            scaler_path = self.models_dir / "feature_scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler chargé: {scaler_path}")
            
            # Métadonnées
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Métadonnées chargées: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèles: {e}")
            sys.exit(1)
    
    def _get_system_info(self):
        """Récupère les informations système"""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": datetime.now().isoformat()
        }
    
    def benchmark_inference_speed(self, n_iterations=1000):
        """Benchmark vitesse d'inférence"""
        print(f"\n⚡ BENCHMARK VITESSE D'INFÉRENCE ({n_iterations} itérations)")
        print("=" * 60)
        
        # Générer des données de test
        test_features = np.random.randn(n_iterations, 40)
        
        # Benchmark RF seul
        rf_times = []
        for i in range(n_iterations):
            features = test_features[i].reshape(1, -1)
            
            start_time = time.perf_counter()
            prediction = self.rf_model.predict(features)
            probabilities = self.rf_model.predict_proba(features)
            end_time = time.perf_counter()
            
            rf_times.append((end_time - start_time) * 1000)
        
        # Statistiques
        rf_stats = {
            'mean_ms': np.mean(rf_times),
            'std_ms': np.std(rf_times),
            'min_ms': np.min(rf_times),
            'max_ms': np.max(rf_times),
            'p95_ms': np.percentile(rf_times, 95),
            'p99_ms': np.percentile(rf_times, 99),
            'fps': 1000 / np.mean(rf_times)
        }
        
        print(f"🌲 Random Forest ({n_iterations} inférences):")
        print(f"   Temps moyen: {rf_stats['mean_ms']:.3f} ± {rf_stats['std_ms']:.3f} ms")
        print(f"   Min/Max: {rf_stats['min_ms']:.3f} / {rf_stats['max_ms']:.3f} ms")
        print(f"   P95/P99: {rf_stats['p95_ms']:.3f} / {rf_stats['p99_ms']:.3f} ms")
        print(f"   FPS théorique: {rf_stats['fps']:.1f}")
        
        return rf_stats, rf_times
    
    def benchmark_memory_usage(self):
        """Benchmark usage mémoire"""
        print("\n💾 BENCHMARK MÉMOIRE")
        print("=" * 30)
        
        # Mémoire avant chargement
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Simuler l'utilisation
        test_data = np.random.randn(1000, 40)
        
        # Mémoire après utilisation
        predictions = self.rf_model.predict(test_data)
        probabilities = self.rf_model.predict_proba(test_data)
        
        memory_after = process.memory_info().rss / (1024**2)  # MB
        
        memory_stats = {
            'before_mb': memory_before,
            'after_mb': memory_after,
            'difference_mb': memory_after - memory_before,
            'total_system_mb': psutil.virtual_memory().total / (1024**2),
            'available_mb': psutil.virtual_memory().available / (1024**2)
        }
        
        print(f"📊 Usage mémoire:")
        print(f"   Avant: {memory_stats['before_mb']:.1f} MB")
        print(f"   Après: {memory_stats['after_mb']:.1f} MB")
        print(f"   Différence: {memory_stats['difference_mb']:.1f} MB")
        print(f"   Disponible: {memory_stats['available_mb']:.1f} MB")
        
        return memory_stats
    
    def benchmark_cpu_usage(self, duration_seconds=30):
        """Benchmark usage CPU"""
        print(f"\n🔥 BENCHMARK CPU ({duration_seconds}s)")
        print("=" * 40)
        
        cpu_percentages = []
        inference_counts = []
        
        start_time = time.time()
        total_inferences = 0
        
        # Monitoring CPU pendant inférence continue
        while time.time() - start_time < duration_seconds:
            # Faire quelques inférences
            test_features = np.random.randn(10, 40)
            
            inference_start = time.time()
            predictions = self.rf_model.predict(test_features)
            inference_time = time.time() - inference_start
            
            total_inferences += 10
            
            # Mesurer CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_percentages.append(cpu_percent)
            inference_counts.append(total_inferences)
            
            # Petite pause pour éviter de saturer
            time.sleep(0.05)
        
        cpu_stats = {
            'mean_cpu_percent': np.mean(cpu_percentages),
            'max_cpu_percent': np.max(cpu_percentages),
            'total_inferences': total_inferences,
            'inferences_per_second': total_inferences / duration_seconds,
            'duration_seconds': duration_seconds
        }
        
        print(f"📊 Usage CPU:")
        print(f"   CPU moyen: {cpu_stats['mean_cpu_percent']:.1f}%")
        print(f"   CPU max: {cpu_stats['max_cpu_percent']:.1f}%")
        print(f"   Inférences/s: {cpu_stats['inferences_per_second']:.1f}")
        print(f"   Total inférences: {cpu_stats['total_inferences']}")
        
        return cpu_stats, cpu_percentages
    
    def benchmark_batch_processing(self, batch_sizes=[1, 10, 50, 100]):
        """Benchmark traitement par batch"""
        print(f"\n📦 BENCHMARK TRAITEMENT PAR BATCH")
        print("=" * 50)
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n🔄 Batch size: {batch_size}")
            
            # Générer données
            test_data = np.random.randn(batch_size, 40)
            
            # Mesurer temps
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                predictions = self.rf_model.predict(test_data)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            # Calculer statistiques
            mean_time = np.mean(times)
            time_per_sample = mean_time / batch_size
            
            batch_results[batch_size] = {
                'mean_time_ms': mean_time,
                'time_per_sample_ms': time_per_sample,
                'throughput_samples_per_second': 1000 / time_per_sample
            }
            
            print(f"   Temps batch: {mean_time:.3f} ms")
            print(f"   Temps/échantillon: {time_per_sample:.3f} ms")
            print(f"   Débit: {1000/time_per_sample:.1f} échantillons/s")
        
        return batch_results
    
    def benchmark_confidence_thresholds(self, n_samples=1000):
        """Benchmark différents seuils de confiance"""
        print(f"\n🎯 BENCHMARK SEUILS DE CONFIANCE")
        print("=" * 45)
        
        # Générer données test
        test_data = np.random.randn(n_samples, 40)
        
        # Calculer probabilités
        probabilities = self.rf_model.predict_proba(test_data)
        confidences = np.max(probabilities, axis=1)
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        threshold_results = {}
        
        for threshold in thresholds:
            confident_predictions = np.sum(confidences >= threshold)
            uncertain_predictions = n_samples - confident_predictions
            
            threshold_results[threshold] = {
                'confident_percent': (confident_predictions / n_samples) * 100,
                'uncertain_percent': (uncertain_predictions / n_samples) * 100,
                'confident_count': confident_predictions,
                'uncertain_count': uncertain_predictions
            }
            
            print(f"   Seuil {threshold}: {confident_predictions}/{n_samples} confiant ({confident_predictions/n_samples*100:.1f}%)")
        
        return threshold_results
    
    def run_full_benchmark(self):
        """Lance le benchmark complet"""
        print("🚀 BENCHMARK COMPLET SIGNALERT MOBILE")
        print("=" * 60)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💻 Système: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"🔧 CPU: {self.system_info['cpu_count']} cores")
        print(f"💾 RAM: {self.system_info['memory_total_gb']:.1f} GB")
        
        results = {
            'system_info': self.system_info,
            'benchmark_timestamp': datetime.now().isoformat()
        }
        
        # 1. Vitesse d'inférence
        inference_stats, inference_times = self.benchmark_inference_speed(1000)
        results['inference_performance'] = inference_stats
        
        # 2. Mémoire
        memory_stats = self.benchmark_memory_usage()
        results['memory_usage'] = memory_stats
        
        # 3. CPU
        cpu_stats, cpu_history = self.benchmark_cpu_usage(30)
        results['cpu_usage'] = cpu_stats
        
        # 4. Batch processing
        batch_results = self.benchmark_batch_processing()
        results['batch_processing'] = batch_results
        
        # 5. Seuils de confiance
        threshold_results = self.benchmark_confidence_thresholds()
        results['confidence_thresholds'] = threshold_results
        
        # Résumé final
        self._print_benchmark_summary(results)
        
        # Sauvegarde
        self._save_benchmark_results(results)
        
        # Graphiques
        self._plot_benchmark_results(inference_times, cpu_history)
        
        return results
    
    def _print_benchmark_summary(self, results):
        """Affiche le résumé du benchmark"""
        print("\n📋 RÉSUMÉ BENCHMARK")
        print("=" * 30)
        
        inf_stats = results['inference_performance']
        mem_stats = results['memory_usage']
        cpu_stats = results['cpu_usage']
        
        print(f"⚡ Performance:")
        print(f"   Inférence: {inf_stats['mean_ms']:.2f}ms ({inf_stats['fps']:.1f} FPS)")
        print(f"   P95 latence: {inf_stats['p95_ms']:.2f}ms")
        
        print(f"💾 Mémoire:")
        print(f"   Usage: {mem_stats['after_mb']:.1f} MB")
        print(f"   Overhead: {mem_stats['difference_mb']:.1f} MB")
        
        print(f"🔥 CPU:")
        print(f"   Usage moyen: {cpu_stats['mean_cpu_percent']:.1f}%")
        print(f"   Débit: {cpu_stats['inferences_per_second']:.1f} inf/s")
        
        # Évaluation mobile
        print(f"\n📱 ÉVALUATION MOBILE:")
        if inf_stats['mean_ms'] < 50:
            print("   ✅ Latence: EXCELLENT (< 50ms)")
        elif inf_stats['mean_ms'] < 100:
            print("   ⚠️ Latence: ACCEPTABLE (50-100ms)")
        else:
            print("   ❌ Latence: PROBLÉMATIQUE (> 100ms)")
        
        if mem_stats['after_mb'] < 50:
            print("   ✅ Mémoire: EXCELLENT (< 50MB)")
        elif mem_stats['after_mb'] < 100:
            print("   ⚠️ Mémoire: ACCEPTABLE (50-100MB)")
        else:
            print("   ❌ Mémoire: PROBLÉMATIQUE (> 100MB)")
        
        if cpu_stats['mean_cpu_percent'] < 20:
            print("   ✅ CPU: EXCELLENT (< 20%)")
        elif cpu_stats['mean_cpu_percent'] < 40:
            print("   ⚠️ CPU: ACCEPTABLE (20-40%)")
        else:
            print("   ❌ CPU: PROBLÉMATIQUE (> 40%)")
    
    def _save_benchmark_results(self, results):
        """Sauvegarde les résultats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mobile_benchmark_{timestamp}.json"
        
        reports_dir = Path("../reports")
        reports_dir.mkdir(exist_ok=True)
        
        output_path = reports_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {output_path}")
    
    def _plot_benchmark_results(self, inference_times, cpu_history):
        """Affiche les graphiques de benchmark"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution des temps d'inférence
        axes[0, 0].hist(inference_times, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution Temps d\'Inférence')
        axes[0, 0].set_xlabel('Temps (ms)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temps d'inférence dans le temps
        axes[0, 1].plot(inference_times[:200], color='red', linewidth=1)
        axes[0, 1].set_title('Temps d\'Inférence (échantillon)')
        axes[0, 1].set_xlabel('Itération')
        axes[0, 1].set_ylabel('Temps (ms)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Usage CPU
        axes[1, 0].plot(cpu_history, color='green', linewidth=2)
        axes[1, 0].set_title('Usage CPU')
        axes[1, 0].set_xlabel('Temps')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot temps d'inférence
        axes[1, 1].boxplot(inference_times, vert=True)
        axes[1, 1].set_title('Box Plot Temps d\'Inférence')
        axes[1, 1].set_ylabel('Temps (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Benchmark mobile Signalert")
    parser.add_argument("--full-benchmark", action="store_true", help="Lance le benchmark complet")
    parser.add_argument("--quick-test", action="store_true", help="Test rapide")
    parser.add_argument("--iterations", type=int, default=1000, help="Nombre d'itérations")
    parser.add_argument("--models-dir", default="../models", help="Dossier des modèles")
    
    args = parser.parse_args()
    
    # Créer le benchmark
    benchmark = MobileBenchmark(models_dir=args.models_dir)
    
    if args.full_benchmark:
        # Benchmark complet
        results = benchmark.run_full_benchmark()
    elif args.quick_test:
        # Test rapide
        print("⚡ TEST RAPIDE")
        inference_stats, _ = benchmark.benchmark_inference_speed(args.iterations)
        memory_stats = benchmark.benchmark_memory_usage()
        print(f"\n✅ Test terminé - Inférence: {inference_stats['mean_ms']:.2f}ms, Mémoire: {memory_stats['after_mb']:.1f}MB")
    else:
        # Benchmark par défaut
        results = benchmark.run_full_benchmark()
    
    print("\n🎉 BENCHMARK TERMINÉ!")

if __name__ == "__main__":
    main()
