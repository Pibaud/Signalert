"""
SIGNALERT - AUDIO TESTER
=========================
Teste le pipeline audio complet en temps réel.

Usage:
    python audio_tester.py --duration 30 --save-results
    python audio_tester.py --test-file ../data/siren/test.wav
"""

import argparse
import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import time
import sys
import json
from pathlib import Path
from datetime import datetime
import threading
import queue

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

class AudioTester:
    """Testeur du pipeline audio Signalert"""
    
    def __init__(self, sample_rate=16000, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.buffer_size = int(sample_rate * duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.results = []
        
    def extract_mfcc_features(self, audio_data):
        """Extrait les features MFCC comme en production"""
        try:
            # Calcul MFCC
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=20,
                fmax=4000
            )
            
            # Statistiques
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_var = np.var(mfccs, axis=1)
            feature_vector = np.concatenate((mfcc_mean, mfcc_var))
            
            return feature_vector
            
        except Exception as e:
            print(f"❌ Erreur extraction MFCC: {e}")
            return None
    
    def audio_callback(self, indata, frames, time, status):
        """Callback pour la capture audio temps réel"""
        if status:
            print(f"⚠️ Status audio: {status}")
        
        # Convertir en mono si nécessaire
        if len(indata.shape) > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Ajouter à la queue
        self.audio_queue.put(audio_data.copy())
    
    def test_real_time_processing(self, duration_seconds=30):
        """Test du traitement audio temps réel"""
        print(f"🎙️ TEST AUDIO TEMPS RÉEL - {duration_seconds}s")
        print("=" * 50)
        
        self.is_recording = True
        processing_times = []
        feature_stats = []
        
        # Démarrer la capture audio
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.buffer_size
        ):
            print("🔴 Enregistrement démarré... (parlez ou faites du bruit)")
            
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Récupérer les données audio
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # Mesurer le temps de traitement
                    process_start = time.perf_counter()
                    features = self.extract_mfcc_features(audio_data)
                    process_time = (time.perf_counter() - process_start) * 1000
                    
                    if features is not None:
                        processing_times.append(process_time)
                        feature_stats.append({
                            'timestamp': time.time(),
                            'rms': np.sqrt(np.mean(audio_data**2)),
                            'max_amplitude': np.max(np.abs(audio_data)),
                            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio_data)),
                            'processing_time_ms': process_time
                        })
                        
                        # Affichage en temps réel
                        if len(processing_times) % 10 == 0:
                            avg_time = np.mean(processing_times[-10:])
                            fps = 1000 / avg_time if avg_time > 0 else 0
                            print(f"⏱️ Traitement: {avg_time:.2f}ms | FPS: {fps:.1f} | Échantillons: {len(processing_times)}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Erreur traitement: {e}")
        
        self.is_recording = False
        
        # Analyse des résultats
        self._analyze_real_time_results(processing_times, feature_stats)
        
        return processing_times, feature_stats
    
    def test_file_processing(self, file_path):
        """Test du traitement d'un fichier audio"""
        print(f"📁 TEST FICHIER: {file_path}")
        print("=" * 50)
        
        try:
            # Charger le fichier
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f"✅ Fichier chargé: {len(audio_data)} échantillons, {sr} Hz")
            
            # Diviser en chunks de 1 seconde
            chunk_size = int(sr * self.duration)
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            processing_times = []
            results = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk) < chunk_size:
                    # Padding si nécessaire
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Traitement
                start_time = time.perf_counter()
                features = self.extract_mfcc_features(chunk)
                process_time = (time.perf_counter() - start_time) * 1000
                
                if features is not None:
                    processing_times.append(process_time)
                    results.append({
                        'chunk': i,
                        'features': features.tolist(),
                        'processing_time_ms': process_time,
                        'rms': float(np.sqrt(np.mean(chunk**2))),
                        'max_amplitude': float(np.max(np.abs(chunk)))
                    })
            
            # Analyse
            self._analyze_file_results(processing_times, results, file_path)
            
            return processing_times, results
            
        except Exception as e:
            print(f"❌ Erreur traitement fichier: {e}")
            return [], []
    
    def _analyze_real_time_results(self, processing_times, feature_stats):
        """Analyse les résultats du test temps réel"""
        print("\n📊 ANALYSE TEMPS RÉEL")
        print("=" * 30)
        
        if not processing_times:
            print("❌ Aucune donnée à analyser")
            return
        
        # Statistiques de performance
        mean_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        max_time = np.max(processing_times)
        min_time = np.min(processing_times)
        
        print(f"⏱️ Temps de traitement:")
        print(f"   Moyenne: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"   Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"   FPS moyen: {1000/mean_time:.1f}")
        
        # Analyse audio
        if feature_stats:
            rms_values = [s['rms'] for s in feature_stats]
            print(f"🔊 Analyse audio:")
            print(f"   RMS moyen: {np.mean(rms_values):.4f}")
            print(f"   Dynamique: {np.max(rms_values)/np.min(rms_values):.2f}")
        
        # Visualisation
        self._plot_real_time_analysis(processing_times, feature_stats)
    
    def _analyze_file_results(self, processing_times, results, file_path):
        """Analyse les résultats du test fichier"""
        print("\n📊 ANALYSE FICHIER")
        print("=" * 30)
        
        if not processing_times:
            print("❌ Aucune donnée à analyser")
            return
        
        mean_time = np.mean(processing_times)
        total_chunks = len(results)
        
        print(f"📁 Fichier: {Path(file_path).name}")
        print(f"⏱️ Temps moyen: {mean_time:.2f} ms/chunk")
        print(f"📊 Chunks traités: {total_chunks}")
        print(f"⚡ Vitesse: {1000/mean_time:.1f} chunks/seconde")
        
        # Détection d'activité
        active_chunks = [r for r in results if r['rms'] > 0.01]
        print(f"🔊 Chunks actifs: {len(active_chunks)}/{total_chunks} ({len(active_chunks)/total_chunks*100:.1f}%)")
    
    def _plot_real_time_analysis(self, processing_times, feature_stats):
        """Affiche les graphiques d'analyse"""
        if not processing_times:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temps de traitement
        axes[0, 0].plot(processing_times)
        axes[0, 0].set_title('Temps de traitement (ms)')
        axes[0, 0].set_xlabel('Échantillon')
        axes[0, 0].set_ylabel('Temps (ms)')
        axes[0, 0].grid(True)
        
        # Histogramme temps
        axes[0, 1].hist(processing_times, bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution des temps')
        axes[0, 1].set_xlabel('Temps (ms)')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].grid(True)
        
        if feature_stats:
            # RMS dans le temps
            timestamps = [s['timestamp'] for s in feature_stats]
            rms_values = [s['rms'] for s in feature_stats]
            
            axes[1, 0].plot(timestamps, rms_values)
            axes[1, 0].set_title('RMS Audio')
            axes[1, 0].set_xlabel('Temps')
            axes[1, 0].set_ylabel('RMS')
            axes[1, 0].grid(True)
            
            # FPS en temps réel
            fps_values = [1000/s['processing_time_ms'] for s in feature_stats]
            axes[1, 1].plot(fps_values)
            axes[1, 1].set_title('FPS Temps Réel')
            axes[1, 1].set_xlabel('Échantillon')
            axes[1, 1].set_ylabel('FPS')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, processing_times, feature_stats, output_dir="../reports"):
        """Sauvegarde les résultats de test"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_test_results_{timestamp}.json"
        
        results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "sample_rate": self.sample_rate,
                "duration": self.duration,
                "total_samples": len(processing_times)
            },
            "performance": {
                "mean_processing_time_ms": float(np.mean(processing_times)) if processing_times else 0,
                "std_processing_time_ms": float(np.std(processing_times)) if processing_times else 0,
                "max_processing_time_ms": float(np.max(processing_times)) if processing_times else 0,
                "min_processing_time_ms": float(np.min(processing_times)) if processing_times else 0,
                "average_fps": float(1000/np.mean(processing_times)) if processing_times else 0
            },
            "audio_analysis": feature_stats if feature_stats else [],
            "raw_processing_times": processing_times
        }
        
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Résultats sauvegardés: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Test le pipeline audio Signalert")
    parser.add_argument("--duration", type=int, default=30, help="Durée test temps réel (secondes)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Fréquence d'échantillonnage")
    parser.add_argument("--test-file", help="Fichier audio à tester")
    parser.add_argument("--save-results", action="store_true", help="Sauvegarder les résultats")
    
    args = parser.parse_args()
    
    # Créer le testeur
    tester = AudioTester(sample_rate=args.sample_rate)
    
    processing_times = []
    feature_stats = []
    
    if args.test_file:
        # Test sur fichier
        processing_times, results = tester.test_file_processing(args.test_file)
        feature_stats = results
    else:
        # Test temps réel
        processing_times, feature_stats = tester.test_real_time_processing(args.duration)
    
    # Sauvegarder si demandé
    if args.save_results:
        tester.save_results(processing_times, feature_stats)
    
    print("\n🎉 TESTS TERMINÉS!")

if __name__ == "__main__":
    main()
