#!/usr/bin/env python
"""Script de lancement VIVE Labeler avec vérifications et dataset de test."""

import sys
import subprocess
from pathlib import Path
import os

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"   {text}")
    print("=" * 60)

def print_step(step, total, text):
    """Print step progress."""
    print(f"\n[{step}/{total}] {text}...")

def check_python():
    """Check Python version."""
    print_step(1, 5, "Vérification Python")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ requis (détecté: {version.major}.{version.minor})")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} détecté")
    return True

def check_dependencies():
    """Check required dependencies."""
    print_step(2, 5, "Vérification dépendances")

    required = {
        'PyQt6': 'PyQt6',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'OpenGL': 'PyOpenGL',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} manquant")
            missing.append(package)

    if missing:
        print(f"\n⚠️  {len(missing)} dépendance(s) manquante(s)")
        response = input("Installer maintenant? (o/n): ")
        if response.lower() == 'o':
            print("\nInstallation des dépendances...")
            for pkg in missing:
                subprocess.run([sys.executable, '-m', 'pip', 'install', pkg],
                             capture_output=True)
            print("✅ Dépendances installées")
            return True
        else:
            print("❌ Installation annulée")
            return False

    print("✅ Toutes les dépendances sont installées")
    return True

def check_dataset():
    """Check and create test dataset if needed."""
    print_step(3, 5, "Vérification dataset de test")

    dataset_path = Path("test_dataset")

    if not dataset_path.exists():
        print("⚠️  Dataset de test non trouvé")
        print("\nGénération du dataset synthétique...")

        try:
            subprocess.run([
                sys.executable,
                'scripts/create_test_dataset.py',
                '--output-dir', './test_dataset',
                '--episodes', '2',
                '--frames', '150',
                '--fps', '30'
            ], check=True)
            print("✅ Dataset généré avec succès")
        except subprocess.CalledProcessError:
            print("❌ Erreur lors de la génération du dataset")
            return False
    else:
        # Verify dataset structure
        required_files = [
            'meta/info.json',
            'meta/stats.json',
            'data/chunk_000000.parquet',
        ]

        all_exist = all((dataset_path / f).exists() for f in required_files)

        if all_exist:
            print(f"✅ Dataset trouvé: {dataset_path.absolute()}")
        else:
            print("⚠️  Dataset incomplet, régénération...")
            try:
                subprocess.run([
                    sys.executable,
                    'scripts/create_test_dataset.py',
                    '--output-dir', './test_dataset',
                    '--episodes', '2',
                    '--frames', '150'
                ], check=True)
                print("✅ Dataset régénéré")
            except subprocess.CalledProcessError:
                print("❌ Erreur lors de la génération")
                return False

    return True

def show_instructions():
    """Show usage instructions."""
    print_step(4, 5, "Instructions d'utilisation")

    print("\n📖 Étapes dans l'application:")
    print("  1. Attendez l'ouverture de la fenêtre")
    print("  2. Menu: File → Open Dataset")
    print("  3. Sélectionnez le dossier: test_dataset")
    print("  4. Cliquez sur 'Ouvrir'")

    print("\n⌨️  Navigation:")
    print("  ← →       : Frame précédent/suivant")
    print("  Espace    : Play/Pause")
    print("  Clic line : Aller à un frame")

    print("\n🏷️  Labellisation:")
    print("  1. Clic 'Add Label' → Créer un label")
    print("  2. Naviguer à un frame")
    print("  3. Clic sur le label → Annoter")

    print("\n💾 Export:")
    print("  File → Export Annotations → Choisir format")

def launch_app():
    """Launch the VIVE Labeler application."""
    print_step(5, 5, "Lancement de l'application")

    print("\n" + "=" * 60)
    print("🚀 Démarrage de VIVE Labeler...")
    print("=" * 60)
    print()

    # Set environment variable for better error messages
    os.environ['PYTHONUNBUFFERED'] = '1'

    try:
        # Import and run
        from vive_labeler.main import main
        main()
    except ImportError:
        # Fallback: run as module
        subprocess.run([sys.executable, '-m', 'vive_labeler'])
    except Exception as e:
        print(f"\n❌ Erreur au lancement: {e}")
        print("\nEssayez:")
        print("  python -m vive_labeler")
        return False

    print("\n✅ Application fermée")
    return True

def main():
    """Main entry point."""
    print_header("VIVE LABELER - Lancement Rapide")

    # Run checks
    if not check_python():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    if not check_dataset():
        sys.exit(1)

    show_instructions()

    print("\n" + "=" * 60)
    input("Appuyez sur Entrée pour lancer l'application...")
    print()

    # Launch
    success = launch_app()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
