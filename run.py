#!/usr/bin/env python3
"""
Main executable for Sequential Music Recommendation with RL.

Usage:
    python run.py preprocess         - Preprocess raw data
    python run.py train-baselines    - Train all 4 baselines (Popularity, Item-CF, User-CF, SASRec)
    python run.py train-rl           - Train RL agent with Item-CF backbone
    python run.py evaluate           - Evaluate all models
    python run.py --help             - Show this help message
"""
import sys
import subprocess
from pathlib import Path


def print_help():
    """Print help message."""
    print(__doc__)
    print("Available commands:")
    print("  preprocess       - Preprocess the raw Last.fm data")
    print("  train-baselines  - Train all 4 baselines (Popularity, Item-CF, User-CF, SASRec)")
    print("  train-rl         - Train RL agent using Item-CF as backbone")
    print("  evaluate         - Evaluate and compare all trained models")
    print("")
    print("Example workflow:")
    print("  1. python run.py preprocess")
    print("  2. python run.py train-baselines")
    print("  3. python run.py train-rl")
    print("  4. python run.py evaluate")


def run_script(script_name: str) -> int:
    """Run a Python script from the src directory."""
    src_dir = Path(__file__).parent / "src"
    script_path = src_dir / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return 1
    
    print(f"Running: {script_path}")
    print("=" * 60)
    
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(Path(__file__).parent))
    return result.returncode


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['--help', '-h', 'help']:
        print_help()
        return 0
    
    command = sys.argv[1].lower()
    
    commands = {
        'preprocess': 'data/preprocess.py',
        'train-baselines': 'train_baselines.py',
        'train-rl': 'train_rl_itemcf.py',
        'evaluate': 'evaluate_all_models.py',
    }
    
    if command not in commands:
        print(f"Error: Unknown command '{command}'")
        print("Use 'python run.py --help' to see available commands.")
        return 1
    
    return run_script(commands[command])


if __name__ == "__main__":
    sys.exit(main())
