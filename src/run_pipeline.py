import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_step(command):
    print(f"\nRunning: {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")

def main():
    python = sys.executable
    run_step([python, "src/generate_data.py"])
    run_step([python, "src/visualize.py"])
    run_step([python, "src/train.py"])
    run_step([python, "src/evaluate.py"])
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
