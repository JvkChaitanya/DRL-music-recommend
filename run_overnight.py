"""
Master execution script for Overnight Experiment (Full Data).
Runs Baselines, SASRec, and RL sequentially.
"""
import shutil
from pathlib import Path
import subprocess
import yaml
import sys
import time
import os

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_step(command, step_name, log_file):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run command and pipe output to log file and stdout
    # Using shell=True for simple piping (or implementing python piping)
    # Safer to run subprocess and write stdout to file + console
    
    with open(log_file, "w") as f_log:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            encoding='utf-8' # for text output
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                f_log.write(line)
                # Flush to ensure log file is updated in real-time
                f_log.flush()
                
    rc = process.poll()
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\n--> {step_name} {'COMPLETED' if rc == 0 else 'FAILED'} in {duration/60:.2f} minutes.")
    
    if rc != 0:
        print("Stopping pipeline due to failure.")
        sys.exit(rc)

def main():
    print("Starting Overnight Experiment...")
    
    # 1. Backup Config
    print("Backing up config.yaml -> config.yaml.bak")
    shutil.copy("config.yaml", "config.yaml.bak")
    
    try:
        # 2. Modify Config for Full Data
        print("Switching config to use FULL DATA (data/processed)...")
        with open("config.yaml", "r") as f:
            lines = f.readlines()
            
        with open("config.yaml", "w") as f:
            for line in lines:
                if 'processed_path: "data/subset"' in line:
                    f.write('  processed_path: "data/processed"\n')
                elif 'sasrec_epochs:' in line:
                    # Optional: increase epochs if previously low?
                    # keeping default 50
                    f.write(line)
                else:
                    f.write(line)
        
        # Verify config
        config = load_config()
        print(f"Verified processed_path: {config['data']['processed_path']}")
        
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # 3. Run Baselines (Sampled)
        # Use limit 100k to avoid OOM on 11M data with ItemCF/UserCF
        run_step(
            f"{sys.executable} src/baselines_gpu.py --limit 100000",
            "Baseline Models (100k Sample)",
            logs_dir / "baselines_full.log"
        )
        
        # 4. Run SASRec (Full)
        run_step(
            f"{sys.executable} src/train_sasrec.py",
            "SASRec Training (Full Data)",
            logs_dir / "sasrec_full.log"
        )
        
        # 5. Run RL Agent (Full)
        run_step(
            f"{sys.executable} src/train_rl.py",
            "RL Agent Training (Full Data)",
            logs_dir / "rl_full.log"
        )
        
        print("\n\n" + "="*60)
        print("OVERNIGHT EXPERIMENT COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n\nPipeline Failed: {e}")
    finally:
        # Restore Config? Maybe better to leave it so user sees state.
        print("\nNote: config.yaml is currently pointing to 'data/processed'.")
        print("Restore from config.yaml.bak if you want to switch back to subset.")

if __name__ == "__main__":
    main()
