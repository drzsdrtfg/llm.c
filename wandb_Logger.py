import re
import wandb
import os
import time
import json
CHECKPOINT_INTERVAL = 5000  # Set this to match your checkpoint interval

def parse_log_line(line):
    pattern = r's:(\d+) trl:([\d.]+) lr:([\d.]+) norm:([\d.]+)'
    match = re.match(pattern, line)
    if match:
        return {
            'step': int(match.group(1)),
            'training_loss': float(match.group(2)),
            'learning_rate': float(match.group(3)),
            'gradient_norm': float(match.group(4))
        }
    return None

def init_wandb():
    return wandb.init(project="log124M-analysis", name="training-progress", resume="allow")

def log_to_wandb(entry, global_step):
    wandb.log(entry, step=global_step)

def follow(file):
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

def get_checkpoint_info(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint.get('global_step', 0), checkpoint.get('last_checkpoint_step', 0)
    return 0, 0

def save_checkpoint(checkpoint_file, global_step, last_checkpoint_step):
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'global_step': global_step,
            'last_checkpoint_step': last_checkpoint_step
        }, f)

def main():
    log_dir = "log124M"
    log_file = os.path.join(log_dir, "main.log")
    checkpoint_file = os.path.join(log_dir, "wandb_checkpoint.json")
    
    if not os.path.exists(log_file):
        print(f"Error: File '{log_file}' not found.")
        return

    run = init_wandb()
    global_step, last_checkpoint_step = get_checkpoint_info(checkpoint_file)
    last_logged_step = -1

    print(f"Starting to process the log file. Global step: {global_step}, Last checkpoint step: {last_checkpoint_step}")
    print("Press Ctrl+C to stop.")
    
    while True:
        with open(log_file, 'r') as file:
            for line in follow(file):
                parsed = parse_log_line(line.strip())
                if parsed:
                    current_step = parsed['step']
                    
                    if current_step <= last_logged_step:
                        # Log file has reset, update global step to last checkpoint
                        global_step = last_checkpoint_step
                        last_logged_step = -1
                    
                    if current_step > last_logged_step:
                        current_global_step = global_step + (current_step - last_checkpoint_step)
                        
                        # Only log if we're past the last checkpoint step
                        if current_step > last_checkpoint_step:
                            log_to_wandb(parsed, current_global_step)
                            print(f"Logged global step {current_global_step}")
                        
                        last_logged_step = current_step
                        
                        # Update checkpoint info if we've reached a new checkpoint
                        if current_step % CHECKPOINT_INTERVAL == 0:
                            last_checkpoint_step = current_step
                            global_step = current_global_step
                            save_checkpoint(checkpoint_file, global_step, last_checkpoint_step)
                            print(f"Updated checkpoint. Global step: {global_step}, Last checkpoint step: {last_checkpoint_step}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping the script...")
    finally:
        wandb.finish()
