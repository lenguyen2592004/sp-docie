#!/usr/bin/env python3
"""Launch training and monitor output."""
import subprocess
import time
import glob
import os

print("Starting training...")
print("="*60)

# Kill any existing training
os.system("pkill -f 'python.*moe.py' 2>/dev/null")
time.sleep(2)

# Start training in background
proc = subprocess.Popen(
    ["python", "moe.py", "--stage", "train", "--epochs", "1"],
    cwd="/workspace",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print(f"Training started (PID: {proc.pid})")
print("Monitoring output...")
print("="*60)

# Monitor for 180 seconds
start_time = time.time()
last_size = 0

while time.time() - start_time < 180:
    # Find most recent wandb run
    runs = sorted(glob.glob("/workspace/wandb/run-*/files/output.log"))
    if runs:
        latest = runs[-1]
        size = os.path.getsize(latest)
        
        # Show new content if file grew
        if size > last_size:
            with open(latest, 'r') as f:
                f.seek(last_size)
                new_content = f.read()
                if new_content:
                    print(new_content, end='')
            last_size = size
    
    # Check if process finished
    if proc.poll() is not None:
        print(f"\n{'='*60}")
        print(f"Training completed with exit code: {proc.returncode}")
        break
    
    time.sleep(2)

# Show final output
if runs:
    print(f"\n{'='*60}")
    print(f"Final output from {runs[-1]}:")
    print("="*60)
    with open(runs[-1], 'r') as f:
        lines = f.readlines()
        for line in lines[-30:]:  # Last 30 lines
            print(line, end='')

# Check for errors
if runs:
    with open(runs[-1], 'r') as f:
        content = f.read()
        if 'Error' in content or 'Traceback' in content:
            print("\n" + "="*60)
            print("❌ ERRORS DETECTED!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("✓ No errors in log")
            print("="*60)
