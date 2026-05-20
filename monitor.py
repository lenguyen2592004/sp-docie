#!/usr/bin/env python3
"""Simple training monitor to check progress"""
import time
import subprocess
import os

print("=" * 60)
print("Training Monitor")
print("=" * 60)

# Check if process is running
result = subprocess.run(
    ["ps", "aux"],
    capture_output=True,
    text=True
)

moe_processes = [line for line in result.stdout.split('\n') if 'python' in line and 'moe.py' in line and 'grep' not in line]

if moe_processes:
    print("\n✅ Training process is running:")
    for proc in moe_processes:
        print(f"   {proc[:120]}")
else:
    print("\n❌ No training process found")

# Check log file
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(workspace_dir, "tmp")
    log_file = os.path.join(tmp_dir, "training_cuda.log")
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    print(f"\n📊 Log file: {len(lines)} lines")
    print(f"Latest output (last 30 lines):")
    print("-" * 60)
    for line in lines[-30:]:
        print(line.rstrip())
else:
    print(f"\n❌ Log file not found: {log_file}")

print("\n" + "=" * 60)
