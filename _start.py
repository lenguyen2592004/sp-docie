#!/usr/bin/env python3
import os
import subprocess
import sys

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(WORKSPACE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# Start training
proc = subprocess.Popen([sys.executable, os.path.join(WORKSPACE_DIR, "moe.py"), "--stage", "train", "--epochs", "1"],
                       stdout=open(os.path.join(TMP_DIR, "train_direct.log"), "w"),
                       stderr=subprocess.STDOUT,
                       cwd=WORKSPACE_DIR)
                       
with open(os.path.join(TMP_DIR, "train_direct.pid"), "w") as f:
    f.write(str(proc.pid))
    
print(f"STARTED:{proc.pid}")
