import subprocess, time, os

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(WORKSPACE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# Kill previous processes
os.system("pkill -f 'python.*moe.py' > /dev/null 2>&1")
time.sleep(3)

# Start training
with open(os.path.join(TMP_DIR, "training.pid"), "w") as f:
    proc = subprocess.Popen(
        ["python", "-u", os.path.join(WORKSPACE_DIR, "moe.py"), "--stage", "train", "--epochs", "1"], 
        stdout=open(os.path.join(TMP_DIR, "train.out"), "w"),
        stderr=subprocess.STDOUT,
        cwd=WORKSPACE_DIR,
    )
    f.write(str(proc.pid))
    print(f"Started training PID: {proc.pid}")
