import os
import subprocess
import sys

def run():
    script_path = r"d:\Desktop\CV\CLIP\NanS-CLIP\cn_clip\preprocess\build_lmdb_dataset.py"
    data_dir = r"d:\Desktop\CV\CLIP\clip_data\datasets\SongDynasty"
    
    cmd = [
        "python", script_path,
        "--data_dir", data_dir,
        "--splits", "train,valid"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    
    if result.returncode == 0:
        print("Success!")
    else:
        print(f"Failed with return code {result.returncode}")

if __name__ == "__main__":
    run()
