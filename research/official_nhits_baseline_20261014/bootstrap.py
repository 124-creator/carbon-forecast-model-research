"""Fixed-version replay installer; never changes Hermes/parent environment."""
from pathlib import Path
import sys,subprocess,os,shutil,json
R=Path(__file__).resolve().parent
if not (R/'.venv/Scripts/python.exe').exists():subprocess.run([sys.executable,'-m','venv',str(R/'.venv')],check=True)
py=str(R/'.venv/Scripts/python.exe')
# CPU index is explicit. No unpinned torch upgrade and no CUDA packages.
cmd=['uv','pip','install','--python',py,'--index','https://download.pytorch.org/whl/cpu','--index-strategy','unsafe-best-match','torch==2.3.1+cpu','numpy==1.26.4']
subprocess.run(cmd,check=True)
subprocess.run(['uv','pip','install','--python',py,'-r',str(R/'requirements.lock.txt')],check=True)
subprocess.run(['uv','pip','check','--python',py],check=True)
