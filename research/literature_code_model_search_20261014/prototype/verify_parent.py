from pathlib import Path
import importlib.util
p=Path(__file__).with_name('mechanism.py')
if not p.exists(): raise AssertionError('independent mechanism implementation missing')
s=importlib.util.spec_from_file_location('mechanism',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(m.verify())
