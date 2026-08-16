import sys
sys.path.insert(0, '/Users/condenast/Downloads/omni_29dof_v260705')
from convert_to_amp import npz_to_visualization_json
import os
import glob

os.chdir('/Users/condenast/Downloads/omni_29dof_v260705/retargeted')
files = [f for f in os.listdir('.') if f.startswith('跳高') and f.endswith('.npz') and '_v' not in f]
files.sort()

# 跳高 06
f = files[5]
print(f"转换：{f}")
output = f.replace('.npz', '_v3.txt')
npz_to_visualization_json(f, output, fps=30.0)
print(f"输出：{output}")
