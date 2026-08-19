#!/usr/bin/env python3
"""批量扫描本地所有 ONNX: 结构/normalizer/与17700权重相似度/与NPZ匹配度"""
import glob, os, datetime
import onnx
import numpy as np
from onnx import numpy_helper

files = sorted(glob.glob('/Users/condenast/Downloads/*.onnx')) + \
        sorted(glob.glob('/Users/condenast/Downloads/omni_29dof_v260705/**/*.onnx', recursive=True))
files = sorted(set(files))

REF = {i.name: numpy_helper.to_array(i)
       for i in onnx.load('/Users/condenast/Downloads/policy-17700step.onnx').graph.initializer}

def arch(inits):
    ws = [v.shape for n, v in inits.items() if n.endswith('weight') and v.ndim == 2]
    return ws

for p in files:
    try:
        g = onnx.load(p).graph
        inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        has_norm = any('normalizer' in n for n in inits)
        layers = arch(inits)
        ins = [(i.name) for i in g.input]
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(p))
        # 与 17700 权重相似度
        cos = None
        wname = [n for n in inits if n.endswith('weight') and inits[n].ndim == 2 and inits[n].shape == (512, 529)]
        if wname:
            w = inits[wname[0]].ravel(); r = REF['actor.0.weight'].ravel()
            cos = float(w @ r / (np.linalg.norm(w) * np.linalg.norm(r)))
        # normalizer command mean 与 NPZ 匹配
        cmd_ok = ''
        if has_norm:
            mn = [n for n in inits if '_mean' in n][0]
            m = inits[mn].reshape(-1)[:29]
            npz = np.load('training_data/jump_high_firstjump_50fps.npz') if os.path.exists('training_data/jump_high_firstjump_50fps.npz') else np.load('/Users/condenast/Downloads/omni_29dof_v260705/training_data/jump_high_firstjump_50fps.npz')
            cmd_ok = f'cmd_mean_diff={np.abs(m - npz["joint_pos"].mean(0)).mean():.3f}'
        print(f'{os.path.basename(p):<58} {mt:%m-%d} norm={int(has_norm)} {str(layers):<38} '
              f'cos17700={cos if cos is None else round(cos,3)} {cmd_ok}')
    except Exception as e:
        print(f'{os.path.basename(p)}: ERR {type(e).__name__} {e}')
