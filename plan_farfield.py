import argparse
import json
import librosa as lr
import math
import numpy as np
import random as rnd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./test_re_3d/test_spherical_shape', type=str,
                    help='Root folder with all audio files')
parser.add_argument('--json', default='./json/farfield.json', type=str, help='JSON with parameters')
parser.add_argument('--farfield_meta', default='./meta/test_re_3d/test_spherical_shape.meta', type=str,
                    help='Root folder with FARFIELD META')
args = parser.parse_args()

with open(args.json, 'r') as f:
    params = json.load(f)

meta_list = []

for idx, path in enumerate(Path(args.root).rglob('*.%s' % params['extension']['audio'])):
    with open(path.with_suffix('.' + params['extension']['meta']), 'r') as f:
        meta = json.load(f)
    # with open(args.json, 'r') as f:
    #     meta = json.load(f)
    with open(args.farfield_meta, mode='a') as f:

        meta['snr'] = [10, 10]
        meta['gain'] = [1.85, 1.8, 0.56, 1.18, 1.81, 0.53, 1.07, 1.01, 1.07, 0.65, 1.56, 1.84, 0.75, 1.58, 1.94, 1.94, 1.05, 0.89, 1.53, 1.06, 1.39, 1.42, 0.61, 0.85, 1.74, 0.84, 0.98, 1.56, 1.76, 0.8, 1.95, 1.77]
        meta ['volume'] = 0.61
        # meta['snr'] = (np.round(
        #     np.random.uniform(params['snr']['min'], params['snr']['max'], len(meta['srcs'])) * 10) / 10).tolist()
        # meta['gain'] = (np.round(
        #     np.random.uniform(params['gain']['min'], params['gain']['max'], len(meta['mics'])) * 100) / 100).tolist()
        # meta['volume'] = round(rnd.uniform(params['volume']['min'], params['volume']['max']) * 100) / 100
        meta['path'] = str(path)
        ##meta_list.append(meta)
        meta_str = json.dumps(meta)
        print(meta_str)
        f.write(meta_str)
        f.write('\n')
