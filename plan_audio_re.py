import argparse
import json
import random as rnd

parser = argparse.ArgumentParser()

parser.add_argument('--farfield', default='./meta/test_re_3d/test_spherical_shape.meta', type=str, help='Meta for farfield')
parser.add_argument('--audio_meta', default='./meta/test_re_3d/AUDIO_test_spherical_shape.meta', type=str,
                    help='Meta for samples of audios')
parser.add_argument('--count', default=24, type=int, help='Number of audio samples')
args = parser.parse_args()

with open(args.farfield) as f:
    farfield_elements = f.read().splitlines()

for i in range(0, args.count):
    meta = {}

    meta['farfield'] = json.loads(farfield_elements[i])

    meta['speech'] = [{"offset": 1.52, "duration": 5.0, "path": "test_audio_wav/2300-131720-0036.wav"}, {"offset": 11.47, "duration": 5.0, "path": "test_audio_wav/5639-40744-0032.wav"}]
    meta_str = json.dumps(meta)
    print(meta_str)
    with open(args.audio_meta, mode='a') as f:
        f.write(meta_str)
        f.write('\n')
