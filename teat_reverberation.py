'''
Applying the maximum SNR beamformer on speech with various reverberation condition but fixed other parameter.

'''
import argparse
import beam_maxSNR
import librosa as lr
import numpy as np
import torch
import json
import scipy
import math
from scipy.io.wavfile import write


from dataset.array import Array
from model.blstm import Blstm


parser = argparse.ArgumentParser()
parser.add_argument('--audio', default='./meta/test_re_3d/AUDIO_test_spherical_shape.meta', type=str, help='Meta for audio')
parser.add_argument('--json', default='./json/features.json', type=str, help='JSON of parameters')
parser.add_argument('--model_src', default='./model/model_saved.bin', type=str, help='Model to evaluate')
parser.add_argument('--wave_dst', default='./test_re_3d/output_test_cylindrical_2f_max4/', type=str, help='Wave file to save result')
args = parser.parse_args()



dataset = Array(file_meta=args.audio, file_json=args.json)

net = Blstm(file_json=args.json)
net.load_state_dict(torch.load(args.model_src))

lengh = 4
gama = 0.96

print(args.audio)
print(args.wave_dst)
print(lengh)
for index in range(4, 6):

    Xs, Ns, Ys, YYs = dataset[index]
    print(index)

    M = beam_maxSNR.mask(YYs, net)

    Ms = np.expand_dims(M, 0).repeat(Ys.shape[0], 0)
    Ts = Ys * Ms
    Is = Ys * (1.0 - Ms)
    Zs = beam_maxSNR.maxSNR(Ys, Ts, Is, lengh, gama)

    Cs = Xs[0, 0, :, :]

    XsTarget = np.transpose(Cs)
    XsMixed = np.transpose(Ys[0, :, :])
    XsOut = np.transpose(Zs)
    xsOut = np.expand_dims(lr.core.istft(XsOut), 1)
    xsTarget = np.expand_dims(lr.core.istft(XsTarget), 1)
    xsMixed = np.expand_dims(lr.core.istft(XsMixed), 1)

    xs = np.concatenate((xsTarget, xsMixed, xsOut), axis=1)

    wave_pre = '{0:>08d}'.format(index)
    write(args.wave_dst+wave_pre+'.wav', 16000, xs)
