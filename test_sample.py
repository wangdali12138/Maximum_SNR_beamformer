import argparse
import beam
import beam_MFmaxSNR
import librosa as lr
import numpy as np
import torch
import json
import scipy
import math
from scipy.io.wavfile import write


from dataset.array import Array
from model.blstm import Blstm
from usedFunction import SNR_cal, Distor_cal

parser = argparse.ArgumentParser()
parser.add_argument('--audio', default='./meta/AUDIO_Respeaker_USB.meta', type=str, help='Meta for audio')
parser.add_argument('--json', default='./json/features.json', type=str, help='JSON of parameters')
parser.add_argument('--model_src', default='./model/model_saved.bin', type=str, help='Model to evaluate from')
parser.add_argument('--wave_dst', default='./test_Respeaker_USB/00000004', type=str, help='Wave file to save result')
parser.add_argument('--index', default= 4, type=int, help='Index of element in dataset')
args = parser.parse_args()

## Dataset

dataset = Array(file_meta=args.audio, file_json=args.json)

## Model

net = Blstm(file_json=args.json)
net.load_state_dict(torch.load(args.model_src))

## Evaluate

with open(args.json, 'r') as f:
    features = json.load(f)
frameSize = features['frame_size']
hopSize = features['hop_size']
Xs, Ns, Ys, YYs, ref_Masks = dataset[args.index]

# set the paramter of maximum SNR beamformer
lengh = 1
gamay = 0.96

# generate the estimation masks from network
M = beam_MFmaxSNR.mask(YYs, net)
Ms = np.expand_dims(M, 0).repeat(Ys.shape[0], 0)

# easimate the component of target and 'noise'(inlcude interference and addictive noise)
Ts = Ys * Ms
Is = Ys * (1.0 - Ms)

# get estimation of target signal
out_z = beam_MFmaxSNR.MFmaxSNR(Ys, Ts, Is, lengh, gamay)

# save the result
Cs = Xs[0, 0, :, :]
XsTarget = np.transpose(Cs)
XsMixed = np.transpose(Ys[0, :, :])
XsOut = np.transpose(out_z)
xsOut = np.expand_dims(lr.core.istft(XsOut), 1)
xsTarget = np.expand_dims(lr.core.istft(XsTarget), 1)
xsMixed = np.expand_dims(lr.core.istft(XsMixed), 1)
xs = np.concatenate((xsTarget, xsMixed, xsOut), axis=1)

write(args.wave_dst+'.wav',16000, xs)


print("process is successful")
