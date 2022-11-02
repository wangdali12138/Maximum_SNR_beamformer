'''
Applying the maximum SNR beamformer with oracle pairwise ratio mask 

'''
import argparse
import beam_maxSNR
import librosa as lr
import numpy as np
import torch
import json
import math
from scipy.io.wavfile import write


from dataset.array import Array
from model.blstm import Blstm

parser = argparse.ArgumentParser()
parser.add_argument('--audio', default='./meta/AUDIO_ReSpeaker_USB.meta', type=str, help='Meta for audio')
parser.add_argument('--json', default='./json/features.json', type=str, help='JSON of parameters')
parser.add_argument('--model_src', default='./model/model_saved.bin', type=str, help='Model to evaluate from')
parser.add_argument('--wave_dst', default='./test_Respeaker_USB_idealM_2/', type=str, help='Wave file to save result')
parser.add_argument('--num_eles', default= 300, type=int, help='number of test elements in dataset')
args = parser.parse_args()

# Dataset

dataset = Array(file_meta=args.audio, file_json=args.json)

# Model

net = Blstm(file_json=args.json)
net.load_state_dict(torch.load(args.model_src))

# Evaluate

with open(args.json, 'r') as f:
    features = json.load(f)
frameSize = features['frame_size']
hopSize = features['hop_size']

lengh = 2
gama = 0.96

for index in range(0, 300):
    
    # ref_Masks:  Oracle pairwise ratio mask
    Xs, Ns, Ys, YYs, ref_Masks = dataset[index]
    print(index)

    n_pairs = len(ref_Masks)
    M = sum(ref_Masks) / n_pairs
    Ms = np.expand_dims(M, 0).repeat(Ys.shape[0], 0)
    
    # easimate the component of target and 'noise'(inlcude interference and addictive noise)
    Ts = Ys * Ms
    Is = Ys * (1.0 - Ms)
    
    # get estimation of target signal with maximum SNR beamforming
    Zs = beam_maxSNR.maxSNR(Ys, Ts, Is, lengh, gama)
    
    # save the result
    Cs = Xs[0,0,:,:]
    XsTarget = np.transpose(Cs)
    XsMixed = np.transpose(Ys[0, :, :])
    XsOut = np.transpose(Zs)
    xsOut = np.expand_dims(lr.core.istft(XsOut), 1)
    xsTarget = np.expand_dims(lr.core.istft(XsTarget), 1)
    xsMixed = np.expand_dims(lr.core.istft(XsMixed), 1)
    xs = np.concatenate((xsTarget, xsMixed, xsOut), axis=1)
    wave_pre = '{0:>08d}'.format(index)
    write(args.wave_dst+wave_pre+'.wav', 16000, xs)

print("process is successful")
