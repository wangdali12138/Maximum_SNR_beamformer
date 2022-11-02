import argparse
from pesq import pesq
from os.path import dirname, join as pjoin
from scipy.io import wavfile


parser = argparse.ArgumentParser()
parser.add_argument('--wave_dst', default='./test_Respeaker_USB_4', type=str, help='folder of wave files')
parser.add_argument('--num_eles', default= 300, type=int, help='number of test elements')
args = parser.parse_args()

pesq_list = []
si_snr_list = []

for index in range (0, args.num_eles):
    print(index)
    wav_name = '{0:>08d}'.format(index)+'.wav'
    file_name = pjoin(args.wav_dst, '/', wav_name)

    sc, data = wavfile.read(file_name)

    target = data[:, 0]
    preds =  data[:, 2]
    nb_pesq = pesq(sc, target, preds, 'nb')
    


    pesq_list.append(nb_pesq)
    

mean_pesq = sum(pesq_list)/len(pesq_list)

print(mean_pesq)

