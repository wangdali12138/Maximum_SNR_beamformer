'''
Maximum SNR beamformer:

1.step: use generate the estimation masks with the cross-spectrum through the network;
2.step: estimate the desired signal with Maximum SNR beamformer

'''

import librosa as lr
import numpy as np
import scipy.linalg as la
import torch


## compute eastimaton masks from network 
def mask(YYs, net):
    net.eval()
    nPairs = YYs.shape[0]

    M = 0.0

    for iPair in range(0, nPairs):
        YY = torch.from_numpy(YYs[iPair, :, :, :]).unsqueeze(0)
        MM = net(YY)
        MM = MM.squeeze(0).detach().cpu().numpy()

        M += MM

    M /= nPairs

    return M

## estimate the desired signal with Maximum SNR beamformer
def maxSNR(Ys, Xs, Ns, L, gama):

    F = Ys.shape[2]
    M = Ys.shape[0]
    T = Ys.shape[1]
    N = L
    num_avgFrm = 100

    ##create and initilize new buffs
    inWinBuf = np.zeros((M * N, F), dtype=np.complex64)
    inWin_xbuf = np.zeros((M * N, F), dtype=np.complex64)
    inWin_nbuf = np.zeros((M * N, F), dtype=np.complex64)
    # fullY = np.zeros((M*N, T, F), dtype=np.complex64)

    out_z = np.zeros((T,F), dtype=np.complex64)
    hmax_M = np.zeros((M*N,T,F), dtype=np.complex64)
    belta_M = np.zeros((T,F), dtype=np.complex64)



    ## correlation matrix

    Rfx = np.zeros((M * N, M * N, F), dtype=np.complex64)
    Rfn = np.zeros((M * N, M * N, F), dtype=np.complex64)
    # Rfy = np.zeros((M * N, M * N, F), dtype=np.complex64)

    ## particular filter
    iN1 = np.zeros((M*N, 1))
    iN1[0, 0] = 1

    ## initilize the correlation matrix
    for t in range(0, num_avgFrm):

        for m in range(0, M):
            inY = np.squeeze(Ys[m, t, :])
            inX = np.squeeze(Xs[m, t, :])
            inN = np.squeeze(Ns[m, t, :])

            inWinBuf[m * N + 1:(m + 1)*N, :] = inWinBuf[m * N:(m + 1) * N - 1, :]
            inWinBuf[m * N, :] = inY

            inWin_xbuf[m * N + 1:(m + 1)*N, :] = inWin_xbuf[m * N:(m + 1) * N - 1, :]
            inWin_xbuf[m * N, :] = inX

            inWin_nbuf[m * N + 1:(m + 1)*N, :] = inWin_nbuf[m * N:(m + 1) * N - 1, :]
            inWin_nbuf[m * N, :] = inN

        for f in range(0, F):

            # Fyvector = np.expand_dims(inWinBuf[:, f], 1)
            Fnvector = np.expand_dims(inWin_nbuf[:, f], 1)
            Fxvector = np.expand_dims(inWin_xbuf[:, f], 1)

            # tmp_RfyPlus = np.matmul(Fyvector, np.conj(np.transpose(Fyvector)))
            # Rfy[:, :, f] = Rfy[:, :, f] + tmp_RfyPlus

            tmp_RfnPlus = np.matmul(Fnvector, np.conj(np.transpose(Fnvector)))
            Rfn[:, :, f] = Rfn[:, :, f] + tmp_RfnPlus

            tmp_RfxPlus = np.matmul(Fxvector, np.conj(np.transpose(Fxvector)))
            Rfx[:, :, f] = Rfx[:, :, f] + tmp_RfxPlus


    # Rfy = Rfy/num_avgFrm
    Rfn = Rfn/num_avgFrm
    Rfx = Rfx/num_avgFrm

    ## update the correlation matrix and compute the Multichannel Maximum SNR filter coefficients
    for t in range(0, T):

        for m in range(0, M):

            inY = np.squeeze(Ys[m, t, :])
            inX = np.squeeze(Xs[m, t, :])
            inN = np.squeeze(Ns[m, t, :])

            inWinBuf[m * N + 1:(m + 1)*N, :] = inWinBuf[m * N:(m + 1) * N - 1, :]
            inWinBuf[m * N, :] = inY

            inWin_xbuf[m * N + 1:(m + 1)*N, :] = inWin_xbuf[m * N:(m + 1) * N - 1, :]
            inWin_xbuf[m * N, :] = inX

            inWin_nbuf[m * N + 1:(m + 1)*N, :] = inWin_nbuf[m * N:(m + 1) * N - 1, :]
            inWin_nbuf[m * N, :] = inN


        for f in range(0, F):

            Fyvector = np.expand_dims(inWinBuf[:, f], 1)
            Fnvector = np.expand_dims(inWin_nbuf[:, f], 1)
            Fxvector = np.expand_dims(inWin_xbuf[:, f], 1)
            # Rfy[:, :, f] = gama*Rfy[:, :, f] + (1-gama)*np.matmul(Fyvector, np.conj(np.transpose(Fyvector)))
            Rfn[:, :, f] = gama*Rfn[:, :, f] + (1 - gama)*np.matmul(Fnvector, np.conj(np.transpose(Fnvector)))
            Rfx[:, :, f] = gama*Rfx[:, :, f] + (1 - gama)*np.matmul(Fxvector, np.conj(np.transpose(Fxvector)))
            # Rfx[:, :, f] = Rfy[:, :, f]-Rfn[:, :, f]

            rRfn = Rfn[:, :, f]*T
            rRfx = Rfx[:, :, f]*T

            ## function thmat:avoid the  diagonal element < 0
            rRfx_diag = np.diag(rRfx)
            row, col = np.diag_indices_from(rRfx)
            rRfx_diag = np.where(rRfx_diag >= 0, rRfx_diag, rRfx_diag.max() * 1e-4)
            rRfx[row, col] = rRfx_diag

            ## inverse the matrix Rfn, where non-positive eigenvalues are set to zero
            [Dn, Vn] = la.eig(rRfn)
            An = np.where(Dn <= 0, 0, 1.0 / Dn)
            AA = np.diag(An)
            irRfn = np.matmul(np.matmul(Vn, AA), np.conj(np.transpose(Vn)))
            
            ## compute the parameter belta and eigenvector b1 and then get maximum SNR filter
            [D, V] = la.eig(np.matmul(irRfn, rRfx))
            index = np.argsort(-D)
            b1 = np.expand_dims(V[:, index[0]], axis=1)
            myeps = 1e-6

            expr_up1 = np.matmul(np.conj(np.transpose(b1)), rRfx)
            expr_up2 = np.squeeze(np.matmul(expr_up1, iN1))
            expr_up = expr_up2 * b1

            expr_blow1 = np.matmul(np.conj(np.transpose(b1)), rRfx)
            expr_blow2 = np.matmul(expr_blow1, b1)
            expr_blow = np.squeeze(expr_blow2.real + myeps)

            hmax = expr_up / expr_blow
            belta = expr_up2 / expr_blow
            hmax_M[:, t, f] = np.squeeze(hmax)
            belta_M[t, f] = belta


            ## apply the filter to the estimate the desired signal 
            out_z[t, f] = np.squeeze(np.matmul(np.conj(np.transpose(hmax)), Fyvector))

    return out_z






