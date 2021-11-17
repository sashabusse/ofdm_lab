import numpy as np


def auto_corr_analysis(s, ofdm_cfg, w_width, out_len=None, tg_pos_lvl=0.7, fr_shift_lvl=0.7):
    if out_len is None:
        out_len = ofdm_cfg.n_fft
    corr_res = np.zeros(out_len, dtype=complex)
    for i in range(out_len):
        corr_res[i] = (s[i:i+w_width] @ s[ofdm_cfg.n_fft+i:ofdm_cfg.n_fft+i+w_width].conj())/np.sqrt((s[i:i+w_width] @ s[i:i+w_width].conj())*(s[ofdm_cfg.n_fft+i:ofdm_cfg.n_fft+i+w_width] @ s[ofdm_cfg.n_fft+i:ofdm_cfg.n_fft+i+w_width].conj()))

    corr_res = corr_res/np.max(np.abs(corr_res))

    corr_ind = np.arange(len(corr_res))
    tg_pos = int(np.rint(np.mean(corr_ind[np.abs(corr_res) > tg_pos_lvl])))
    tg_mid_shift = int(0.5*ofdm_cfg.t_guard - (ofdm_cfg.t_guard - w_width)/2)
    tg_pos += tg_mid_shift
    freq_offset = np.mean(  -np.angle(corr_res[np.abs(corr_res) > fr_shift_lvl])/(2*np.pi*ofdm_cfg.n_fft)  )

    return corr_res, tg_pos, freq_offset
