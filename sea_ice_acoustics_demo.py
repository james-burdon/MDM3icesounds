#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sea-Ice Acoustics MVP — Source→Environment→Propagation→Sensitivity
Author: (you)
Date: 2025-10-03

What this script does
---------------------
1) Synthesizes 3 sea-ice event source signals (time + spectrum):
   - crack (short, high-frequency noisy pulse)
   - moan  (longer, narrow-band chirp)
   - icequake (low-mid frequency decaying tone)

2) Defines 3 idealized Sound Speed Profiles (SSP) and 2 surface types:
   SSPs:  isothermal / stratified (surface warm) / weak sound channel
   Surface: open water / ice cover (stronger surface loss)

3) Computes analytical transmission loss (TL) as:
   TL(f, r) = n*log10(r) + alpha_Thorp(f[kHz]) * r_km  [dB]
   where n = 20 (spherical), 10 (cylindrical), or adjusted per SSP

4) Band-aggregates RL using source spectrum |S(f)|^2 as weights:
   RL_B(r) = SL_B - < TL(f, r) >_{weights=|S(f)|^2}

5) Produces figures:
   - fig1_sources.png: waveforms + spectra (panel grid)
   - fig2_ssp.png: idealized c(z) curves
   - fig3_tl_examples.png: TL vs distance for one SSP/surface per source
   - fig4_sensitivity.png: RL_B vs distance for 3x3 (source x SSP) under two surfaces

This is an MVP for coursework. For accuracy, later swap analytic TL with BELLHOP/RAM.

Usage
-----
$ python sea_ice_acoustics_mvp.py

Outputs under ./figs/
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Utilities -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def db10(x, eps=1e-12):
    return 10*np.log10(np.maximum(x, eps))

def thorp_alpha_db_per_km(f_hz):
    """Thorp absorption in dB/km. f_hz: array-like (Hz). Valid roughly 100 Hz–1 MHz.
       Below ~100 Hz, this overestimates slightly; still fine for MVP.
    """
    f_khz = np.maximum(f_hz, 1.0) / 1000.0  # avoid zero
    term1 = 0.11 * (f_khz**2) / (1 + f_khz**2)
    term2 = 44   * (f_khz**2) / (4100 + f_khz**2)
    term3 = 2.75e-4 * (f_khz**2)
    term4 = 0.003
    return term1 + term2 + term3 + term4

# ------------------------- Sources ---------------------------

def synthesize_sources(fs=2000, dur_s=6.0):
    t = np.arange(0, dur_s, 1/fs)

    # crack: band-pass noisy pulse with fast decay
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(t.size)
    # window for pulse (0.5 s)
    tau_c = 0.15
    env_c = np.exp(-t/tau_c)
    bp_low, bp_high = 300, 1200
    # simple FFT-domain bandpass
    fft_noise = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(t.size, 1/fs)
    mask = (freqs>=bp_low) & (freqs<=bp_high)
    fft_noise_filtered = np.zeros_like(fft_noise)
    fft_noise_filtered[mask] = fft_noise[mask]
    crack = np.fft.irfft(fft_noise_filtered, n=t.size).real
    crack = crack/np.max(np.abs(crack)+1e-12) * env_c

    # moan: linear chirp narrow-band, smooth window
    f0, f1 = 80, 300
    k = (f1 - f0) / dur_s
    phase = 2*np.pi*(f0*t + 0.5*k*t**2)
    moan = np.sin(phase)
    # apply Hann window spanning 4 s in the middle
    win = np.zeros_like(t)
    t0, t1 = 1.0, 5.0
    inside = (t>=t0) & (t<=t1)
    win[inside] = 0.5*(1 - np.cos(2*np.pi*(t[inside]-t0)/(t1-t0)))
    moan = moan * win

    # icequake: low-mid frequency decaying tone, allow slight AM jitter
    fq = 120.0
    tau_q = 2.0
    am = 1.0 + 0.1*np.sin(2*np.pi*0.8*t)
    icequake = am * np.sin(2*np.pi*fq*t) * np.exp(-t/tau_q)

    sources = {
        "crack": crack,
        "moan": moan,
        "icequake": icequake
    }
    return t, sources, fs

def source_spectrum_weights(sig, fs, fmin=20, fmax=500):
    """Return frequency bins (Hz) and normalized |S(f)|^2 weights within [fmin,fmax]."""
    n = len(sig)
    S = np.fft.rfft(sig * np.hanning(n))
    f = np.fft.rfftfreq(n, 1/fs)
    band = (f>=fmin) & (f<=fmax)
    P = np.abs(S[band])**2
    if P.sum() <= 0:
        P = np.ones_like(P)
    w = P / P.sum()
    return f[band], w

# ------------------------- SSPs ------------------------------

def ssp_isothermal(z, c0=1450.0):
    return np.full_like(z, c0, dtype=float)

def ssp_stratified(z, c0=1445.0, grad=0.017):
    """Surface warm → higher c at top, decreasing with z (m)."""
    return c0 - grad*z

def ssp_weak_channel(z, c_mid=1440.0, amp=6.0, z_mid=300.0, width=400.0):
    """Parabolic minimum at z_mid (simplified SOFAR-like)."""
    return c_mid + amp*((z - z_mid)/width)**2

def build_ssps(H=1000.0, dz=10.0):
    z = np.arange(0, H+dz, dz)
    return z, {
        "isothermal": ssp_isothermal(z),
        "stratified": ssp_stratified(z),
        "weak_channel": ssp_weak_channel(z)
    }

# ------------------------- TL model --------------------------

def tl_analytic(freqs_hz, r_km, mode, surface="open"):
    """
    Analytic TL(f, r) = n*log10(r_m) + alpha(f)*r_km
      mode in {"isothermal","stratified","weak_channel"}
      surface in {"open","ice"}
    Heuristic geometric exponent n:
      isothermal: 20 (spherical)
      stratified: 22 (slightly worse near-surface due to downward refraction away)
      weak_channel: 18 (slightly better due to ducting tendency)
    Surface loss (ice): +3 dB one-way penalty (heuristic) beyond 1 km
    """
    r_m = np.maximum(r_km*1000.0, 1.0)
    if mode == "isothermal":
        n = 20.0
    elif mode == "stratified":
        n = 22.0
    elif mode == "weak_channel":
        n = 18.0
    else:
        n = 20.0
    geom = n*np.log10(r_m)
    alpha = thorp_alpha_db_per_km(freqs_hz)  # shape (F,)
    # Broadcast to (F, R)
    TL = geom[None, :] + alpha[:, None]*r_km[None, :]
    if surface == "ice":
        # Extra surface loss after 1 km (coarse heuristic)
        extra = np.where(r_km >= 1.0, 3.0, 0.0)
        TL = TL + extra[None, :]
    return TL  # (F, R)

def rl_band_aggregated(SL_B, freqs_hz, weights, r_km, mode, surface):
    """Compute band-aggregated RL at distances r_km given spectrum weights."""
    TL = tl_analytic(freqs_hz, r_km, mode=mode, surface=surface)  # (F,R)
    TL_w = (weights[:, None] * TL).sum(axis=0)  # weighted average over f
    RL_B = SL_B - TL_w
    return RL_B, TL

# ------------------------- Plotting --------------------------

def plot_sources(t, sources, fs, outdir):
    fig = plt.figure(figsize=(10, 8))
    names = list(sources.keys())
    for i, name in enumerate(names, 1):
        sig = sources[name]
        # waveform
        ax = plt.subplot(3,2,2*i-1)
        ax.plot(t, sig)
        ax.set_title(f"{name} — waveform")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Amp.")
        # spectrum
        f, w = source_spectrum_weights(sig, fs)
        ax2 = plt.subplot(3,2,2*i)
        ax2.plot(f, db10(w))
        ax2.set_title(f"{name} — normalized |S(f)|^2 (dB)")
        ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Rel. power [dB]")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_sources.png"), dpi=200)
    plt.close(fig)

def plot_ssps(z, ssps, outdir):
    fig = plt.figure(figsize=(6,5))
    for k, c in ssps.items():
        plt.plot(c, z, label=k)
    plt.gca().invert_yaxis()
    plt.xlabel("Sound speed c(z) [m/s]"); plt.ylabel("Depth z [m]")
    plt.title("Idealized SSPs")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_ssp.png"), dpi=200)
    plt.close(fig)

def plot_tl_examples(freqs, r_km, TL_dict, outdir):
    fig = plt.figure(figsize=(6,5))
    for label, TL in TL_dict.items():
        # Take median over freq as a 1D representative curve
        TL_med = np.median(TL, axis=0)
        plt.plot(r_km, TL_med, label=label)
    plt.xlabel("Range r [km]"); plt.ylabel("Median TL over band [dB]")
    plt.title("TL vs Range (band-median)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_tl_examples.png"), dpi=200)
    plt.close(fig)

def plot_sensitivity(r_km, RL_results, outdir):
    """RL_results: dict[(source, ssp, surface)] -> RL_B(r)"""
    surfaces = sorted({k[2] for k in RL_results.keys()})
    ssps = sorted({k[1] for k in RL_results.keys()})
    sources = sorted({k[0] for k in RL_results.keys()})
    for surf in surfaces:
        fig = plt.figure(figsize=(8,6))
        for ssp in ssps:
            for src in sources:
                RL = RL_results[(src, ssp, surf)]
                plt.plot(r_km, RL, label=f"{src} | {ssp}")
        plt.xlabel("Range r [km]"); plt.ylabel("Band-agg RL [dB re 1 µPa]")
        plt.title(f"Sensitivity: RL vs r  ({surf})")
        plt.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig4_sensitivity_{surf}.png"), dpi=200)
        plt.close(fig)

# ------------------------- Main ------------------------------

def main():
    outdir = ensure_dir("figs")
    # 1) sources
    t, sources, fs = synthesize_sources(fs=2000, dur_s=6.0)
    plot_sources(t, sources, fs, outdir)

    # Choose nominal broadband source levels (relative). You can calibrate later.
    SL_map = {"crack": 165.0, "moan": 170.0, "icequake": 180.0}  # dB re 1 µPa @1m (nominal)

    # 2) SSPs (idealized only used to pick TL geometric exponent & surface loss heuristics in this MVP)
    z, ssps = build_ssps(H=1000.0, dz=10.0)
    plot_ssps(z, ssps, outdir)

    # 3) Ranges & band
    r_km = np.linspace(0.1, 50.0, 400)  # 0.1–50 km
    fband = (20, 500)  # analysis band

    # Example TL panel for one SSP/surface per source
    TL_examples = {}

    # 4) RL band-aggregated across (source, ssp, surface)
    RL_results = {}
    for ssp_name in ssps.keys():
        for surface in ["open", "ice"]:
            for src_name, sig in sources.items():
                freqs, weights = source_spectrum_weights(sig, fs, *fband)
                RL_B, TL = rl_band_aggregated(
                    SL_B=SL_map[src_name],
                    freqs_hz=freqs,
                    weights=weights,
                    r_km=r_km,
                    mode=ssp_name,
                    surface=surface
                )
                RL_results[(src_name, ssp_name, surface)] = RL_B
                # populate TL_examples with one curve per source (only for a chosen ssp/surface to avoid clutter)
                if (ssp_name == "isothermal") and (surface == "open"):
                    TL_examples[f"{src_name}"] = TL

    # 5) Plots
    plot_tl_examples(freqs, r_km, TL_examples, outdir)
    plot_sensitivity(r_km, RL_results, outdir)

    # Save a small JSON summary for the report
    summary = {
        "band_hz": fband,
        "ranges_km": [float(r_km[0]), float(r_km[-1])],
        "sources_SL_dB": SL_map,
        "surfaces": ["open", "ice"],
        "ssps": list(ssps.keys())
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)

    print("Done. Figures saved under ./figs/ :")
    for fn in ["fig1_sources.png","fig2_ssp.png","fig3_tl_examples.png","fig4_sensitivity_open.png","fig4_sensitivity_ice.png","summary.json"]:
        print(" -", os.path.join(outdir, fn))

if __name__ == "__main__":
    main()
