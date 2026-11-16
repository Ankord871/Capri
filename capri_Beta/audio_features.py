"""Признаки звука с нуля: log-mel спектрограмма + дельты на чистом numpy.

Никаких готовых аудио-моделей и библиотек обработки речи.
Улучшения: дельта-признаки (динамика), более устойчивая нормализация,
чуть более длинный контекст.
"""
import numpy as np

SR = 16000
WIN, HOP, NFFT, N_MELS = 400, 160, 512, 40        # 25 мс окно, шаг 10 мс
CLIP_SAMPLES = int(SR * 1.6)                      # чуть длиннее: 1.6 с
POOL = 4                                          # усреднение по времени
N_FRAMES = (1 + (CLIP_SAMPLES - WIN) // HOP) // POOL
# статичные mel + дельты → в 2 раза больше признаков
FEAT_DIM = N_FRAMES * N_MELS * 2


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=NFFT, sr=SR, fmin=60.0, fmax=7600.0):
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        c = max(c, l + 1)
        r = max(r, c + 1)
        fb[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(WIN).astype(np.float32)


def log_mel(x):
    """Сигнал -> (кадры, 40) логарифм энергии в mel-полосах."""
    x = np.asarray(x, dtype=np.float32)
    # предыскажение + лёгкая нормализация амплитуды
    peak = np.max(np.abs(x)) + 1e-8
    x = x / peak
    x = np.append(x[:1], x[1:] - 0.97 * x[:-1])
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n_frames = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WINDOW
    # избегаем переполнения на очень тихих сигналах
    spec = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)) ** 2
    return np.log(spec @ _FB.T + 1e-6)


def _deltas(feat, width=2):
    """Простые дельты по времени (центральная разность)."""
    # feat: (T, F)
    t = feat.shape[0]
    if t < 3:
        return np.zeros_like(feat)
    pad = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    d = np.zeros_like(feat)
    for i in range(t):
        # взвешенная сумма как в HTK/kaldi (упрощённо)
        num = 0.0
        den = 0.0
        for n in range(1, width + 1):
            num += n * (pad[i + width + n] - pad[i + width - n])
            den += 2 * n * n
        d[i] = num / (den + 1e-8)
    return d


def fit_window(x, jitter=0):
    """Вырезает 1.6 с вокруг середины речи. Недостающее дополняет шумом комнаты.

    jitter (в отсчётах) сдвигает окно: используется только при обучении.
    """
    x = np.asarray(x, dtype=np.float32)
    hop = 320
    if len(x) < hop:
        x = np.pad(x, (0, hop - len(x)))
    n = len(x) // hop
    e = (x[: n * hop].reshape(n, hop) ** 2).mean(1)
    active = np.nonzero(e >= 0.08 * e.max())[0]   # чуть чувствительнее
    if len(active) == 0:
        mid = len(x) // 2
    else:
        mid = (active[0] + active[-1] + 1) * hop // 2
    # оценка шума по самым тихим фреймам
    quiet = np.sort(e)[: max(1, n // 8)]
    noise_rms = float(np.sqrt(quiet.mean() + 1e-12))
    rng = np.random.default_rng(0)
    out = rng.normal(0, noise_rms, CLIP_SAMPLES).astype(np.float32)
    start = int(mid + jitter - CLIP_SAMPLES // 2)
    s0, s1 = max(0, start), min(len(x), start + CLIP_SAMPLES)
    if s1 > s0:
        out[s0 - start: s1 - start] = x[s0:s1]
    return out


def features_from_clip(clip):
    """log-mel + deltas → плоский вектор фиксированной длины."""
    lm = log_mel(clip)
    # относительная нормализация по максимуму (громкость-инвариантность)
    lm = np.maximum(lm - lm.max(), -16.0)
    # лёгкая пер-канальная центрировка
    lm = lm - lm.mean(axis=0, keepdims=True)

    t = (lm.shape[0] // POOL) * POOL
    if t < POOL:
        # слишком короткий — дополняем
        pad = np.full((POOL - t, N_MELS), lm.min() if t > 0 else -16.0, dtype=np.float32)
        lm = np.vstack([lm[:t], pad]) if t > 0 else pad
        t = POOL
    pooled = lm[:t].reshape(-1, POOL, N_MELS).mean(1)   # (T', 40)

    # дельты
    d = _deltas(pooled)
    feat = np.concatenate([pooled, d], axis=1)          # (T', 80)

    # финальная шкала примерно в [-1, 1]
    return (feat / 6.0).astype(np.float32).ravel()


def waveform_features(x, jitter=0):
    return features_from_clip(fit_window(x, jitter))
