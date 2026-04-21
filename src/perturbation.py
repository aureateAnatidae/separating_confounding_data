"""
Perturbation of input to determine importance of inputs

Schirrmeister 2017 - https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730
"""

import numpy as np
import torch


def _predict_class_scores(model, X, class_idx, batch_size=64, use_softmax=True):
    """
    Predict class score for each trial.

    X shape: (n_trials, n_chans, n_times)
    """
    model.eval()
    device = next(model.parameters()).device

    scores = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            logits = model(xb)

            if use_softmax:
                out = torch.softmax(logits, dim=1)
            else:
                out = logits

            scores.append(out[:, class_idx].detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def amplitude_perturbation_importance(
    model,
    X,
    class_idx,
    sfreq,
    n_iterations=30,
    noise_std=0.02,
    batch_size=64,
    seed=2017,
    use_softmax=True,
):
    """
    Reimplementation of the old Braindecode-style amplitude perturbation idea.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X : np.ndarray
        Shape: (n_trials, n_chans, n_times).
        Use the same window length the model was trained on.
    class_idx : int
        Class score to explain.
    sfreq : float
        Sampling frequency.
    n_iterations : int
        Number of random perturbation rounds.
    noise_std : float
        Standard deviation of additive amplitude noise.
        Old Braindecode docs used N(0, 0.02) for additive amplitude perturbation.
    batch_size : int
        Prediction batch size.
    seed : int
        Random seed.
    use_softmax : bool
        If True, correlate perturbations with class probabilities.
        If False, correlate with raw logits.

    Returns
    -------
    corr : np.ndarray
        Shape: (n_chans, n_freqs).
        Correlation between amplitude perturbation and class-score change.
    freqs : np.ndarray
        Frequencies corresponding to corr columns.
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X shape (n_trials, n_chans, n_times), got {X.shape}"
        )

    n_trials, n_chans, n_times = X.shape
    rng = np.random.default_rng(seed)

    fft = np.fft.rfft(X, axis=-1)
    amps = np.abs(fft)
    phases = np.angle(fft)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    base_scores = _predict_class_scores(
        model,
        X,
        class_idx=class_idx,
        batch_size=batch_size,
        use_softmax=use_softmax,
    )

    n_freqs = len(freqs)

    sum_x = np.zeros((n_chans, n_freqs), dtype=np.float64)
    sum_x2 = np.zeros((n_chans, n_freqs), dtype=np.float64)
    sum_xy = np.zeros((n_chans, n_freqs), dtype=np.float64)

    sum_y = 0.0
    sum_y2 = 0.0
    n_total = 0

    for _ in range(n_iterations):
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=amps.shape,
        ).astype(np.float32)

        amps_pert = amps + noise
        amps_pert = np.maximum(amps_pert, 0.0)

        fft_pert = amps_pert * np.exp(1j * phases)
        X_pert = np.fft.irfft(fft_pert, n=n_times, axis=-1).astype(np.float32)

        pert_scores = _predict_class_scores(
            model,
            X_pert,
            class_idx=class_idx,
            batch_size=batch_size,
            use_softmax=use_softmax,
        )

        score_change = pert_scores - base_scores
        # shape: (n_trials,)

        sum_x += noise.sum(axis=0)
        sum_x2 += (noise**2).sum(axis=0)
        sum_xy += (noise * score_change[:, None, None]).sum(axis=0)

        sum_y += score_change.sum()
        sum_y2 += (score_change**2).sum()
        n_total += n_trials

    numerator = n_total * sum_xy - sum_x * sum_y

    denom_x = n_total * sum_x2 - sum_x**2
    denom_y = n_total * sum_y2 - sum_y**2

    denominator = np.sqrt(np.maximum(denom_x, 0)) * np.sqrt(max(denom_y, 0))

    corr = numerator / (denominator + 1e-12)

    return corr, freqs
