"""Positive-class F1, balanced accuracy, train-only balancing, and paired participant inference."""
import numpy as np
import torch
from torch.utils.data import Sampler


def f1(y, probability, cut):
    y = np.asarray(y, bool)
    pred = np.asarray(probability) >= cut
    return 2 * (pred & y).sum(axis=-1) / np.maximum(1, pred.sum(axis=-1) + y.sum())


def threshold(y, probability):
    """Maximize validation F1; use the highest threshold when scores tie."""
    p = np.asarray(probability, dtype=float); y = np.asarray(y, dtype=int)
    order = np.argsort(-p, kind='stable'); p = p[order]; y = y[order]
    ends = np.flatnonzero(np.r_[p[1:] != p[:-1], True])
    scores = 2 * y.cumsum()[ends] / np.maximum(1, ends + 1 + y.sum())
    return float(p[ends[np.argmax(scores)]])


def balanced_indices(y, seed):
    """Keep every training participant, duplicate the smaller class to equality."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Training needs both outcome classes for oversampling')
    size = max(map(len, groups))
    indices = np.concatenate([np.r_[g, rng.choice(g, size - len(g), replace=True)] for g in groups])
    return rng.permutation(indices)


class BalancedSampler(Sampler):
    def __init__(self, y, seed):
        self.y = np.asarray(y); self.seed = seed; self.epoch = 0
    def __len__(self):
        return 2 * max(int((self.y == k).sum()) for k in (0, 1))
    def __iter__(self):
        return iter(balanced_indices(self.y, self.seed + self.epoch).tolist())


def bh_adjust(p):
    p = np.asarray(p, float); order = np.argsort(p); n = len(p)
    adjusted = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n); result[order] = np.clip(adjusted, 0, 1)
    return result


class PairedF1:
    """Shared stratified bootstrap draws; paired prediction-swap permutation p.

    Evaluation participants occur once in the input. Resampling here estimates
    uncertainty, and never changes the evaluated prevalence or trains a model.
    Each swap exchanges the two decisions for one participant. It tests paired
    prediction exchangeability conditional on the fitted models/thresholds.
    """
    def __init__(self, y, samples=10000, seed=42, device='cpu'):
        self.y = np.asarray(y, int); self.device = device
        self.samples = samples; rng = np.random.default_rng(seed)
        weights = np.ones((samples + 1, len(y)), np.float64)
        for k in (0, 1):
            idx = np.flatnonzero(self.y == k)
            if not len(idx): raise ValueError('Evaluation needs both outcome classes')
            weights[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)), samples)
        self.w = torch.as_tensor(weights, device=device)
        self.truth = torch.as_tensor(self.y, dtype=torch.float64, device=device)
        self.positive = self.w @ self.truth
        self.swaps = torch.as_tensor(rng.integers(0, 2, (samples, len(y))),
                                     dtype=torch.float64, device=device)

    def decisions(self, p, cut):
        return torch.as_tensor(np.atleast_2d(np.asarray(p) >= cut), dtype=torch.float64, device=self.device)

    @staticmethod
    def score(tp, predicted, positive, total, metric):
        if metric == 'balanced_accuracy':
            return .5*(tp/positive + (total-positive-predicted+tp)/(total-positive))
        return 2*tp/(predicted+positive).clamp_min(1)

    def draws(self, p, cut, metric='f1'):
        d = self.decisions(p, cut)
        tp = (d * self.truth) @ self.w.T
        predicted = d @ self.w.T
        return self.score(tp, predicted, self.positive, self.w.sum(1), metric).cpu().numpy()

    def pvalues(self, a, b, cut_a, cut_b, metric='f1'):
        a = self.decisions(a, cut_a); b = self.decisions(b, cut_b)
        a, b = torch.broadcast_tensors(a, b)
        # Swapping only changes predicted positives and true positives.
        change = (b-a) @ self.swaps.T
        change_tp = ((b-a)*self.truth) @ self.swaps.T
        ap, bp = a.sum(1, keepdim=True), b.sum(1, keepdim=True)
        at, bt = (a*self.truth).sum(1, keepdim=True), (b*self.truth).sum(1, keepdim=True)
        npos = self.truth.sum()
        def score(tp, predicted): return self.score(tp, predicted, npos, len(self.y), metric)
        observed = score(at, ap) - score(bt, bp)
        null = score(at+change_tp, ap+change) - score(bt-change_tp, bp-change)
        return ((1+(null.abs() >= observed.abs()-1e-12).sum(1))/(self.samples+1)).cpu().numpy()

    @staticmethod
    def intervals(draws):
        draws = np.atleast_2d(draws)
        ci = np.quantile(draws[:, 1:], [.025, .975], axis=1)
        return [dict(value=float(d[0]), ci_lower=float(lo), ci_upper=float(hi))
                for d, lo, hi in zip(draws, *ci)]

    def performance(self, p, cut, metric='f1'):
        return self.intervals(self.draws(p, cut, metric))[0]

    def effects(self, a, b, cut_a, cut_b, metric='f1'):
        """Effect = metric(a) - metric(b); all draws use the same participants."""
        rows = self.intervals(self.draws(a, cut_a, metric)-self.draws(b, cut_b, metric))
        for row, p in zip(rows, self.pvalues(a, b, cut_a, cut_b, metric)):
            row['p_value'] = float(p)
        return rows
