

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
from scipy import signal
from scipy.linalg import eigh, fractional_matrix_power
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    message=r"Please use the new API settings to control TF32 behavior.*",
    category=UserWarning,
)

BCI2020_CONFIG = {
    "filter_band": [30, 90],
    "psd_band": [30, 90],
    "segment_type": "sliding",
    "win_len_s": 2.0,
    "overlap": 0.5,
}
FEATURE_ORDER = ("CSP", "PSD", "LogBP")
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass

def filter_data(data, lowcut, highcut, fs=256, notch_freq=None):
   
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    y = filtfilt(b, a, data, axis=2)
    
    if notch_freq is not None:
        if lowcut < notch_freq < highcut:
            b_notch, a_notch = iirnotch(w0=notch_freq, Q=30.0, fs=fs)
            y = filtfilt(b_notch, a_notch, y, axis=2)      
    return y

def extract_covariance_features(X):
    n_trials, n_channels, n_times = X.shape
    features = []
    triu_indices = np.triu_indices(n_channels)
    for i in range(n_trials):
        x = X[i]
        cov = np.dot(x, x.T) / (n_times - 1)
        if np.trace(cov) != 0:
            cov = cov / np.trace(cov)
        features.append(cov[triu_indices])
    return np.array(features)

def euclidean_alignment(X_data):
    n_trials, n_channels, n_times = X_data.shape
    covs = []
    for i in range(n_trials):
        x = X_data[i]
        c = np.dot(x, x.T)
        t = np.trace(c)
        covs.append(c / t if t != 0 else c)
    R = np.mean(covs, axis=0)
    try:
        P = fractional_matrix_power(R + 1e-6 * np.eye(n_channels), -0.5)
    except:
        return X_data
    X_aligned = np.zeros_like(X_data)
    for i in range(n_trials):
        X_aligned[i] = np.matmul(P, X_data[i])
    return X_aligned

def compute_log_bp(X):
    power = np.var(X, axis=2)
    return np.log(power + 1e-8)

def extract_psd_features_raw(X, sfreq=256, fmin=30, fmax=90):
    freqs, psd = signal.welch(X, fs=sfreq, nperseg=sfreq, axis=-1)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    psd_roi = psd[:, :, freq_mask]
    psd_log = 10 * np.log10(psd_roi + 1e-10)
    X_flat = psd_log.reshape(X.shape[0], -1)
    
    return X_flat

def train_weighted_pca(X, weights, n_components=20):
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(X)) / len(X)
    mean_vec = np.average(X, axis=0, weights=weights)
    X_centered = X - mean_vec
    
    X_weighted = X_centered * np.sqrt(weights[:, None])
    cov_matrix = np.dot(X_weighted.T, X_weighted)
    vals, vecs = scipy.linalg.eigh(cov_matrix)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    vals = vals[idx]
    components = vecs[:, :n_components]
    
    return components, mean_vec

def apply_pca(X_in, filters, mean_vec):
    X_centered = X_in - mean_vec
    return np.dot(X_centered, filters)

def fit_ea_matrix(X_data):
    n_trials, n_channels, n_times = X_data.shape
    covs = []
    for i in range(n_trials):
        x = X_data[i]
        c = np.dot(x, x.T)
        t = np.trace(c)
        covs.append(c / t if t != 0 else c)

    R = np.mean(covs, axis=0)

    try:
        P = fractional_matrix_power(R + 1e-6 * np.eye(n_channels), -0.5)
    except Exception:
        P = np.eye(n_channels)

    return np.real(P)

def apply_ea_matrix(X_data, P):
    X_aligned = np.zeros_like(X_data)
    for i in range(X_data.shape[0]):
        X_aligned[i] = np.matmul(P, X_data[i])
    return X_aligned

def calculate_hsfkc_theta(X, y, sample_weights=None, eta=2.0, tau=1.0):
    N, D = X.shape
    classes = np.unique(y)
    if sample_weights is None:
        s = np.ones(N)
    else:
        s = np.array(sample_weights)
        if np.sum(s) > 0:
            s = s / np.sum(s) * N
    eps = 1e-8
    mu_global = np.average(X, axis=0, weights=s)
    Sb = np.zeros(D)
    Sw = np.zeros(D)
    for c in classes:
        mask = (y == c)
        X_k = X[mask]       
        s_k = s[mask]       
        
        if len(X_k) == 0: continue
        pk = np.sum(s_k)
        mu_k = np.average(X_k, axis=0, weights=s_k)
        Sb += pk * ((mu_k - mu_global) ** 2)
        diff_sq = (X_k - mu_k) ** 2  
        Sw += np.dot(s_k, diff_sq)         
    log_ratio = np.log(Sb + eps) - np.log(Sw + eps)
    scaled_log = log_ratio * (eta / tau)   
    scaled_log -= np.max(scaled_log)
    a = np.exp(scaled_log)
    theta = a / (np.sum(a) + eps)*D
    return theta

class WeightedCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg='ledoit_wolf', log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.filters_ = None

    def _compute_covariance(self, X, weights=None):
        n_trials, n_channels, n_times = X.shape
        covs = []
        for i in range(n_trials):
            x = X[i]
            c = np.dot(x, x.T)
            if self.norm_trace:
                c /= np.trace(c)
            else:
                c /= (n_times - 1)
            covs.append(c)
        covs = np.array(covs)

        if weights is not None:
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum() * n_trials
            covs = covs * weights[:, np.newaxis, np.newaxis]
        return np.mean(covs, axis=0)

    def fit(self, X, y, sample_weight=None):
        classes = np.unique(y)
        n_channels = X.shape[1]
        covs_by_class = []
        for cls in classes:
            mask = (y == cls)
            X_cls = X[mask]
            w_cls = sample_weight[mask] if sample_weight is not None else None
            C_cls = self._compute_covariance(X_cls, w_cls)
            if self.reg == 'ledoit_wolf' or (isinstance(self.reg, float)):
                 reg_val = 1e-4 if isinstance(self.reg, str) else self.reg
                 C_cls = (1 - reg_val) * C_cls + reg_val * np.eye(n_channels)
            covs_by_class.append(C_cls)
            
        C_sum = np.sum(covs_by_class, axis=0)
        pick_filters = []
        for i, cls in enumerate(classes):
            C_cls = covs_by_class[i]
            vals, vecs = eigh(C_cls, C_sum)
            ix = np.argsort(vals)[::-1]
            vecs = vecs[:, ix]
            pick_filters.append(vecs[:, :self.n_components])
        self.filters_ = np.concatenate(pick_filters, axis=1).T 
        self.patterns_ = scipy.linalg.pinv(self.filters_).T
        return self

    def transform(self, X, return_signal=False):
        X_filt = np.asarray([np.dot(self.filters_, x) for x in X])
        if return_signal:
            return X_filt
        X_feat = (X_filt ** 2).mean(axis=2)
        if self.log:
            X_feat = np.log(X_feat + 1e-8)
        return X_feat

class JointThetaWeightKRR(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        gamma=None,
        lambda_reg=1e-2,
        alt_rounds=5,
        phi_steps=10,
        theta_steps=5,
        lr_phi=0.03,
        lr_theta=0.01,
        val_per_class=2,
        w_floor=1e-4,
        reg_theta=5e-3,
        reg_phi=1e-2,
        device="cpu",
        random_state=42,
        verbose=False,
    ):
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.alt_rounds = alt_rounds
        self.phi_steps = phi_steps
        self.theta_steps = theta_steps
        self.lr_phi = lr_phi
        self.lr_theta = lr_theta
        self.val_per_class = val_per_class
        self.w_floor = w_floor
        self.reg_theta = reg_theta
        self.reg_phi = reg_phi
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.classes_ = None
        self.class_to_idx_ = None
        self.theta_ = None
        self.w_src_ = None
        self.alpha_ = None
        self.X_train_ = None
        self.beta_ = None
        self.a_ = None
        self.b_ = None

    @staticmethod
    def _split_calibration(y, val_per_class=2, random_state=42):
        rng = np.random.RandomState(random_state)
        y = np.asarray(y)

        tr_idx, val_idx = [], []
        for c in np.unique(y):
            idx = np.where(y == c)[0].copy()
            rng.shuffle(idx)
            n_val = min(val_per_class, max(1, len(idx) // 5))
            val_idx.extend(idx[:n_val].tolist())
            tr_idx.extend(idx[n_val:].tolist())

        return np.array(tr_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)

    def _gamma_value(self, D):
        return (1.0 / D) if self.gamma is None else self.gamma

    def _weighted_rbf(self, X1, X2, theta):
        theta_sqrt = torch.sqrt(theta + 1e-12)
        X1w = X1 * theta_sqrt[None, :]
        X2w = X2 * theta_sqrt[None, :]

        x1_sq = torch.sum(X1w * X1w, dim=1, keepdim=True)
        x2_sq = torch.sum(X2w * X2w, dim=1, keepdim=True).T
        dist2 = x1_sq + x2_sq - 2.0 * (X1w @ X2w.T)
        dist2 = torch.clamp(dist2, min=0.0)

        gamma_val = self._gamma_value(X1.shape[1])
        return torch.exp(-gamma_val * dist2)

    def _to_onehot(self, y_idx, C):
        Y = torch.zeros((len(y_idx), C), dtype=torch.float32, device=self.device)
        Y[torch.arange(len(y_idx), device=self.device), y_idx] = 1.0
        return Y

    def _theta_from_q(self, q, D):
        return D * torch.softmax(q, dim=0)

    def _weights_from_phi(self, w_macro_t, w_micro_t, s_beta, s_a, s_b):
        beta = 0.05 + 0.95 * torch.sigmoid(s_beta)
        a = 1.5 * torch.sigmoid(s_a)
        b = 1.5 * torch.sigmoid(s_b)

        w_src = beta * (w_macro_t ** a) * (w_micro_t ** b)
        w_src = torch.clamp(w_src, min=self.w_floor, max=1.0)
        return w_src, beta, a, b

    def _solve_alpha(self, X_joint, y_joint_idx, theta, w_src, n_t_train):
        C = len(self.classes_)
        K = self._weighted_rbf(X_joint, X_joint, theta)

        W_diag = torch.cat([
            torch.ones(n_t_train, dtype=torch.float32, device=self.device),
            w_src
        ], dim=0)

        W_inv = torch.diag(1.0 / (W_diag + 1e-6))
        Y = self._to_onehot(y_joint_idx, C)

        A = K + self.lambda_reg * W_inv + 1e-6 * torch.eye(K.shape[0], device=self.device)
        alpha = torch.linalg.solve(A, Y)
        return alpha

    def fit(self, X_cal, y_cal, X_src, y_src, theta_init, w_macro, w_micro,
            beta_init=0.6, a_init=0.5, b_init=0.5):
        X_cal = np.asarray(X_cal, dtype=np.float32)
        y_cal = np.asarray(y_cal)
        X_src = np.asarray(X_src, dtype=np.float32)
        y_src = np.asarray(y_src)
        theta_init = np.asarray(theta_init, dtype=np.float32)
        w_macro = np.asarray(w_macro, dtype=np.float32)
        w_micro = np.asarray(w_micro, dtype=np.float32)

        self.classes_ = np.unique(np.concatenate([y_cal, y_src]))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}

        y_cal_idx = np.array([self.class_to_idx_[c] for c in y_cal], dtype=np.int64)
        y_src_idx = np.array([self.class_to_idx_[c] for c in y_src], dtype=np.int64)

        D = X_cal.shape[1]
        X_src_t = torch.tensor(X_src, dtype=torch.float32, device=self.device)
        y_src_t = torch.tensor(y_src_idx, dtype=torch.long, device=self.device)

        theta_init = np.clip(theta_init, 1e-8, None)
        theta_init = D * theta_init / (np.sum(theta_init) + 1e-12)
        theta_init_prob = theta_init / (np.sum(theta_init) + 1e-12)

        q0 = np.log(theta_init_prob + 1e-12)
        q = torch.nn.Parameter(torch.tensor(q0, dtype=torch.float32, device=self.device))

        w_macro_t = torch.tensor(np.clip(w_macro, 1e-6, None), dtype=torch.float32, device=self.device)
        w_micro_t = torch.tensor(np.clip(w_micro, 1e-6, None), dtype=torch.float32, device=self.device)
        theta_init_prob_t = torch.tensor(theta_init_prob, dtype=torch.float32, device=self.device)

        beta_init = float(np.clip(beta_init, 0.051, 0.999))
        a_init = float(np.clip(a_init / 1.5, 1e-4, 1 - 1e-4))
        b_init = float(np.clip(b_init / 1.5, 1e-4, 1 - 1e-4))

        s_beta = torch.nn.Parameter(torch.tensor(np.log((beta_init - 0.05) / (1.0 - beta_init + 1e-12)), dtype=torch.float32, device=self.device))
        s_a = torch.nn.Parameter(torch.tensor(np.log(a_init / (1 - a_init + 1e-12)), dtype=torch.float32, device=self.device))
        s_b = torch.nn.Parameter(torch.tensor(np.log(b_init / (1 - b_init + 1e-12)), dtype=torch.float32, device=self.device))

        opt_phi = torch.optim.Adam([s_beta, s_a, s_b], lr=self.lr_phi)
        opt_theta = torch.optim.Adam([q], lr=self.lr_theta)

        best_loss = np.inf
        best_state = None
        for rnd in range(self.alt_rounds):
            tr_idx, val_idx = self._split_calibration(
                y_cal, val_per_class=self.val_per_class, random_state=self.random_state + rnd
            )
            if len(tr_idx) == 0 or len(val_idx) == 0:
                tr_idx = np.arange(len(y_cal))
                val_idx = np.arange(len(y_cal))

            X_cal_tr = torch.tensor(X_cal[tr_idx], dtype=torch.float32, device=self.device)
            y_cal_tr = torch.tensor(y_cal_idx[tr_idx], dtype=torch.long, device=self.device)
            X_cal_val = torch.tensor(X_cal[val_idx], dtype=torch.float32, device=self.device)
            y_cal_val = torch.tensor(y_cal_idx[val_idx], dtype=torch.long, device=self.device)

            X_joint_tr = torch.cat([X_cal_tr, X_src_t], dim=0)
            y_joint_tr = torch.cat([y_cal_tr, y_src_t], dim=0)
            n_t_train = len(y_cal_tr)
            for _ in range(self.phi_steps):
                opt_phi.zero_grad()
                theta = self._theta_from_q(q.detach(), D)
                w_src, beta, a, b = self._weights_from_phi(w_macro_t, w_micro_t, s_beta, s_a, s_b)

                alpha = self._solve_alpha(X_joint_tr, y_joint_tr, theta, w_src, n_t_train)
                K_val = self._weighted_rbf(X_cal_val, X_joint_tr, theta)
                logits_val = K_val @ alpha

                loss_val = F.cross_entropy(logits_val, y_cal_val)
                loss_phi_reg = (beta - 0.6) ** 2 + (a - 0.5) ** 2 + (b - 0.5) ** 2
                loss = loss_val + self.reg_phi * loss_phi_reg
                loss.backward()
                opt_phi.step()

            for _ in range(self.theta_steps):
                opt_theta.zero_grad()
                w_src, beta, a, b = self._weights_from_phi(
                    w_macro_t, w_micro_t, s_beta.detach(), s_a.detach(), s_b.detach()
                )
                theta = self._theta_from_q(q, D)

                alpha = self._solve_alpha(X_joint_tr, y_joint_tr, theta, w_src, n_t_train)
                K_val = self._weighted_rbf(X_cal_val, X_joint_tr, theta)
                logits_val = K_val @ alpha

                loss_val = F.cross_entropy(logits_val, y_cal_val)
                theta_prob = theta / (torch.sum(theta) + 1e-12)
                loss_theta_reg = torch.sum(theta_prob * torch.log((theta_prob + 1e-12) / (theta_init_prob_t + 1e-12)))
                loss = loss_val + self.reg_theta * loss_theta_reg
                loss.backward()
                opt_theta.step()

            with torch.no_grad():
                w_src, beta, a, b = self._weights_from_phi(w_macro_t, w_micro_t, s_beta, s_a, s_b)
                theta = self._theta_from_q(q, D)
                alpha = self._solve_alpha(X_joint_tr, y_joint_tr, theta, w_src, n_t_train)
                K_val = self._weighted_rbf(X_cal_val, X_joint_tr, theta)
                logits_val = K_val @ alpha
                loss_val = F.cross_entropy(logits_val, y_cal_val)

                curr = float(loss_val.detach().cpu().item())
                if curr < best_loss:
                    best_loss = curr
                    best_state = {
                        "theta": theta.detach().cpu().numpy().copy(),
                        "w_src": w_src.detach().cpu().numpy().copy(),
                        "beta": float(beta.detach().cpu().item()),
                        "a": float(a.detach().cpu().item()),
                        "b": float(b.detach().cpu().item()),
                    }


        if best_state is None:
            best_state = {
                "theta": theta.detach().cpu().numpy().copy(),
                "w_src": w_src.detach().cpu().numpy().copy(),
                "beta": float(beta.detach().cpu().item()),
                "a": float(a.detach().cpu().item()),
                "b": float(b.detach().cpu().item()),
            }

        self.theta_ = best_state["theta"]
        self.w_src_ = best_state["w_src"]
        self.beta_ = best_state["beta"]
        self.a_ = best_state["a"]
        self.b_ = best_state["b"]

                                                  
        X_cal_full = torch.tensor(X_cal, dtype=torch.float32, device=self.device)
        y_cal_full = torch.tensor(y_cal_idx, dtype=torch.long, device=self.device)

        X_joint_full = torch.cat([X_cal_full, X_src_t], dim=0)
        y_joint_full = torch.cat([y_cal_full, y_src_t], dim=0)

        theta_t = torch.tensor(self.theta_, dtype=torch.float32, device=self.device)
        w_t = torch.tensor(self.w_src_, dtype=torch.float32, device=self.device)
        alpha = self._solve_alpha(X_joint_full, y_joint_full, theta_t, w_t, n_t_train=len(y_cal_idx))

        self.alpha_ = alpha.detach()
        self.X_train_ = X_joint_full.detach()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        theta_t = torch.tensor(self.theta_, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            K = self._weighted_rbf(X_t, self.X_train_, theta_t)
            logits = K @ self.alpha_
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

def compute_kernel_embedding(X, anchors, gamma):
    return rbf_kernel(X, anchors, gamma=gamma)

def compute_class_prototypes(Z_cal, y_cal):
    prototypes = {}
    for c in np.unique(y_cal):
        prototypes[c] = Z_cal[y_cal == c].mean(axis=0)
    return prototypes

def compute_micro_weights(Z_src, y_src, prototypes, sigma=1.0):
    w_micro = np.zeros(len(y_src), dtype=np.float64)
    for i in range(len(y_src)):
        c = y_src[i]
        mu_c = prototypes[c]
        dist_sq = np.sum((Z_src[i] - mu_c) ** 2)
        w_micro[i] = np.exp(-dist_sq / (2 * sigma ** 2))
    w_micro = w_micro / (np.max(w_micro) + 1e-12)
    return w_micro


@dataclass
class SourceCache:
    top_ids: list
    top_scores: list
    y_epochs: np.ndarray
    macro_weights: np.ndarray
    pca_filters: np.ndarray
    pca_mean: np.ndarray
    source_blocks: dict
    csp: object = None


def segment_into_epochs_with_ids(X, y, config, fs=256):
    epoch_len = int(float(config["win_len_s"]) * fs)
    step = int(epoch_len * (1.0 - float(config["overlap"])))
    starts = []
    current = 0
    while current + epoch_len <= X.shape[2]:
        starts.append(current)
        current += step
    epochs, labels, trial_ids = [], [], []
    for trial_idx in range(X.shape[0]):
        for start in starts:
            end = start + epoch_len
            if end <= X.shape[2]:
                epochs.append(X[trial_idx, :, start:end])
                labels.append(y[trial_idx])
                trial_ids.append(trial_idx)
    return np.asarray(epochs), np.asarray(labels), np.asarray(trial_ids, dtype=np.int64)


def compute_dte_from_features(source_features, source_labels, target_features):
    mean_total = np.mean(source_features, axis=0)
    discriminability = 0.0
    for class_id in np.unique(source_labels):
        indices = np.where(source_labels == class_id)[0]
        if len(indices):
            class_mean = np.mean(source_features[indices], axis=0)
            discriminability += len(indices) * np.sum(np.abs(class_mean - mean_total))
    discriminability /= len(source_features)
    difference = np.sum(np.abs(np.mean(source_features, axis=0) - np.mean(target_features, axis=0)))
    return float(discriminability / (difference + 1e-8))


def rank_source_subjects(X_all, y_all, subjects):
    covariance_features = extract_covariance_features(X_all)
    unique_subjects = np.unique(subjects)
    subject_features = {s: covariance_features[subjects == s] for s in unique_subjects}
    subject_labels = {s: y_all[subjects == s] for s in unique_subjects}
    rankings = {}
    for target in unique_subjects:
        scores = [
            (source, compute_dte_from_features(subject_features[source], subject_labels[source], subject_features[target]))
            for source in unique_subjects if source != target
        ]
        rankings[int(target)] = sorted(scores, key=lambda item: item[1], reverse=True)
    return rankings


def create_fixed_indices(y_all, subjects, seeds, calibration_shots=10):
    all_splits = {}
    for seed in seeds:
        setup_seed(seed)
        seed_splits = {}
        for subject in np.unique(subjects):
            labels = y_all[subjects == subject]
            calibration, test = [], []
            for class_id in np.unique(labels):
                locations = np.where(labels == class_id)[0]
                np.random.shuffle(locations)
                if len(locations) > calibration_shots:
                    calibration.extend(locations[:calibration_shots])
                    test.extend(locations[calibration_shots:])
                else:
                    test.extend(locations)
            seed_splits[int(subject)] = {
                "cal": np.asarray(calibration, dtype=np.int64),
                "test": np.asarray(test, dtype=np.int64),
            }
        all_splits[int(seed)] = seed_splits
    return all_splits


def build_source_cache(target_subject, k, X_all, y_all, subjects, rankings):
    selected = rankings[target_subject][:k]
    top_ids = [item[0] for item in selected]
    top_scores = [float(item[1]) for item in selected]
    max_score = top_scores[0]
    source_data, source_labels = [], []
    pca_weights, macro_weights = [], []
    for source_id, score in zip(top_ids, top_scores):
        mask = subjects == source_id
        source_data.append(X_all[mask])
        source_labels.append(y_all[mask])
        pca_weights.append(np.full(np.sum(mask), (score / max_score) ** 5.0))
        macro_weights.append(np.full(np.sum(mask), (score / max_score) ** 3.0))
    source_data = euclidean_alignment(np.concatenate(source_data))
    source_labels = np.concatenate(source_labels)
    pca_weights = np.concatenate(pca_weights)
    macro_weights = np.concatenate(macro_weights)
    source_epochs, epoch_labels, epoch_to_trial = segment_into_epochs_with_ids(
        source_data, source_labels, BCI2020_CONFIG
    )
    pca_epoch_weights = pca_weights[epoch_to_trial]
    macro_epoch_weights = macro_weights[epoch_to_trial]
    low, high = BCI2020_CONFIG["psd_band"]
    source_psd = extract_psd_features_raw(source_epochs, 256, low, high)
    pca_filters, pca_mean = train_weighted_pca(source_psd, pca_epoch_weights, 50)
    source_blocks = {
        "PSD": apply_pca(source_psd, pca_filters, pca_mean),
        "LogBP": compute_log_bp(source_epochs),
    }
    return SourceCache(top_ids, top_scores, epoch_labels, macro_epoch_weights,
                       pca_filters, pca_mean, source_blocks), source_epochs, pca_epoch_weights


def build_target_feature_blocks(X_target, y_target, split, source_cache):
    X_cal, y_cal = X_target[split["cal"]], y_target[split["cal"]]
    X_test, y_test = X_target[split["test"]], y_target[split["test"]]
    alignment = fit_ea_matrix(X_cal)
    X_cal = apply_ea_matrix(X_cal, alignment)
    X_test = apply_ea_matrix(X_test, alignment)
    cal_epochs, y_cal_epochs, _ = segment_into_epochs_with_ids(X_cal, y_cal, BCI2020_CONFIG)
    test_epochs, _, test_trial_ids = segment_into_epochs_with_ids(X_test, y_test, BCI2020_CONFIG)
    low, high = BCI2020_CONFIG["psd_band"]
    cal_psd = extract_psd_features_raw(cal_epochs, 256, low, high)
    test_psd = extract_psd_features_raw(test_epochs, 256, low, high)
    return {
        "source": source_cache.source_blocks,
        "calibration": {
            "PSD": apply_pca(cal_psd, source_cache.pca_filters, source_cache.pca_mean),
            "LogBP": compute_log_bp(cal_epochs),
        },
        "test": {
            "PSD": apply_pca(test_psd, source_cache.pca_filters, source_cache.pca_mean),
            "LogBP": compute_log_bp(test_epochs),
        },
        "y_source": source_cache.y_epochs,
        "y_calibration": y_cal_epochs,
        "y_test": y_test,
        "test_trial_ids": test_trial_ids,
        "calibration_epochs": cal_epochs,
        "test_epochs": test_epochs,
    }


def concatenate_features(blocks):
    return np.concatenate([blocks[name] for name in FEATURE_ORDER], axis=1)



SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data.npy"
LABEL_PATH = SCRIPT_DIR / "label.npy"
SUBJECT_PATH = SCRIPT_DIR / "data_subject.npy"
TARGET_SUBJECT = 0
K = 6
CSP_COMPONENTS = 6
SEEDS = (2023, 42, 100)

MODEL_SETTINGS = dict(
    gamma=None, lambda_reg=0.01, alt_rounds=8, phi_steps=10,
    theta_steps=10, lr_phi=0.02, lr_theta=0.02, val_per_class=2,
    w_floor=1e-4, reg_theta=1e-4, reg_phi=1e-2, verbose=False,
)


def fit_csp6(source_cache, source_epochs, source_weights):
    csp = WeightedCSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf")
    csp.fit(source_epochs, source_cache.y_epochs, sample_weight=source_weights)
    source_cache.csp = csp
    source_cache.source_blocks["CSP"] = csp.transform(source_epochs, return_signal=False)
    return source_cache


def prepare_features(feature_data, source_cache):
    calibration_blocks = dict(feature_data["calibration"])
    calibration_blocks["CSP"] = source_cache.csp.transform(
        feature_data["calibration_epochs"], return_signal=False
    )
    test_blocks = dict(feature_data["test"])
    test_blocks["CSP"] = source_cache.csp.transform(
        feature_data["test_epochs"], return_signal=False
    )
    scaler = StandardScaler()
    calibration = scaler.fit_transform(concatenate_features(calibration_blocks))
    source = scaler.transform(concatenate_features(source_cache.source_blocks))
    test = scaler.transform(concatenate_features(test_blocks))
    theta = calculate_hsfkc_theta(
        calibration, feature_data["y_calibration"], sample_weights=None, eta=0.5, tau=1.0
    )
    root = np.sqrt(theta)
    gamma = 1.0 / calibration.shape[1]
    z_cal = compute_kernel_embedding(calibration * root, calibration * root, gamma)
    z_src = compute_kernel_embedding(source * root, calibration * root, gamma)
    prototypes = compute_class_prototypes(z_cal, feature_data["y_calibration"])
    distances = []
    for class_id in np.unique(feature_data["y_calibration"]):
        values = z_cal[feature_data["y_calibration"] == class_id]
        if len(values) >= 2:
            center = values.mean(axis=0, keepdims=True)
            distances.extend(np.sqrt(np.sum((values - center) ** 2, axis=1)).tolist())
    sigma = 1.3 * (np.median(distances) + 1e-8) if distances else 1.0
    micro = compute_micro_weights(z_src, feature_data["y_source"], prototypes, sigma=sigma)
    return calibration, source, test, theta, micro


def trial_predictions(epoch_probabilities, classes, labels, trial_ids):
    probabilities = np.stack([
        epoch_probabilities[trial_ids == trial].mean(axis=0)
        for trial in range(len(labels))
    ])
    return classes[probabilities.argmax(axis=1)]


def load_data():
    X = np.load(DATA_PATH, allow_pickle=False)
    y = np.load(LABEL_PATH, allow_pickle=False)
    subjects = np.load(SUBJECT_PATH, allow_pickle=False)
    if not (X.shape[0] == len(y) == len(subjects)):
        raise ValueError("data.npy, label.npy and data_subject.npy are not aligned")
    if X.ndim != 3:
        raise ValueError("data.npy must be a three-dimensional EEG array")
    if X.shape[1] != 64 and X.shape[2] == 64:
        X = X.transpose(0, 2, 1)
    if X.shape[1] != 64:
        raise ValueError("BCI2020 data must have 64 EEG channels")

    if len(np.unique(y)) != 5:
        raise ValueError("BCI2020 must contain five classes")
    return X, y, subjects


def select_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = select_device()
    X_all, y_all, subjects = load_data()
    X_all = filter_data(X_all, 30, 90, fs=256, notch_freq=60.0)
    rankings = rank_source_subjects(X_all, y_all, subjects)
    ranking = rankings[TARGET_SUBJECT]
    selected_source = [int(item[0]) for item in ranking[:K]]

    source_cache, source_epochs, source_weights = build_source_cache(
        TARGET_SUBJECT, K, X_all, y_all, subjects, rankings
    )
    source_cache = fit_csp6(source_cache, source_epochs, source_weights)
    splits = create_fixed_indices(y_all, subjects, SEEDS, calibration_shots=10)
    target_mask = subjects == TARGET_SUBJECT
    target_X, target_y = X_all[target_mask], y_all[target_mask]
    accuracies = []
    for seed in SEEDS:
        setup_seed(seed)
        feature_data = build_target_feature_blocks(
            target_X, target_y, splits[seed][TARGET_SUBJECT], source_cache
        )
        calibration, source, test, theta, micro = prepare_features(feature_data, source_cache)
        model = JointThetaWeightKRR(**MODEL_SETTINGS, device=device, random_state=seed)
        model.fit(
            calibration, feature_data["y_calibration"], source, feature_data["y_source"],
            theta, source_cache.macro_weights, micro,
            beta_init=0.6, a_init=0.5, b_init=0.5,
        )
        probabilities = model.predict_proba(test)
        predictions = trial_predictions(
            probabilities, model.classes_, feature_data["y_test"], feature_data["test_trial_ids"]
        )
        accuracy = 100.0 * float(np.mean(predictions == feature_data["y_test"]))
        accuracies.append(accuracy)

    mean_accuracy = float(np.mean(accuracies))
    print(f"target={TARGET_SUBJECT}, mean accuracy={mean_accuracy:.2f}%")


if __name__ == "__main__":
    main()
