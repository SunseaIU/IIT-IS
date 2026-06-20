
import numpy as np
import sys
import os
import datetime
import random
import scipy.linalg
from scipy import signal
from sklearn.svm import SVC
from sklearn.decomposition import PCA 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh, fractional_matrix_power
from scipy.signal import butter, filtfilt
from sklearn.feature_selection import SelectKBest, f_classif 
from scipy.linalg import eigh, fractional_matrix_power
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.signal import butter, filtfilt, iirnotch  

DATASET_CONFIGS = {
    'korean': {
        'filter_band': [30, 90],      
        'psd_band':[30,90],
        'csp_components': 8,          
        'segment_type': 'sliding',    
        'win_len_s': 2.0,
        'overlap': 0.5
    },
    'asu': {
        'filter_band': [8, 70],       
        'psd_band':[8,70],
        'csp_components': 8,          
        'segment_type': 'fixed_asu',  
        'win_len_s': None,            
        'overlap': None
    }
}

DIMENSION_THRESHOLD = 150 
SDFR_ACTIVATION_THRESHOLD = 100

class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

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


def segment_into_epochs(X, y, config, weights=None, fs=256):
    n_trials = X.shape[0]
    X_epochs = []
    y_epochs = []
    w_epochs = []
    
    starts = []
    epoch_len = 0
    
    if config['segment_type'] == 'sliding':
        win_s = config['win_len_s']
        ovlap = config['overlap']
        epoch_len = int(win_s * fs)  
        step_pts = int(epoch_len * (1 - ovlap))  
        curr = 0
        while curr + epoch_len <= X.shape[2]:
            starts.append(curr)
            curr += step_pts
            
    elif config['segment_type'] == 'fixed_asu':
        epoch_len = 2 * fs 
        starts = [0, int(1.25 * fs), int(2.50 * fs)]
        
    for i in range(n_trials):
        for start in starts:
            end = start + epoch_len
            if end <= X.shape[2]:
                X_epochs.append(X[i, :, start:end])
                y_epochs.append(y[i])
                if weights is not None:
                    w_epochs.append(weights[i])
                
    if weights is not None:
        return np.array(X_epochs), np.array(y_epochs), np.array(w_epochs)
    else:
        return np.array(X_epochs), np.array(y_epochs)

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

def compute_dte(X_source, y_source, X_target):
    feat_s = extract_covariance_features(X_source)
    feat_t = extract_covariance_features(X_target)
    classes = np.unique(y_source)
    mean_total_s = np.mean(feat_s, axis=0)
    dis_score = 0
    for c in classes:
        idx = np.where(y_source == c)[0]
        if len(idx) == 0: continue
        mu_c = np.mean(feat_s[idx], axis=0)
        n_c = len(idx)
        dis_score += n_c * np.sum(np.abs(mu_c - mean_total_s))
    DIS = dis_score / len(feat_s)
    mean_s = np.mean(feat_s, axis=0)
    mean_t = np.mean(feat_t, axis=0)
    DIF = np.sum(np.abs(mean_s - mean_t))
    return DIS / (DIF + 1e-8)
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

def align_features_mean(X_train, X_test):
    mean_train = np.mean(X_train, axis=0)
    mean_test = np.mean(X_test, axis=0)
    X_train_centered = X_train - mean_train
    return X_train_centered, mean_train 
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
def compute_target_calibration_reliability(X_cal_feat, y_cal):
  
    X_cal_feat = np.asarray(X_cal_feat)
    y_cal = np.asarray(y_cal)

    classes = np.unique(y_cal)
    n_classes = len(classes)

    class_means = {}
    for c in classes:
        X_c = X_cal_feat[y_cal == c]
        if len(X_c) == 0:
            continue
        class_means[c] = np.mean(X_c, axis=0)

  
    intra_sum = 0.0
    intra_count = 0

    for c in classes:
        X_c = X_cal_feat[y_cal == c]
        if len(X_c) == 0:
            continue
        mu_c = class_means[c]

  
        dist_sq = np.sum((X_c - mu_c) ** 2, axis=1)
        intra_sum += np.sum(dist_sq)
        intra_count += len(X_c)

    intra_class_var = intra_sum / (intra_count + 1e-8)

    inter_sum = 0.0
    inter_count = 0

    class_list = list(class_means.keys())
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            mu_i = class_means[class_list[i]]
            mu_j = class_means[class_list[j]]

            dist_sq = np.sum((mu_i - mu_j) ** 2)
            inter_sum += dist_sq
            inter_count += 1

    inter_class_sep = inter_sum / (inter_count + 1e-8)

    reliability = inter_class_sep / (intra_class_var + 1e-8)

    return inter_class_sep, intra_class_var, reliability
def reliability_to_beta_src(reliability, beta_min=0.3, beta_max=0.9):
 
    r_norm = reliability / (reliability + 1.0)
    beta_src = beta_max - r_norm * (beta_max - beta_min)
    beta_src = np.clip(beta_src, beta_min, beta_max)
    return beta_src
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_micro_weights_margin(Z_src, y_src, prototypes, sigma_micro=1.0):
 
    y_src = np.asarray(y_src)
    classes = list(prototypes.keys())
    w_micro = np.zeros(len(Z_src), dtype=np.float64)

    for i, (z_i, y_i) in enumerate(zip(Z_src, y_src)):
   
        mu_same = prototypes[y_i]
        dist_same = np.sum((z_i - mu_same) ** 2)

        dist_other_list = []
        for c in classes:
            if c == y_i:
                continue
            mu_other = prototypes[c]
            dist_other = np.sum((z_i - mu_other) ** 2)
            dist_other_list.append(dist_other)

        if len(dist_other_list) == 0:
            w_micro[i] = 1.0
            continue

        dist_other = np.min(dist_other_list)
        margin = dist_other - dist_same
        w_micro[i] = sigmoid(margin / (sigma_micro + 1e-8))

    return w_micro

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DeepRBFNet(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=None, device='cuda'):
        self.gamma = gamma 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.classes_ = None
        self.centers = None 
    class RBFLayer(nn.Module):
        def __init__(self, centers, gamma):
            super().__init__()
            self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=False)
            self.gamma = gamma

        def forward(self, x):
            diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            return torch.exp(-self.gamma * dist_sq)

    class Network(nn.Module):
        def __init__(self, centers, n_classes, gamma):
            super().__init__()
            self.rbf = DeepRBFNet.RBFLayer(centers, gamma)
            self.linear = nn.Linear(len(centers), n_classes, bias=False) 
        
        def forward(self, x):
            features = self.rbf(x) 
            return self.linear(features)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        if self.gamma is None:
            self.gamma = 1.0 / n_features
        X_t = torch.FloatTensor(X).to(self.device)

        Y_onehot = torch.full((n_samples, n_classes), -1.0, device=self.device)
        for i, c in enumerate(self.classes_):
            Y_onehot[y == c, i] = 1.0
        rbf_layer = self.RBFLayer(X, self.gamma).to(self.device)
        K = rbf_layer(X_t) 
        self.K_ = K
        lambda_reg = 0.1 
        I = torch.eye(n_samples, device=self.device)
        weights = torch.linalg.solve(K + lambda_reg * I, Y_onehot)
        self.model = self.Network(X, n_classes, self.gamma).to(self.device)
        with torch.no_grad():
            self.model.linear.weight.data = weights.T 
            
        return self

    def predict_proba(self, X):
        self.model.eval() 
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]
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

class TangentSpace(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.reference_ = None

    def _compute_cov(self, X):
        n_epochs, n_ch, n_time = X.shape
        covs = np.zeros((n_epochs, n_ch, n_ch))
        for i in range(n_epochs):
            x_centered = X[i] - np.mean(X[i], axis=1, keepdims=True)
            covs[i] = np.dot(x_centered, x_centered.T) / (n_time - 1)
            covs[i] += 1e-6 * np.eye(n_ch)
        return covs

    def fit(self, X, y=None):
        covs = self._compute_cov(X)
        log_covs = []
        for c in covs:
            vals, vecs = scipy.linalg.eigh(c)
            vals = np.maximum(vals, 1e-8)
            log_covs.append(vecs @ np.diag(np.log(vals)) @ vecs.T)
        mean_log = np.mean(log_covs, axis=0)
        vals, vecs = scipy.linalg.eigh(mean_log)
        self.reference_ = vecs @ np.diag(np.exp(vals)) @ vecs.T
        return self

    def transform(self, X):
        covs = self._compute_cov(X)
        n_epochs, n_ch, _ = covs.shape
        vals, vecs = scipy.linalg.eigh(self.reference_)
        vals = np.maximum(vals, 1e-8)
        ref_inv_sqrt = vecs @ np.diag(vals**(-0.5)) @ vecs.T
        tangent_vecs = []
        triu_idx = np.triu_indices(n_ch)
        for i in range(n_epochs):
            s_mapped = ref_inv_sqrt @ covs[i] @ ref_inv_sqrt
            vals_s, vecs_s = scipy.linalg.eigh(s_mapped)
            vals_s = np.maximum(vals_s, 1e-8)
            log_s = vecs_s @ np.diag(np.log(vals_s)) @ vecs_s.T
            tangent_vecs.append(log_s[triu_idx])
        return np.array(tangent_vecs)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

class WeightedJointKRRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=None, lambda_reg=0.1):
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.classes_ = None
        self.X_train_ = None
        self.alpha_ = None

    def _compute_gamma(self, X):
        if self.gamma is not None:
            return self.gamma
        return 1.0 / X.shape[1]

    def fit(self, X_joint, y_joint, sample_weights):
        self.classes_ = np.unique(y_joint)
        self.X_train_ = X_joint.copy()

        gamma_val = self._compute_gamma(X_joint)

        # Kernel matrix
        K = rbf_kernel(X_joint, X_joint, gamma=gamma_val)
        N = len(y_joint)
        C = len(self.classes_)
        Y = -np.ones((N, C), dtype=np.float64)
        for idx, c in enumerate(self.classes_):
            Y[y_joint == c, idx] = 1.0

        # W^{-1}
        w = np.asarray(sample_weights, dtype=np.float64).copy()
        w = np.clip(w, 1e-6, None)
        W_inv = np.diag(1.0 / w)
        A = K + self.lambda_reg * W_inv
        self.alpha_ = np.linalg.solve(A, Y)
        self.gamma_used_ = gamma_val
        return self

    def decision_function(self, X):
        K_test = rbf_kernel(X, self.X_train_, gamma=self.gamma_used_)
        return K_test @ self.alpha_

    def predict_proba(self, X):
        logits = self.decision_function(X)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


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

                if self.verbose:
                    print(
                        f"        [Alt round {rnd+1:02d}] val={curr:.4f}, "
                        f"beta={float(beta.detach().cpu()):.3f}, a={float(a.detach().cpu()):.3f}, "
                        f"b={float(b.detach().cpu()):.3f}, theta_mean={float(theta.mean().detach().cpu()):.4f}"
                    )
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
from sklearn.metrics.pairwise import rbf_kernel

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
def run_experiment_final(data_path, label_path, sub_path):
    print(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    SEEDS = [2023, 42, 100]
    print(f"使用的 Seeds: {SEEDS}")

    print("加载数据...")
    X_all = np.load(data_path)
    y_all = np.load(label_path)
    sub_all = np.load(sub_path)
    
    if X_all.shape[2] < X_all.shape[1]:
        X_all = X_all.transpose(0, 2, 1)

    n_classes = len(np.unique(y_all))
    current_config = {}
    dataset_name = ""
    
    if n_classes == 5:
        dataset_name = 'korean'
        current_config = DATASET_CONFIGS['korean']
        weight_power = 5 
        calibration_shots = 10 
        macro_weight_power = 3
        target_notch = 60.0 
  
    elif n_classes == 2:
        dataset_name = 'asu'
        current_config = DATASET_CONFIGS['asu']
        weight_power = 5 
        macro_weight_power = 3
        calibration_shots = 10 
        target_notch = None 

    elif n_classes == 3:
        dataset_name = 'asu'
        current_config = DATASET_CONFIGS['asu']
        weight_power = 5 
        calibration_shots = 10 
        macro_weight_power = 3
        target_notch = None 

    else:
  
        dataset_name = 'korean'
        current_config = DATASET_CONFIGS['korean']
        weight_power = 5
        macro_weight_power = 3
        target_notch = None


    f_band = current_config['filter_band']
    try:
         X_all = filter_data(X_all, f_band[0], f_band[1], fs=256,notch_freq=target_notch)
    except:
        pass

    unique_subjects = np.unique(sub_all)
    n_subjects = len(unique_subjects)

    
    #  DTE 排名 ---
    print("\n" + "="*60)
    print("Step 1: 计算 DTE 排名")
    dte_ranks = {} 
    for target_sub in unique_subjects:
        t_idx = sub_all == target_sub
        X_target = X_all[t_idx]
        scores = []
        for source_sub in unique_subjects:
            if source_sub == target_sub: continue
            s_idx = sub_all == source_sub
            X_source = X_all[s_idx]
            y_source = y_all[s_idx]
            score = compute_dte(X_source, y_source, X_target)
            scores.append((source_sub, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        dte_ranks[target_sub] = scores

    # --- Step 2: 主循环 ---
    
    max_sources = n_subjects - 1 
    final_k_stats = {}
    for k in [7]:
    # for k in range(1, max_sources + 1):
        print(f"\n>>> 正在评估源域数量 K = {k} ...")
        seed_accuracies = [] 
        seed_medians = []   
        
        for seed in SEEDS:
            setup_seed(seed)
            
            # 数据划分
            fixed_indices = {}
            for sub in unique_subjects:
                sub_mask = sub_all == sub
                y_sub = y_all[sub_mask]
                cal_relative = []
                test_relative = []
                classes = np.unique(y_sub)
                for cls in classes:
                    cls_locs = np.where(y_sub == cls)[0]
                    np.random.shuffle(cls_locs) 
                    if len(cls_locs) > calibration_shots:
                        cal_relative.extend(cls_locs[:calibration_shots])
                        test_relative.extend(cls_locs[calibration_shots:])
                    else:
                        test_relative.extend(cls_locs)
                fixed_indices[sub] = {'cal': np.array(cal_relative), 'test': np.array(test_relative)}
            
            subject_accuracies_this_seed = []
            
            print(f"  [Seed {seed}] Source Details (Top-{k}):")

            for test_sub in unique_subjects:
                top_k_info = dte_ranks[test_sub][:k]
                top_k_ids = [info[0] for info in top_k_info]
                top_k_scores = [info[1] for info in top_k_info]
                max_score = top_k_scores[0]
     
                X_source_list = []
                y_source_list = []
                w_source_list = []
                w_source_list_pca = []
                w_source_list_macro = []

                for s_id, s_score in zip(top_k_ids, top_k_scores):
                    s_mask = sub_all == s_id
                    X_s = X_all[s_mask]
                    y_s = y_all[s_mask]
                    X_source_list.append(X_s)
                    y_source_list.append(y_s)
                    w_val = (s_score / max_score) ** weight_power
                    w_source_list.append(np.full(len(y_s), w_val))

                    w_val_pca = (s_score / max_score) ** 5.0 
                    w_source_list_pca.append(np.full(len(y_s), w_val_pca))

                  
                    w_val_macro = (s_score / max_score) ** macro_weight_power
                    w_source_list_macro.append(np.full(len(y_s), w_val_macro))

                X_source_raw = np.concatenate(X_source_list, axis=0)
                y_source = np.concatenate(y_source_list, axis=0)
                weights_csp_source = np.concatenate(w_source_list, axis=0)
                weights_source_pca = np.concatenate(w_source_list_pca, axis=0)
                weights_macro_source = np.concatenate(w_source_list_macro, axis=0)
         
                target_mask = sub_all == test_sub
                X_target_raw_all = X_all[target_mask]
                y_target_all = y_all[target_mask]

       
                X_source_aligned = euclidean_alignment(X_source_raw)
       

                idxs = fixed_indices[test_sub]
                if len(idxs['cal']) == 0: continue

                X_cal = X_target_raw_all[idxs['cal']]
                y_cal = y_target_all[idxs['cal']]
                X_test = X_target_raw_all[idxs['test']]
                y_test = y_target_all[idxs['test']]

                P_target = fit_ea_matrix(X_cal)

         
                X_cal = apply_ea_matrix(X_cal, P_target)
                X_test = apply_ea_matrix(X_test, P_target)

           
                X_src_eps, y_src_eps, w_src_eps = segment_into_epochs(
                    X_source_aligned, y_source, config=current_config, weights=weights_csp_source, fs=256
                )
             
               
                _, _, w_src_eps_pca = segment_into_epochs(
                    X_source_aligned, y_source, config=current_config, 
                    weights=weights_source_pca, 
                    fs=256
                )
                _, _, w_src_eps_macro = segment_into_epochs(
                    X_source_aligned, y_source, config=current_config,
                    weights=weights_macro_source,
                    fs=256
                )
                X_cal_eps, y_cal_eps = segment_into_epochs(
                    X_cal, y_cal, config=current_config, fs=256
                )
                p_min, p_max = current_config['psd_band']
                X_src_psd_raw = extract_psd_features_raw(
                    X_src_eps, sfreq=256, fmin=p_min, fmax=p_max
                )
                n_psd_components = 50 
                pca_filters, pca_mean = train_weighted_pca(
                    X_src_psd_raw, 
                    weights=w_src_eps_pca, 
                    n_components=n_psd_components
                )

                n_csp_comp = current_config['csp_components']

                n_csp_comp = current_config['csp_components']
                n_channels_out = n_classes * n_csp_comp 
                ts_feature_dim = int(n_channels_out * (n_channels_out + 1) / 2)
                
                use_tangent = False
                
                if ts_feature_dim <= DIMENSION_THRESHOLD:
                    use_tangent = True
                else:
                    use_tangent = False

                try:
                    # 1. 训练 CSP
                    csp = WeightedCSP(n_components=n_csp_comp, reg='ledoit_wolf')
                    csp.fit(X_src_eps, y_src_eps, sample_weight=w_src_eps)
                    if use_tangent:
                        S_src = csp.transform(X_src_eps, return_signal=True)
                        S_cal = csp.transform(X_cal_eps, return_signal=True)
                        ts = TangentSpace()
                        ts.fit(S_cal) 
                        X_src_feat_csp = ts.transform(S_src) 
                        X_cal_feat_csp = ts.transform(S_cal)
                    else:
                        X_src_feat_csp = csp.transform(X_src_eps, return_signal=False)
                        X_cal_feat_csp = csp.transform(X_cal_eps, return_signal=False)
                    X_cal_feat_bp = compute_log_bp(X_cal_eps)
                    X_src_feat_bp = compute_log_bp(X_src_eps)
                    X_cal_psd_raw = extract_psd_features_raw(
                        X_cal_eps, 
                        sfreq=256, 
                        fmin=p_min, 
                        fmax=p_max
                    )
                    X_src_psd_raw = extract_psd_features_raw(
                        X_src_eps, sfreq=256, fmin=p_min, fmax=p_max
                    )
                    X_src_feat_psd = apply_pca(X_src_psd_raw, pca_filters, pca_mean)
                    X_cal_feat_psd = apply_pca(X_cal_psd_raw, pca_filters, pca_mean)
                    dim_csp = X_cal_feat_csp.shape[1]
                    dim_psd = X_cal_feat_psd.shape[1]
                    dim_bp  = X_cal_feat_bp.shape[1]
                    feature_slices = {
                        'CSP': slice(0, dim_csp),
                        'PSD': slice(dim_csp, dim_csp + dim_psd),
                        'LogBP': slice(dim_csp + dim_psd, dim_csp + dim_psd + dim_bp)
                    }
                    X_cal_combined = np.concatenate([X_cal_feat_csp,X_cal_feat_psd, X_cal_feat_bp], axis=1)
                    X_src_combined = np.concatenate([X_src_feat_csp, X_src_feat_psd, X_src_feat_bp], axis=1)
                    scaler = StandardScaler()
                    X_cal_std = scaler.fit_transform(X_cal_combined)
                    X_src_std = scaler.transform(X_src_combined)

                    total_dims = X_cal_std.shape[1]
                    selector = None

                    theta_init = calculate_hsfkc_theta(
                        X_cal_std,
                        y_cal_eps,
                        sample_weights=None,
                        eta=0.5,
                        tau=1.0
                    )
                    scale_factor_init = np.sqrt(theta_init)
                    X_cal_init = X_cal_std * scale_factor_init
                    X_src_init = X_src_std * scale_factor_init

                    w_macro = w_src_eps_macro.copy()
                    gamma_embed = 1.0 / X_cal_init.shape[1]

                    Z_cal = compute_kernel_embedding(X_cal_init, X_cal_init, gamma_embed)
                    Z_src = compute_kernel_embedding(X_src_init, X_cal_init, gamma_embed)

                    prototypes = compute_class_prototypes(Z_cal, y_cal_eps)

                    pair_dists = []
                    for c in np.unique(y_cal_eps):
                        Zc = Z_cal[y_cal_eps == c]
                        if len(Zc) >= 2:
                            center = Zc.mean(axis=0, keepdims=True)
                            d = np.sqrt(np.sum((Zc - center) ** 2, axis=1))
                            pair_dists.extend(d.tolist())

                    sigma_scale = 1.3
                    sigma_micro = sigma_scale * (np.median(pair_dists) + 1e-8) if len(pair_dists) > 0 else 1.0

                    w_micro = compute_micro_weights(Z_src, y_src_eps, prototypes, sigma=sigma_micro)

                    beta_src = 0.6
                    a = 0.5
                    b = 0.5

                    w_init = beta_src * (w_macro ** a) * (w_micro ** b)
                    w_init = np.clip(w_init, 1e-4, 1.0)
                    clf = JointThetaWeightKRR(           
                        gamma=None,           
                        lambda_reg=0.01,           
                        alt_rounds=8,           
                        phi_steps=10,           
                        theta_steps=10,         
                        lr_phi=0.02,         
                        lr_theta=0.02,          
                        val_per_class=2,           
                        w_floor=1e-4,           
                        reg_theta=1e-4,       
                        reg_phi=1e-2,           
                        device="cpu",           
                        random_state=seed,           
                        verbose=False
                    )  

                    clf.fit(
                        X_cal=X_cal_std,
                        y_cal=y_cal_eps,
                        X_src=X_src_std,
                        y_src=y_src_eps,
                        theta_init=theta_init,
                        w_macro=w_macro,
                        w_micro=w_micro,       
                        beta_init=beta_src,
                        a_init=a,
                        b_init=b
                    )
                    print(
                        f"beta={clf.beta_:.4f}, a={clf.a_:.4f}, b={clf.b_:.4f}, "
                        f"theta_mean={clf.theta_.mean():.4f}, "
                        f"wsrc_mean={clf.w_src_.mean():.4f}, "
                        f"wsrc_min={clf.w_src_.min():.4f}, "
                        f"wsrc_max={clf.w_src_.max():.4f}"
                    )
                    trial_correct = 0
                    n_test_trials = len(X_test)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(n_test_trials):
                        single_trial = X_test[i:i+1] 
                        x_te_eps, _ = segment_into_epochs(single_trial, np.array([0]), config=current_config, fs=256)
                        if len(x_te_eps) == 0: continue
                        if use_tangent:
                            s_te = csp.transform(x_te_eps, return_signal=True)
                            f_te_csp = ts.transform(s_te)
                        else:
                            f_te_csp = csp.transform(x_te_eps, return_signal=False)
                        f_te_bp = compute_log_bp(x_te_eps)
                        f_te_psd_raw = extract_psd_features_raw(
                            x_te_eps, 
                            sfreq=256, 
                            fmin=p_min, 
                            fmax=p_max
                        )
                        f_te_psd = apply_pca(f_te_psd_raw, pca_filters, pca_mean)
                        f_te_combined = np.concatenate([f_te_csp,f_te_psd, f_te_bp], axis=1)
                        f_te_std = scaler.transform(f_te_combined)
                        f_te_selected = f_te_std # 不做 transform
                        epoch_probs = clf.predict_proba(f_te_selected)
                        trial_prob = np.mean(epoch_probs, axis=0)
                        final_pred_idx = np.argmax(trial_prob)
                        final_pred = clf.classes_[final_pred_idx]

                        # [新增] 2. 收集结果
                        y_true_all.append(y_test[i])
                        y_pred_all.append(final_pred)
                        
                        if final_pred == y_test[i]:
                            trial_correct += 1
                    
                    score = trial_correct / n_test_trials
                    subject_accuracies_this_seed.append(score)
                  
                    
                    source_str = ", ".join([f"S{int(sid)}({sc:.3f})" for sid, sc in zip(top_k_ids, top_k_scores)])
                    print(f"gamma_embed = {gamma_embed:.6f}")
                    print(f"sigma_micro = {sigma_micro:.6f}")
                    print(f"w_macro: min={w_macro.min():.4f}, mean={w_macro.mean():.4f}, max={w_macro.max():.4f}")
                    print(f"w_micro: min={w_micro.min():.4f}, mean={w_micro.mean():.4f}, max={w_micro.max():.4f}")
                    print(f"w_micro p10/p50/p90 = "
                        f"{np.percentile(w_micro,10):.4f}/"
                        f"{np.percentile(w_micro,50):.4f}/"
                        f"{np.percentile(w_micro,90):.4f}")
                    print(f"    Target {int(test_sub):<2} | Sources: [{source_str}] | Acc: {score:.4f} | TS: {use_tangent} ")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    subject_accuracies_this_seed.append(0.0)
                    print(f"    Target {int(test_sub):<2} | Error | Acc: 0.0000")

            mean_acc_seed = np.mean(subject_accuracies_this_seed)
            median_acc_seed = np.median(subject_accuracies_this_seed) 
            seed_accuracies.append(mean_acc_seed)
            seed_medians.append(median_acc_seed) 
            print(f"  -> Seed {seed} Summary: Avg Acc = {mean_acc_seed:.4f}\n")

        k_mean = np.mean(seed_accuracies)
        k_std = np.std(seed_accuracies)
        k_median_avg = np.mean(seed_medians) 
        final_k_stats[k] = {'mean': k_mean, 'std': k_std, 'median': k_median_avg, 'raw': seed_accuracies}
        print(f"  [Result] K={k} | Mean: {k_mean:.4f} | Median: {k_median_avg:.4f} | Std: {k_std:.4f}")

    print("\n" + "="*60)
    best_k = max(final_k_stats, key=lambda x: final_k_stats[x]['mean'])
    print(f"最佳 K 值: {best_k}, 平均准确率: {final_k_stats[best_k]['mean']:.4f}")

d_path = r'D:\speech_dataset\korean\product_npy\data.npy'
l_path = r'D:\speech_dataset\korean\product_npy\label.npy'
s_path = r'D:\speech_dataset\korean\product_npy\data_subject.npy'

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_Adaptive_Final_{timestamp}.txt"
    sys.stdout = Logger(log_filename)
    
    print(f"*** 日志保存至: {log_filename} ***")
    
    run_experiment_final(d_path, l_path, s_path)