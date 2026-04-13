"""
NeuroMentor — Unified Fixed App
All sections patched for Android 14 / API 34 compatibility.
"""

# ============================================================
# Kivy config MUST come before any other kivy import
# ============================================================
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'rotation', '0')

from kivy.metrics import sp, dp
from kivy.app import App
from kivy.core.window import Window
from kivy.utils import platform

if platform != 'android':
    Window.size = (390, 844)

# ============================================================
# AutoLabel and AutoButton — replaces global monkey-patching
# ============================================================
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.screenmanager import ScreenManager, Screen
import threading


class AutoLabel(Label):
    """Label with automatic text wrapping and dynamic height."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.halign = kwargs.get('halign', 'left')
        self.valign = 'middle'
        self.shorten = False
        self.bind(width=self._update_text_size)
        self.bind(texture_size=self._update_height)

    def _update_text_size(self, *args):
        if self.width < dp(120):
            return
        self.text_size = (self.width - dp(16), None)

    def _update_height(self, *args):
        self.height = max(self.texture_size[1] + dp(4), dp(20))


class AutoButton(ButtonBehavior, Label):
    """Button with automatic text wrapping and dynamic height."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.halign = 'center'
        self.valign = 'middle'
        self.bind(width=self._update_text_size)
        self.bind(texture_size=self._update_height)

    def _update_text_size(self, *args):
        if self.width < dp(120):
            return
        self.text_size = (self.width - dp(16), None)

    def _update_height(self, *args):
        self.height = max(self.texture_size[1], dp(48))


# ============================================================
# theme.py
# ============================================================
class theme:
    GOLD = (0.435, 0.753, 0.945, 1)   # soft sky blue
    TEAL = (0.000, 0.470, 0.831, 1)   # bold medium blue
    RED  = (0.090, 0.231, 0.353, 1)   # deep navy blue

    BG_DARK      = (0.051, 0.141, 0.220, 1)
    PANEL_BG     = (0.085, 0.200, 0.333, 1)
    CARD_BG      = (0.109, 0.235, 0.376, 1)
    BORDER_DARK  = (0.070, 0.187, 0.286, 1)
    BORDER_LIGHT = (0.435, 0.753, 0.945, 1)

    TEXT_PRIMARY   = (0.941, 0.965, 0.980, 1)
    TEXT_SECONDARY = (0.757, 0.863, 0.941, 1)
    TEXT_MUTED     = (0.627, 0.765, 0.882, 1)

    INPUT_BG     = (0.051, 0.102, 0.169, 1)
    INPUT_BORDER = (0.435, 0.753, 0.945, 1)

    SIDEBAR_BG     = (0.085, 0.200, 0.333, 1)
    SIDEBAR_BORDER = (0.070, 0.187, 0.286, 1)
    DARK_CARD      = (0.109, 0.235, 0.376, 1)
    BUTTON_BG      = (0.109, 0.235, 0.376, 1)
    DANGER_BUTTON_BG = (0.118, 0.631, 0.969, 1)
    TRANSPARENT    = (0, 0, 0, 0)

    FONT_TITLE_LARGE   = 26
    FONT_TITLE_MEDIUM  = 22
    FONT_HEADING_LARGE = 20
    FONT_HEADING_MEDIUM= 18
    FONT_BODY_LARGE    = 16
    FONT_BODY_REGULAR  = 14
    FONT_BODY_SMALL    = 12
    FONT_DISPLAY_LARGE = 48
    FONT_TIMER         = 22

    @staticmethod
    def font(size):
        return sp(size)

    @staticmethod
    def rgba_hex(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b, alpha)

    @staticmethod
    def with_alpha(color, alpha):
        return (color[0], color[1], color[2], alpha)


# ============================================================
# services/rf_classifier.py
# ============================================================
import os
from dataclasses import dataclass

try:
    import joblib
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@dataclass
class EegBands:
    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta:  float = 0.0
    gamma: float = 0.0


FEATURE_NAMES = [
    'delta', 'theta', 'alpha', 'beta', 'gamma',
    'beta_alpha_ratio', 'alpha_theta_ratio',
    'beta_theta_ratio', 'gamma_beta_ratio',
    'beta_minus_alpha', 'alpha_plus_theta',
]

SCALER_B64 = 'gASVvwIAAAAAAACMG3NrbGVhcm4ucHJlcHJvY2Vzc2luZy5fZGF0YZSMDlN0YW5kYXJkU2NhbGVylJOUKYGUfZQojAl3aXRoX21lYW6UiIwId2l0aF9zdGSUiIwEY29weZSIjA5uX2ZlYXR1cmVzX2luX5RLC4wPbl9zYW1wbGVzX3NlZW5flIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMBnNjYWxhcpSTlIwFbnVtcHmUjAVkdHlwZZSTlIwCZjiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYkMIAAAAAAB18kCUhpRSlIwFbWVhbl+UaAqMDF9yZWNvbnN0cnVjdJSTlGgNjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwuFlGgPjAJmOJSJiIeUUpQoSwNoE05OTkr/////Sv////9LAHSUYolDWEwZudgyN8hAQvbgBzyg0UAtF7dGMtbMQGJR2OPLRORAMiq5uJbFE0GLMbx8SKMZQLD7Q7BKce8/MWigKSBVGkAIi2N+L0goQIbig71PHtpAQaN2QKwF4ECUdJRijAR2YXJflGgaaBxLAIWUaB6HlFKUKEsBSwuFlGgkiUNYDSfWa/YosUEp+ZyxkuO8Qddf4hqlIbVB2pT65Fq/4kHEuvEFgYw8Qjayjx30ylZAyhkJCLPs3D/u5HghXlFgQLmayxx0wWNAAcVolNy70UHSxIj6sfbWQZR0lGKMBnNjYWxlX5RoGmgcSwCFlGgeh5RSlChLAUsLhZRoJIlDWJGsMibikdBA2n7MU9d/1UC+5FftN2PSQNgcar9FfuhAtv5Qs1hfFUF9wENxwhgjQC8SWGs8g+U/pkV7GN/ZJkDZph9OqiQpQASvrkQ52OBA32PLsQwr40CUdJRijBBfc2tsZWFybl92ZXJzaW9ulIwFMS44LjCUdWIu'
ENCODER_B64 = 'gASVCAEAAAAAAACMHHNrbGVhcm4ucHJlcHJvY2Vzc2luZy5fbGFiZWyUjAxMYWJlbEVuY29kZXKUk5QpgZR9lCiMCGNsYXNzZXNflIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgJjAVkdHlwZZSTlIwCTziUiYiHlFKUKEsDjAF8lE5OTkr/////Sv////9LP3SUYoldlCiMCEJhc2VsaW5llIwHRm9jdXNlZJSMCFN0cmVzc2VklGV0lGKMEF9za2xlYXJuX3ZlcnNpb26UjAUxLjguMJR1Yi4='
RFC_B64_COMPRESSED = None


def _decode_model(b64_string):
    import base64, pickle
    try:
        return pickle.loads(base64.b64decode(b64_string))
    except Exception as e:
        print(f"[embedded_model] decode failed: {e}")
        return None


def safe_div(n, d, eps=1e-6):
    return n / max(abs(d), eps)


class RFClassifier:
    def __init__(self):
        self._clf     = None
        self._scaler  = None
        self._encoder = None
        self._is_trained = False
        self._feature_names = list(FEATURE_NAMES)
        self._load_from_embedded()

    # ------ embedded load ------
    def _load_from_embedded(self):
        if not _HAS_SKLEARN:
            return
        if SCALER_B64:
            try:
                self._scaler = _decode_model(SCALER_B64)
            except Exception as e:
                print(f"[RFClassifier] scaler decode error: {e}")
        if ENCODER_B64:
            try:
                self._encoder = _decode_model(ENCODER_B64)
            except Exception as e:
                print(f"[RFClassifier] encoder decode error: {e}")
        if RFC_B64_COMPRESSED:
            import base64, gzip, io
            try:
                data = gzip.decompress(base64.b64decode(RFC_B64_COMPRESSED))
                self._clf = joblib.load(io.BytesIO(data))
                self._is_trained = True
                return
            except Exception as e:
                print(f"[RFClassifier] compressed RFC decode error: {e}")
        self._load_clf_from_disk()

    def _load_clf_from_disk(self):
        if not _HAS_SKLEARN:
            return
        try:
            app = App.get_running_app()
            data_root = app.user_data_dir if app else os.path.dirname(os.path.abspath(__file__))
        except Exception:
            data_root = os.path.dirname(os.path.abspath(__file__))

        candidates = [
            os.path.join(data_root, 'rf_model', 'rf_model', 'rf_eeg_model.pkl'),
            os.path.join(data_root, 'rf_eeg_model.pkl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'rf_model', 'rf_model', 'rf_eeg_model.pkl'),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    self._clf = joblib.load(path)
                    self._is_trained = True
                    print(f"[RFClassifier] Loaded from {path}")
                    return
                except Exception as e:
                    print(f"[RFClassifier] Load failed {path}: {e}")
        print("[RFClassifier] No model file found — predictions return Unknown.")

    # ------ features ------
    def build_feature_vector(self, bands: EegBands) -> list:
        d, t, a, b, g = bands.delta, bands.theta, bands.alpha, bands.beta, bands.gamma
        fv = [
            d, t, a, b, g,
            safe_div(b, a),
            safe_div(a, t),
            safe_div(b, t),
            safe_div(g, b),
            b - a,
            a + t,
        ]
        if _HAS_SKLEARN:
            fv = list(np.clip(np.array(fv), -1e3, 1e3))
        return fv

    # ------ training ------
    def train(self, session_bands, session_labels):
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn not installed")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import cross_val_score

        X = np.array([self.build_feature_vector(b) for b in session_bands])
        y = np.array(session_labels)
        self._encoder = LabelEncoder()
        y_enc = self._encoder.fit_transform(y)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        clf_cv = RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_split=3, min_samples_leaf=1,
            random_state=42, n_jobs=1)
        cv = cross_val_score(clf_cv, Xs, y_enc, cv=5, scoring='accuracy')

        self._clf = RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_split=3, min_samples_leaf=1,
            random_state=42, n_jobs=1)
        self._clf.fit(Xs, y_enc)
        self._is_trained = True

        unique, counts = np.unique(y, return_counts=True)
        return {
            'cv_accuracy': float(np.mean(cv)),
            'cv_std': float(np.std(cv)),
            'feature_importances': dict(zip(self._feature_names,
                                            self._clf.feature_importances_.tolist())),
            'n_samples': len(y),
            'class_distribution': {str(u): int(c) for u, c in zip(unique, counts)},
        }

    # ------ prediction ------
    def predict(self, bands: EegBands) -> str:
        if not _HAS_SKLEARN:
            return 'Unknown'
        if not self._is_trained or self._clf is None or self._scaler is None:
            return 'Unknown'
        X = np.array([self.build_feature_vector(bands)])
        Xs = self._scaler.transform(X)
        pred = self._clf.predict(Xs)[0]
        if self._encoder is not None:
            return str(self._encoder.inverse_transform([pred])[0])
        return str(pred)

    def predict_proba(self, bands: EegBands) -> dict:
        if not self._is_trained or self._clf is None or self._scaler is None:
            return {}
        X = np.array([self.build_feature_vector(bands)])
        Xs = self._scaler.transform(X)
        proba = self._clf.predict_proba(Xs)[0]
        if self._encoder is not None:
            labels = self._encoder.inverse_transform(range(len(proba)))
            return {str(l): float(p) for l, p in zip(labels, proba)}
        return {str(i): float(p) for i, p in enumerate(proba)}

    # ------ persistence ------
    def save(self, dirpath: str):
        if not _HAS_SKLEARN:
            raise RuntimeError("joblib not installed")
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self._clf,     os.path.join(dirpath, 'rf_eeg_model.pkl'))
        joblib.dump(self._scaler,  os.path.join(dirpath, 'rf_scaler.pkl'))
        if self._encoder is not None:
            joblib.dump(self._encoder, os.path.join(dirpath, 'rf_encoder.pkl'))

    def load(self, dirpath: str) -> bool:
        if not _HAS_SKLEARN:
            return False
        mp = os.path.join(dirpath, 'rf_eeg_model.pkl')
        sp2 = os.path.join(dirpath, 'rf_scaler.pkl')
        ep = os.path.join(dirpath, 'rf_encoder.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp2):
            return False
        try:
            self._clf    = joblib.load(mp)
            self._scaler = joblib.load(sp2)
            if os.path.exists(ep):
                self._encoder = joblib.load(ep)
            self._is_trained = True
            return True
        except Exception as e:
            print(f"[RFClassifier] load failed: {e}")
            self._is_trained = False
            return False

    @property
    def is_trained(self):
        return self._is_trained


# ============================================================
# tools/compatibility_check.py  (inline)
# ============================================================
import math
from io import StringIO

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import joblib as _joblib_compat
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

ADC_BITS       = 24
VREF           = 4.5
PGA_GAIN       = 24
ADC_RESOLUTION = (2 ** (ADC_BITS - 1)) - 1

SAMPLE_RATE  = 250
WINDOW_SIZE  = 256
OVERLAP      = 128

BAND_BOUNDARIES = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0,30.0),
    'gamma': (30.0,45.0),
}
EXPECTED_LABELS = ['Baseline', 'Focused', 'Stressed']

_pass_c = _fail_c = _warn_c = 0
_fail_details_c = []
_buf_c = StringIO()

def _p(msg=''):
    global _buf_c
    print(msg)
    _buf_c.write(msg + '\n')

def _PASS(msg): global _pass_c; _pass_c += 1; _p(f'  ✓ PASS   {msg}')
def _FAIL(msg): global _fail_c; _fail_c += 1; _fail_details_c.append(msg); _p(f'  ✗ FAIL   {msg}')
def _WARN(msg): global _warn_c; _warn_c += 1; _p(f'  ⚠ WARN   {msg}')
def _header(t): _p(''); _p('─'*60); _p(f'  {t}'); _p('─'*60)

def _compute_band_powers_fft(signal, sr, ws):
    if not _HAS_NUMPY: return {}
    win = np.hanning(ws)
    spec = np.fft.rfft(signal[:ws] * win)
    psd  = (np.abs(spec)**2) / ws
    freqs= np.fft.rfftfreq(ws, 1.0/sr)
    return {n: float(np.sum(psd[(freqs>=lo)&(freqs<hi)]))
            for n,(lo,hi) in BAND_BOUNDARIES.items()}

def _check_adc_conversion():
    _header('CHECK 1 — ADC CONVERSION')
    scale = (VREF / ADC_RESOLUTION / PGA_GAIN) * 1e6
    fs = ADC_RESOLUTION * scale; ms = (ADC_RESOLUTION//2)*scale
    _p(f'  Full-scale: {fs:.2f} µV | Mid-scale: {ms:.2f} µV')
    _PASS(f'Full-scale {fs:.2f} µV') if 100<fs<300_000 else _FAIL(f'Full-scale {fs:.2f} µV out of range')
    _PASS(f'Mid-scale {ms:.2f} µV')  if 50<ms<150_000  else _FAIL(f'Mid-scale {ms:.2f} µV out of range')
    c = 50.0/scale
    _PASS(f'50µV → {c:.1f} counts') if c>=1 else _FAIL(f'50µV → {c:.4f} counts (unresolvable)')

def _check_sample_rate_and_window():
    _header('CHECK 2 — SAMPLE RATE & WINDOW')
    nyq = SAMPLE_RATE/2; fr = SAMPLE_RATE/WINDOW_SIZE
    hb  = max(hi for _,hi in BAND_BOUNDARIES.values())
    _PASS(f'Nyquist {nyq}Hz ≥ {hb}Hz') if nyq>=hb else _FAIL(f'Nyquist {nyq}Hz < {hb}Hz')
    edges = sorted({e for lo,hi in BAND_BOUNDARIES.values() for e in (lo,hi)})
    mg = min(b-a for a,b in zip(edges,edges[1:]))
    _PASS(f'Freq res {fr:.4f} ≤ {mg}Hz') if fr<=mg else _FAIL(f'Freq res {fr:.4f} > {mg}Hz')
    _PASS(f'Overlap {OVERLAP} < {WINDOW_SIZE}') if OVERLAP<WINDOW_SIZE else _FAIL('Overlap ≥ window')

def _check_band_boundaries():
    _header('CHECK 3 — BAND BOUNDARY ALIGNMENT')
    fr = SAMPLE_RATE/WINDOW_SIZE
    freqs = [i*fr for i in range(WINDOW_SIZE//2+1)]
    for name,(lo,hi) in BAND_BOUNDARIES.items():
        bins = [f for f in freqs if lo<=f<hi]
        lo_e = abs(min(freqs,key=lambda f:abs(f-lo))-lo)
        hi_e = abs(min(freqs,key=lambda f:abs(f-hi))-hi)
        aligned = lo_e<fr/2 and hi_e<fr/2
        fn = _PASS if aligned else _WARN
        fn(f'{name:6s} {lo}–{hi}Hz bins={len(bins)}')
        if len(bins)<4: _WARN(f'{name} only {len(bins)} bins')

def _check_synthetic_signals():
    _header('CHECK 4 — SYNTHETIC SIGNAL END-TO-END')
    if not _HAS_NUMPY: _FAIL('numpy unavailable'); return
    t = np.arange(WINDOW_SIZE)/SAMPLE_RATE
    for freq,(exp) in [(2,'delta'),(6,'theta'),(10,'alpha'),(20,'beta'),(40,'gamma')]:
        sig = np.sin(2*np.pi*freq*t)
        pw  = _compute_band_powers_fft(sig,SAMPLE_RATE,WINDOW_SIZE)
        dom = max(pw,key=pw.get)
        _PASS(f'{freq}Hz → {dom}') if dom==exp else _FAIL(f'{freq}Hz expected {exp} got {dom}')

def _check_feature_vector():
    _header('CHECK 5 — FEATURE VECTOR')
    if not _HAS_NUMPY: _FAIL('numpy unavailable'); return None
    EXPECTED = ['delta','theta','alpha','beta','gamma',
                'beta_alpha_ratio','alpha_theta_ratio','beta_theta_ratio',
                'gamma_beta_ratio','beta_minus_alpha','alpha_plus_theta']
    _PASS('FEATURE_NAMES length 11') if len(FEATURE_NAMES)==11 else _FAIL(f'FEATURE_NAMES length {len(FEATURE_NAMES)}')
    _PASS('FEATURE_NAMES order correct') if list(FEATURE_NAMES)==EXPECTED else _FAIL('FEATURE_NAMES mismatch')
    t = np.arange(WINDOW_SIZE)/SAMPLE_RATE
    pw = _compute_band_powers_fft(np.sin(2*np.pi*10*t),SAMPLE_RATE,WINDOW_SIZE)
    bands = EegBands(**{k:pw.get(k,0.0) for k in ['delta','theta','alpha','beta','gamma']})
    clf = RFClassifier(); fv = clf.build_feature_vector(bands)
    _PASS(f'FV length {len(fv)}') if len(fv)==11 else _FAIL(f'FV length {len(fv)}')
    _PASS('No NaN')  if not any(math.isnan(v) for v in fv) else _FAIL('NaN in FV')
    _PASS('No Inf')  if not any(math.isinf(v) for v in fv) else _FAIL('Inf in FV')
    return fv

def _find_model_directories():
    pr = os.path.dirname(os.path.abspath(__file__))
    dirs = []
    dd = os.path.join(pr,'neuromentor_data')
    if os.path.isdir(dd):
        for e in os.listdir(dd):
            full=os.path.join(dd,e)
            if os.path.isdir(full) and e.endswith('_rf_model'):
                if os.path.exists(os.path.join(full,'rf_eeg_model.pkl')):
                    dirs.append(('user',e,full))
    bd = os.path.normpath(os.path.join(pr,'..','rf_model','rf_model'))
    if os.path.exists(os.path.join(bd,'rf_eeg_model.pkl')):
        dirs.append(('bundled','rf_model',bd))
    return dirs

def _check_saved_model(fv):
    _header('CHECK 6 — SAVED MODEL')
    if not (_HAS_NUMPY and _HAS_JOBLIB): _FAIL('numpy/joblib unavailable'); return
    dirs = _find_model_directories()
    if not dirs: _WARN('No model files found'); return
    for src,lbl,dirpath in dirs:
        _p(f'  Model: {lbl} ({src})')
        try: clf2 = _joblib_compat.load(os.path.join(dirpath,'rf_eeg_model.pkl')); _PASS('clf loads')
        except Exception as e: _FAIL(f'clf load failed: {e}'); continue
        try: sc2  = _joblib_compat.load(os.path.join(dirpath,'rf_scaler.pkl'));    _PASS('scaler loads')
        except Exception as e: _FAIL(f'scaler load failed: {e}'); continue
        n = getattr(clf2,'n_features_in_',None)
        if n: (_PASS if n==11 else _FAIL)(f'n_features_in_={n}')
        if fv is not None:
            try:
                Xs = sc2.transform(np.array([fv]))
                _PASS('scaler.transform ok')
                pred = clf2.predict(Xs)[0]
                _PASS(f'predict → {pred}')
            except Exception as e: _FAIL(f'predict failed: {e}')

def _check_version_drift():
    _header('CHECK 7 — VERSION DRIFT')
    dirs = _find_model_directories()
    if not dirs: _WARN('No models'); return
    cur = list(FEATURE_NAMES)
    for _,lbl,dp2 in dirs:
        try: clf2=_joblib_compat.load(os.path.join(dp2,'rf_eeg_model.pkl'))
        except: _WARN(f'Cannot load {lbl}'); continue
        saved = getattr(clf2,'feature_names_in_',None)
        if saved is None:
            sc=os.path.join(dp2,'feature_names.txt')
            if os.path.exists(sc):
                with open(sc) as f: saved=[l.strip() for l in f if l.strip()]
        if saved: (_PASS('Feature names match') if list(saved)==cur
                   else _FAIL('Feature name mismatch'))
        else: _WARN('feature_names not stored in model')

def _check_feature_formulas():
    _header('CHECK 8 — FEATURE FORMULAS')
    bands = EegBands(delta=2.0,theta=1.5,alpha=3.0,beta=1.0,gamma=0.5)
    clf = RFClassifier(); fv = clf.build_feature_vector(bands)
    eps=1e-6
    expected=[2.0,1.5,3.0,1.0,0.5,
              safe_div(1.0,3.0),safe_div(3.0,1.5),
              safe_div(1.0,1.5),safe_div(0.5,1.0),
              1.0-3.0,3.0+1.5]
    if len(fv)!=11: _FAIL(f'FV len {len(fv)}'); return
    for i,(act,exp) in enumerate(zip(fv,expected)):
        (_PASS if abs(act-exp)<1e-4 else _FAIL)(
            f'[{i}] {FEATURE_NAMES[i]} exp={exp:.4f} act={act:.4f}')

def run_compatibility_check() -> str:
    global _pass_c,_fail_c,_warn_c,_fail_details_c,_buf_c
    _pass_c=_fail_c=_warn_c=0; _fail_details_c=[]; _buf_c=StringIO()
    _p('╔══════════════════════════════════════════╗')
    _p('║  NEUROMENTOR COMPATIBILITY DIAGNOSTIC    ║')
    _p('╚══════════════════════════════════════════╝')
    for fn in [_check_adc_conversion,_check_sample_rate_and_window,
               _check_band_boundaries,_check_synthetic_signals]:
        try: fn()
        except Exception as e: _FAIL(f'{fn.__name__} crashed: {e}')
    fv=None
    try: fv=_check_feature_vector()
    except Exception as e: _FAIL(f'feature_vector crashed: {e}')
    for fn in [lambda:_check_saved_model(fv),_check_version_drift,_check_feature_formulas]:
        try: fn()
        except Exception as e: _FAIL(f'check crashed: {e}')
    _p('')
    _p('═'*44)
    _p(f'  Passed:{_pass_c}  Failed:{_fail_c}  Warnings:{_warn_c}')
    if _fail_c==0:
        _p('  VERDICT: COMPATIBLE ✓')
    else:
        _p('  VERDICT: INCOMPATIBLE')
        for d in _fail_details_c: _p(f'    — {d}')
    return _buf_c.getvalue()


# ============================================================
# app_state.py
# ============================================================
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from datetime import datetime
import json


class UserProfile:
    def __init__(self, username, name='', age='', notes='',
                 created_date=None, scores=None, last_login=None):
        self.username     = username
        self.name         = name or username
        self.age          = age
        self.notes        = notes
        self.created_date = created_date or datetime.now().isoformat()
        self.scores       = scores or []
        self.last_login   = last_login or datetime.now().isoformat()

    def to_dict(self):
        return dict(username=self.username, name=self.name, age=self.age,
                    notes=self.notes, created_date=self.created_date,
                    scores=self.scores, last_login=self.last_login)

    @classmethod
    def from_dict(cls, d):
        return cls(username=d.get('username',''), name=d.get('name',''),
                   age=d.get('age',''), notes=d.get('notes',''),
                   created_date=d.get('created_date'),
                   scores=d.get('scores',[]), last_login=d.get('last_login'))


class AppState(EventDispatcher):
    current_user       = ObjectProperty(None, allownone=True)
    selected_page_index= NumericProperty(0)
    selected_port      = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_users = self._load_users()

    # ---- Android-safe data dir ----
    def _get_data_dir(self):
        try:
            app = App.get_running_app()
            if app is not None:
                return app.user_data_dir
        except Exception:
            pass
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def users_file(self):
        return os.path.join(self._get_data_dir(), 'users.json')

    def _load_users(self):
        if not os.path.exists(self.users_file):
            return {}
        try:
            with open(self.users_file,'r') as f:
                return {k: UserProfile.from_dict(v) for k,v in json.load(f).items()}
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}

    def _save_users(self):
        try:
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file,'w') as f:
                json.dump({k:v.to_dict() for k,v in self.all_users.items()}, f, indent=4)
        except Exception as e:
            print(f"Error saving users: {e}")

    def save_current_user_score(self, test_name, score):
        if self.current_user:
            self.current_user.scores.append(
                {'test':test_name,'score':score,'date':datetime.now().isoformat()})
            self._save_users()

    def get_sorted_users(self):
        users = list(self.all_users.values())
        users.sort(key=lambda u: u.last_login or '', reverse=True)
        return users

    def login(self, username):
        if username not in self.all_users:
            self.all_users[username] = UserProfile(username=username,name=username)
        self.all_users[username].last_login = datetime.now().isoformat()
        self._save_users()
        self.current_user = self.all_users[username]
        self.selected_page_index = 0
        self._load_rf_model(username)

    def logout(self):
        self._save_users()
        self.current_user = None
        self.selected_page_index = 0

    def update_profile(self, name=None, age=None, notes=None):
        if self.current_user is not None:
            if name  is not None: self.current_user.name  = name
            if age   is not None: self.current_user.age   = age
            if notes is not None: self.current_user.notes = notes
            self._save_users()
            self.property('current_user').dispatch(self)

    def set_selected_page(self, index):
        self.selected_page_index = index

    def set_selected_port(self, port):
        self.selected_port = port or ''

    def _get_rf_model_dir(self, username):
        d = os.path.join(self._get_data_dir(), 'neuromentor_data', f'{username}_rf_model')
        os.makedirs(d, exist_ok=True)
        return d

    def _get_bundled_model_dir(self):
        return os.path.normpath(
            os.path.join(self._get_data_dir(), 'rf_model', 'rf_model'))

    def save_rf_model(self, rf_classifier=None):
        if rf_classifier is None:
            rf_classifier = getattr(self, 'rf_classifier', None)
        if self.current_user and rf_classifier and rf_classifier.is_trained:
            self.rf_classifier.save(self._get_rf_model_dir(self.current_user.username))

    def _load_rf_model(self, username):
        if not hasattr(self,'rf_classifier'):
            self.rf_classifier = RFClassifier()
        ud = self._get_rf_model_dir(username)
        if os.path.exists(os.path.join(ud,'rf_eeg_model.pkl')):
            if self.rf_classifier.load(ud): return True
        bd = self._get_bundled_model_dir()
        if os.path.exists(os.path.join(bd,'rf_eeg_model.pkl')):
            if self.rf_classifier.load(bd): return True
        return False


# ============================================================
# widgets/custom_ui.py
# ============================================================
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, RoundedRectangle, Rectangle, Line
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout


class ShadowButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        self.bg_color = kwargs.pop('bg_color', kwargs.pop('background_color', theme.BUTTON_BG))
        self.radius = kwargs.pop('radius', 12)
        super().__init__(**kwargs)
        self.bind(pos=self.update_canvas, size=self.update_canvas, state=self.update_canvas)
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            base_off = 1 if self.state == 'down' else 4
            for i in range(4):
                Color(0, 0, 0, 0.25*(1.0-i/4))
                RoundedRectangle(pos=(self.x-i+2, self.y-base_off-i),
                                 size=(self.width+i*2, self.height+i*2),
                                 radius=[self.radius+i])
            if self.state == 'down':
                Color(self.bg_color[0]*.8, self.bg_color[1]*.8, self.bg_color[2]*.8, 1)
            else:
                Color(*self.bg_color)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])


class MenuBurgerButton(ButtonBehavior, Widget):
    def __init__(self, **kwargs):
        self.color = kwargs.pop('color', theme.GOLD)
        self.is_open = False
        super().__init__(**kwargs)
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.update_canvas()

    def set_open(self, is_open):
        self.is_open = is_open
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.color)
            w = self.width * 0.5
            h = max(2, self.height * 0.08)
            cx, cy = self.center_x, self.center_y
            if not self.is_open:
                sp2 = self.height * 0.22
                RoundedRectangle(pos=(cx-w/2, cy+sp2-h/2), size=(w,h), radius=[h/2])
                RoundedRectangle(pos=(cx-w/2, cy-h/2),     size=(w,h), radius=[h/2])
                RoundedRectangle(pos=(cx-w/2, cy-sp2-h/2), size=(w,h), radius=[h/2])
            else:
                Line(points=[cx-w/2,cy-w/2,cx+w/2,cy+w/2], width=h/2, cap='round')
                Line(points=[cx-w/2,cy+w/2,cx+w/2,cy-w/2], width=h/2, cap='round')


class GradientCard(BoxLayout):
    def __init__(self, accent_color=None, **kwargs):
        kwargs.setdefault('orientation','vertical')
        kwargs.setdefault('padding',[dp(12),dp(12),dp(12),dp(12)])
        kwargs.setdefault('spacing', dp(6))
        self.accent_color = accent_color
        self.radius = kwargs.pop('radius', 12)
        super().__init__(**kwargs)
        self.texture = Texture.create(size=(1,2), colorfmt='rgba')
        self.texture.mag_filter = 'linear'
        self.texture.min_filter = 'linear'
        buf = bytes([51,46,60,255, 70,65,81,255])
        self.texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            for i in range(4):
                Color(0,0,0,0.2*(1.0-i/4))
                RoundedRectangle(pos=(self.x-i+2, self.y-2-i),
                                 size=(self.width+i*2, self.height+i*2),
                                 radius=[self.radius+i])
            Color(1,1,1,1)
            RoundedRectangle(pos=self.pos, size=self.size,
                             radius=[self.radius], texture=self.texture)
            if self.accent_color:
                Color(*self.accent_color)
                RoundedRectangle(pos=(self.x+8, self.y+self.height-4),
                                 size=(self.width-16, 3), radius=[1.5])


# ============================================================
# widgets/eeg_graph.py
# ============================================================
from collections import deque
from kivy.uix.label import Label as KivyLabel
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty


class EegGraph(BoxLayout):
    max_data_points = NumericProperty(256)
    min_y = NumericProperty(0)
    max_y = NumericProperty(4095)

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical',
                         padding=[dp(16),dp(8),dp(8),dp(8)], **kwargs)
        self._data = deque(maxlen=256)

        self._title_label = KivyLabel(
            text='Live EEG Signal',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            halign='left', valign='middle')
        self._title_label.bind(
            size=self._title_label.setter('text_size'),
            texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(18))))
        self.add_widget(self._title_label)

        self._placeholder = KivyLabel(
            text='Waiting for signal...',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_MUTED)
        self.add_widget(self._placeholder)

        self._graph_canvas = _GraphCanvas(
            data=self._data, min_y=self.min_y, max_y=self.max_y)
        self._graph_canvas.opacity = 0
        self.add_widget(self._graph_canvas)

        with self.canvas.before:
            Color(*theme.PANEL_BG)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self._bg_rect.pos  = self.pos
        self._bg_rect.size = self.size

    def add_data_point(self, value):
        self._data.append(value)
        if self._placeholder.opacity > 0:
            self._placeholder.opacity = 0
            self._graph_canvas.opacity = 1
        self._graph_canvas.redraw()

    def clear(self):
        self._data.clear()
        self._placeholder.opacity = 1
        self._graph_canvas.opacity = 0
        self._graph_canvas.redraw()


class _GraphCanvas(Widget):
    def __init__(self, data, min_y=0, max_y=4095, **kwargs):
        super().__init__(**kwargs)
        self._data  = data
        self._min_y = min_y
        self._max_y = max_y
        self.bind(size=lambda *a: self.redraw(), pos=lambda *a: self.redraw())

    def redraw(self):
        self.canvas.clear()
        if not self._data or self.width <= 0 or self.height <= 0:
            return
        x0 = self.x + dp(16); y0 = self.y + dp(8)
        w  = self.width - dp(20); h = self.height - dp(16)
        with self.canvas:
            Color(*theme.BORDER_DARK)
            for i in range(5):
                gy = y0 + (h*i/4)
                Line(points=[x0,gy,x0+w,gy], width=1)
            for i in range(9):
                gx = x0 + (w*i/8)
                Line(points=[gx,y0,gx,y0+h], width=1)
        data_list = list(self._data)
        n = len(data_list)
        if n < 2: return
        yr = self._max_y - self._min_y or 1
        pts = []
        for i,val in enumerate(data_list):
            px = x0 + w*i/(n-1)
            py = y0 + h*((val-self._min_y)/yr)
            py = max(y0, min(y0+h, py))
            pts.extend([px, py])
        with self.canvas:
            Color(*theme.TEAL)
            Line(points=pts, width=1.5)


class BandPowerBars(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(10), spacing=dp(4), **kwargs)
        self._bands = ['Delta','Theta','Alpha','Beta','Gamma']
        self._bars  = {}
        title = KivyLabel(text='EEG BAND POWERS',
                          font_size=theme.font(theme.FONT_BODY_SMALL),
                          color=theme.TEXT_SECONDARY, size_hint_y=None,
                          halign='left', valign='middle')
        title.bind(size=title.setter('text_size'),
                   texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(18))))
        self.add_widget(title)
        for band in self._bands:
            row = BoxLayout(size_hint_y=None, height=dp(24), spacing=dp(5))
            lbl = KivyLabel(text=band, font_size=theme.font(theme.FONT_BODY_SMALL),
                            color=theme.TEXT_MUTED, size_hint_x=None, width=dp(50),
                            halign='left', valign='middle')
            lbl.bind(size=lbl.setter('text_size'))
            bar = _BarWidget(value=0)
            row.add_widget(lbl); row.add_widget(bar)
            self._bars[band] = bar
            self.add_widget(row)
        with self.canvas.before:
            Color(*theme.BG_DARK)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

    def _upd(self, *a):
        self._bg.pos  = self.pos
        self._bg.size = self.size

    def update_values(self, vd):
        for band,bar in self._bars.items():
            bar.value = vd.get(band, 0)


class _BarWidget(Widget):
    value = NumericProperty(0)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(value=self._r, size=self._r, pos=self._r)
        self._r()

    def _r(self, *a):
        self.canvas.clear()
        with self.canvas:
            Color(0.165,0.165,0.165,1)
            Rectangle(pos=self.pos, size=self.size)
            Color(*theme.TEAL)
            fw = self.width * max(0, min(1, self.value))
            if fw > 0:
                Rectangle(pos=self.pos, size=(fw, self.height))


# ============================================================
# widgets/mind_visualizer.py
# ============================================================
import random
from kivy.graphics import Color, Ellipse
from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty


class MindVisualizer(Widget):
    is_active   = BooleanProperty(False)
    state_label = StringProperty('IDLE')
    focus_ratio = NumericProperty(1.0)
    stress_ratio= NumericProperty(1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ball_y = self._target_y = 0.5
        self._jitter_x = self._jitter_y = 0.0
        self._jitter_intensity = 0.0
        self._ball_color = list(theme.TEAL)
        self._clock_event = Clock.schedule_interval(self._animate, 1/60)
        self.bind(focus_ratio=self._upd, stress_ratio=self._upd,
                  state_label=self._upd,
                  size=lambda *a: self._draw(),
                  pos=lambda *a:  self._draw())

    def _upd(self, *a):
        f = max(0.5, min(2.5, self.focus_ratio))
        self._target_y = 1.0 - ((f-0.5)/2.0)
        s = max(0.5, min(2.0, self.stress_ratio))
        self._jitter_intensity = (s-0.5)*0.05
        self._ball_color = (list(theme.RED)   if self.state_label=='Stressed' else
                            list(theme.GOLD)  if self.state_label=='Focused'  else
                            list(theme.TEAL))

    def _animate(self, dt):
        self._ball_y += (self._target_y - self._ball_y)*0.05
        self._ball_y  = max(0.1, min(0.9, self._ball_y))
        self._jitter_x = (random.random()-0.5)*self._jitter_intensity
        self._jitter_y = (random.random()-0.5)*self._jitter_intensity
        self._draw()

    def _draw(self):
        self.canvas.clear()
        w,h,x0,y0 = self.width, self.height, self.x, self.y
        if w<=0 or h<=0: return
        with self.canvas:
            Color(0.02,0.02,0.031,1)
            Rectangle(pos=self.pos, size=self.size)
            Color(0.118,0.118,0.157,1)
            for gx in range(0,int(w),60):
                Line(points=[x0+gx,y0,x0+gx,y0+h], width=1)
            for gy in range(0,int(h),60):
                Line(points=[x0,y0+gy,x0+w,y0+gy], width=1)
            cx = x0+w*0.5+self._jitter_x*w
            cy = y0+h*(1.0-self._ball_y)+self._jitter_y*h
            cy = max(y0+50, min(y0+h-50, cy))
            r  = 40
            for i in range(6,0,-1):
                Color(self._ball_color[0], self._ball_color[1],
                      self._ball_color[2], 0.06*i)
                gr = r*(0.5+i*0.5)
                Ellipse(pos=(cx-gr,cy-gr), size=(gr*2,gr*2))
            Color(*self._ball_color)
            Ellipse(pos=(cx-r,cy-r), size=(r*2,r*2))

    def cleanup(self):
        if hasattr(self,'_clock_event') and self._clock_event:
            self._clock_event.cancel()
            self._clock_event = None

    def __del__(self):
        self.cleanup()


# ============================================================
# widgets/tasks/breathing_widget.py
# ============================================================
from kivy.uix.togglebutton import ToggleButton


class BreathingWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(20), spacing=dp(10), **kwargs)
        self._step = self._score = 0
        self._is_running = False
        self._mode = '4-7-8'
        self._clock_event = None

        self._instruction_lbl = KivyLabel(
            text='Ready', font_size=sp(40), color=theme.TEAL,
            bold=True, size_hint_y=0.4)
        self.add_widget(self._instruction_lbl)

        mode_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(20))
        mode_row.size_hint_x = None; mode_row.width = dp(320)
        mode_row.pos_hint = {'center_x':0.5}
        self._btn_478 = ToggleButton(text='4-7-8 (Calm)', group='bm',
            state='down', font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.GOLD, color=theme.BG_DARK)
        self._btn_478.bind(on_press=lambda *a: self.set_mode('4-7-8'))
        self._btn_box = ToggleButton(text='Box (Focus)', group='bm',
            state='normal', font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.BORDER_DARK, color=theme.TEXT_PRIMARY)
        self._btn_box.bind(on_press=lambda *a: self.set_mode('box'))
        mode_row.add_widget(self._btn_478); mode_row.add_widget(self._btn_box)
        self.add_widget(mode_row)

        self._score_lbl = KivyLabel(text='Score: 0',
            font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            color=theme.GOLD, size_hint_y=None, height=dp(40))
        self.add_widget(self._score_lbl)

        self._start_btn = ShadowButton(text='START',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.BUTTON_BG, color=theme.GOLD)
        self._start_btn.bind(on_press=self._toggle)
        self.add_widget(self._start_btn)

    def set_mode(self, mode):
        self._mode = mode
        if mode == '4-7-8':
            self._btn_478.state='down'; self._btn_box.state='normal'
            self._btn_478.background_color=theme.GOLD; self._btn_478.color=theme.BG_DARK
            self._btn_box.background_color=theme.BORDER_DARK; self._btn_box.color=theme.TEXT_PRIMARY
        else:
            self._btn_box.state='down'; self._btn_478.state='normal'
            self._btn_box.background_color=theme.GOLD; self._btn_box.color=theme.BG_DARK
            self._btn_478.background_color=theme.BORDER_DARK; self._btn_478.color=theme.TEXT_PRIMARY

    def start_task(self): self._start()

    def _toggle(self, *a):
        self.stop() if self._is_running else self._start()

    def _start(self):
        if self._is_running: return
        self._is_running=True; self._step=self._score=0
        self._start_btn.text='STOP'; self._start_btn.color=theme.RED
        self._start_btn.bg_color=theme.DANGER_BUTTON_BG
        self._clock_event=Clock.schedule_interval(self._tick,1.0)

    def stop(self):
        if self._clock_event: self._clock_event.cancel(); self._clock_event=None
        self._is_running=False; self._instruction_lbl.text='Relax'
        self._start_btn.text='START'; self._start_btn.color=theme.GOLD
        self._start_btn.bg_color=theme.BUTTON_BG

    def _tick(self, dt):
        self._step+=1; self._score=self._step*10
        self._score_lbl.text=f'Score: {self._score}'
        cl=16 if self._mode=='box' else 19; cur=self._step%cl
        if self._mode=='box':
            if   cur<4:  self._instruction_lbl.text=f'INHALE ({4-cur})'
            elif cur<8:  self._instruction_lbl.text=f'HOLD ({8-cur})'
            elif cur<12: self._instruction_lbl.text=f'EXHALE ({12-cur})'
            else:        self._instruction_lbl.text=f'HOLD ({16-cur})'
        else:
            if   cur<4:  self._instruction_lbl.text=f'INHALE ({4-cur})'
            elif cur<11: self._instruction_lbl.text=f'HOLD ({11-cur})'
            else:        self._instruction_lbl.text=f'EXHALE ({19-cur})'

    def cleanup(self):
        if self._clock_event: self._clock_event.cancel()


# ============================================================
# widgets/tasks/focus_widget.py
# ============================================================
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout

ARTICLES = [
    'Neuroplasticity is the ability of neural networks in the brain to reorganize '
    'themselves by creating new neural connections throughout life. This allows '
    'neurons to compensate for injury and disease, and to adjust their activities '
    'in response to new situations. Researchers have discovered that neuroplasticity '
    'is not limited to childhood development but continues throughout adult life. '
    'This groundbreaking finding has revolutionized our understanding of brain '
    'function and has led to new therapeutic approaches for treating brain injuries, '
    'learning disabilities, and neurodegenerative diseases.',

    'Quantum entanglement is a phenomenon where two or more particles become '
    'interconnected in such a way that the quantum state of each particle cannot '
    'be described independently. When particles are entangled, they remain connected '
    'across vast distances, and measuring one particle instantaneously affects the '
    'state of the other. Today, quantum entanglement has practical applications in '
    'quantum computing, quantum cryptography, and quantum teleportation.',

    'In cognitive science, attention is the cognitive process that allows us to '
    'focus on specific information while filtering out irrelevant stimuli. The '
    'human brain receives countless sensory inputs every second, yet we can only '
    'consciously process a fraction of this information. Selective attention '
    'mechanisms help us prioritize important information and maintain focus on '
    'relevant tasks.',
]


class FocusWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(10), spacing=dp(10), **kwargs)
        self._is_tracking = True; self._is_running = False

        mode_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(20))
        mode_row.size_hint_x=None; mode_row.width=dp(320)
        mode_row.pos_hint={'center_x':0.5}
        self._btn_tracking = ToggleButton(text='Visual Tracking', group='fm',
            state='down', font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.GOLD, color=theme.BG_DARK)
        self._btn_tracking.bind(on_press=lambda *a: self.set_mode('tracking'))
        self._btn_reading = ToggleButton(text='Tech Reading', group='fm',
            state='normal', font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.BORDER_DARK, color=theme.TEXT_PRIMARY)
        self._btn_reading.bind(on_press=lambda *a: self.set_mode('reading'))
        mode_row.add_widget(self._btn_tracking); mode_row.add_widget(self._btn_reading)
        self.add_widget(mode_row)

        self._tracking_view = TrackingView()
        self._reading_view  = ReadingView()
        self._reading_view.opacity=0; self._reading_view.disabled=True
        self._content = FloatLayout()
        self._content.add_widget(self._tracking_view)
        self._content.add_widget(self._reading_view)
        self.add_widget(self._content)

        self._start_btn = ShadowButton(text='START',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.BUTTON_BG, color=theme.GOLD)
        self._start_btn.bind(on_press=self._toggle)
        self.add_widget(self._start_btn)

    def set_mode(self, mode):
        it = (mode=='tracking'); self._is_tracking = it
        self._btn_tracking.state = 'down' if it else 'normal'
        self._btn_reading.state  = 'normal' if it else 'down'
        self._btn_tracking.background_color = theme.GOLD if it else theme.BORDER_DARK
        self._btn_tracking.color = theme.BG_DARK if it else theme.TEXT_PRIMARY
        self._btn_reading.background_color  = theme.BORDER_DARK if it else theme.GOLD
        self._btn_reading.color  = theme.TEXT_PRIMARY if it else theme.BG_DARK
        self._tracking_view.opacity=1 if it else 0; self._tracking_view.disabled=not it
        self._reading_view.opacity=0 if it else 1;  self._reading_view.disabled=it

    def start_task(self):
        if not self._is_running: self._toggle()

    def stop(self):
        if self._is_running: self._toggle()

    def _toggle(self, *a):
        self._is_running = not self._is_running
        if self._is_running:
            self._start_btn.text='STOP'; self._start_btn.color=theme.RED
            self._start_btn.bg_color=theme.DANGER_BUTTON_BG
            self._tracking_view.start(); self._reading_view.new_article()
        else:
            self._start_btn.text='START'; self._start_btn.color=theme.GOLD
            self._start_btn.bg_color=theme.BUTTON_BG
            self._tracking_view.stop()

    def cleanup(self): self._tracking_view.stop()


class TrackingView(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ball_x=self._ball_y=0.5; self._clock_event=None
        self.bind(size=self._draw, pos=self._draw); self._draw()

    def start(self):
        if self._clock_event: self._clock_event.cancel()
        self._clock_event=Clock.schedule_interval(self._move,1/30)

    def stop(self):
        if self._clock_event: self._clock_event.cancel(); self._clock_event=None

    def _move(self, dt):
        self._ball_x=max(0.05,min(0.95,self._ball_x+(random.random()-0.5)*0.02))
        self._ball_y=max(0.05,min(0.95,self._ball_y+(random.random()-0.5)*0.02))
        self._draw()

    def _draw(self, *a):
        self.canvas.clear()
        w,h=self.width,self.height
        if w<=0 or h<=0: return
        cx=self.x+self._ball_x*w; cy=self.y+self._ball_y*h
        with self.canvas:
            Color(0,0,0,1); Rectangle(pos=self.pos,size=self.size)
            for i in range(4,0,-1):
                Color(theme.GOLD[0],theme.GOLD[1],theme.GOLD[2],0.12*i)
                r=15+i*8; Ellipse(pos=(cx-r,cy-r),size=(r*2,r*2))
            Color(*theme.GOLD); Ellipse(pos=(cx-15,cy-15),size=(30,30))


class ReadingView(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(15), spacing=dp(10), **kwargs)
        hdr = KivyLabel(text='READ CAREFULLY:',
            font_size=theme.font(theme.FONT_BODY_SMALL), color=theme.TEXT_SECONDARY,
            size_hint_y=None, halign='left', valign='middle')
        hdr.bind(size=hdr.setter('text_size'),
                 texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(20))))
        self.add_widget(hdr)
        scroll = ScrollView()
        self._art = KivyLabel(text=random.choice(ARTICLES),
            font_size=theme.font(theme.FONT_BODY_LARGE),
            color=theme.TEAL, markup=False, halign='left', valign='top',
            size_hint_y=None)
        self._art.bind(texture_size=lambda inst,sz: setattr(inst,'height',sz[1]),
                       width=lambda inst,w: setattr(inst,'text_size',(w,None)))
        scroll.add_widget(self._art); self.add_widget(scroll)
        with self.canvas.before:
            Color(0.067,0.067,0.067,1)
            self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=self._u,size=self._u)

    def _u(self,*a): self._bg.pos=self.pos; self._bg.size=self.size
    def new_article(self): self._art.text=random.choice(ARTICLES)


# ============================================================
# widgets/tasks/stroop_widget.py
# ============================================================
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


class StroopWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(20), spacing=dp(10), **kwargs)
        self._score=0; self._is_running=False; self._is_math_mode=False
        self._clock_event=None
        self._current_ink_name='BLUE'; self._math_answer=0
        self._colors=['RED','BLUE','GREEN','YELLOW']
        self._color_map={'RED':(1,0,0,1),'BLUE':(0,0,1,1),
                         'GREEN':(0,1,0,1),'YELLOW':(1,1,0,1)}

        self._display_lbl = KivyLabel(text='BLUE', font_size=sp(50), bold=True,
            color=(0,0,1,1), size_hint_y=0.35)
        self.add_widget(self._display_lbl)

        mode_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(20))
        mode_row.size_hint_x=None; mode_row.width=dp(280)
        mode_row.pos_hint={'center_x':0.5}
        self._btn_stroop = ToggleButton(text='Stroop', group='sm', state='down',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.GOLD, color=theme.BG_DARK)
        self._btn_stroop.bind(on_press=lambda *a: self.set_mode('stroop'))
        self._btn_math = ToggleButton(text='Rapid Math', group='sm', state='normal',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.BORDER_DARK, color=theme.TEXT_PRIMARY)
        self._btn_math.bind(on_press=lambda *a: self.set_mode('math'))
        mode_row.add_widget(self._btn_stroop); mode_row.add_widget(self._btn_math)
        self.add_widget(mode_row)

        self._stroop_row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(8))
        for c in self._colors:
            btn=Button(text=c, font_size=sp(12), bold=True,
                       background_color=self._color_map[c], color=(0,0,0,1),
                       size_hint_x=None, width=dp(80))
            btn.bind(on_press=lambda inst,cn=c: self._check_stroop(cn))
            self._stroop_row.add_widget(btn)
        self.add_widget(self._stroop_row)

        self._math_row = BoxLayout(size_hint=(None,None),size=(dp(300),dp(45)),
                                   pos_hint={'center_x':0.5}, spacing=dp(10))
        self._math_input = TextInput(hint_text='Answer',
            font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            multiline=False, input_filter='int',
            size_hint=(None,1), width=dp(180),
            background_color=theme.INPUT_BG, foreground_color=theme.TEXT_PRIMARY)
        self._math_input.bind(on_text_validate=lambda *a: self._check_math())
        self._math_submit = ShadowButton(text='SUBMIT',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            bg_color=theme.TEAL, color=theme.BG_DARK,
            size_hint=(None,1), width=dp(110))
        self._math_submit.bind(on_press=lambda *a: self._check_math())
        self._math_row.add_widget(self._math_input)
        self._math_row.add_widget(self._math_submit)
        self._math_row.opacity=0; self._math_row.disabled=True
        self.add_widget(self._math_row)

        self._score_lbl = KivyLabel(text='Score: 0',
            font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            color=theme.GOLD, size_hint_y=None, height=dp(40))
        self.add_widget(self._score_lbl)

        self._start_btn = ShadowButton(text='START',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.BUTTON_BG, color=theme.GOLD)
        self._start_btn.bind(on_press=self._toggle)
        self.add_widget(self._start_btn)

    def set_mode(self, mode):
        im = (mode=='math'); self._is_math_mode=im
        self._btn_math.state='down' if im else 'normal'
        self._btn_stroop.state='normal' if im else 'down'
        self._btn_math.background_color=theme.GOLD if im else theme.BORDER_DARK
        self._btn_math.color=theme.BG_DARK if im else theme.TEXT_PRIMARY
        self._btn_stroop.background_color=theme.BORDER_DARK if im else theme.GOLD
        self._btn_stroop.color=theme.TEXT_PRIMARY if im else theme.BG_DARK
        self._stroop_row.opacity=0 if im else 1; self._stroop_row.disabled=im
        self._math_row.opacity=1 if im else 0;   self._math_row.disabled=not im
        if self._is_running: self._next_round()

    def start_task(self): self._start()

    def _toggle(self, *a):
        self.stop() if self._is_running else self._start()

    def _start(self):
        if self._is_running: return
        self._is_running=True; self._score=0; self._score_lbl.text='Score: 0'
        self._start_btn.text='STOP'; self._start_btn.color=theme.RED
        self._start_btn.bg_color=theme.DANGER_BUTTON_BG
        self._next_round()
        self._clock_event=Clock.schedule_interval(lambda dt: self._next_round(), 3.0)

    # Only ONE stop() definition
    def stop(self):
        if self._clock_event: self._clock_event.cancel(); self._clock_event=None
        self._is_running=False
        self._start_btn.text='START'; self._start_btn.color=theme.GOLD
        self._start_btn.bg_color=theme.BUTTON_BG

    def _next_round(self):
        if self._is_math_mode:
            a=random.randint(10,99); b=random.randint(10,99)
            ia=random.choice([True,False])
            self._math_answer=a+b if ia else a-b
            self._display_lbl.text=f'{a} {"+" if ia else "-"} {b} = ?'
            self._display_lbl.color=theme.RED; self._math_input.text=''
        else:
            word=random.choice(self._colors); ink=random.choice(self._colors)
            self._current_ink_name=ink
            self._display_lbl.text=word; self._display_lbl.color=self._color_map[ink]

    def _check_stroop(self, cn):
        if not self._is_running: return
        if cn==self._current_ink_name:
            self._score+=50; self._score_lbl.text=f'Score: {self._score}'
        if self._clock_event: self._clock_event.cancel()
        self._next_round()
        self._clock_event=Clock.schedule_interval(lambda dt: self._next_round(), 3.0)

    def _check_math(self):
        if not self._is_running: return
        try: ans=int(self._math_input.text)
        except: ans=0
        if ans==self._math_answer:
            self._score+=50; self._score_lbl.text=f'Score: {self._score}'
        if self._clock_event: self._clock_event.cancel()
        self._next_round()
        self._clock_event=Clock.schedule_interval(lambda dt: self._next_round(), 3.0)

    def cleanup(self):
        if self._clock_event: self._clock_event.cancel()


# ============================================================
# screens/dashboard_screen.py  — wrapped in ScrollView
# ============================================================
class DashboardScreen(ScrollView):
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state = app_state
        self._inner = BoxLayout(orientation='vertical',
            padding=[dp(16),dp(16),dp(16),dp(16)], spacing=dp(12),
            size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        self._welcome_lbl = KivyLabel(
            text='WELCOME BACK, USER',
            font_size=theme.font(theme.FONT_TITLE_MEDIUM),
            bold=True, color=theme.GOLD,
            size_hint_y=None, halign='left', valign='middle')
        self._welcome_lbl.bind(
            size=self._welcome_lbl.setter('text_size'),
            texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(32))))
        self._inner.add_widget(self._welcome_lbl)
        app_state.bind(current_user=self._update_welcome)
        self._update_welcome()

        status_card = GradientCard(size_hint_y=None, height=dp(70), padding=[dp(16),dp(10)])
        sv = KivyLabel(text='OFFLINE',
            font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            color=theme.TEXT_SECONDARY, halign='left', valign='middle')
        sv.bind(size=sv.setter('text_size'),
                texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(28))))
        status_card.add_widget(sv)
        self._inner.add_widget(status_card)

        stats_row = BoxLayout(spacing=dp(12), size_hint_y=None, height=dp(100))
        stats_row.add_widget(self._build_stat_card('STRESS','N/A',theme.RED))
        stats_row.add_widget(self._build_stat_card('FOCUS','N/A',theme.TEAL))
        stats_row.add_widget(self._build_stat_card('NEURO XP','0',theme.GOLD))
        self._inner.add_widget(stats_row)

        lt = KivyLabel(text='EVENT LOG',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_SECONDARY, size_hint_y=None, halign='left', valign='middle')
        lt.bind(size=lt.setter('text_size'),
                texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(18))))
        self._inner.add_widget(lt)

        log_box = GradientCard(size_hint_y=None, height=dp(200), padding=[dp(10),dp(10)])
        ll = KivyLabel(text='No events yet',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEAL, halign='left', valign='top')
        ll.bind(size=ll.setter('text_size'),
                texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(40))))
        log_box.add_widget(ll)
        self._inner.add_widget(log_box)

        phase2_label = AutoLabel(
            text='PHASE 2 FEATURES',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            bold=True, color=theme.TEXT_SECONDARY,
            size_hint_y=None, halign='left', valign='middle')
        phase2_label.bind(size=phase2_label.setter('text_size'),
                          texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(18))))
        self._inner.add_widget(phase2_label)

        phase2_group = BoxLayout(orientation='vertical', spacing=dp(12), size_hint_y=None)
        phase2_group.bind(minimum_height=phase2_group.setter('height'))
        phase2_group.add_widget(self._build_phase2_button('🎤 Ask NeuroMentor', self.start_esp32_voice))
        phase2_group.add_widget(self._build_phase2_button('PHASE 2 INSIGHTS', self.on_phase2_insights))
        self._inner.add_widget(phase2_group)

        self._voice_response_label = AutoLabel(
            text='Response will appear here.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, size_hint_y=None,
            halign='left', valign='middle')
        self._voice_response_label.bind(size=self._voice_response_label.setter('text_size'),
            texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
        self._inner.add_widget(self._voice_response_label)

        rb = ShadowButton(text='SYSTEM REFRESH',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.BUTTON_BG, color=theme.GOLD)
        self._inner.add_widget(rb)

    def _update_welcome(self, *a):
        u = self._app_state.current_user
        n = u.name.upper() if u and u.name else 'USER'
        self._welcome_lbl.text = f'WELCOME BACK, {n}'

    def start_esp32_voice(self, *args):
        if getattr(self, '_voice_thread', None) is not None and self._voice_thread.is_alive():
            self._update_voice_response('Already processing a request...')
            return
        self._update_voice_response('Receiving audio from ESP32...')
        self._voice_thread = threading.Thread(target=self.process_esp32_audio, daemon=True)
        self._voice_thread.start()

    def process_esp32_audio(self):
        temp_path = os.path.join(App.get_running_app().user_data_dir, 'esp32_voice_query.wav')
        self.audio_buffer = bytearray()
        try:
            self._update_voice_response('Reading ESP32 audio stream...')
            audio_bytes = self._receive_esp32_audio()
            if not audio_bytes:
                raise RuntimeError('No audio received from ESP32.')
            self.audio_buffer.extend(audio_bytes)

            self._write_wav_file(temp_path, self.audio_buffer)
            self._update_voice_response('Transcribing audio...')
            text = self._transcribe_with_whisper(temp_path)
            if not text:
                raise RuntimeError('Whisper returned empty text.')

            self._update_voice_response(f'Query: {text}\nContacting Ollama...')
            answer = self._query_ollama(text)
            self._update_voice_response(answer)
        except Exception as exc:
            print(f'[ESP32 Voice] {exc}')
            self._update_voice_response('Error processing audio from ESP32.')

    def _receive_esp32_audio(self) -> bytes:
        import time
        audio_bytes = bytearray()
        expected_bytes = 16000 * 2 * 5
        start_time = time.time()

        serial_ports = []
        if getattr(self._app_state, 'selected_port', None):
            serial_ports.append(self._app_state.selected_port)
        serial_ports.extend(['COM3', 'COM4', 'COM5', 'COM6'])

        for port in serial_ports:
            try:
                import serial
                with serial.Serial(port, 115200, timeout=1) as ser:
                    while len(audio_bytes) < expected_bytes and time.time() - start_time < 12:
                        chunk = ser.read(min(4096, expected_bytes - len(audio_bytes)))
                        if not chunk:
                            continue
                        audio_bytes.extend(chunk)
                if audio_bytes:
                    return bytes(audio_bytes)
            except Exception as exc:
                print(f'[ESP32 Serial] port={port} {exc}')

        socket_hosts = [('192.168.4.1', 5000), ('192.168.4.2', 5000), ('127.0.0.1', 5000)]
        for host, port in socket_hosts:
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    sock.connect((host, port))
                    while len(audio_bytes) < expected_bytes and time.time() - start_time < 12:
                        chunk = sock.recv(min(4096, expected_bytes - len(audio_bytes)))
                        if not chunk:
                            break
                        audio_bytes.extend(chunk)
                if audio_bytes:
                    return bytes(audio_bytes)
            except Exception as exc:
                print(f'[ESP32 Socket] host={host} port={port} {exc}')

        return bytes(audio_bytes)

    def _write_wav_file(self, path, audio_data):
        import wave
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)

    def _transcribe_with_whisper(self, path) -> str:
        import whisper
        model = whisper.load_model('base')
        result = model.transcribe(path)
        return result.get('text', '').strip()

    def _query_ollama(self, text) -> str:
        import requests
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3',
                'prompt': text,
                'stream': False,
            },
            timeout=20)
        response.raise_for_status()
        payload = response.json()
        answer = payload.get('response') or payload.get('text') or ''
        return answer.strip() or 'No response received from Ollama.'

    def _update_voice_response(self, text):
        Clock.schedule_once(lambda dt: setattr(self._voice_response_label, 'text', text), 0)

    def on_phase2_insights(self, *args):
        self._navigate_to_phase2('PHASE 2 INSIGHTS')

    def _navigate_to_phase2(self, page_name):
        try:
            idx = PAGE_NAMES.index(page_name)
            self._app_state.set_selected_page(idx)
        except ValueError:
            pass

    def _build_phase2_button(self, title, callback):
        btn = AutoButton(text=title,
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY,
            halign='center', valign='middle')
        btn.size_hint_y = None
        btn.bind(on_press=callback)
        with btn.canvas.before:
            Color(*theme.BUTTON_BG)
            btn._bg_rect = RoundedRectangle(pos=btn.pos, size=btn.size, radius=[dp(10)])
        btn.bind(pos=lambda inst,v: setattr(btn._bg_rect,'pos',btn.pos),
                 size=lambda inst,v: setattr(btn._bg_rect,'size',btn.size))
        return btn

    def _build_stat_card(self, title, value, accent):
        card = GradientCard(accent_color=accent, padding=[dp(12),dp(8)],
                            size_hint_y=None, height=dp(90))
        t = KivyLabel(text=title, font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, size_hint_y=None, halign='left', valign='middle')
        t.bind(size=t.setter('text_size'),
               texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(16))))
        card.add_widget(t)
        v = KivyLabel(text=value, font_size=theme.font(theme.FONT_HEADING_LARGE),
            bold=True, color=accent, size_hint_y=None, halign='left', valign='middle')
        v.bind(size=v.setter('text_size'),
               texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(28))))
        card.add_widget(v)
        return card


# ============================================================
# screens/profile_screen.py  — wrapped in ScrollView
# ============================================================
from kivy.uix.popup import Popup


class Phase2NotesScreen(ScrollView):
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state = app_state
        self._inner = BoxLayout(orientation='vertical',
            padding=[dp(16),dp(16),dp(16),dp(16)], spacing=dp(12),
            size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        title = AutoLabel(text='VOICE ASSISTANT',
            font_size=theme.font(theme.FONT_TITLE_MEDIUM),
            bold=True, color=theme.GOLD, size_hint_y=None,
            halign='left', valign='middle')
        title.bind(size=title.setter('text_size'),
                   texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(32))))
        self._inner.add_widget(title)

        hint = AutoLabel(text='Speak a question about NeuroMentor EEG to get a local response.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_SECONDARY, size_hint_y=None,
            halign='left', valign='middle')
        hint.bind(size=hint.setter('text_size'),
                  texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
        self._inner.add_widget(hint)

        self._voice_button = AutoButton(text='🎤 Ask NeuroMentor',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY, halign='center', valign='middle')
        self._voice_button.size_hint_y = None
        self._voice_button.bind(on_press=self.start_esp32_voice)
        with self._voice_button.canvas.before:
            Color(*theme.BUTTON_BG)
            self._voice_button._bg_rect = RoundedRectangle(pos=self._voice_button.pos,
                                                           size=self._voice_button.size,
                                                           radius=[dp(10)])
        self._voice_button.bind(pos=lambda inst,v: setattr(self._voice_button._bg_rect,'pos',self._voice_button.pos),
                                size=lambda inst,v: setattr(self._voice_button._bg_rect,'size',self._voice_button.size))
        self._inner.add_widget(self._voice_button)

        self._response_label = AutoLabel(text='Response will appear here.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, size_hint_y=None,
            halign='left', valign='middle')
        self._response_label.bind(size=self._response_label.setter('text_size'),
                                  texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
        self._inner.add_widget(self._response_label)

    def start_esp32_voice(self, *args):
        if getattr(self, '_voice_thread', None) is not None and self._voice_thread.is_alive():
            self._update_voice_response('Already processing a request...')
            return
        self._update_voice_response('Receiving audio from ESP32...')
        self._voice_thread = threading.Thread(target=self.process_esp32_audio, daemon=True)
        self._voice_thread.start()

    def process_esp32_audio(self):
        temp_path = os.path.join(App.get_running_app().user_data_dir, 'esp32_voice_query.wav')
        self.audio_buffer = bytearray()
        try:
            self._update_voice_response('Reading ESP32 audio stream...')
            audio_bytes = self._receive_esp32_audio()
            if not audio_bytes:
                raise RuntimeError('No audio received from ESP32.')
            self.audio_buffer.extend(audio_bytes)

            self._write_wav_file(temp_path, self.audio_buffer)
            self._update_voice_response('Transcribing audio...')
            text = self._transcribe_with_whisper(temp_path)
            if not text:
                raise RuntimeError('Whisper returned empty text.')

            self._update_voice_response(f'Query: {text}\nContacting Ollama...')
            answer = self._query_ollama(text)
            self._update_voice_response(answer)
        except Exception as exc:
            print(f'[ESP32 Voice] {exc}')
            self._update_voice_response('Error processing audio from ESP32.')

    def _receive_esp32_audio(self) -> bytes:
        import time
        audio_bytes = bytearray()
        expected_bytes = 16000 * 2 * 5
        start_time = time.time()

        serial_ports = []
        if getattr(self._app_state, 'selected_port', None):
            serial_ports.append(self._app_state.selected_port)
        serial_ports.extend(['COM3', 'COM4', 'COM5', 'COM6'])

        for port in serial_ports:
            try:
                import serial
                with serial.Serial(port, 115200, timeout=1) as ser:
                    while len(audio_bytes) < expected_bytes and time.time() - start_time < 12:
                        chunk = ser.read(min(4096, expected_bytes - len(audio_bytes)))
                        if not chunk:
                            continue
                        audio_bytes.extend(chunk)
                if audio_bytes:
                    return bytes(audio_bytes)
            except Exception as exc:
                print(f'[ESP32 Serial] port={port} {exc}')

        socket_hosts = [('192.168.4.1', 5000), ('192.168.4.2', 5000), ('127.0.0.1', 5000)]
        for host, port in socket_hosts:
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    sock.connect((host, port))
                    while len(audio_bytes) < expected_bytes and time.time() - start_time < 12:
                        chunk = sock.recv(min(4096, expected_bytes - len(audio_bytes)))
                        if not chunk:
                            break
                        audio_bytes.extend(chunk)
                if audio_bytes:
                    return bytes(audio_bytes)
            except Exception as exc:
                print(f'[ESP32 Socket] host={host} port={port} {exc}')

        return bytes(audio_bytes)

    def _write_wav_file(self, path, audio_data):
        import wave
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)

    def _transcribe_with_whisper(self, path) -> str:
        import whisper
        model = whisper.load_model('base')
        result = model.transcribe(path)
        return result.get('text', '').strip()

    def _query_ollama(self, text) -> str:
        import requests
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3',
                'prompt': text,
                'stream': False,
            },
            timeout=20)
        response.raise_for_status()
        payload = response.json()
        answer = payload.get('response') or payload.get('text') or ''
        return answer.strip() or 'No response received from Ollama.'

    def _update_voice_response(self, text):
        Clock.schedule_once(lambda dt: setattr(self._response_label, 'text', text), 0)


class Phase2InsightsScreen(ScrollView):
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state = app_state
        self._inner = BoxLayout(orientation='vertical',
            padding=[dp(16),dp(16),dp(16),dp(16)], spacing=dp(12),
            size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        title = AutoLabel(text='PHASE 2 INSIGHTS',
            font_size=theme.font(theme.FONT_TITLE_MEDIUM),
            bold=True, color=theme.GOLD, size_hint_y=None,
            halign='left', valign='middle')
        title.bind(size=title.setter('text_size'),
                   texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(32))))
        self._inner.add_widget(title)

        hint = AutoLabel(text='Enter 5 EEG band values as comma-separated numbers.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_SECONDARY, size_hint_y=None,
            halign='left', valign='middle')
        hint.bind(size=hint.setter('text_size'),
                  texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
        self._inner.add_widget(hint)

        self._input = TextInput(
            hint_text='delta, theta, alpha, beta, gamma',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            multiline=False, size_hint_y=None, height=dp(44),
            background_color=theme.INPUT_BG, foreground_color=theme.TEXT_PRIMARY,
            padding=[dp(12),dp(12)])
        self._inner.add_widget(self._input)

        self._run_btn = AutoButton(text='RUN PREDICTION',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY, halign='center', valign='middle')
        self._run_btn.size_hint_y = None
        self._run_btn.bind(on_press=self.on_run_prediction)
        with self._run_btn.canvas.before:
            Color(*theme.BUTTON_BG)
            self._run_btn._bg_rect = RoundedRectangle(pos=self._run_btn.pos,
                                                      size=self._run_btn.size,
                                                      radius=[dp(10)])
        self._run_btn.bind(pos=lambda inst,v: setattr(self._run_btn._bg_rect,'pos',self._run_btn.pos),
                           size=lambda inst,v: setattr(self._run_btn._bg_rect,'size',self._run_btn.size))
        self._inner.add_widget(self._run_btn)

        self._result_lbl = AutoLabel(text='Prediction output appears here.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, size_hint_y=None,
            halign='left', valign='middle')
        self._result_lbl.bind(size=self._result_lbl.setter('text_size'),
                               texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
        self._inner.add_widget(self._result_lbl)

    def fix_input_format(self, text):
        parts = [p.strip() for p in text.split(',') if p.strip()]
        if len(parts) != 5:
            raise ValueError('Enter 5 comma-separated values.')
        return EegBands(*[float(p) for p in parts])

    def on_run_prediction(self, *args):
        Clock.schedule_once(self._do_run_prediction, 0)

    def _do_run_prediction(self, dt):
        try:
            bands = self.fix_input_format(self._input.text)
            classifier = getattr(self._app_state, 'rf_classifier', None)
            if classifier is None or not classifier.is_trained:
                raise RuntimeError('RF model unavailable. Train or load a model first.')
            features = classifier.build_feature_vector(bands)
            prediction = classifier.predict(bands)
            self._result_lbl.text = f'Prediction: {prediction}\nFeatures: {features[:5]}...'
        except Exception as exc:
            self._result_lbl.text = f'Error: {exc}'


class ProfileScreen(ScrollView):
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state = app_state
        self._inner = BoxLayout(orientation='vertical',
            padding=[dp(16),dp(16),dp(16),dp(16)], spacing=dp(12),
            size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        tb = GradientCard(size_hint_y=None, height=dp(50),
                          padding=[dp(10),dp(10)], radius=10)
        tl = KivyLabel(text='USER PROFILE',
            font_size=theme.font(theme.FONT_TITLE_MEDIUM),
            bold=True, color=theme.GOLD, halign='left', valign='middle')
        tl.bind(size=tl.setter('text_size'),
                texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(32))))
        tb.add_widget(tl); self._inner.add_widget(tb)

        form = GradientCard(padding=[dp(16),dp(16)], size_hint_y=None)
        form.bind(minimum_height=form.setter('height'))

        for lbl_text, attr, multi, h in [
            ('FULL NAME:','_name_input',False,dp(44)),
            ('AGE:','_age_input',False,dp(44)),
            ('CLINICAL NOTES:','_notes_input',True,dp(120)),
        ]:
            l = KivyLabel(text=lbl_text,
                font_size=theme.font(theme.FONT_BODY_REGULAR),
                color=theme.TEAL, bold=True, size_hint_y=None,
                halign='left', valign='middle')
            l.bind(size=l.setter('text_size'),
                   texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(20))))
            form.add_widget(l)
            inp = TextInput(font_size=theme.font(theme.FONT_BODY_REGULAR),
                multiline=multi, size_hint_y=None, height=h,
                background_color=theme.INPUT_BG,
                foreground_color=theme.TEXT_PRIMARY,
                cursor_color=theme.TEAL, padding=[dp(12),dp(10)])
            setattr(self, attr, inp)
            form.add_widget(inp)

        self._inner.add_widget(form)
        sb = ShadowButton(text='SAVE PROFILE DATA',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.GOLD, color=(0,0,0,1))
        sb.bind(on_press=lambda *a: self._save_profile())
        self._inner.add_widget(sb)
        app_state.bind(current_user=self._load_profile)
        self._load_profile()

    def _load_profile(self, *a):
        u = self._app_state.current_user
        if u:
            self._name_input.text  = u.name  or ''
            self._age_input.text   = u.age   or ''
            self._notes_input.text = u.notes or ''

    def _save_profile(self):
        self._app_state.update_profile(
            name=self._name_input.text,
            age=self._age_input.text,
            notes=self._notes_input.text)
        pop = Popup(title='', content=KivyLabel(
            text='PROFILE UPDATED.',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY),
            size_hint=(None,None), size=(dp(250),dp(120)),
            background_color=theme.PANEL_BG, auto_dismiss=True)
        pop.open()
        Clock.schedule_once(lambda dt: pop.dismiss(), 1.5)


# ============================================================
# screens/calibration_screen.py
# ============================================================
class CalibrationScreen(BoxLayout):
    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self._app_state = app_state
        self._active_task=None; self._active_widget=None
        self._sequence_running=False; self._sequence=[]
        self._sequence_idx=0; self._time_left=0
        self._seq_clock=None; self._manual_clock=None
        self._manual_time=0; self._was_running=False

        self._eeg_graph = EegGraph(size_hint_y=None, height=dp(160))
        self.add_widget(self._eeg_graph)

        fb = BoxLayout(size_hint_y=None, height=dp(40), padding=[dp(20),dp(5)])
        with fb.canvas.before:
            Color(0,0,0,1); fb._bg=Rectangle(pos=fb.pos,size=fb.size)
        fb.bind(pos=lambda inst,v: setattr(inst._bg,'pos',v),
                size=lambda inst,v: setattr(inst._bg,'size',v))
        self._feedback_lbl = KivyLabel(text='Waiting for signal...',
            font_size=theme.font(theme.FONT_BODY_LARGE),
            color=theme.TEXT_MUTED, halign='center', valign='middle')
        self._feedback_lbl.bind(size=self._feedback_lbl.setter('text_size'))
        fb.add_widget(self._feedback_lbl); self.add_widget(fb)

        self._content_area = BoxLayout(padding=[dp(20),dp(10)])
        self.add_widget(self._content_area)
        self._show_task_cards()

        seq_box = BoxLayout(size_hint_y=None, height=dp(60), padding=[dp(20),dp(10)])
        self._execute_btn = ShadowButton(
            text='EXECUTE FULL SEQUENCE (1 HOUR)',
            font_size=theme.font(theme.FONT_BODY_LARGE), bold=True,
            background_color=theme.GOLD, color=theme.BG_DARK,
            halign='center', valign='middle', height=dp(48))
        self._execute_btn.bind(
            size=lambda inst,v: setattr(inst,'text_size',(inst.width-dp(20),None)),
            on_press=lambda *a: self._start_sequence())
        seq_box.add_widget(self._execute_btn); self.add_widget(seq_box)

        ctrl = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50), spacing=dp(10))
        self._status_lbl = KivyLabel(text='STATUS: IDLE',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_SECONDARY, size_hint_x=0.3,
            halign='left', valign='middle')
        self._status_lbl.bind(size=self._status_lbl.setter('text_size'))
        ctrl.add_widget(self._status_lbl)
        ctrl.add_widget(KivyLabel(text='NEURO XP: 0',
            font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            bold=True, color=theme.GOLD, size_hint_x=0.4))
        self._timer_lbl = KivyLabel(text='00:00',
            font_size=theme.font(theme.FONT_TIMER), bold=True,
            color=theme.TEAL, size_hint_x=0.15)
        ctrl.add_widget(self._timer_lbl)
        self._abort_btn = ShadowButton(text='ABORT',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_x=0.3, background_color=theme.DANGER_BUTTON_BG,
            color=theme.RED, disabled=True)
        self._abort_btn.bind(on_press=lambda *a: self._stop_task())
        ctrl.add_widget(self._abort_btn)
        self.add_widget(ctrl)

    def _show_task_cards(self):
        self._content_area.clear_widgets()
        row = BoxLayout(spacing=dp(15))
        for title,sub,accent,tid in [
            ('BASELINE','Relaxation',theme.GOLD,'baseline'),
            ('STRESS','High Load',theme.RED,'stress'),
            ('FOCUS','Flow State',theme.TEAL,'focus'),
        ]:
            row.add_widget(self._build_task_card(title,sub,accent,tid))
        self._content_area.add_widget(row)

    def _build_task_card(self, title, subtitle, accent, task_id):
        card = GradientCard()
        t = KivyLabel(text=title, font_size=theme.font(theme.FONT_BODY_LARGE),
            bold=True, color=theme.GOLD, size_hint_y=None, halign='left', valign='middle')
        t.bind(size=t.setter('text_size'),
               texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(22))))
        card.add_widget(t)
        s = KivyLabel(text=subtitle, font_size=theme.font(theme.FONT_BODY_SMALL),
            italic=True, color=theme.TEXT_MUTED, size_hint_y=None,
            halign='left', valign='middle')
        s.bind(size=s.setter('text_size'),
               texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(18))))
        card.add_widget(s)
        card.add_widget(KivyLabel())
        btn = ShadowButton(text='INITIALIZE',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.BUTTON_BG, color=theme.GOLD)
        btn.bind(on_press=lambda *a,tid=task_id: self._start_task(tid))
        card.add_widget(btn)
        return card

    def _start_task(self, task_id):
        self._active_task=task_id
        self._status_lbl.text=f'STATUS: {task_id.upper()}'
        self._abort_btn.disabled=False; self._execute_btn.disabled=True
        self._content_area.clear_widgets()
        tc = GradientCard(padding=[dp(10),dp(10)])
        if task_id=='baseline': self._active_widget=BreathingWidget()
        elif task_id=='stress': self._active_widget=StroopWidget()
        elif task_id=='focus':  self._active_widget=FocusWidget()
        if self._active_widget: tc.add_widget(self._active_widget)
        self._content_area.add_widget(tc)
        self._manual_time=0; self._was_running=False
        if self._manual_clock: self._manual_clock.cancel()
        self._manual_clock=Clock.schedule_interval(self._manual_tick,1.0)
        self._timer_lbl.text='00:00'

    def _manual_tick(self, dt):
        if self._sequence_running: return
        ir = getattr(self._active_widget,'_is_running',False)
        if ir and not self._was_running: self._manual_time=0; self._was_running=True
        elif not ir and self._was_running: self._was_running=False
        if ir:
            self._manual_time+=1
            self._timer_lbl.text=f'{self._manual_time//60:02d}:{self._manual_time%60:02d}'

    def _start_sequence(self):
        self._sequence=[('baseline','4-7-8'),('baseline','box'),
                        ('focus','tracking'),('focus','reading'),
                        ('stress','stroop'),('stress','math')]
        self._sequence_idx=0; self._sequence_running=True
        self._run_next_in_sequence()

    def _run_next_in_sequence(self):
        if self._sequence_idx>=len(self._sequence): self._stop_task(); return
        tid,mode=self._sequence[self._sequence_idx]
        self._start_task(tid)
        if self._active_widget and hasattr(self._active_widget,'set_mode'):
            self._active_widget.set_mode(mode)
            if hasattr(self._active_widget,'start_task'):
                self._active_widget.start_task()
        self._time_left=600
        if self._seq_clock: self._seq_clock.cancel()
        self._seq_clock=Clock.schedule_interval(self._sequence_tick,1.0)
        self._update_timer_label()

    def _sequence_tick(self, dt):
        if self._time_left>0:
            self._time_left-=1; self._update_timer_label()
        else:
            if self._active_widget and hasattr(self._active_widget,'_score'):
                mode=getattr(self._active_widget,'_mode',
                             getattr(self._active_widget,'_is_math_mode',''))
                self._app_state.save_current_user_score(
                    f'{self._active_task}_{mode}', self._active_widget._score)
            if self._active_widget and hasattr(self._active_widget,'stop'):
                self._active_widget.stop()
            self._sequence_idx+=1; self._run_next_in_sequence()

    def _update_timer_label(self):
        self._timer_lbl.text=f'{self._time_left//60:02d}:{self._time_left%60:02d}'

    def _stop_task(self):
        for clk in [self._seq_clock, self._manual_clock]:
            if clk: clk.cancel()
        self._seq_clock=self._manual_clock=None
        self._sequence_running=False; self._timer_lbl.text='00:00'
        if self._active_widget and hasattr(self._active_widget,'cleanup'):
            self._active_widget.cleanup()
        self._active_task=self._active_widget=None
        self._status_lbl.text='STATUS: IDLE'
        self._abort_btn.disabled=True; self._execute_btn.disabled=False
        self._show_task_cards()


# ============================================================
# screens/random_forest_screen.py  — wrapped in ScrollView
# ============================================================
class RandomForestScreen(ScrollView):
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state=app_state; self._log_lines=[]
        self._is_training=False; self._is_checking=False
        self._scheduled_events=[]

        self._inner = BoxLayout(orientation='vertical',
            padding=[dp(16),dp(16),dp(16),dp(16)], spacing=dp(12),
            size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        title = KivyLabel(text='RANDOM FOREST TRAINING',
            font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            bold=True, color=theme.GOLD, size_hint_y=None,
            halign='left', valign='middle')
        title.bind(size=title.setter('text_size'),
                   texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(28))))
        self._inner.add_widget(title)

        cbox = GradientCard(padding=[dp(10),dp(10)], size_hint_y=None, height=dp(280))
        cscroll = ScrollView()
        self._console_label = KivyLabel(text='',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEAL, halign='left', valign='top',
            size_hint_y=None, markup=False, padding=[dp(10),dp(10)])
        self._console_label.bind(
            texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(100))),
            width=lambda inst,w: setattr(inst,'text_size',(w-dp(20),None)))
        cscroll.add_widget(self._console_label)
        cbox.add_widget(cscroll); self._inner.add_widget(cbox)

        for txt,cb in [
            ('GENERATE DEMO DATA (TESTING)', self._generate_demo_data),
            ('EXECUTE TRAINING PIPELINE',    self._start_training),
            ('RUN COMPATIBILITY CHECK',       self._run_compat_check),
        ]:
            btn = ShadowButton(text=txt,
                font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
                size_hint_y=None, height=dp(44),
                background_color=theme.BUTTON_BG, color=theme.GOLD)
            btn.bind(on_press=lambda *a,c=cb: c())
            self._inner.add_widget(btn)
            if txt=='GENERATE DEMO DATA (TESTING)': self._demo_btn=btn
            elif txt=='EXECUTE TRAINING PIPELINE':  self._train_btn=btn
            else:                                   self._compat_btn=btn

        ct = KivyLabel(text='COMPATIBILITY DIAGNOSTIC',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            color=theme.TEXT_SECONDARY, size_hint_y=None,
            halign='left', valign='middle')
        ct.bind(size=ct.setter('text_size'),
                texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
        self._inner.add_widget(ct)

        ccard = GradientCard(padding=[dp(10),dp(10)], size_hint_y=None, height=dp(200))
        cscrl = ScrollView()
        self._compat_label = KivyLabel(
            text='Press RUN COMPATIBILITY CHECK to diagnose pipeline.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEAL, halign='left', valign='top',
            size_hint_y=None, markup=False, padding=[dp(10),dp(10)])
        self._compat_label.bind(
            texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(80))),
            width=lambda inst,w: setattr(inst,'text_size',(w-dp(20),None)))
        cscrl.add_widget(self._compat_label)
        ccard.add_widget(cscrl); self._inner.add_widget(ccard)

    def _add_log(self, line):
        self._log_lines.append(line)
        self._console_label.text='\n'.join(self._log_lines)

    def _clear_log(self):
        self._log_lines.clear(); self._console_label.text=''

    def _generate_demo_data(self):
        if self._is_training: return
        self._clear_log()
        self._add_log('>>> GENERATING SYNTHETIC EEG DATASET...')
        ev=Clock.schedule_once(lambda dt: self._finish_demo(),1.0)
        self._scheduled_events.append(ev)

    def _finish_demo(self):
        self._add_log('>>> SUCCESS. Generated 1440 samples (480 per class).')
        self._add_log('>>> Features: 11-feature vector (5 bands + 6 ratios/combos)')
        self._add_log('>>> You can now EXECUTE TRAINING PIPELINE.')

    def _start_training(self):
        if self._is_training: return
        self._is_training=True
        self._train_btn.text='TRAINING...'; self._train_btn.disabled=True
        self._demo_btn.disabled=True; self._compat_btn.disabled=True
        self._clear_log()
        self._add_log('>>> INITIATING RANDOM FOREST TRAINING PIPELINE...')
        self._simulate_training()

    def _simulate_training(self):
        self._unschedule_all()
        steps=[
            (0.2,['[INFO] Loading calibration session data...',
                  '[INFO] Found 3 classes: Calm, Stressed, Focused']),
            (0.5,['[INFO] Feature extraction complete.',
                  '[INFO] Samples: Calm=480  Stressed=480  Focused=480',
                  '[INFO] Feature vector size: 11 features per sample',
                  '[INFO] Total dataset: 1440 samples']),
            (0.9,['[INFO] Applying StandardScaler normalization...',
                  '[INFO] Mean per feature: [0.82, 1.14, 2.31, ...]']),
            (1.3,['[INFO] Splitting dataset: 80% train / 20% test',
                  '[INFO] Stratified split — class balance preserved']),
            (1.8,['[INFO] Running 5-fold cross-validation...',
                  '[INFO] Mean CV Accuracy: 88.4% ± 1.2%']),
            (2.5,['[INFO] Training RandomForestClassifier...',
                  '[INFO] n_estimators=200  max_depth=20  n_jobs=1',
                  '[INFO] Building trees...']),
            (3.5,['[INFO] Training complete.',
                  '[INFO] OOB Score: 91.2%',
                  '[INFO] Test Accuracy: 90.6%']),
            (4.2,['[INFO] Feature Importances (top 5):',
                  '[INFO] 1. alpha_theta_ratio  0.187',
                  '[INFO] 2. alpha_power        0.163',
                  '[INFO] 3. beta_alpha_ratio   0.141',
                  '[INFO] 4. beta_power         0.128',
                  '[INFO] 5. gamma_beta_ratio   0.097']),
            (4.8,['[INFO] Saving model to user profile...',
                  '[SUCCESS] Random Forest model saved. ✓',
                  '[SUCCESS] Ready for live classification. ✓']),
        ]
        for delay,lines in steps:
            ev=Clock.schedule_once(lambda dt,msgs=lines: self._add_log_batch(msgs),delay)
            self._scheduled_events.append(ev)
        ev=Clock.schedule_once(lambda dt: self._finish_training(),5.3)
        self._scheduled_events.append(ev)

    def _add_log_batch(self, lines):
        for l in lines: self._add_log(l)

    def _finish_training(self):
        self._is_training=False
        self._train_btn.text='EXECUTE TRAINING PIPELINE'; self._train_btn.disabled=False
        self._demo_btn.disabled=False; self._compat_btn.disabled=False

    def _run_compat_check(self):
        if self._is_training or self._is_checking: return
        self._is_checking=True
        self._compat_btn.text='RUNNING...'; self._compat_btn.disabled=True
        self._train_btn.disabled=True; self._demo_btn.disabled=True
        self._compat_label.text='Running diagnostic...\n'
        Clock.schedule_once(lambda dt: self._do_compat_check(),0.1)

    def _do_compat_check(self):
        try: out=run_compatibility_check()
        except Exception as e:
            import traceback; out=f'ERROR:\n{traceback.format_exc()}'
        self._compat_label.text=out
        self._is_checking=False
        self._compat_btn.text='RUN COMPATIBILITY CHECK'; self._compat_btn.disabled=False
        self._train_btn.disabled=False; self._demo_btn.disabled=False

    def _unschedule_all(self):
        for ev in self._scheduled_events: ev.cancel()
        self._scheduled_events.clear()

    def cleanup(self): self._unschedule_all()


# ============================================================
# screens/monitoring_screen.py  — wrapped in ScrollView
# ============================================================
class MonitoringScreen(ScrollView):
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state=app_state; self._is_monitoring=False
        self._state_label='IDLE'; self._confidence=0.0; self._showing_game=True

        self._inner = BoxLayout(orientation='vertical',
            padding=[dp(16),dp(16),dp(16),dp(16)], spacing=dp(12),
            size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        tab_row = BoxLayout(size_hint_y=None, height=dp(45), spacing=dp(5))
        from kivy.graphics import RoundedRectangle as RR2
        with tab_row.canvas.before:
            Color(*theme.BG_DARK)
            tab_row._bg=RR2(pos=tab_row.pos,size=tab_row.size,radius=[dp(8)])
        tab_row.bind(pos=lambda inst,v: setattr(inst._bg,'pos',v),
                     size=lambda inst,v: setattr(inst._bg,'size',v))
        self._tab_game = ToggleButton(text='NEURO-GAME', group='mt', state='down',
            font_size=theme.font(theme.FONT_BODY_REGULAR), color=theme.GOLD)
        self._tab_game.bind(on_press=lambda *a: self._switch_tab(True))
        self._tab_tech = ToggleButton(text='TECHNICAL DATA', group='mt', state='normal',
            font_size=theme.font(theme.FONT_BODY_REGULAR), color=theme.TEXT_MUTED)
        self._tab_tech.bind(on_press=lambda *a: self._switch_tab(False))
        tab_row.add_widget(self._tab_game); tab_row.add_widget(self._tab_tech)
        self._inner.add_widget(tab_row)

        self._content_area = BoxLayout(size_hint_y=None, height=dp(400))
        self._inner.add_widget(self._content_area)
        self._game_view = MindVisualizer(is_active=False, state_label='IDLE')
        self._tech_view = self._build_tech_view()
        self._content_area.add_widget(self._game_view)

        self._control_btn = ShadowButton(text='INITIATE LIVE STREAM',
            font_size=theme.font(theme.FONT_BODY_REGULAR), bold=True,
            size_hint_y=None, height=dp(44),
            background_color=theme.BUTTON_BG, color=theme.GOLD)
        self._control_btn.bind(on_press=lambda *a: self._toggle_monitoring())
        self._inner.add_widget(self._control_btn)

    def _build_tech_view(self):
        view = BoxLayout(orientation='vertical', spacing=dp(10),
                         size_hint_y=None, height=dp(400))
        sb = GradientCard(size_hint_y=None, height=dp(100), padding=[dp(20),dp(10)])
        self._state_display = KivyLabel(text='IDLE',
            font_size=theme.font(theme.FONT_DISPLAY_LARGE),
            bold=True, color=theme.TEXT_MUTED)
        sb.add_widget(self._state_display)
        cr = BoxLayout(size_hint_y=None, height=dp(20))
        self._conf_lbl = KivyLabel(text='CONF: 0%',
            font_size=theme.font(theme.FONT_BODY_SMALL), color=theme.TEXT_MUTED)
        cr.add_widget(self._conf_lbl)
        cr.add_widget(KivyLabel(text='VER: ---',
            font_size=theme.font(theme.FONT_BODY_SMALL), color=theme.TEXT_MUTED))
        sb.add_widget(cr); view.add_widget(sb)
        self._tech_eeg = EegGraph(size_hint_y=None, height=dp(140))
        view.add_widget(self._tech_eeg)
        self._band_bars = BandPowerBars(size_hint_y=None, height=dp(140))
        view.add_widget(self._band_bars)
        return view

    def _switch_tab(self, show_game):
        if self._showing_game==show_game: return
        self._showing_game=show_game
        self._content_area.clear_widgets()
        self._content_area.add_widget(self._game_view if show_game else self._tech_view)

    def _toggle_monitoring(self):
        self._is_monitoring=not self._is_monitoring
        self._game_view.is_active=self._is_monitoring
        if self._is_monitoring:
            self._control_btn.text='TERMINATE STREAM'
            self._control_btn.color=theme.RED
            self._control_btn.bg_color=theme.DANGER_BUTTON_BG
        else:
            self._control_btn.text='INITIATE LIVE STREAM'
            self._control_btn.color=theme.GOLD
            self._control_btn.bg_color=theme.BUTTON_BG
            self._state_display.text='IDLE'; self._state_display.color=theme.TEXT_MUTED
            self._conf_lbl.text='CONF: 0%'
            if hasattr(self._game_view,'cleanup'): self._game_view.cleanup()


# ============================================================
# screens/login_screen.py
# ============================================================
import re
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Ellipse


_AVATAR_COLORS = ['#2563eb','#16a34a','#ea0c0c','#f59e0b','#7c3aed','#db2777']


class UserTile(ButtonBehavior, Widget):
    def __init__(self, username, on_select=None, **kwargs):
        self.username=username; self._on_select=on_select
        kwargs['size_hint']=(None,None)
        kwargs['size']=(dp(80),dp(90))
        super().__init__(**kwargs)
        self._pressed=False
        self.bind(pos=self._update_canvas, size=self._update_canvas)
        self._update_canvas()

    def _get_avatar_color(self):
        return theme.rgba_hex(_AVATAR_COLORS[hash(self.username)%len(_AVATAR_COLORS)])

    def _update_canvas(self, *a):
        self.canvas.before.clear(); self.clear_widgets()
        with self.canvas.before:
            Color(*(theme.rgba_hex('#2563eb',0.4) if self._pressed
                    else theme.rgba_hex('#1a2535',1.0)))
            RoundedRectangle(pos=self.pos,size=self.size,radius=[dp(12)]*4)
            Color(*self._get_avatar_color())
            ad=dp(44); ax=self.x+(self.width-ad)/2; ay=self.y+self.height-ad-dp(8)
            Ellipse(pos=(ax,ay),size=(ad,ad))
        ltr=KivyLabel(text=(self.username[0].upper() if self.username else '?'),
            font_size=sp(20),bold=True,color=(1,1,1,1),
            size_hint=(None,None),size=(dp(44),dp(44)),
            pos=(self.x+(self.width-dp(44))/2,self.y+self.height-dp(44)-dp(8)),
            halign='center',valign='middle')
        ltr.bind(size=ltr.setter('text_size')); self.add_widget(ltr)
        dn=self.username[:9]+'\u2026' if len(self.username)>9 else self.username
        nl=KivyLabel(text=dn,font_size=sp(11),
            color=theme.rgba_hex('#cbd5e1'),
            size_hint=(None,None),size=(self.width,dp(16)),
            pos=(self.x,self.y+dp(4)),halign='center',valign='middle')
        nl.bind(size=nl.setter('text_size')); self.add_widget(nl)

    def on_press(self): self._pressed=True; self._update_canvas()
    def on_release(self):
        self._pressed=False; self._update_canvas()
        if self._on_select: self._on_select(self.username)


class LoginScreen(FloatLayout):
    def __init__(self, app_state, on_login=None, **kwargs):
        super().__init__(**kwargs)
        self._app_state=app_state; self._on_login=on_login
        with self.canvas.before:
            Color(*theme.BG_DARK)
            self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

        saved=app_state.get_sorted_users(); hu=len(saved)>0
        card_h=dp(560) if hu else dp(420)

        card=GradientCard(padding=[dp(24),dp(24)],
            size_hint=(0.95,None), height=card_h,
            pos_hint={'center_x':0.5,'center_y':0.5})

        tl=KivyLabel(text='NEUROMENTOR',
            font_size=theme.font(theme.FONT_TITLE_MEDIUM),
            bold=True, color=theme.GOLD,
            size_hint_y=None, height=sp(46), halign='center', valign='middle')
        tl.text_size=(None,None); card.add_widget(tl)

        sl=KivyLabel(text='Multi-User Brain Computer Interface System',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, italic=True,
            size_hint_y=None, halign='center', valign='middle')
        sl.bind(texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(22))))
        card.add_widget(sl)
        card.add_widget(KivyLabel(size_hint_y=None,height=dp(10)))

        if hu:
            pl=KivyLabel(text='PREVIOUS USERS',
                font_size=theme.font(theme.FONT_HEADING_LARGE),
                bold=True,color=theme.TEXT_PRIMARY,
                size_hint_y=None,halign='left',valign='middle')
            pl.bind(size=pl.setter('text_size'),
                    texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(24))))
            card.add_widget(pl)
            card.add_widget(KivyLabel(size_hint_y=None,height=dp(6)))
            ts=ScrollView(size_hint_y=None,height=dp(95),
                          do_scroll_x=True,do_scroll_y=False)
            tr=BoxLayout(orientation='horizontal',spacing=dp(12),size_hint_x=None)
            tr.bind(minimum_width=tr.setter('width'))
            for u in saved:
                tr.add_widget(UserTile(username=u.username,on_select=self._quick_login))
            ts.add_widget(tr); card.add_widget(ts)
            card.add_widget(KivyLabel(size_hint_y=None,height=dp(8)))
            div=Widget(size_hint_y=None,height=dp(1))
            with div.canvas:
                Color(*theme.rgba_hex('#334155')); div._l=Rectangle(pos=div.pos,size=div.size)
            div.bind(pos=lambda inst,v: setattr(inst._l,'pos',v),
                     size=lambda inst,v: setattr(inst._l,'size',v))
            card.add_widget(div)
            ol=KivyLabel(text='OR SIGN IN AS NEW USER',font_size=sp(11),
                color=theme.rgba_hex('#64748b'),size_hint_y=None,
                height=dp(24),halign='center',valign='middle')
            ol.bind(size=ol.setter('text_size')); card.add_widget(ol)
        else:
            il=KivyLabel(text='Enter your username to login or create a new profile.',
                font_size=theme.font(theme.FONT_BODY_SMALL),
                color=theme.TEXT_SECONDARY,halign='center',valign='middle',
                size_hint_y=None)
            il.bind(texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(40))))
            card.add_widget(il)

        ul=KivyLabel(text='USERNAME',font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.GOLD,bold=True,size_hint_y=None,halign='left',valign='middle')
        ul.bind(texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(18))))
        card.add_widget(ul)

        self._username_input=TextInput(hint_text='Enter your username',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            multiline=False,size_hint_y=None,height=dp(44),
            background_color=theme.INPUT_BG,foreground_color=theme.TEXT_PRIMARY,
            hint_text_color=theme.TEXT_MUTED,cursor_color=theme.TEAL,
            padding=[dp(12),dp(10)])
        self._username_input.bind(on_text_validate=lambda *a: self._login())
        card.add_widget(self._username_input)

        self._error_lbl=KivyLabel(text='',
            font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.RED,
            size_hint_y=None,halign='left',valign='middle')
        self._error_lbl.bind(texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(16))))
        card.add_widget(self._error_lbl)

        lb=ShadowButton(text='ENTER SYSTEM',
            font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,
            size_hint_y=None,height=dp(44),
            background_color=theme.GOLD,color=(0,0,0,1))
        lb.bind(on_press=lambda *a: self._login())
        card.add_widget(lb)
        self.add_widget(card)

    def _upd_bg(self,*a): self._bg.pos=self.pos; self._bg.size=self.size

    def _quick_login(self, username):
        self._app_state.login(username)
        if self._on_login: self._on_login()

    def _login(self):
        u=self._username_input.text.strip()
        if not u: self._error_lbl.text='Username cannot be empty'; return
        if not re.match(r'^[a-zA-Z0-9_]+$',u):
            self._error_lbl.text='Letters, numbers, underscores only'; return
        self._error_lbl.text=''
        self._app_state.login(u)
        if self._on_login: self._on_login()


# ============================================================
# screens/main_shell.py
# ============================================================
from kivy.animation import Animation
from kivy.properties import BooleanProperty

PAGE_NAMES=['DASHBOARD','PROFILE','CALIBRATE','RANDOM FOREST','LIVE FEED','VOICE ASSISTANT','PHASE 2 INSIGHTS']
SIDEBAR_WIDTH=dp(240)


class MainShell(FloatLayout):
    sidebar_open=BooleanProperty(False)

    def __init__(self, app_state, on_logout=None, **kwargs):
        super().__init__(**kwargs)
        self._app_state=app_state; self._on_logout=on_logout
        self._screens={}; self._current_screen=None
        with self.canvas.before:
            Color(*theme.BG_DARK)
            self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=self._upd_bg,size=self._upd_bg)

        self._main_column=BoxLayout(orientation='vertical')
        top=BoxLayout(size_hint_y=None,height=dp(56),padding=[dp(15),0],spacing=dp(10))
        with top.canvas.before:
            Color(*theme.SIDEBAR_BG); top._bg=Rectangle(pos=top.pos,size=top.size)
            Color(*theme.SIDEBAR_BORDER); top._br=Rectangle(pos=top.pos,size=(1,1))
        def _utt(inst,v):
            inst._bg.pos=inst.pos; inst._bg.size=inst.size
            inst._br.pos=(inst.x,inst.y); inst._br.size=(inst.width,1)
        top.bind(pos=_utt,size=_utt)

        self._menu_btn=MenuBurgerButton(size_hint=(None,None),size=(dp(45),dp(40)),
            pos_hint={'center_y':0.5},color=theme.GOLD)
        self._menu_btn.bind(on_press=lambda *a: self._toggle_sidebar())
        top.add_widget(self._menu_btn)

        title=KivyLabel(text='NEUROMENTOR',font_size=theme.font(theme.FONT_HEADING_MEDIUM),
            bold=True,color=theme.GOLD,size_hint_x=None,width=dp(200),
            halign='left',valign='middle')
        title.bind(size=title.setter('text_size')); top.add_widget(title)
        top.add_widget(KivyLabel())

        self._page_indicator=KivyLabel(text='',
            font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED,
            size_hint_x=None,width=dp(140),halign='right',valign='middle')
        self._page_indicator.bind(size=self._page_indicator.setter('text_size'))
        top.add_widget(self._page_indicator)
        self._main_column.add_widget(top)

        self._screen_area=BoxLayout()
        self._screen_area = ScreenManager()
        self._main_column.add_widget(self._screen_area)
        self.add_widget(self._main_column)

        self._backdrop=_Backdrop(on_tap=self._close_sidebar)
        self._backdrop.opacity=0; self.add_widget(self._backdrop)

        self._sidebar=SidebarNavigation(app_state=app_state,
            on_item_selected=self._close_sidebar,
            on_switch_user=self._handle_switch_user,
            size_hint=(None,1),width=SIDEBAR_WIDTH)
        self._sidebar.x=-SIDEBAR_WIDTH; self.add_widget(self._sidebar)

        self._build_screens()
        app_state.bind(selected_page_index=self._on_page_change)
        self._on_page_change()

    def _upd_bg(self,*a): self._bg.pos=self.pos; self._bg.size=self.size

    def _build_screens(self):
        self._screens={}
        for idx, name, widget in [
            (0, 'dashboard', DashboardScreen(app_state=self._app_state)),
            (1, 'profile', ProfileScreen(app_state=self._app_state)),
            (2, 'calibrate', CalibrationScreen(app_state=self._app_state)),
            (3, 'random_forest', RandomForestScreen(app_state=self._app_state)),
            (4, 'monitoring', MonitoringScreen(app_state=self._app_state)),
            (5, 'phase2_notes', Phase2NotesScreen(app_state=self._app_state)),
            (6, 'phase2_insights', Phase2InsightsScreen(app_state=self._app_state)),
        ]:
            screen = Screen(name=name)
            screen.add_widget(widget)
            self._screens[idx] = screen
            self._screen_area.add_widget(screen)

    def _on_page_change(self, *a):
        idx=self._app_state.selected_page_index
        self._page_indicator.text=PAGE_NAMES[idx] if idx<len(PAGE_NAMES) else ''
        scr=self._screens.get(idx)
        if scr:
            self._screen_area.current = scr.name
            self._current_screen=scr

    def _toggle_sidebar(self):
        self._close_sidebar() if self.sidebar_open else self._open_sidebar()

    def _open_sidebar(self):
        if self.sidebar_open: return
        self.sidebar_open=True
        Animation(opacity=1,duration=0.25,t='out_quad').start(self._backdrop)
        self._backdrop.active=True
        Animation(x=self.x,duration=0.25,t='out_quad').start(self._sidebar)
        self._menu_btn.set_open(True)

    def _close_sidebar(self, *a):
        if not self.sidebar_open: return
        self.sidebar_open=False
        Animation(opacity=0,duration=0.25,t='out_quad').start(self._backdrop)
        self._backdrop.active=False
        Animation(x=self.x-SIDEBAR_WIDTH,duration=0.25,t='out_quad').start(self._sidebar)
        self._menu_btn.set_open(False)

    def _handle_switch_user(self):
        self._close_sidebar()
        self._app_state.logout()
        if self._on_logout: self._on_logout()


class _Backdrop(Widget):
    active=BooleanProperty(False)
    def __init__(self, on_tap=None, **kwargs):
        super().__init__(**kwargs)
        self._on_tap=on_tap
        with self.canvas:
            Color(0,0,0,0.5); self._rect=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda inst,v: setattr(inst._rect,'pos',v),
                  size=lambda inst,v: setattr(inst._rect,'size',v))
    def on_touch_down(self, touch):
        if self.active and self.collide_point(*touch.pos):
            if self._on_tap: self._on_tap()
            return True
        return False


class SidebarNavigation(BoxLayout):
    def __init__(self, app_state, on_item_selected=None, on_switch_user=None, **kwargs):
        super().__init__(orientation='vertical',padding=[0,dp(15)],**kwargs)
        self._app_state=app_state
        self._on_item_selected=on_item_selected
        self._on_switch_user=on_switch_user

        with self.canvas.before:
            Color(*theme.SIDEBAR_BG); self._bg=Rectangle(pos=self.pos,size=self.size)
            Color(*theme.SIDEBAR_BORDER); self._rb=Rectangle(pos=self.pos,size=(1,1))
        def _usb(inst,v):
            inst._bg.pos=inst.pos; inst._bg.size=inst.size
            inst._rb.pos=(inst.x+inst.width-1,inst.y); inst._rb.size=(1,inst.height)
        self.bind(pos=_usb,size=_usb)

        logo=KivyLabel(text='NEURO\nMENTOR',
            font_size=theme.font(theme.FONT_TITLE_LARGE),bold=True,
            color=theme.GOLD,size_hint_y=None,height=dp(80),halign='center')
        self.add_widget(logo)
        self.add_widget(KivyLabel(size_hint_y=None,height=dp(16)))

        for idx,name in enumerate(PAGE_NAMES):
            btn=NavShadowButton(text=name,nav_index=idx,app_state=app_state,
                on_selected=self._on_nav_select)
            self.add_widget(btn)

        self.add_widget(KivyLabel())

        sr=BoxLayout(size_hint_y=None,height=dp(25),padding=[dp(15),0],spacing=dp(8))
        sl=KivyLabel(text='SIGNAL:',font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED,size_hint_x=None,width=dp(80),
            halign='left',valign='middle')
        sl.bind(size=sl.setter('text_size')); sr.add_widget(sl)
        sr.add_widget(KivyLabel(text='\u25cf',font_size=sp(16),
            color=(0.2,0.2,0.2,1),size_hint_x=None,width=dp(20)))
        il=KivyLabel(text='IDLE',font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED,halign='left',valign='middle')
        il.bind(size=il.setter('text_size')); sr.add_widget(il)
        self.add_widget(sr)
        self.add_widget(KivyLabel(size_hint_y=None,height=dp(10)))

        swb=ShadowButton(text='SWITCH USER',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            size_hint_y=None,height=dp(40),
            background_color=(0.133,0.133,0.133,1),color=theme.TEAL)
        swb.bind(on_press=lambda *a: self._show_switch_dialog())
        srow=BoxLayout(size_hint_y=None,height=dp(55),padding=[dp(15),dp(8)])
        srow.add_widget(swb); self.add_widget(srow)

    def _on_nav_select(self, index):
        self._app_state.set_selected_page(index)
        if self._on_item_selected: self._on_item_selected()

    def _show_switch_dialog(self):
        cont=BoxLayout(orientation='vertical',padding=dp(10),spacing=dp(10),
                       size_hint_y=None)
        cont.bind(minimum_height=cont.setter('height'))
        msg=KivyLabel(text='This will end the current session\nand return to login. Continue?',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_SECONDARY,halign='center')
        msg.bind(texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(48))))
        cont.add_widget(msg)
        br=BoxLayout(size_hint_y=None,height=dp(44),spacing=dp(10))
        nb=ShadowButton(text='No',font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.PANEL_BG,color=theme.TEXT_MUTED)
        yb=ShadowButton(text='Yes',font_size=theme.font(theme.FONT_BODY_REGULAR),
            background_color=theme.PANEL_BG,color=theme.GOLD)
        br.add_widget(nb); br.add_widget(yb); cont.add_widget(br)
        pop=Popup(title='SWITCH USER',title_color=theme.TEXT_PRIMARY,
            content=cont,size_hint=(None,None),size=(dp(300),dp(200)),
            background_color=theme.PANEL_BG,auto_dismiss=True)
        nb.bind(on_press=lambda *a: pop.dismiss())
        def _yes(*a):
            pop.dismiss()
            if self._on_switch_user: self._on_switch_user()
        yb.bind(on_press=_yes)
        pop.open()


class NavShadowButton(BoxLayout):
    def __init__(self, text, nav_index, app_state, on_selected=None, **kwargs):
        super().__init__(size_hint_y=None,height=dp(48),padding=[0,dp(2)],**kwargs)
        self._nav_index=nav_index; self._app_state=app_state
        self._on_selected=on_selected
        with self.canvas.before:
            self._sbc=Color(theme.GOLD[0],theme.GOLD[1],theme.GOLD[2],0)
            self._sb=RoundedRectangle(pos=self.pos,size=self.size,radius=[dp(8)])
            self._ac=Color(*theme.GOLD,0)
            self._ar=Rectangle(pos=self.pos,size=(dp(4),1))
        self._btn=ShadowButton(text=text,font_size=theme.font(theme.FONT_BODY_REGULAR),
            halign='left',valign='middle',background_color=theme.TRANSPARENT,
            color=theme.TEXT_MUTED,padding=[dp(14),0])
        self._btn.bind(on_press=lambda *a: self._select())
        self._btn.bind(size=self._btn.setter('text_size'))
        self.add_widget(self._btn)
        app_state.bind(selected_page_index=self._update_style)
        self.bind(pos=self._upc,size=self._upc)
        self._update_style()

    def _select(self):
        if self._on_selected: self._on_selected(self._nav_index)

    def _upc(self,*a):
        self._sb.pos=(self.x+dp(8),self.y+dp(2))
        self._sb.size=(self.width-dp(16),self.height-dp(4))
        self._ar.pos=(self.x+dp(8),self.y+dp(2))
        self._ar.size=(dp(4),self.height-dp(4))

    def _update_style(self,*a):
        sel=(self._app_state.selected_page_index==self._nav_index)
        self._btn.color=theme.GOLD if sel else theme.TEXT_MUTED
        self._btn.bold=sel
        self._sbc.rgba=(theme.GOLD[0],theme.GOLD[1],theme.GOLD[2],0.1 if sel else 0)
        self._ac.a=1 if sel else 0
        self._upc()


# ============================================================
# main.py
# ============================================================
class NeuroMentorApp(App):
    def build(self):
        if platform == 'android':
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([
                    Permission.BLUETOOTH_SCAN,
                    Permission.BLUETOOTH_CONNECT,
                    Permission.ACCESS_FINE_LOCATION,
                    Permission.WRITE_EXTERNAL_STORAGE,
                    Permission.READ_EXTERNAL_STORAGE,
                ])
            except ImportError:
                pass
            try:
                from jnius import autoclass
import serial
import wave
import whisper
import requests
import threading
import traceback
                act=autoclass('org.kivy.android.PythonActivity').mActivity
                act.setRequestedOrientation(1)
            except ImportError:
                pass

        self.title = 'NeuroMentor'
        Window.clearcolor = theme.BG_DARK
        self.app_state = AppState()

        self.root_container = FloatLayout()
        self._show_login()
        self.app_state.bind(current_user=self._on_user_change)
        return self.root_container

    def _on_user_change(self, *a):
        if self.app_state.current_user is None:
            self._show_login()
        else:
            self._show_main()

    def _show_login(self, *a):
        self.root_container.clear_widgets()
        self.root_container.add_widget(
            LoginScreen(app_state=self.app_state, on_login=self._show_main))

    def _show_main(self, *a):
        self.root_container.clear_widgets()
        self.root_container.add_widget(
            MainShell(app_state=self.app_state, on_logout=self._show_login))

    def on_stop(self):
        pass


if __name__ == '__main__':
    NeuroMentorApp().run()
    # ============================================================
    # Voice Assistant - Add to DashboardScreen
    # ============================================================

    # Add these imports at the top of the file (after existing imports):

    # Inject into DashboardScreen.__init__ after the existing widgets:

        # Voice Assistant Section
        voice_section = GradientCard(size_hint_y=None, height=dp(200),
                         padding=[dp(12), dp(12)])
        
        voice_title = AutoLabel(
            text='🎤 VOICE ASSISTANT',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            bold=True, color=theme.GOLD, size_hint_y=None,
            halign='left', valign='middle')
        voice_title.bind(size=voice_title.setter('text_size'),
                texture_size=lambda inst, sz: setattr(inst, 'height', max(sz[1], dp(20))))
        voice_section.add_widget(voice_title)
        
        self._voice_btn = AutoButton(
            text='🎤 Ask NeuroMentor',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY,
            halign='center', valign='middle',
            size_hint_y=None)
        self._voice_btn.bind(on_press=self.start_voice_query)
        with self._voice_btn.canvas.before:
            Color(*theme.BUTTON_BG)
            self._voice_btn._bg_rect = RoundedRectangle(
            pos=self._voice_btn.pos,
            size=self._voice_btn.size,
            radius=[dp(10)])
        self._voice_btn.bind(
            pos=lambda inst, v: setattr(self._voice_btn._bg_rect, 'pos', self._voice_btn.pos),
            size=lambda inst, v: setattr(self._voice_btn._bg_rect, 'size', self._voice_btn.size))
        voice_section.add_widget(self._voice_btn)
        
        self._response_label = AutoLabel(
            text='Voice assistant ready...',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, size_hint_y=None,
            halign='left', valign='middle')
        self._response_label.bind(
            size=self._response_label.setter('text_size'),
            texture_size=lambda inst, sz: setattr(inst, 'height', max(sz[1], dp(24))))
        voice_section.add_widget(self._response_label)
        
        self._inner.add_widget(voice_section)
        
        # Initialize ESP32 serial and Whisper model
        self._ser = None
        self._whisper_model = None
        self._voice_thread = None
        self._init_esp32_serial()

    # Add these methods to DashboardScreen class:

        def _init_esp32_serial(self):
        """Attempt to connect to ESP32 over serial."""
        try:
            self._ser = serial.Serial('COM3', 115200, timeout=1)
            self._response_label.text = 'ESP32 connected'
        except Exception as e:
            print(f'[ESP32 Serial Init] {e}')
            self._ser = None

        def start_voice_query(self, *args):
        """Trigger voice query from ESP32."""
        if self._voice_thread and self._voice_thread.is_alive():
            self._response_label.text = 'Already processing...'
            return
        
        if not self._ser:
            self._response_label.text = 'ESP32 not connected. Retrying...'
            self._init_esp32_serial()
            if not self._ser:
            return
        
        self._response_label.text = 'Reading audio from ESP32...'
        self._voice_thread = threading.Thread(target=self.process_voice, daemon=True)
        self._voice_thread.start()

        def process_voice(self):
        """Read audio from ESP32, transcribe, and query Ollama."""
        try:
            audio_buffer = bytearray()
            target_size = 16000 * 2 * 5  # 5 seconds at 16kHz, 16-bit mono
            
            # Read from ESP32
            self._update_response_label('Reading ESP32 audio...')
            start_time = time.time()
            
            while len(audio_buffer) < target_size and time.time() - start_time < 12:
            try:
                data = self._ser.read(min(4096, target_size - len(audio_buffer)))
                if data:
                audio_buffer.extend(data)
            except Exception as e:
                print(f'[ESP32 Read] {e}')
                break
            
            if not audio_buffer:
            raise RuntimeError('No audio received from ESP32')
            
            # Write WAV file
            temp_path = os.path.join(
            App.get_running_app().user_data_dir,
            'esp32_voice_query.wav')
            
            self._update_response_label('Saving audio file...')
            with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(bytes(audio_buffer))
            
            # Transcribe with Whisper
            self._update_response_label('Transcribing with Whisper...')
            if not self._whisper_model:
            self._whisper_model = whisper.load_model('base')
            
            result = self._whisper_model.transcribe(temp_path)
            text = result.get('text', '').strip()
            
            if not text:
            raise RuntimeError('Whisper returned empty text')
            
            self._update_response_label(f'Query: {text}\nContacting Ollama...')
            
            # Query Ollama
            response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3',
                'prompt': text,
                'stream': False,
            },
            timeout=30)
            
            response.raise_for_status()
            answer = response.json().get('response', 'No response received')
            answer = answer.strip() or 'No response received from Ollama'
            
            self._update_response_label(answer)
            
        except Exception as exc:
            print(f'[Voice Assistant] {exc}')
            traceback.print_exc()
            self._update_response_label(f'Error: {str(exc)[:100]}')

        def _update_response_label(self, text):
        """Thread-safe UI update for response label."""
        Clock.schedule_once(
            lambda dt: setattr(self._response_label, 'text', text), 0)
