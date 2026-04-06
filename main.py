"""
Unified NeuroMentor App
"""

# ============================================================
# File: theme.py
# ============================================================

class theme:
    """
    NeuroMentor Theme Configuration
    Charcoal / Gold / Amber palette.
    All colors are RGBA tuples (0-1 range) for Kivy.
    """


    # ============================================================
    # COLOR PALETTE
    # ============================================================

    # Primary accent - Gold Yellow
    GOLD = (0.910, 0.753, 0.180, 1)          # #e8c02e

    # Secondary accent - Lighter Gold
    TEAL = (0.933, 0.800, 0.310, 1)          # #eecc4f

    # Danger/Stress - Amber Orange
    RED = (0.863, 0.608, 0.157, 1)           # #dc9b28

    # Backgrounds
    BG_DARK = (0.200, 0.180, 0.235, 1)       # #332e3c
    PANEL_BG = (0.235, 0.216, 0.275, 1)      # #3c3746
    CARD_BG = (0.275, 0.255, 0.318, 1)       # #464151

    # Borders
    BORDER_DARK = (0.310, 0.290, 0.357, 1)   # #4f4a5b
    BORDER_LIGHT = (0.357, 0.337, 0.400, 1)  # #5b5666

    # Text colors
    TEXT_PRIMARY = (0.910, 0.902, 0.925, 1)   # #e8e6ec
    TEXT_SECONDARY = (0.753, 0.737, 0.773, 1) # #c0bcc5
    TEXT_MUTED = (0.475, 0.455, 0.502, 1)     # #797480

    # Input
    INPUT_BG = (0.173, 0.157, 0.208, 1)      # #2c2835
    INPUT_BORDER = (0.357, 0.337, 0.400, 1)   # #5b5666

    # Additional UI colors
    SIDEBAR_BG = (0.180, 0.161, 0.212, 1)    # #2e2936
    SIDEBAR_BORDER = (0.310, 0.290, 0.357, 1) # #4f4a5b
    DARK_CARD = (0.157, 0.141, 0.188, 1)     # #282430
    BUTTON_BG = (0.275, 0.255, 0.318, 1)     # #464151
    DANGER_BUTTON_BG = (0.250, 0.200, 0.100, 1) # #40331a

    # Transparent
    TRANSPARENT = (0, 0, 0, 0)


    # ============================================================
    # FONT SIZES
    # ============================================================

    FONT_TITLE_LARGE = 28
    FONT_TITLE_MEDIUM = 22
    FONT_HEADING_LARGE = 24
    FONT_HEADING_MEDIUM = 18
    FONT_BODY_LARGE = 16
    FONT_BODY_REGULAR = 14
    FONT_BODY_SMALL = 12
    FONT_DISPLAY_LARGE = 72
    FONT_TIMER = 24


    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def rgba_hex(hex_color, alpha=1.0):
        """Convert hex color string to RGBA tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b, alpha)


    @staticmethod
    def with_alpha(color, alpha):
        """Return a color tuple with modified alpha."""
        return (color[0], color[1], color[2], alpha)


# ============================================================
# File: services/rf_classifier.py
# ============================================================

"""
Random Forest Classifier for EEG-based mental state classification.
Wraps a pre-trained scikit-learn RandomForestClassifier with EEG band power features.

The pre-trained model expects 11 features:
  Delta, Theta, Alpha, Beta, Gamma,
  beta_alpha, alpha_theta, beta_theta, gamma_beta,
  beta_minus_alpha, alpha_plus_theta

Labels: Baseline=0, Focused=1, Stressed=2  (from LabelEncoder)
"""
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
    """EEG frequency band power values from a single FFT window."""
    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0


FEATURE_NAMES = [
    'delta', 'theta', 'alpha', 'beta', 'gamma',
    'beta_alpha_ratio', 'alpha_theta_ratio',
    'beta_theta_ratio', 'gamma_beta_ratio',
    'beta_minus_alpha', 'alpha_plus_theta',
]


class RFClassifier:
    """Random Forest classifier for EEG mental state prediction.

    Classifies EEG band power features into three states:
      Baseline (Calm), Stressed, Focused

    Uses an 11-feature vector extracted from EegBands objects,
    matching the training script's feature engineering.
    """

    def __init__(self):
        self._clf = None          # RandomForestClassifier
        self._scaler = None       # StandardScaler
        self._encoder = None      # LabelEncoder
        self._is_trained: bool = False
        self._feature_names: list = list(FEATURE_NAMES)

    # ----------------------------------------------------------
    # Feature extraction — matches training script exactly
    # ----------------------------------------------------------

    def build_feature_vector(self, bands: EegBands) -> list:
        """Extract all 11 features from an EegBands object.

        Feature engineering matches rf_model_training.py:
          5 raw bands + 4 ratios + 2 arithmetic combinations

        Returns:
            list of 11 floats
        """
        d, t, a, b, g = bands.delta, bands.theta, bands.alpha, bands.beta, bands.gamma

        beta_alpha = b / (a + 1e-9)
        alpha_theta = a / (t + 1e-9)
        beta_theta = b / (t + 1e-9)
        gamma_beta = g / (b + 1e-9)
        beta_minus_alpha = b - a
        alpha_plus_theta = a + t

        return [
            d, t, a, b, g,
            beta_alpha, alpha_theta,
            beta_theta, gamma_beta,
            beta_minus_alpha, alpha_plus_theta,
        ]

    # ----------------------------------------------------------
    # Training (for future re-training from app)
    # ----------------------------------------------------------

    def train(self, session_bands: list, session_labels: list) -> dict:
        """Train the Random Forest on labelled EEG sessions.

        Args:
            session_bands: list of EegBands objects
            session_labels: list of string labels ('Baseline', 'Stressed', 'Focused')

        Returns:
            dict with training metrics
        """
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn / joblib is not installed")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import cross_val_score, train_test_split

        # Build feature matrix
        X = np.array([self.build_feature_vector(b) for b in session_bands])
        y = np.array(session_labels)

        # Encode labels
        self._encoder = LabelEncoder()
        y_encoded = self._encoder.fit_transform(y)

        # Fit scaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Cross-validation
        clf_cv = RandomForestClassifier(
            n_estimators=700, max_depth=None, min_samples_split=3,
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        )
        cv_scores = cross_val_score(clf_cv, X_scaled, y_encoded, cv=5, scoring='accuracy')

        # Final training
        self._clf = RandomForestClassifier(
            n_estimators=700, max_depth=None, min_samples_split=3,
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        )
        self._clf.fit(X_scaled, y_encoded)
        self._is_trained = True

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {str(u): int(c) for u, c in zip(unique, counts)}

        return {
            'cv_accuracy': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'feature_importances': dict(zip(self._feature_names, self._clf.feature_importances_.tolist())),
            'n_samples': len(y),
            'class_distribution': class_dist,
        }

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def predict(self, bands: EegBands) -> str:
        """Predict mental state label from a single EEG window.

        Returns:
            str label ('Baseline', 'Stressed', 'Focused') or 'Unknown'
        """
        if not self._is_trained or self._clf is None or self._scaler is None:
            return 'Unknown'

        fv = self.build_feature_vector(bands)
        if _HAS_SKLEARN:
            X = np.array([fv])
            X_scaled = self._scaler.transform(X)
            pred = self._clf.predict(X_scaled)[0]
            if self._encoder is not None:
                return str(self._encoder.inverse_transform([pred])[0])
            return str(pred)
        return 'Unknown'

    def predict_proba(self, bands: EegBands) -> dict:
        """Predict class probabilities from a single EEG window.

        Returns:
            dict e.g. {'Baseline': 0.7, 'Stressed': 0.1, 'Focused': 0.2}
        """
        if not self._is_trained or self._clf is None or self._scaler is None:
            return {}

        fv = self.build_feature_vector(bands)
        if _HAS_SKLEARN:
            X = np.array([fv])
            X_scaled = self._scaler.transform(X)
            proba = self._clf.predict_proba(X_scaled)[0]
            if self._encoder is not None:
                labels = self._encoder.inverse_transform(range(len(proba)))
                return {str(l): float(p) for l, p in zip(labels, proba)}
            return {str(i): float(p) for i, p in enumerate(proba)}
        return {}

    # ----------------------------------------------------------
    # Persistence — uses joblib (matches training script)
    # ----------------------------------------------------------

    def save(self, dirpath: str):
        """Save trained model, scaler, and encoder to directory via joblib."""
        if not _HAS_SKLEARN:
            raise RuntimeError("joblib is not installed")
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self._clf, os.path.join(dirpath, 'rf_eeg_model.pkl'))
        joblib.dump(self._scaler, os.path.join(dirpath, 'rf_scaler.pkl'))
        if self._encoder is not None:
            joblib.dump(self._encoder, os.path.join(dirpath, 'rf_encoder.pkl'))

    def load(self, dirpath: str) -> bool:
        """Load model, scaler, and encoder from a directory of joblib pkl files.

        Expects:
          dirpath/rf_eeg_model.pkl
          dirpath/rf_scaler.pkl
          dirpath/rf_encoder.pkl

        Returns:
            True if loaded successfully.
        """
        if not _HAS_SKLEARN:
            print("[RFClassifier] joblib/numpy not available — cannot load model")
            return False

        model_path = os.path.join(dirpath, 'rf_eeg_model.pkl')
        scaler_path = os.path.join(dirpath, 'rf_scaler.pkl')
        encoder_path = os.path.join(dirpath, 'rf_encoder.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"[RFClassifier] Model files not found in {dirpath}")
            return False

        try:
            self._clf = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            if os.path.exists(encoder_path):
                self._encoder = joblib.load(encoder_path)
            self._is_trained = True
            print(f"[RFClassifier] Model loaded successfully from {dirpath}")
            return True
        except Exception as e:
            print(f"[RFClassifier] Failed to load model: {e}")
            self._is_trained = False
            return False

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# ============================================================
# File: tools/compatibility_check.py
# ============================================================

"""
NeuroMentor — Pipeline × Model Compatibility Diagnostic
========================================================
Standalone tool that verifies the EXG sensor data pipeline
is fully aligned with the saved Random Forest model.

Run standalone:
    python -m tools.compatibility_check

Or call from app code:
    output_text = run_compatibility_check()
"""
import os
import sys
import math
import glob
import traceback
from io import StringIO

# ---------------------------------------------------------------------------
# Ensure project root is on path so we can import services/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Import project modules (read-only — never modify them)
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# ═══════════════════════════════════════════════════════════════════════════
# DSP PIPELINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
# These mirror the standard NeuroMentor EXG pipeline.  Because
# dsp_pipeline.py does not yet exist in the Kivy codebase, we define
# the canonical values here so every check is self-contained.
# When dsp_pipeline.py is eventually added these MUST be kept in sync.
# ═══════════════════════════════════════════════════════════════════════════

# ADC conversion — ADS1299 defaults
ADC_BITS       = 24
VREF           = 4.5         # Volts
PGA_GAIN       = 24
ADC_RESOLUTION = (2 ** (ADC_BITS - 1)) - 1  # 8388607

# Sampling & windowing
SAMPLE_RATE    = 250         # Hz
WINDOW_SIZE    = 256         # samples
OVERLAP        = 128         # samples

# Standard clinical EEG bands (Hz)
BAND_BOUNDARIES = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

# Expected classifier labels
EXPECTED_LABELS = ['Baseline', 'Focused', 'Stressed']

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
_pass = 0
_fail = 0
_warn = 0
_fail_details: list = []
_buf = StringIO()


def _p(msg: str = ''):
    """Print to both stdout and buffer."""
    print(msg)
    _buf.write(msg + '\n')


def _PASS(msg: str):
    global _pass
    _pass += 1
    _p(f'  ✓ PASS   {msg}')


def _FAIL(msg: str):
    global _fail
    _fail += 1
    _fail_details.append(msg)
    _p(f'  ✗ FAIL   {msg}')


def _WARN(msg: str):
    global _warn
    _warn += 1
    _p(f'  ⚠ WARN   {msg}')


def _header(title: str):
    _p('')
    _p('─' * 60)
    _p(f'  {title}')
    _p('─' * 60)


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 1: ADC CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def _check_adc_conversion():
    _header('CHECK 1 — ADC CONVERSION')

    scale = (VREF / ADC_RESOLUTION / PGA_GAIN) * 1e6  # µV per count

    full_scale_uv = ADC_RESOLUTION * scale
    mid_scale_uv  = (ADC_RESOLUTION // 2) * scale

    _p(f'  ADC bits          : {ADC_BITS}')
    _p(f'  VREF              : {VREF} V')
    _p(f'  PGA gain          : {PGA_GAIN}')
    _p(f'  ADC resolution    : {ADC_RESOLUTION}')
    _p(f'  µV per count      : {scale:.6f}')
    _p(f'  Full-scale output : {full_scale_uv:.2f} µV')
    _p(f'  Mid-scale output  : {mid_scale_uv:.2f} µV')

    # Full-scale must be within absolute EEG range (1 – 500 µV is generous;
    # the ADC full-scale will be much larger — that is expected because the
    # ADC can represent larger signals.  What matters is that the *scale
    # factor* is correct, i.e. a typical 50 µV scalp signal uses a
    # meaningful portion of the ADC range.)
    if 100 < full_scale_uv < 300_000:
        _PASS(f'Full-scale {full_scale_uv:.2f} µV is within sensor range')
    else:
        _FAIL(f'Full-scale {full_scale_uv:.2f} µV is outside expected sensor range')

    # Mid-scale should be within a broadly reasonable range
    if 50 < mid_scale_uv < 150_000:
        _PASS(f'Mid-scale {mid_scale_uv:.2f} µV is within reasonable range')
    else:
        _FAIL(f'Mid-scale {mid_scale_uv:.2f} µV is outside expected range')

    # Check that 50 µV (typical scalp EEG) maps to a sensible ADC count
    counts_for_50uv = 50.0 / scale
    if counts_for_50uv >= 1:
        _PASS(f'50 µV signal → {counts_for_50uv:.1f} ADC counts (resolvable)')
    else:
        _FAIL(f'50 µV signal → {counts_for_50uv:.4f} ADC counts (too few — cannot resolve)')


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 2: SAMPLE RATE AND WINDOW
# ═══════════════════════════════════════════════════════════════════════════

def _check_sample_rate_and_window():
    _header('CHECK 2 — SAMPLE RATE & WINDOW')

    nyquist = SAMPLE_RATE / 2.0
    freq_res = SAMPLE_RATE / WINDOW_SIZE  # Hz per bin
    highest_band = max(hi for _, hi in BAND_BOUNDARIES.values())

    _p(f'  Sample rate       : {SAMPLE_RATE} Hz')
    _p(f'  Window size       : {WINDOW_SIZE} samples')
    _p(f'  Overlap           : {OVERLAP} samples')
    _p(f'  Nyquist frequency : {nyquist} Hz')
    _p(f'  FFT resolution    : {freq_res:.4f} Hz/bin')
    _p(f'  Highest band edge : {highest_band} Hz')

    # Nyquist covers highest band
    if nyquist >= highest_band:
        _PASS(f'Nyquist {nyquist} Hz ≥ highest band edge {highest_band} Hz')
    else:
        _FAIL(f'Nyquist {nyquist} Hz < highest band edge {highest_band} Hz — aliasing!')

    # Frequency resolution fine enough to separate adjacent boundaries
    all_edges = sorted(set(
        edge for lo, hi in BAND_BOUNDARIES.values() for edge in (lo, hi)
    ))
    min_gap = min(b - a for a, b in zip(all_edges, all_edges[1:]))
    if freq_res <= min_gap:
        _PASS(f'Freq resolution {freq_res:.4f} Hz ≤ min band gap {min_gap} Hz')
    else:
        _FAIL(f'Freq resolution {freq_res:.4f} Hz > min band gap {min_gap} Hz — bins fall between bands')

    # Overlap < window
    if OVERLAP < WINDOW_SIZE:
        _PASS(f'Overlap {OVERLAP} < window size {WINDOW_SIZE}')
    else:
        _FAIL(f'Overlap {OVERLAP} ≥ window size {WINDOW_SIZE}')


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 3: BAND BOUNDARY ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def _check_band_boundaries():
    _header('CHECK 3 — BAND BOUNDARY ALIGNMENT')

    freq_res = SAMPLE_RATE / WINDOW_SIZE
    freqs = [i * freq_res for i in range(WINDOW_SIZE // 2 + 1)]

    for name, (lo, hi) in BAND_BOUNDARIES.items():
        bins_in_band = [f for f in freqs if lo <= f < hi]
        n_bins = len(bins_in_band)

        # Check if lo and hi land on (or very near) an FFT bin edge
        lo_snap = min(freqs, key=lambda f: abs(f - lo))
        hi_snap = min(freqs, key=lambda f: abs(f - hi))
        lo_err = abs(lo_snap - lo)
        hi_err = abs(hi_snap - hi)

        aligned = lo_err < freq_res / 2 and hi_err < freq_res / 2

        if aligned:
            _PASS(f'{name:6s}  {lo:5.1f}–{hi:5.1f} Hz  bins={n_bins}  (aligned)')
        else:
            _WARN(f'{name:6s}  {lo:5.1f}–{hi:5.1f} Hz  bins={n_bins}  (lo_err={lo_err:.3f}, hi_err={hi_err:.3f})')

        if n_bins < 4:
            _WARN(f'{name:6s}  only {n_bins} bins — power estimate unreliable')


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 4: SYNTHETIC SIGNAL END-TO-END
# ═══════════════════════════════════════════════════════════════════════════

def _compute_band_powers_fft(signal, sample_rate, window_size):
    """Compute band powers from a signal using the same FFT approach the
    live pipeline will use.  Returns dict of band→power."""
    if not _HAS_NUMPY:
        return {}

    # Hann window + FFT
    window = np.hanning(window_size)
    windowed = signal[:window_size] * window
    spectrum = np.fft.rfft(windowed)
    psd = (np.abs(spectrum) ** 2) / window_size
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)

    powers = {}
    for name, (lo, hi) in BAND_BOUNDARIES.items():
        mask = (freqs >= lo) & (freqs < hi)
        powers[name] = float(np.sum(psd[mask]))

    return powers


def _check_synthetic_signals():
    _header('CHECK 4 — SYNTHETIC SIGNAL END-TO-END')

    if not _HAS_NUMPY:
        _FAIL('numpy not available — cannot run synthetic signal tests')
        return

    test_cases = [
        (2,  'delta'),
        (6,  'theta'),
        (10, 'alpha'),
        (20, 'beta'),
        (40, 'gamma'),
    ]

    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    all_ok = True

    for freq_hz, expected_band in test_cases:
        signal = np.sin(2 * np.pi * freq_hz * t)
        powers = _compute_band_powers_fft(signal, SAMPLE_RATE, WINDOW_SIZE)

        dominant = max(powers, key=powers.get)
        vals = '  '.join(f'{k}={v:.4f}' for k, v in powers.items())

        if dominant == expected_band:
            _PASS(f'{freq_hz:2d} Hz → dominates {dominant:6s}   [{vals}]')
        else:
            _FAIL(f'{freq_hz:2d} Hz → expected {expected_band}, got {dominant}   [{vals}]')
            all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 5: FEATURE VECTOR CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

def _check_feature_vector():
    _header('CHECK 5 — FEATURE VECTOR CONSISTENCY')

    if not _HAS_NUMPY:
        _FAIL('numpy not available — cannot build feature vector')
        return None

    # ── Validate FEATURE_NAMES constant itself ──
    EXPECTED_FEATURE_NAMES = [
        'delta', 'theta', 'alpha', 'beta', 'gamma',
        'beta_alpha_ratio', 'alpha_theta_ratio',
        'beta_theta_ratio', 'gamma_beta_ratio',
        'beta_minus_alpha', 'alpha_plus_theta',
    ]
    EXPECTED_N_FEATURES = 11

    if len(FEATURE_NAMES) == EXPECTED_N_FEATURES:
        _PASS(f'FEATURE_NAMES has exactly {EXPECTED_N_FEATURES} entries')
    else:
        _FAIL(f'FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected {EXPECTED_N_FEATURES}')

    if list(FEATURE_NAMES) == EXPECTED_FEATURE_NAMES:
        _PASS('FEATURE_NAMES matches canonical order and spelling')
    else:
        _FAIL('FEATURE_NAMES does not match canonical list')
        for i, (got, exp) in enumerate(zip(FEATURE_NAMES, EXPECTED_FEATURE_NAMES)):
            if got != exp:
                _p(f'    [{i}] got "{got}", expected "{exp}"')

    # ── Build a feature vector from a 10 Hz alpha sine ──
    t = np.arange(WINDOW_SIZE) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * 10 * t)
    powers = _compute_band_powers_fft(signal, SAMPLE_RATE, WINDOW_SIZE)

    bands = EegBands(
        delta=powers.get('delta', 0.0),
        theta=powers.get('theta', 0.0),
        alpha=powers.get('alpha', 0.0),
        beta=powers.get('beta', 0.0),
        gamma=powers.get('gamma', 0.0),
    )

    clf = RFClassifier()
    fv = clf.build_feature_vector(bands)

    _p(f'  Expected feature count : {EXPECTED_N_FEATURES}')
    _p(f'  Actual feature count   : {len(fv)}')

    if len(fv) == EXPECTED_N_FEATURES:
        _PASS(f'Feature vector length {len(fv)} == {EXPECTED_N_FEATURES}')
    else:
        _FAIL(f'Feature vector length {len(fv)} ≠ {EXPECTED_N_FEATURES}')

    has_nan = any(math.isnan(v) for v in fv)
    has_inf = any(math.isinf(v) for v in fv)
    all_finite = all(math.isfinite(v) for v in fv)

    if not has_nan:
        _PASS('No NaN values in feature vector')
    else:
        _FAIL('Feature vector contains NaN values')

    if not has_inf:
        _PASS('No Inf values in feature vector')
    else:
        _FAIL('Feature vector contains Inf values')

    if all_finite:
        _PASS('All values are finite')
    else:
        _FAIL('Feature vector contains non-finite values')

    # Report feature values with position labels
    _p('')
    _p('  Feature vector values:')
    for i, val in enumerate(fv):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'idx_{i}'
        sign = '  ' if val >= 0 else ''
        _p(f'    [{i:2d}] {name:22s} = {sign}{val:.8f}')

    return fv


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 6: SAVED MODEL FILE
# ═══════════════════════════════════════════════════════════════════════════

def _find_model_directories():
    """Find all model directories (user-specific and bundled)."""
    data_dir = os.path.join(_PROJECT_ROOT, 'neuromentor_data')
    bundled_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, '..', 'rf_model', 'rf_model'))

    dirs = []

    # User-specific model dirs
    if os.path.isdir(data_dir):
        for entry in os.listdir(data_dir):
            full = os.path.join(data_dir, entry)
            if os.path.isdir(full) and entry.endswith('_rf_model'):
                model_file = os.path.join(full, 'rf_eeg_model.pkl')
                if os.path.exists(model_file):
                    dirs.append(('user', entry, full))

    # Bundled model
    bundled_model = os.path.join(bundled_dir, 'rf_eeg_model.pkl')
    if os.path.exists(bundled_model):
        dirs.append(('bundled', 'rf_model', bundled_dir))

    return dirs


def _check_saved_model(feature_vector):
    _header('CHECK 6 — SAVED MODEL FILES')

    if not _HAS_NUMPY or not _HAS_JOBLIB:
        _FAIL('numpy/joblib not available — cannot inspect model files')
        return

    model_dirs = _find_model_directories()

    if not model_dirs:
        _WARN('No model files found.')
        _p('  → Complete a calibration session, then re-run this check.')
        _p('  → Expected locations:')
        _p(f'     neuromentor_data/<user>_rf_model/rf_eeg_model.pkl')
        _p(f'     ../rf_model/rf_model/rf_eeg_model.pkl')
        return

    expected_n_features = 11  # hardcoded — must match the model's training format

    for source, label, dirpath in model_dirs:
        _p('')
        _p(f'  ── Model: {label} ({source}) ──')
        _p(f'  Path: {dirpath}')

        model_path   = os.path.join(dirpath, 'rf_eeg_model.pkl')
        scaler_path  = os.path.join(dirpath, 'rf_scaler.pkl')
        encoder_path = os.path.join(dirpath, 'rf_encoder.pkl')

        # Load model
        try:
            clf = joblib.load(model_path)
            _PASS(f'rf_eeg_model.pkl loads without error')
        except Exception as e:
            _FAIL(f'rf_eeg_model.pkl failed to load: {e}')
            continue

        # Load scaler
        try:
            scaler = joblib.load(scaler_path)
            _PASS(f'rf_scaler.pkl loads without error')
        except Exception as e:
            _FAIL(f'rf_scaler.pkl failed to load: {e}')
            continue

        # Load encoder (optional but expected)
        encoder = None
        if os.path.exists(encoder_path):
            try:
                encoder = joblib.load(encoder_path)
                _PASS(f'rf_encoder.pkl loads without error')
            except Exception as e:
                _WARN(f'rf_encoder.pkl failed to load: {e}')

        # Check clf.n_features_in_
        clf_n = getattr(clf, 'n_features_in_', None)
        if clf_n is not None:
            if clf_n == expected_n_features:
                _PASS(f'clf.n_features_in_ = {clf_n} matches feature vector ({expected_n_features})')
            else:
                _FAIL(f'clf.n_features_in_ = {clf_n} ≠ feature vector ({expected_n_features})')
        else:
            _WARN('clf.n_features_in_ not available')

        # Check scaler.n_features_in_
        scaler_n = getattr(scaler, 'n_features_in_', None)
        if scaler_n is not None:
            if scaler_n == expected_n_features:
                _PASS(f'scaler.n_features_in_ = {scaler_n} matches feature vector ({expected_n_features})')
            else:
                _FAIL(f'scaler.n_features_in_ = {scaler_n} ≠ feature vector ({expected_n_features})')
        else:
            _WARN('scaler.n_features_in_ not available')

        # Check classes
        clf_classes = getattr(clf, 'classes_', None)
        n_classes = getattr(clf, 'n_classes_', None)

        if encoder is not None:
            # The trained model uses integer labels; encoder maps them back
            decoded_classes = sorted(encoder.inverse_transform(clf_classes).tolist()) if clf_classes is not None else []
            expected_sorted = sorted(EXPECTED_LABELS)

            if n_classes is not None:
                if n_classes == len(EXPECTED_LABELS):
                    _PASS(f'clf.n_classes_ = {n_classes} matches expected ({len(EXPECTED_LABELS)})')
                else:
                    _FAIL(f'clf.n_classes_ = {n_classes} ≠ expected ({len(EXPECTED_LABELS)})')

            if decoded_classes == expected_sorted:
                _PASS(f'clf.classes_ (decoded) = {decoded_classes}')
            else:
                _FAIL(f'clf.classes_ (decoded) = {decoded_classes}, expected {expected_sorted}')
        else:
            # No encoder — classes are raw
            if clf_classes is not None:
                _p(f'  clf.classes_ (raw) = {list(clf_classes)}')
            if n_classes is not None:
                if n_classes == len(EXPECTED_LABELS):
                    _PASS(f'clf.n_classes_ = {n_classes}')
                else:
                    _FAIL(f'clf.n_classes_ = {n_classes} ≠ expected {len(EXPECTED_LABELS)}')

        # Scaler transform
        if feature_vector is not None:
            try:
                X = np.array([feature_vector])
                X_scaled = scaler.transform(X)
                _PASS('scaler.transform() succeeded')

                max_abs = float(np.max(np.abs(X_scaled)))
                _p(f'  Max |scaled value| : {max_abs:.4f}')
                if max_abs <= 10.0:
                    _PASS(f'Max |scaled value| {max_abs:.4f} ≤ 10 — amplitude scale consistent')
                else:
                    _WARN(f'Max |scaled value| {max_abs:.4f} > 10 — possible amplitude scale mismatch')

                # Predict
                try:
                    pred = clf.predict(X_scaled)
                    pred_label = pred[0]
                    if encoder is not None:
                        pred_label = encoder.inverse_transform(pred)[0]
                    pred_label = str(pred_label)

                    if pred_label in EXPECTED_LABELS:
                        _PASS(f'clf.predict() → "{pred_label}" (valid label)')
                    else:
                        _FAIL(f'clf.predict() → "{pred_label}" (not in expected labels)')
                except Exception as e:
                    _FAIL(f'clf.predict() failed: {e}')

                # Predict proba
                try:
                    proba = clf.predict_proba(X_scaled)[0]
                    prob_sum = float(np.sum(proba))
                    if abs(prob_sum - 1.0) < 1e-6:
                        _PASS(f'predict_proba() sums to {prob_sum:.6f} (≈1.0)')
                    else:
                        _FAIL(f'predict_proba() sums to {prob_sum:.6f} (≠1.0)')
                except Exception as e:
                    _FAIL(f'clf.predict_proba() failed: {e}')

            except Exception as e:
                _FAIL(f'scaler.transform() failed: {e}')

        # Report scaler means
        scaler_mean = getattr(scaler, 'mean_', None)
        if scaler_mean is not None:
            _p('')
            _p('  Scaler mean per feature position:')
            for i, m in enumerate(scaler_mean):
                name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'idx_{i}'
                _p(f'    [{i:2d}] {name:20s} = {m:.8f}')


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 7: VERSION DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _check_version_drift():
    _header('CHECK 7 — VERSION DRIFT DETECTION')

    model_dirs = _find_model_directories()

    if not model_dirs:
        _WARN('No model files to check for version drift.')
        return

    if not _HAS_JOBLIB:
        _FAIL('joblib not available — cannot inspect models')
        return

    current_names = [
        'delta', 'theta', 'alpha', 'beta', 'gamma',
        'beta_alpha_ratio', 'alpha_theta_ratio',
        'beta_theta_ratio', 'gamma_beta_ratio',
        'beta_minus_alpha', 'alpha_plus_theta',
    ]

    for source, label, dirpath in model_dirs:
        _p(f'  ── Model: {label} ({source}) ──')

        # The RFClassifier stores _feature_names on the instance, but
        # it is NOT persisted inside the pkl files.  We check for a
        # feature_names.txt sidecar or any attribute on the saved clf.
        model_path = os.path.join(dirpath, 'rf_eeg_model.pkl')
        try:
            clf = joblib.load(model_path)
        except Exception:
            _WARN(f'Cannot load model at {model_path}')
            continue

        saved_names = getattr(clf, 'feature_names_in_', None)
        if saved_names is None:
            # Try sidecar file
            sidecar = os.path.join(dirpath, 'feature_names.txt')
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    saved_names = [line.strip() for line in f if line.strip()]

        if saved_names is not None:
            saved_list = list(saved_names)
            if saved_list == current_names:
                _PASS(f'Feature names match current FEATURE_NAMES')
            else:
                _FAIL('Feature name mismatch detected:')
                for i, (s, c) in enumerate(zip(saved_list, current_names)):
                    if s != c:
                        _p(f'    [{i}] model="{s}"  current="{c}"')
                if len(saved_list) != len(current_names):
                    _p(f'    Length mismatch: model={len(saved_list)} current={len(current_names)}')
        else:
            _WARN(f'Model does not store feature names — version drift cannot be auto-detected.')
            _p('  → Recommend adding feature name storage to RFClassifier.save()')
            _p(f'  → Current FEATURE_NAMES: {current_names}')


# ═══════════════════════════════════════════════════════════════════════════
# CHECK GROUP 8: FEATURE FORMULA VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _check_feature_formulas():
    _header('CHECK 8 — FEATURE FORMULA VERIFICATION')

    # Synthetic EegBands with known values
    bands = EegBands(
        delta=2.0,
        theta=1.5,
        alpha=3.0,
        beta=1.0,
        gamma=0.5,
    )

    clf = RFClassifier()
    fv = clf.build_feature_vector(bands)

    # Expected results computed from the canonical formulas
    eps = 1e-9
    expected = [
        2.0,                         # [0] delta
        1.5,                         # [1] theta
        3.0,                         # [2] alpha
        1.0,                         # [3] beta
        0.5,                         # [4] gamma
        1.0 / (3.0 + eps),           # [5] beta / alpha
        3.0 / (1.5 + eps),           # [6] alpha / theta
        1.0 / (1.5 + eps),           # [7] beta / theta
        0.5 / (1.0 + eps),           # [8] gamma / beta
        1.0 - 3.0,                   # [9] beta - alpha = -2.0
        3.0 + 1.5,                   # [10] alpha + theta = 4.5
    ]

    feature_labels = [
        'delta',            'theta',             'alpha',
        'beta',             'gamma',             'beta_alpha_ratio',
        'alpha_theta_ratio','beta_theta_ratio',  'gamma_beta_ratio',
        'beta_minus_alpha', 'alpha_plus_theta',
    ]

    if len(fv) != 11:
        _FAIL(f'Feature vector has {len(fv)} elements, expected 11')
        return

    _p(f'  Input bands: delta=2.0, theta=1.5, alpha=3.0, beta=1.0, gamma=0.5')
    _p(f'  Epsilon used in denominators: {eps}')
    _p('')

    all_ok = True
    for i in range(11):
        actual = fv[i]
        exp = expected[i]
        tol = 1e-6
        match = abs(actual - exp) < tol
        status = 'PASS' if match else 'FAIL'
        symbol = '✓' if match else '✗'

        if match:
            _PASS(f'[{i:2d}] {feature_labels[i]:22s}  expected={exp:12.8f}  actual={actual:12.8f}')
        else:
            _FAIL(f'[{i:2d}] {feature_labels[i]:22s}  expected={exp:12.8f}  actual={actual:12.8f}')
            all_ok = False

    if all_ok:
        _p('')
        _p('  All 11 feature formulas produce correct results.')
    else:
        _p('')
        _p('  ⚠ Some feature formulas produced incorrect results!')


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary():
    _p('')
    _p('═' * 60)
    _p('  COMPATIBILITY CHECK SUMMARY')
    _p('═' * 60)
    total = _pass + _fail + _warn
    _p(f'  Total checks  : {total}')
    _p(f'  Passed        : {_pass}')
    _p(f'  Failed        : {_fail}')
    _p(f'  Warnings      : {_warn}')

    if _fail_details:
        _p('')
        _p('  ── FAILED CHECKS ──')
        for i, detail in enumerate(_fail_details, 1):
            _p(f'  {i}. {detail}')

    _p('')
    if _fail == 0:
        _p('  ╔══════════════════════════════════════════════════╗')
        _p('  ║   VERDICT: COMPATIBLE                           ║')
        _p('  ║   Safe to run live classification.               ║')
        _p('  ╚══════════════════════════════════════════════════╝')
    else:
        _p('  ╔══════════════════════════════════════════════════╗')
        _p('  ║   VERDICT: INCOMPATIBLE                         ║')
        _p('  ║   Fix issues above before connecting hardware.   ║')
        _p('  ╚══════════════════════════════════════════════════╝')
    _p('')


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def run_compatibility_check() -> str:
    """Run all checks and return the full output as a string.

    This is the entry point for both standalone use and in-app integration.
    """
    global _pass, _fail, _warn, _fail_details, _buf
    _pass = 0
    _fail = 0
    _warn = 0
    _fail_details = []
    _buf = StringIO()

    _p('╔════════════════════════════════════════════════════════╗')
    _p('║  NEUROMENTOR PIPELINE × MODEL COMPATIBILITY DIAGNOSTIC ║')
    _p('╚════════════════════════════════════════════════════════╝')

    try:
        _check_adc_conversion()
    except Exception as e:
        _FAIL(f'ADC conversion check crashed: {e}')

    try:
        _check_sample_rate_and_window()
    except Exception as e:
        _FAIL(f'Sample rate check crashed: {e}')

    try:
        _check_band_boundaries()
    except Exception as e:
        _FAIL(f'Band boundary check crashed: {e}')

    try:
        _check_synthetic_signals()
    except Exception as e:
        _FAIL(f'Synthetic signal check crashed: {e}')

    fv = None
    try:
        fv = _check_feature_vector()
    except Exception as e:
        _FAIL(f'Feature vector check crashed: {e}')

    try:
        _check_saved_model(fv)
    except Exception as e:
        _FAIL(f'Saved model check crashed: {e}')

    try:
        _check_version_drift()
    except Exception as e:
        _FAIL(f'Version drift check crashed: {e}')

    try:
        _check_feature_formulas()
    except Exception as e:
        _FAIL(f'Feature formula check crashed: {e}')

    _print_summary()

    return _buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if False:
    run_compatibility_check()


# ============================================================
# File: app_state.py
# ============================================================

"""
NeuroMentor Application State
Observable state management using Kivy EventDispatcher (replaces Flutter Provider).
"""
from kivy.event import EventDispatcher
from kivy.properties import (
    ObjectProperty, NumericProperty, StringProperty
)
from datetime import datetime
import json
import os

class UserProfile:
    """User profile data."""
    def __init__(self, username, name='', age='', notes='', created_date=None, scores=None, last_login=None):
        self.username = username
        self.name = name or username
        self.age = age
        self.notes = notes
        self.created_date = created_date or datetime.now().isoformat()
        self.scores = scores or []
        self.last_login = last_login or datetime.now().isoformat()

    def to_dict(self):
        return {
            'username': self.username,
            'name': self.name,
            'age': self.age,
            'notes': self.notes,
            'created_date': self.created_date,
            'scores': self.scores,
            'last_login': self.last_login,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            username=data.get('username', ''),
            name=data.get('name', ''),
            age=data.get('age', ''),
            notes=data.get('notes', ''),
            created_date=data.get('created_date'),
            scores=data.get('scores', []),
            last_login=data.get('last_login'),
        )


class AppState(EventDispatcher):
    """
    Main application state.
    Uses Kivy properties for automatic UI binding.
    """
    current_user = ObjectProperty(None, allownone=True)
    selected_page_index = NumericProperty(0)
    selected_port = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')
        self.all_users = self._load_users()

    def _load_users(self):
        if not os.path.exists(self.users_file):
            return {}
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
                return {k: UserProfile.from_dict(v) for k, v in data.items()}
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}

    def _save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump({k: v.to_dict() for k, v in self.all_users.items()}, f, indent=4)
        except Exception as e:
            print(f"Error saving users: {e}")

    def save_current_user_score(self, test_name, score):
        if self.current_user:
            self.current_user.scores.append({'test': test_name, 'score': score, 'date': datetime.now().isoformat()})
            self._save_users()

    def get_sorted_users(self):
        """Return list of UserProfile sorted by last_login descending (most recent first)."""
        users = list(self.all_users.values())
        users.sort(key=lambda u: u.last_login or '', reverse=True)
        return users

    # ============================================================
    # USER MANAGEMENT
    # ============================================================

    def login(self, username):
        """Login with username - creates profile if not exists."""
        if username not in self.all_users:
            self.all_users[username] = UserProfile(username=username, name=username)
        self.all_users[username].last_login = datetime.now().isoformat()
        self._save_users()
        self.current_user = self.all_users[username]
        self.selected_page_index = 0

        # Attempt to load RF model for this user
        self._load_rf_model(username)

    def logout(self):
        """Logout current user."""
        self._save_users()
        self.current_user = None
        self.selected_page_index = 0

    def update_profile(self, name=None, age=None, notes=None):
        """Update user profile fields."""
        if self.current_user is not None:
            if name is not None:
                self.current_user.name = name
            if age is not None:
                self.current_user.age = age
            if notes is not None:
                self.current_user.notes = notes
            self._save_users()
            # Force property change notification
            self.property('current_user').dispatch(self)

    # ============================================================
    # NAVIGATION
    # ============================================================

    def set_selected_page(self, index):
        """Set the currently selected page index."""
        self.selected_page_index = index

    def set_selected_port(self, port):
        """Set the selected device port."""
        self.selected_port = port or ''

    # ============================================================
    # RF MODEL PERSISTENCE
    # ============================================================

    def _get_rf_model_dir(self, username):
        """Get the directory path for a user's RF model."""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neuromentor_data')
        user_model_dir = os.path.join(data_dir, f'{username}_rf_model')
        os.makedirs(user_model_dir, exist_ok=True)
        return user_model_dir

    def _get_bundled_model_dir(self):
        """Get the path to the bundled pre-trained RF model."""
        # The bundled model lives at: Kivy_Section/rf_model/rf_model/
        project_root = os.path.dirname(os.path.abspath(__file__))
        bundled = os.path.join(project_root, '..', 'rf_model', 'rf_model')
        return os.path.normpath(bundled)

    def save_rf_model(self, rf_classifier=None):
        """Save the RF model for the current user."""
        if rf_classifier is None:
            rf_classifier = getattr(self, 'rf_classifier', None)
        if self.current_user and rf_classifier and rf_classifier.is_trained:
            dirpath = self._get_rf_model_dir(self.current_user.username)
            rf_classifier.save(dirpath)
            print(f"[AppState] RF model saved to {dirpath}")

    def _load_rf_model(self, username):
        """Attempt to load an RF model for the given user.

        Tries user-specific model first, then falls back to bundled model.
        Returns True if loaded.
        """

        if not hasattr(self, 'rf_classifier'):
            self.rf_classifier = RFClassifier()

        # 1. Try user-specific model
        user_dir = self._get_rf_model_dir(username)
        model_file = os.path.join(user_dir, 'rf_eeg_model.pkl')
        if os.path.exists(model_file):
            try:
                success = self.rf_classifier.load(user_dir)
                if success:
                    print(f"[AppState] User RF model loaded for {username}")
                    return True
            except Exception as e:
                print(f"[AppState] Could not load user RF model: {e}")

        # 2. Fall back to bundled pre-trained model
        bundled_dir = self._get_bundled_model_dir()
        bundled_model = os.path.join(bundled_dir, 'rf_eeg_model.pkl')
        if os.path.exists(bundled_model):
            try:
                success = self.rf_classifier.load(bundled_dir)
                if success:
                    print(f"[AppState] Bundled RF model loaded for {username}")
                    return True
            except Exception as e:
                print(f"[AppState] Could not load bundled RF model: {e}")

        print(f"[AppState] No RF model available for {username}")
        return False


# ============================================================
# File: widgets/custom_ui.py
# ============================================================

"""
Custom UI Elements with shadows and gradients.
"""
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle

class ShadowButton(ButtonBehavior, Label):
    """A button with a soft drop-shadow and rounded corners."""
    
    def __init__(self, **kwargs):
        self.bg_color = kwargs.pop('bg_color', kwargs.pop('background_color', theme.BUTTON_BG))
        self.radius = kwargs.pop('radius', 12)
        super().__init__(**kwargs)
        self.bind(pos=self.update_canvas, size=self.update_canvas, state=self.update_canvas)
        self.update_canvas()
        
    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            # Soft shadow effect
            shadow_steps = 4
            base_offset = 1 if self.state == 'down' else 4
            
            for i in range(shadow_steps):
                alpha = 0.25 * (1.0 - i/shadow_steps)
                r_exp = self.radius + i
                Color(0, 0, 0, alpha)
                RoundedRectangle(
                    pos=(self.x - i + 2, self.y - base_offset - i),
                    size=(self.width + i*2, self.height + i*2),
                    radius=[r_exp]
                )
            
            # Main button background
            if self.state == 'down':
                # Dim the color slightly
                Color(self.bg_color[0]*0.8, self.bg_color[1]*0.8, self.bg_color[2]*0.8, 1)
            else:
                Color(*self.bg_color)
            
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[self.radius]
            )

from kivy.uix.widget import Widget
from kivy.graphics import Line

class MenuBurgerButton(ButtonBehavior, Widget):
    """A perfect hamburger menu icon that transitions to an X."""
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
                # Hamburger: 3 lines
                spacing = self.height * 0.22
                RoundedRectangle(pos=(cx - w/2, cy + spacing - h/2), size=(w, h), radius=[h/2])
                RoundedRectangle(pos=(cx - w/2, cy - h/2), size=(w, h), radius=[h/2])
                RoundedRectangle(pos=(cx - w/2, cy - spacing - h/2), size=(w, h), radius=[h/2])
            else:
                # X shape
                Line(points=[cx - w/2, cy - w/2, cx + w/2, cy + w/2], width=h/2, cap='round')
                Line(points=[cx - w/2, cy + w/2, cx + w/2, cy - w/2], width=h/2, cap='round')

from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle

from kivy.graphics.texture import Texture

class GradientCard(BoxLayout):
    """A card with a soft gradient background, rounded corners, and a drop shadow."""
    def __init__(self, accent_color=None, **kwargs):
        kwargs.setdefault('orientation', 'vertical')
        kwargs.setdefault('padding', 15)
        kwargs.setdefault('spacing', 8)
        self.accent_color = accent_color
        self.radius = kwargs.pop('radius', 12)
        super().__init__(**kwargs)
        
        # Create a basic 1x2 vertical gradient texture mapping top #202020 to bottom #0E0E0E
        self.texture = Texture.create(size=(1, 2), colorfmt='rgba')
        self.texture.mag_filter = 'linear'
        self.texture.min_filter = 'linear'
        
        # Bottom color, Top color (charcoal tones)
        buf = bytes([
            51, 46, 60, 255,   # bottom (darker charcoal #332e3c)
            70, 65, 81, 255,   # top (lighter charcoal #464151)
        ])
        self.texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            # Soft dark drop shadow effect
            shadow_steps = 4
            for i in range(shadow_steps):
                alpha = 0.2 * (1.0 - i/shadow_steps)
                r_exp = self.radius + i
                Color(0, 0, 0, alpha)
                RoundedRectangle(
                    pos=(self.x - i + 2, self.y - 2 - i),
                    size=(self.width + i*2, self.height + i*2),
                    radius=[r_exp]
                )
            
            # Gradient rounded background
            Color(1, 1, 1, 1)  # White to allow natural texture colors
            RoundedRectangle(
                pos=self.pos, size=self.size, radius=[self.radius], texture=self.texture
            )

            # Optional accent color indicator strip
            if self.accent_color:
                Color(*self.accent_color)
                RoundedRectangle(
                    pos=(self.x + 8, self.y + self.height - 4), 
                    size=(self.width - 16, 3), 
                    radius=[1.5]
                )


# ============================================================
# File: widgets/eeg_graph.py
# ============================================================

"""
EEG Line Graph widget for live signal visualization.
Uses Kivy Canvas for smooth real-time plotting (replaces fl_chart).
"""
from collections import deque
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import NumericProperty


class EegGraph(BoxLayout):
    """
    EEG Graph widget that shows a live line chart.
    Data is added via add_data_point() and auto-scrolls.
    """
    max_data_points = NumericProperty(256)
    min_y = NumericProperty(0)
    max_y = NumericProperty(4095)

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self._data = deque(maxlen=256)

        # Title label
        self._title_label = Label(
            text='Live EEG Signal Check',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        self._title_label.bind(size=self._title_label.setter('text_size'))
        self.add_widget(self._title_label)

        # Placeholder label (shown when no data)
        self._placeholder = Label(
            text='Waiting for signal...',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEXT_MUTED,
        )
        self.add_widget(self._placeholder)

        # Canvas widget for drawing the graph
        self._graph_canvas = _GraphCanvas(
            data=self._data,
            min_y=self.min_y,
            max_y=self.max_y,
        )
        # Initially hidden; shown when data arrives
        self._graph_canvas.opacity = 0
        self.add_widget(self._graph_canvas)

        # Background
        with self.canvas.before:
            Color(*theme.PANEL_BG)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
            Color(*theme.BORDER_DARK)
            self._border_rect = Rectangle(pos=self.pos, size=self.size)

        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border_rect.pos = self.pos
        self._border_rect.size = self.size

    def add_data_point(self, value):
        """Add a new data point to the graph."""
        self._data.append(value)
        if self._placeholder.opacity > 0:
            self._placeholder.opacity = 0
            self._graph_canvas.opacity = 1
        self._graph_canvas.redraw()

    def clear(self):
        """Clear all data points."""
        self._data.clear()
        self._placeholder.opacity = 1
        self._graph_canvas.opacity = 0
        self._graph_canvas.redraw()


class _GraphCanvas(Widget):
    """Internal widget that draws the EEG line on its canvas."""

    def __init__(self, data, min_y=0, max_y=4095, **kwargs):
        super().__init__(**kwargs)
        self._data = data
        self._min_y = min_y
        self._max_y = max_y
        self.bind(size=lambda *a: self.redraw(), pos=lambda *a: self.redraw())

    def redraw(self):
        self.canvas.clear()
        if not self._data or self.width <= 0 or self.height <= 0:
            return

        x0 = self.x + 5
        y0 = self.y + 5
        w = self.width - 10
        h = self.height - 10

        # Draw grid lines
        with self.canvas:
            Color(*theme.BORDER_DARK)
            for i in range(5):
                gy = y0 + (h * i / 4)
                Line(points=[x0, gy, x0 + w, gy], width=1)
            for i in range(9):
                gx = x0 + (w * i / 8)
                Line(points=[gx, y0, gx, y0 + h], width=1)

        # Draw data line
        data_list = list(self._data)
        n = len(data_list)
        if n < 2:
            return

        y_range = self._max_y - self._min_y
        if y_range == 0:
            y_range = 1

        points = []
        for i, val in enumerate(data_list):
            px = x0 + (w * i / (n - 1))
            py = y0 + h * ((val - self._min_y) / y_range)
            py = max(y0, min(y0 + h, py))
            points.extend([px, py])

        with self.canvas:
            Color(*theme.TEAL)
            Line(points=points, width=1.5)


class BandPowerBars(BoxLayout):
    """EEG Band Power horizontal bar display."""

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=4, **kwargs)
        self._bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        self._bars = {}

        # Title
        title = Label(
            text='EEG BAND POWERS',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        title.bind(size=title.setter('text_size'))
        self.add_widget(title)

        for band in self._bands:
            row = BoxLayout(size_hint_y=None, height=24, spacing=5)
            lbl = Label(
                text=band,
                font_size=theme.FONT_BODY_SMALL,
                color=theme.TEXT_MUTED,
                size_hint_x=None,
                width=50,
                halign='left',
                valign='middle',
            )
            lbl.bind(size=lbl.setter('text_size'))
            bar = _BarWidget(value=0)
            row.add_widget(lbl)
            row.add_widget(bar)
            self._bars[band] = bar
            self.add_widget(row)

        # Background
        with self.canvas.before:
            Color(*theme.BG_DARK)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def update_values(self, values_dict):
        """Update bar values. values_dict maps band name to 0-1 float."""
        for band, bar in self._bars.items():
            bar.value = values_dict.get(band, 0)


class _BarWidget(Widget):
    """A single horizontal progress bar."""
    value = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(value=self._redraw, size=self._redraw, pos=self._redraw)
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            # Track
            Color(0.165, 0.165, 0.165, 1)
            Rectangle(pos=self.pos, size=self.size)
            # Fill
            Color(*theme.TEAL)
            fill_w = self.width * max(0, min(1, self.value))
            if fill_w > 0:
                Rectangle(pos=self.pos, size=(fill_w, self.height))


# ============================================================
# File: widgets/mind_visualizer.py
# ============================================================

"""
MindVisualizer widget matching Flutter's MindVisualizer.
Animated orb that moves based on focus level with jitter based on stress.
Uses Kivy Canvas for drawing.
"""
import random
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.clock import Clock
from kivy.properties import (
    BooleanProperty, StringProperty, NumericProperty, ListProperty
)


class MindVisualizer(Widget):
    """
    Animated mind visualizer with a glowing orb.
    Ball Y position responds to focus_ratio, jitter responds to stress_ratio.
    """
    is_active = BooleanProperty(False)
    state_label = StringProperty('IDLE')
    focus_ratio = NumericProperty(1.0)
    stress_ratio = NumericProperty(1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ball_y = 0.5
        self._target_y = 0.5
        self._jitter_x = 0
        self._jitter_y = 0
        self._jitter_intensity = 0.0
        self._ball_color = list(theme.TEAL)

        # Start animation loop at ~60fps
        self._clock_event = Clock.schedule_interval(self._animate, 1.0 / 60.0)

        self.bind(
            focus_ratio=self._update_data,
            stress_ratio=self._update_data,
            state_label=self._update_data,
            size=lambda *a: self._draw(),
            pos=lambda *a: self._draw(),
        )

    def _update_data(self, *args):
        """Recalculate target position and jitter from ratios."""
        f_val = max(0.5, min(2.5, self.focus_ratio))
        self._target_y = 1.0 - ((f_val - 0.5) / 2.0)

        s_val = max(0.5, min(2.0, self.stress_ratio))
        self._jitter_intensity = (s_val - 0.5) * 0.05

        if self.state_label == 'Stressed':
            self._ball_color = list(theme.RED)
        elif self.state_label == 'Focused':
            self._ball_color = list(theme.GOLD)
        else:
            self._ball_color = list(theme.TEAL)

    def _animate(self, dt):
        """Per-frame animation update."""
        # Smooth interpolation
        self._ball_y += (self._target_y - self._ball_y) * 0.05
        self._ball_y = max(0.1, min(0.9, self._ball_y))

        # Apply jitter
        self._jitter_x = (random.random() - 0.5) * self._jitter_intensity
        self._jitter_y = (random.random() - 0.5) * self._jitter_intensity

        self._draw()

    def _draw(self):
        """Redraw the visualizer."""
        self.canvas.clear()
        w = self.width
        h = self.height
        x0 = self.x
        y0 = self.y

        if w <= 0 or h <= 0:
            return

        with self.canvas:
            # Background
            Color(0.02, 0.02, 0.031, 1)  # #050508
            Rectangle(pos=self.pos, size=self.size)

            # Grid
            Color(0.118, 0.118, 0.157, 1)  # #1E1E28
            for gx in range(0, int(w), 60):
                Line(points=[x0 + gx, y0, x0 + gx, y0 + h], width=1)
            for gy in range(0, int(h), 60):
                Line(points=[x0, y0 + gy, x0 + w, y0 + gy], width=1)

            # Ball position
            cx = x0 + w * 0.5 + self._jitter_x * w
            # Kivy Y is bottom-up, so invert
            cy = y0 + h * (1.0 - self._ball_y) + self._jitter_y * h
            cy = max(y0 + 50, min(y0 + h - 50, cy))
            radius = 40

            # Glow (concentric circles with decreasing alpha)
            for i in range(6, 0, -1):
                alpha = 0.06 * i
                Color(self._ball_color[0], self._ball_color[1],
                      self._ball_color[2], alpha)
                gr = radius * (0.5 + i * 0.5)
                Ellipse(pos=(cx - gr, cy - gr), size=(gr * 2, gr * 2))

            # Ball
            Color(*self._ball_color)
            Ellipse(pos=(cx - radius, cy - radius),
                    size=(radius * 2, radius * 2))

        # Status label is drawn separately so it's always on top
        self.canvas.after.clear()
        with self.canvas.after:
            pass  # Text is handled via an overlay Label if needed

    def on_parent(self, *args):
        """Ensure we have a status label overlay."""
        pass

    def __del__(self):
        if hasattr(self, '_clock_event') and self._clock_event:
            self._clock_event.cancel()


# ============================================================
# File: widgets/tasks/breathing_widget.py
# ============================================================

"""
Breathing task widget matching Flutter's BreathingWidget.
Supports 4-7-8 (Calm) and Box (Focus) breathing modes.
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, RoundedRectangle


class BreathingWidget(BoxLayout):
    """Breathing exercise task with modes and score tracking."""

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=10, **kwargs)
        self._step = 0
        self._score = 0
        self._is_running = False
        self._mode = '4-7-8'
        self._clock_event = None

        # Instruction label
        self._instruction_lbl = Label(
            text='Ready',
            font_size=40,
            color=theme.TEAL,
            bold=True,
            size_hint_y=0.4,
        )
        self.add_widget(self._instruction_lbl)

        # Mode selector row
        mode_row = BoxLayout(
            size_hint_y=None, height=40,
            spacing=20, padding=[0, 0, 0, 0]
        )
        mode_row.size_hint_x = None
        mode_row.width = 320
        mode_row.pos_hint = {'center_x': 0.5}

        self._btn_478 = ToggleButton(
            text='4-7-8 (Calm)',
            group='breathing_mode',
            state='down',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.GOLD,
            color=theme.BG_DARK,
        )
        self._btn_478.bind(on_press=lambda *a: self.set_mode('4-7-8'))

        self._btn_box = ToggleButton(
            text='Box (Focus)',
            group='breathing_mode',
            state='normal',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.BORDER_DARK,
            color=theme.TEXT_PRIMARY,
        )
        self._btn_box.bind(on_press=lambda *a: self.set_mode('box'))

        mode_row.add_widget(self._btn_478)
        mode_row.add_widget(self._btn_box)
        self.add_widget(mode_row)

        # Score label
        self._score_lbl = Label(
            text='Score: 0',
            font_size=theme.FONT_HEADING_MEDIUM,
            color=theme.GOLD,
            size_hint_y=None,
            height=40,
        )
        self.add_widget(self._score_lbl)

        # Start/Stop button
        self._start_btn = ShadowButton(
            text='START',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint=(None, None),
            size=(200, 45),
            pos_hint={'center_x': 0.5},
            bg_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._start_btn.bind(on_press=self._toggle)
        self.add_widget(self._start_btn)

    def set_mode(self, mode):
        self._mode = mode
        if mode == '4-7-8':
            self._btn_478.state = 'down'
            self._btn_box.state = 'normal'
            self._btn_478.background_color = theme.GOLD
            self._btn_478.color = theme.BG_DARK
            self._btn_box.background_color = theme.BORDER_DARK
            self._btn_box.color = theme.TEXT_PRIMARY
        elif mode == 'box':
            self._btn_box.state = 'down'
            self._btn_478.state = 'normal'
            self._btn_box.background_color = theme.GOLD
            self._btn_box.color = theme.BG_DARK
            self._btn_478.background_color = theme.BORDER_DARK
            self._btn_478.color = theme.TEXT_PRIMARY
            
    def start_task(self):
        self._start()

    def _toggle(self, *args):
        if self._is_running:
            self.stop()
        else:
            self._start()

    def _start(self):
        if self._is_running:
            return
        self._is_running = True
        self._step = 0
        self._score = 0
        self._start_btn.text = 'STOP'
        self._start_btn.color = theme.RED
        self._start_btn.bg_color = theme.DANGER_BUTTON_BG
        self._clock_event = Clock.schedule_interval(self._tick, 1.0)

    def stop(self):
        if self._clock_event:
            self._clock_event.cancel()
            self._clock_event = None
        self._is_running = False
        self._instruction_lbl.text = 'Relax'
        self._start_btn.text = 'START'
        self._start_btn.color = theme.GOLD
        self._start_btn.bg_color = theme.BUTTON_BG

    def _tick(self, dt):
        self._step += 1
        self._score = self._step * 10
        self._score_lbl.text = f'Score: {self._score}'

        cycle_length = 16 if self._mode == 'box' else 19
        curr = self._step % cycle_length

        if self._mode == 'box':
            if curr < 4:
                self._instruction_lbl.text = f'INHALE ({4 - curr})'
            elif curr < 8:
                self._instruction_lbl.text = f'HOLD ({8 - curr})'
            elif curr < 12:
                self._instruction_lbl.text = f'EXHALE ({12 - curr})'
            else:
                self._instruction_lbl.text = f'HOLD ({16 - curr})'
        else:
            if curr < 4:
                self._instruction_lbl.text = f'INHALE ({4 - curr})'
            elif curr < 11:
                self._instruction_lbl.text = f'HOLD ({11 - curr})'
            else:
                self._instruction_lbl.text = f'EXHALE ({19 - curr})'

    def cleanup(self):
        """Call when removing widget to stop timers."""
        if self._clock_event:
            self._clock_event.cancel()


# ============================================================
# File: widgets/tasks/focus_widget.py
# ============================================================

"""
Focus task widget matching Flutter's FocusWidget.
Supports Visual Tracking and Tech Reading modes.
"""
import random
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.clock import Clock


ARTICLES = [
    'Neuroplasticity is the ability of neural networks in the brain to reorganize '
    'themselves by creating new neural connections throughout life. This allows '
    'neurons to compensate for injury and disease, and to adjust their activities '
    'in response to new situations. Researchers have discovered that neuroplasticity '
    'is not limited to childhood development but continues throughout adult life. '
    'This groundbreaking finding has revolutionized our understanding of brain '
    'function and has led to new therapeutic approaches for treating brain injuries, '
    'learning disabilities, and neurodegenerative diseases. The brain\'s remarkable '
    'ability to adapt and change forms the biological basis for learning new skills '
    'and forming new memories.',

    'Quantum entanglement is a phenomenon where two or more particles become '
    'interconnected in such a way that the quantum state of each particle cannot '
    'be described independently. When particles are entangled, they remain connected '
    'across vast distances, and measuring one particle instantaneously affects the '
    'state of the other. This counterintuitive phenomenon puzzled even Einstein, '
    'who called it \'spooky action at a distance.\' Today, quantum entanglement is '
    'recognized as a fundamental aspect of quantum mechanics and has practical '
    'applications in quantum computing, quantum cryptography, and quantum '
    'teleportation. Scientists continue to explore the implications of entanglement '
    'for our understanding of reality.',

    'In cognitive science, attention is the cognitive process that allows us to '
    'focus on specific information while filtering out irrelevant stimuli. The '
    'human brain receives countless sensory inputs every second, yet we can only '
    'consciously process a fraction of this information. Selective attention '
    'mechanisms help us prioritize important information and maintain focus on '
    'relevant tasks. Research has shown that attention is not a single unified '
    'process but involves multiple neural systems and brain regions. Understanding '
    'attention mechanisms has profound implications for education, workplace '
    'productivity, mental health treatment, and the design of technology interfaces.',
]


class FocusWidget(BoxLayout):
    """Focus task with Visual Tracking and Tech Reading modes."""

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=10, **kwargs)
        self._is_tracking = True
        self._is_running = False

        # Mode selector row
        mode_row = BoxLayout(
            size_hint_y=None, height=40,
            spacing=20,
        )
        mode_row.size_hint_x = None
        mode_row.width = 320
        mode_row.pos_hint = {'center_x': 0.5}

        self._btn_tracking = ToggleButton(
            text='Visual Tracking',
            group='focus_mode',
            state='down',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.GOLD,
            color=theme.BG_DARK,
        )
        self._btn_tracking.bind(on_press=lambda *a: self.set_mode('tracking'))

        self._btn_reading = ToggleButton(
            text='Tech Reading',
            group='focus_mode',
            state='normal',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.BORDER_DARK,
            color=theme.TEXT_PRIMARY,
        )
        self._btn_reading.bind(on_press=lambda *a: self.set_mode('reading'))

        mode_row.add_widget(self._btn_tracking)
        mode_row.add_widget(self._btn_reading)
        self.add_widget(mode_row)

        # Content area
        self._tracking_view = TrackingView()
        self._reading_view = ReadingView()
        self._reading_view.opacity = 0
        self._reading_view.disabled = True

        self._content = FloatLayout()
        self._content.add_widget(self._tracking_view)
        self._content.add_widget(self._reading_view)
        self.add_widget(self._content)

        # Start/Stop button
        self._start_btn = ShadowButton(
            text='START',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint=(None, None),
            size=(200, 45),
            pos_hint={'center_x': 0.5},
            bg_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._start_btn.bind(on_press=self._toggle)
        self.add_widget(self._start_btn)

    def set_mode(self, mode):
        is_tracking = (mode == 'tracking')
        self._is_tracking = is_tracking
        if is_tracking:
            self._btn_tracking.state = 'down'
            self._btn_reading.state = 'normal'
            self._btn_tracking.background_color = theme.GOLD
            self._btn_tracking.color = theme.BG_DARK
            self._btn_reading.background_color = theme.BORDER_DARK
            self._btn_reading.color = theme.TEXT_PRIMARY

            self._tracking_view.opacity = 1
            self._tracking_view.disabled = False
            self._reading_view.opacity = 0
            self._reading_view.disabled = True
        else:
            self._btn_reading.state = 'down'
            self._btn_tracking.state = 'normal'
            self._btn_reading.background_color = theme.GOLD
            self._btn_reading.color = theme.BG_DARK
            self._btn_tracking.background_color = theme.BORDER_DARK
            self._btn_tracking.color = theme.TEXT_PRIMARY

            self._tracking_view.opacity = 0
            self._tracking_view.disabled = True
            self._reading_view.opacity = 1
            self._reading_view.disabled = False

    def start_task(self):
        if not self._is_running:
            self._toggle()

    def stop(self):
        if self._is_running:
            self._toggle()

    def _toggle(self, *args):
        self._is_running = not self._is_running
        if self._is_running:
            self._start_btn.text = 'STOP'
            self._start_btn.color = theme.RED
            self._start_btn.bg_color = theme.DANGER_BUTTON_BG
            self._tracking_view.start()
            self._reading_view.new_article()
        else:
            self._start_btn.text = 'START'
            self._start_btn.color = theme.GOLD
            self._start_btn.bg_color = theme.BUTTON_BG
            self._tracking_view.stop()

    def cleanup(self):
        self._tracking_view.stop()


class TrackingView(Widget):
    """Moving orb for visual tracking exercise."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ball_x = 0.5
        self._ball_y = 0.5
        self._clock_event = None
        self.bind(size=self._draw, pos=self._draw)
        self._draw()

    def start(self):
        if self._clock_event:
            self._clock_event.cancel()
        self._clock_event = Clock.schedule_interval(self._move_ball, 1.0 / 30.0)

    def stop(self):
        if self._clock_event:
            self._clock_event.cancel()
            self._clock_event = None

    def _move_ball(self, dt):
        self._ball_x += (random.random() - 0.5) * 0.02
        self._ball_y += (random.random() - 0.5) * 0.02
        self._ball_x = max(0.05, min(0.95, self._ball_x))
        self._ball_y = max(0.05, min(0.95, self._ball_y))
        self._draw()

    def _draw(self, *args):
        self.canvas.clear()
        w = self.width
        h = self.height
        if w <= 0 or h <= 0:
            return

        with self.canvas:
            # Background
            Color(0, 0, 0, 1)
            Rectangle(pos=self.pos, size=self.size)

            # Orb glow
            cx = self.x + self._ball_x * w
            cy = self.y + self._ball_y * h
            for i in range(4, 0, -1):
                alpha = 0.12 * i
                Color(theme.GOLD[0], theme.GOLD[1], theme.GOLD[2], alpha)
                r = 15 + i * 8
                Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))

            # Orb
            Color(*theme.GOLD)
            Ellipse(pos=(cx - 15, cy - 15), size=(30, 30))


class ReadingView(BoxLayout):
    """Tech article reading exercise."""

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=15, spacing=10, **kwargs)

        self._header = Label(
            text='READ CAREFULLY:',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        self._header.bind(size=self._header.setter('text_size'))
        self.add_widget(self._header)

        scroll = ScrollView()
        self._article_lbl = Label(
            text=random.choice(ARTICLES),
            font_size=theme.FONT_BODY_LARGE,
            color=theme.TEAL,
            markup=False,
            halign='left',
            valign='top',
            size_hint_y=None,
        )
        self._article_lbl.bind(
            texture_size=lambda inst, sz: setattr(inst, 'height', sz[1]),
            width=lambda inst, w: setattr(inst, 'text_size', (w, None)),
        )
        scroll.add_widget(self._article_lbl)
        self.add_widget(scroll)

        # Background
        with self.canvas.before:
            Color(0.067, 0.067, 0.067, 1)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def new_article(self):
        self._article_lbl.text = random.choice(ARTICLES)


# ============================================================
# File: widgets/tasks/stroop_widget.py
# ============================================================

"""
Stroop/Math task widget matching Flutter's StroopWidget.
Supports Stroop color test and Rapid Math modes.
"""
import random
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock


class StroopWidget(BoxLayout):
    """Stroop color + rapid math stress task."""

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=10, **kwargs)
        self._score = 0
        self._is_running = False
        self._is_math_mode = False
        self._clock_event = None

        # Stroop state
        self._display_word = 'BLUE'
        self._ink_color = (0, 0, 1, 1)
        self._current_ink_name = 'BLUE'

        # Math state
        self._math_answer = 0
        self._math_problem = ''

        self._colors = ['RED', 'BLUE', 'GREEN', 'YELLOW']
        self._color_map = {
            'RED': (1, 0, 0, 1),
            'BLUE': (0, 0, 1, 1),
            'GREEN': (0, 1, 0, 1),
            'YELLOW': (1, 1, 0, 1),
        }

        # Display label (word or math problem)
        self._display_lbl = Label(
            text='BLUE',
            font_size=60,
            bold=True,
            color=(0, 0, 1, 1),
            size_hint_y=0.35,
        )
        self.add_widget(self._display_lbl)

        # Mode selector row
        mode_row = BoxLayout(
            size_hint_y=None, height=40,
            spacing=20
        )
        mode_row.size_hint_x = None
        mode_row.width = 280
        mode_row.pos_hint = {'center_x': 0.5}

        self._btn_stroop = ToggleButton(
            text='Stroop',
            group='stroop_mode',
            state='down',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.GOLD,
            color=theme.BG_DARK,
        )
        self._btn_stroop.bind(on_press=lambda *a: self.set_mode('stroop'))

        self._btn_math = ToggleButton(
            text='Rapid Math',
            group='stroop_mode',
            state='normal',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.BORDER_DARK,
            color=theme.TEXT_PRIMARY,
        )
        self._btn_math.bind(on_press=lambda *a: self.set_mode('math'))

        mode_row.add_widget(self._btn_stroop)
        mode_row.add_widget(self._btn_math)
        self.add_widget(mode_row)

        # Answer area: color buttons for Stroop, text input for Math
        self._stroop_row = BoxLayout(
            size_hint_y=None, height=50,
            spacing=8,
            pos_hint={'center_x': 0.5},
        )
        for c in self._colors:
            btn = Button(
                text=c,
                font_size=12,
                bold=True,
                background_color=self._color_map[c],
                color=(0, 0, 0, 1),
                size_hint_x=None,
                width=80,
            )
            btn.bind(on_press=lambda inst, cn=c: self._check_stroop(cn))
            self._stroop_row.add_widget(btn)
        self.add_widget(self._stroop_row)

        # Math input (hidden by default)
        self._math_row = BoxLayout(
            size_hint=(None, None), size=(300, 45),
            pos_hint={'center_x': 0.5},
            spacing=10
        )
        
        self._math_input = TextInput(
            hint_text='Answer',
            font_size=theme.FONT_HEADING_MEDIUM,
            multiline=False,
            input_filter='int',
            size_hint=(None, 1),
            width=180,
            background_color=theme.INPUT_BG,
            foreground_color=theme.TEXT_PRIMARY,
        )
        self._math_input.bind(on_text_validate=lambda *a: self._check_math())
        
        self._math_submit = ShadowButton(
            text='SUBMIT',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            bg_color=theme.TEAL,
            color=theme.BG_DARK,
            size_hint=(None, 1),
            width=110,
        )
        self._math_submit.bind(on_press=lambda *a: self._check_math())
        
        self._math_row.add_widget(self._math_input)
        self._math_row.add_widget(self._math_submit)
        
        self._math_row.opacity = 0
        self._math_row.disabled = True
        self.add_widget(self._math_row)

        # Score label
        self._score_lbl = Label(
            text='Score: 0',
            font_size=theme.FONT_HEADING_MEDIUM,
            color=theme.GOLD,
            size_hint_y=None,
            height=40,
        )
        self.add_widget(self._score_lbl)

        # Start/Stop button
        self._start_btn = ShadowButton(
            text='START',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint=(None, None),
            size=(200, 45),
            pos_hint={'center_x': 0.5},
            bg_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._start_btn.bind(on_press=self._toggle)
        self.add_widget(self._start_btn)

    def set_mode(self, mode):
        is_math = (mode == 'math')
        self._is_math_mode = is_math
        if is_math:
            self._btn_math.state = 'down'
            self._btn_stroop.state = 'normal'
            self._btn_math.background_color = theme.GOLD
            self._btn_math.color = theme.BG_DARK
            self._btn_stroop.background_color = theme.BORDER_DARK
            self._btn_stroop.color = theme.TEXT_PRIMARY

            self._stroop_row.opacity = 0
            self._stroop_row.disabled = True
            self._math_row.opacity = 1
            self._math_row.disabled = False
        else:
            self._btn_stroop.state = 'down'
            self._btn_math.state = 'normal'
            self._btn_stroop.background_color = theme.GOLD
            self._btn_stroop.color = theme.BG_DARK
            self._btn_math.background_color = theme.BORDER_DARK
            self._btn_math.color = theme.TEXT_PRIMARY

            self._stroop_row.opacity = 1
            self._stroop_row.disabled = False
            self._math_row.opacity = 0
            self._math_row.disabled = True
        if self._is_running:
            self._next_round()

    def start_task(self):
        self._start()

    def _toggle(self, *args):
        if self._is_running:
            self.stop()
        else:
            self._start()

    def _start(self):
        if self._is_running:
            return
        self._is_running = True
        self._score = 0
        self._score_lbl.text = 'Score: 0'
        self._start_btn.text = 'STOP'
        self._start_btn.color = theme.RED
        self._start_btn.bg_color = theme.DANGER_BUTTON_BG
        self._next_round()
        self._clock_event = Clock.schedule_interval(
            lambda dt: self._next_round(), 3.0
        )

    def stop(self):
        if self._clock_event:
            self._clock_event.cancel()
            self._clock_event = None
        self._is_running = False
        self._start_btn.text = 'START'
        self._start_btn.color = theme.GOLD
        self._start_btn.bg_color = theme.BUTTON_BG

    def _next_round(self):
        if self._is_math_mode:
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            is_add = random.choice([True, False])
            self._math_answer = a + b if is_add else a - b
            op = '+' if is_add else '-'
            self._math_problem = f'{a} {op} {b} = ?'
            self._display_lbl.text = self._math_problem
            self._display_lbl.color = theme.RED
            self._math_input.text = ''
        else:
            word = random.choice(self._colors)
            ink_name = random.choice(self._colors)
            self._display_word = word
            self._ink_color = self._color_map[ink_name]
            self._current_ink_name = ink_name
            self._display_lbl.text = word
            self._display_lbl.color = self._ink_color

    def _check_stroop(self, color_name):
        if not self._is_running:
            return
        if color_name == self._current_ink_name:
            self._score += 50
            self._score_lbl.text = f'Score: {self._score}'
        # Reset timer
        if self._clock_event:
            self._clock_event.cancel()
        self._next_round()
        self._clock_event = Clock.schedule_interval(
            lambda dt: self._next_round(), 3.0
        )

    def _check_math(self):
        if not self._is_running:
            return
        try:
            answer = int(self._math_input.text)
        except (ValueError, TypeError):
            answer = 0
        if answer == self._math_answer:
            self._score += 50
            self._score_lbl.text = f'Score: {self._score}'
        # Reset timer
        if self._clock_event:
            self._clock_event.cancel()
        self._next_round()
        self._clock_event = Clock.schedule_interval(
            lambda dt: self._next_round(), 3.0
        )

    def cleanup(self):
        """Call when removing widget to stop timers."""
        if self._clock_event:
            self._clock_event.cancel()


# ============================================================
# File: screens/dashboard_screen.py
# ============================================================

"""
Dashboard screen matching Flutter's DashboardScreen.
Shows welcome title, live status, stat cards, event log, and refresh button.
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle, RoundedRectangle


class DashboardScreen(BoxLayout):
    """Dashboard with status overview and event log."""

    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)
        self._app_state = app_state

        # Welcome title
        self._welcome_lbl = Label(
            text='WELCOME BACK, USER',
            font_size=theme.FONT_TITLE_MEDIUM,
            bold=True,
            color=theme.GOLD,
            size_hint_y=None,
            height=35,
            halign='left',
            valign='middle',
        )
        self._welcome_lbl.bind(size=self._welcome_lbl.setter('text_size'))
        self.add_widget(self._welcome_lbl)

        # Update welcome text when user changes
        app_state.bind(current_user=self._update_welcome)
        self._update_welcome()

        # Live status card
        status_card = GradientCard(
            size_hint_y=None,
            height=80,
            padding=[20, 15]
        )
        status_title = Label(
            text='LIVE STATUS',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=18,
            halign='left',
            valign='middle',
        )
        status_title.bind(size=status_title.setter('text_size'))
        status_card.add_widget(status_title)
        status_value = Label(
            text='OFFLINE',
            font_size=theme.FONT_HEADING_MEDIUM,
            color=theme.TEXT_SECONDARY,
            halign='left',
            valign='middle',
        )
        status_value.bind(size=status_value.setter('text_size'))
        status_card.add_widget(status_value)
        self.add_widget(status_card)

        # Stat cards row
        stats_row = BoxLayout(spacing=15, size_hint_y=None, height=100)
        stats_row.add_widget(self._build_stat_card('STRESS THRESHOLD', 'N/A', theme.RED))
        stats_row.add_widget(self._build_stat_card('FOCUS THRESHOLD', 'N/A', theme.TEAL))
        stats_row.add_widget(self._build_stat_card('NEURO XP', '0', theme.GOLD))
        self.add_widget(stats_row)

        # Event log title
        log_title = Label(
            text='EVENT LOG',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=18,
            halign='left',
            valign='middle',
        )
        log_title.bind(size=log_title.setter('text_size'))
        self.add_widget(log_title)

        # Event log box
        log_box = GradientCard(size_hint_y=0.4, padding=[10, 10])
        log_label = Label(
            text='No events yet',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEAL,
            halign='left',
            valign='top',
        )
        log_label.bind(size=log_label.setter('text_size'))
        log_box.add_widget(log_label)
        self.add_widget(log_box)

        # Refresh button
        refresh_btn = ShadowButton(
            text='SYSTEM REFRESH',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=45,
            background_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self.add_widget(refresh_btn)

    def _update_welcome(self, *args):
        user = self._app_state.current_user
        name = (user.name.upper() if user and user.name else 'USER')
        self._welcome_lbl.text = f'WELCOME BACK, {name}'

    def _build_stat_card(self, title, value, accent_color):
        card = GradientCard(
            accent_color=accent_color,
            padding=[15, 10]
        )

        t = Label(
            text=title,
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
            size_hint_y=None,
            height=16,
            halign='left',
            valign='middle',
        )
        t.bind(size=t.setter('text_size'))
        card.add_widget(t)

        v = Label(
            text=value,
            font_size=theme.FONT_HEADING_LARGE,
            bold=True,
            color=accent_color,
            halign='left',
            valign='middle',
        )
        v.bind(size=v.setter('text_size'))
        card.add_widget(v)

        return card


# ============================================================
# File: screens/profile_screen.py
# ============================================================

"""
Profile screen matching Flutter's ProfileScreen.
Form with name, age, notes fields and save button.
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.graphics import Color, RoundedRectangle, Rectangle


class ProfileScreen(BoxLayout):
    """User profile editing form."""

    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)
        self._app_state = app_state

        # Title banner
        title_box = GradientCard(
            size_hint_y=None, height=50, padding=[10, 10], radius=10
        )
        title_lbl = Label(
            text='USER PROFILE',
            font_size=theme.FONT_TITLE_MEDIUM,
            bold=True,
            color=theme.GOLD,
            halign='left',
            valign='middle',
        )
        title_lbl.bind(size=title_lbl.setter('text_size'))
        title_box.add_widget(title_lbl)
        self.add_widget(title_box)

        # Form container
        form_box = GradientCard(
            padding=[20, 20]
        )
        # Full Name
        name_lbl = Label(
            text='FULL NAME:',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEAL,
            bold=True,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        name_lbl.bind(size=name_lbl.setter('text_size'))
        form_box.add_widget(name_lbl)

        self._name_input = TextInput(
            font_size=theme.FONT_BODY_REGULAR,
            multiline=False,
            size_hint_y=None,
            height=40,
            background_color=theme.INPUT_BG,
            foreground_color=theme.TEXT_PRIMARY,
            cursor_color=theme.TEAL,
            padding=[12, 10],
        )
        form_box.add_widget(self._name_input)

        # Age
        age_lbl = Label(
            text='AGE:',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEAL,
            bold=True,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        age_lbl.bind(size=age_lbl.setter('text_size'))
        form_box.add_widget(age_lbl)

        self._age_input = TextInput(
            font_size=theme.FONT_BODY_REGULAR,
            multiline=False,
            input_filter='int',
            size_hint_y=None,
            height=40,
            background_color=theme.INPUT_BG,
            foreground_color=theme.TEXT_PRIMARY,
            cursor_color=theme.TEAL,
            padding=[12, 10],
        )
        form_box.add_widget(self._age_input)

        # Clinical Notes
        notes_lbl = Label(
            text='CLINICAL NOTES:',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEAL,
            bold=True,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        notes_lbl.bind(size=notes_lbl.setter('text_size'))
        form_box.add_widget(notes_lbl)

        self._notes_input = TextInput(
            font_size=theme.FONT_BODY_REGULAR,
            multiline=True,
            size_hint_y=None,
            height=120,
            background_color=theme.INPUT_BG,
            foreground_color=theme.TEXT_PRIMARY,
            cursor_color=theme.TEAL,
            padding=[12, 10],
        )
        form_box.add_widget(self._notes_input)

        self.add_widget(form_box)

        # Save button
        save_btn = ShadowButton(
            text='SAVE PROFILE DATA',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=45,
            background_color=theme.GOLD,
            color=(0, 0, 0, 1),
        )
        save_btn.bind(on_press=lambda *a: self._save_profile())
        self.add_widget(save_btn)

        # Load initial profile data
        app_state.bind(current_user=self._load_profile)
        self._load_profile()

    def _load_profile(self, *args):
        user = self._app_state.current_user
        if user:
            self._name_input.text = user.name or ''
            self._age_input.text = user.age or ''
            self._notes_input.text = user.notes or ''

    def _save_profile(self):
        self._app_state.update_profile(
            name=self._name_input.text,
            age=self._age_input.text,
            notes=self._notes_input.text,
        )
        # Show feedback popup
        popup = Popup(
            title='',
            content=Label(
                text='PROFILE UPDATED.',
                font_size=theme.FONT_BODY_REGULAR,
                color=theme.TEXT_PRIMARY,
            ),
            size_hint=(None, None),
            size=(250, 120),
            background_color=theme.PANEL_BG,
            auto_dismiss=True,
        )
        popup.open()
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: popup.dismiss(), 1.5)


# ============================================================
# File: screens/calibration_screen.py
# ============================================================

"""
Calibration screen matching Flutter's CalibrationScreen.
Contains EEG graph, task cards (baseline/stress/focus), and active task display.
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.clock import Clock


class CalibrationScreen(BoxLayout):
    """Calibration module with EEG graph, task selector, and active task display."""

    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self._app_state = app_state
        self._active_task = None
        self._active_widget = None
        self._sequence_running = False
        self._sequence = []
        self._sequence_idx = 0
        self._time_left = 0
        self._seq_clock = None
        self._manual_clock = None
        self._manual_time = 0
        self._was_running = False

        # EEG Graph (top)
        self._eeg_graph = EegGraph(size_hint_y=None, height=160)
        self.add_widget(self._eeg_graph)

        # Feedback label
        self._feedback_box = BoxLayout(
            size_hint_y=None, height=40, padding=[20, 5],
        )
        with self._feedback_box.canvas.before:
            Color(0, 0, 0, 1)
            self._feedback_box._bg = Rectangle(
                pos=self._feedback_box.pos, size=self._feedback_box.size
            )
        self._feedback_box.bind(
            pos=lambda inst, val: setattr(inst._bg, 'pos', val),
            size=lambda inst, val: setattr(inst._bg, 'size', val),
        )
        self._feedback_lbl = Label(
            text='Waiting for signal...',
            font_size=theme.FONT_BODY_LARGE,
            color=theme.TEXT_MUTED,
            halign='center',
            valign='middle',
        )
        self._feedback_lbl.bind(size=self._feedback_lbl.setter('text_size'))
        self._feedback_box.add_widget(self._feedback_lbl)
        self.add_widget(self._feedback_box)

        # Content area (task cards or active task)
        self._content_area = BoxLayout(padding=[20, 10])
        self.add_widget(self._content_area)
        self._show_task_cards()

        # Execute Sequence Button
        self._seq_btn_box = BoxLayout(size_hint_y=None, height=60, padding=[20, 10])
        self._execute_btn = ShadowButton(
            text='EXECUTE FULL SEQUENCE (1 HOUR)',
            font_size=theme.FONT_BODY_LARGE,
            bold=True,
            background_color=theme.GOLD,
            color=theme.BG_DARK,
        )
        self._execute_btn.bind(on_press=lambda *a: self._start_sequence())
        self._seq_btn_box.add_widget(self._execute_btn)
        self.add_widget(self._seq_btn_box)

        # Control bar
        control_bar = GradientCard(
            orientation='horizontal',
            size_hint_y=None, height=50, padding=[15, 5], spacing=10,
        )

        self._status_lbl = Label(
            text='STATUS: IDLE',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_SECONDARY,
            size_hint_x=0.3,
            halign='left',
            valign='middle',
        )
        self._status_lbl.bind(size=self._status_lbl.setter('text_size'))
        control_bar.add_widget(self._status_lbl)

        xp_lbl = Label(
            text='NEURO XP: 0',
            font_size=theme.FONT_HEADING_MEDIUM,
            bold=True,
            color=theme.GOLD,
            size_hint_x=0.25,
        )
        control_bar.add_widget(xp_lbl)

        self._timer_lbl = Label(
            text='00:00',
            font_size=theme.FONT_TIMER,
            bold=True,
            color=theme.TEAL,
            size_hint_x=0.15,
        )
        control_bar.add_widget(self._timer_lbl)

        self._abort_btn = ShadowButton(
            text='ABORT',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_x=0.3,
            background_color=theme.DANGER_BUTTON_BG,
            color=theme.RED,
            disabled=True,
        )
        self._abort_btn.bind(on_press=lambda *a: self._stop_task())
        control_bar.add_widget(self._abort_btn)
        self.add_widget(control_bar)

    def _show_task_cards(self):
        self._content_area.clear_widgets()
        cards_row = BoxLayout(spacing=15)

        cards_row.add_widget(self._build_task_card(
            'BASELINE', 'Relaxation', 'Breathing Exercises',
            theme.GOLD, 'baseline'
        ))
        cards_row.add_widget(self._build_task_card(
            'STRESS', 'High Load', 'Math / Stroop',
            theme.RED, 'stress'
        ))
        cards_row.add_widget(self._build_task_card(
            'FOCUS', 'Flow State', 'Tracking / Reading',
            theme.TEAL, 'focus'
        ))
        self._content_area.add_widget(cards_row)

    def _build_task_card(self, title, subtitle, desc, accent, task_id):
        card = GradientCard(padding=[15, 15], spacing=5)

        t = Label(
            text=title,
            font_size=theme.FONT_HEADING_MEDIUM,
            bold=True,
            color=theme.GOLD,
            size_hint_y=None,
            height=25,
            halign='left',
            valign='middle',
        )
        t.bind(size=t.setter('text_size'))
        card.add_widget(t)

        s = Label(
            text=subtitle,
            font_size=theme.FONT_BODY_SMALL,
            italic=True,
            color=theme.TEXT_MUTED,
            size_hint_y=None,
            height=18,
            halign='left',
            valign='middle',
        )
        s.bind(size=s.setter('text_size'))
        card.add_widget(s)

        d = Label(
            text=desc,
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        d.bind(size=d.setter('text_size'))
        card.add_widget(d)

        # Spacer
        card.add_widget(Label())

        # Spacer
        card.add_widget(Label())

        btn = ShadowButton(
            text='INITIALIZE',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=40,
            background_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        btn.bind(on_press=lambda *a, tid=task_id: self._start_task(tid))
        card.add_widget(btn)

        return card

    def _start_task(self, task_id):
        self._active_task = task_id
        self._status_lbl.text = f'STATUS: {task_id.upper()}'
        self._abort_btn.disabled = False
        self._execute_btn.disabled = True

        self._content_area.clear_widgets()

        task_container = GradientCard(padding=[10, 10])

        if task_id == 'baseline':
            self._active_widget = BreathingWidget()
        elif task_id == 'stress':
            self._active_widget = StroopWidget()
        elif task_id == 'focus':
            self._active_widget = FocusWidget()

        if self._active_widget:
            task_container.add_widget(self._active_widget)
        self._content_area.add_widget(task_container)

        self._manual_time = 0
        self._was_running = False
        if self._manual_clock:
            self._manual_clock.cancel()
        self._manual_clock = Clock.schedule_interval(self._manual_tick, 1.0)
        self._timer_lbl.text = '00:00'

    def _manual_tick(self, dt):
        if self._sequence_running:
            return

        is_running = getattr(self._active_widget, '_is_running', False)
        
        if is_running and not self._was_running:
            self._manual_time = 0
            self._was_running = True
        elif not is_running and self._was_running:
            self._was_running = False
            
        if is_running:
            self._manual_time += 1
            mins = self._manual_time // 60
            secs = self._manual_time % 60
            self._timer_lbl.text = f'{mins:02d}:{secs:02d}'

    def _start_sequence(self):
        self._sequence = [
            ('baseline', '4-7-8'),
            ('baseline', 'box'),
            ('focus', 'tracking'),
            ('focus', 'reading'),
            ('stress', 'stroop'),
            ('stress', 'math')
        ]
        self._sequence_idx = 0
        self._sequence_running = True
        self._run_next_in_sequence()

    def _run_next_in_sequence(self):
        if self._sequence_idx >= len(self._sequence):
            self._stop_task()
            return

        task_id, mode = self._sequence[self._sequence_idx]
        self._start_task(task_id)
        
        if self._active_widget and hasattr(self._active_widget, 'set_mode'):
            self._active_widget.set_mode(mode)
            if hasattr(self._active_widget, 'start_task'):
                self._active_widget.start_task()

        self._time_left = 600 # 10 minutes * 60 seconds
        if self._seq_clock:
            self._seq_clock.cancel()
        self._seq_clock = Clock.schedule_interval(self._sequence_tick, 1.0)
        self._update_timer_label()

    def _sequence_tick(self, dt):
        if self._time_left > 0:
            self._time_left -= 1
            self._update_timer_label()
        else:
            if self._active_widget and hasattr(self._active_widget, '_score'):
                mode = getattr(self._active_widget, '_mode', getattr(self._active_widget, '_is_math_mode', ''))
                self._app_state.save_current_user_score(f'{self._active_task}_{mode}', self._active_widget._score)
            
            if self._active_widget and hasattr(self._active_widget, 'stop'):
                self._active_widget.stop()

            self._sequence_idx += 1
            self._run_next_in_sequence()

    def _update_timer_label(self):
        mins = self._time_left // 60
        secs = self._time_left % 60
        self._timer_lbl.text = f'{mins:02d}:{secs:02d}'

    def _stop_task(self):
        if self._seq_clock:
            self._seq_clock.cancel()
            self._seq_clock = None
        if self._manual_clock:
            self._manual_clock.cancel()
            self._manual_clock = None
        self._sequence_running = False
        self._timer_lbl.text = '00:00'

        if self._active_widget and hasattr(self._active_widget, 'cleanup'):
            self._active_widget.cleanup()
        self._active_task = None
        self._active_widget = None
        self._status_lbl.text = 'STATUS: IDLE'
        self._abort_btn.disabled = True
        self._execute_btn.disabled = False
        self._show_task_cards()


# ============================================================
# File: screens/random_forest_screen.py
# ============================================================

"""
Random Forest training screen.
Simulated Random Forest training pipeline with console output.
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.clock import Clock


class RandomForestScreen(BoxLayout):
    """Random Forest training UI with console output."""

    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)
        self._app_state = app_state
        self._log_lines = []
        self._is_training = False
        self._is_checking = False
        self._scheduled_events = []

        # Title
        title = Label(
            text='RANDOM FOREST TRAINING',
            font_size=theme.FONT_TITLE_MEDIUM,
            bold=True,
            color=theme.GOLD,
            size_hint_y=None,
            height=35,
            halign='left',
            valign='middle',
        )
        title.bind(size=title.setter('text_size'))
        self.add_widget(title)

        # Console output
        console_box = GradientCard(padding=[10, 10])

        scroll = ScrollView()
        self._console_label = Label(
            text='',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEAL,
            halign='left',
            valign='top',
            size_hint_y=None,
            markup=False,
            padding=[10, 10],
        )
        self._console_label.bind(
            texture_size=lambda inst, sz: setattr(inst, 'height', max(sz[1], 100)),
            width=lambda inst, w: setattr(inst, 'text_size', (w - 20, None)),
        )
        scroll.add_widget(self._console_label)
        console_box.add_widget(scroll)
        self.add_widget(console_box)

        # Generate demo data button
        self._demo_btn = ShadowButton(
            text='GENERATE DEMO DATA (TESTING)',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=44,
            background_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._demo_btn.bind(on_press=lambda *a: self._generate_demo_data())
        self.add_widget(self._demo_btn)

        # Train button
        self._train_btn = ShadowButton(
            text='EXECUTE TRAINING PIPELINE',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=44,
            background_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._train_btn.bind(on_press=lambda *a: self._start_training())
        self.add_widget(self._train_btn)

        # ── Compatibility check button ──
        self._compat_btn = ShadowButton(
            text='RUN COMPATIBILITY CHECK',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=44,
            background_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._compat_btn.bind(on_press=lambda *a: self._run_compat_check())
        self.add_widget(self._compat_btn)

        # ── Compatibility check output area ──
        compat_title = Label(
            text='COMPATIBILITY DIAGNOSTIC',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            color=theme.TEXT_SECONDARY,
            size_hint_y=None,
            height=25,
            halign='left',
            valign='middle',
        )
        compat_title.bind(size=compat_title.setter('text_size'))
        self.add_widget(compat_title)

        compat_card = GradientCard(padding=[10, 10])
        compat_scroll = ScrollView()
        self._compat_label = Label(
            text='Press RUN COMPATIBILITY CHECK to diagnose pipeline.',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEAL,
            halign='left',
            valign='top',
            size_hint_y=None,
            markup=False,
            padding=[10, 10],
        )
        self._compat_label.bind(
            texture_size=lambda inst, sz: setattr(inst, 'height', max(sz[1], 80)),
            width=lambda inst, w: setattr(inst, 'text_size', (w - 20, None)),
        )
        compat_scroll.add_widget(self._compat_label)
        compat_card.add_widget(compat_scroll)
        self.add_widget(compat_card)

    def _add_log(self, line):
        self._log_lines.append(line)
        self._console_label.text = '\n'.join(self._log_lines)

    def _clear_log(self):
        self._log_lines.clear()
        self._console_label.text = ''

    def _generate_demo_data(self):
        if self._is_training:
            return
        self._clear_log()
        self._add_log('>>> GENERATING SYNTHETIC EEG DATASET...')
        self._add_log('>>> Simulating 12-feature band power vectors...')
        ev = Clock.schedule_once(lambda dt: self._finish_demo(), 1.0)
        self._scheduled_events.append(ev)

    def _finish_demo(self):
        self._add_log('>>> SUCCESS. Generated 1440 samples (480 per class).')
        self._add_log('>>> Features: delta, theta, alpha, beta, gamma power,')
        self._add_log('    focus_index, stress_index, alpha_theta_ratio,')
        self._add_log('    beta_alpha_ratio, delta_alpha_ratio,')
        self._add_log('    spectral_entropy, mean_power')
        self._add_log('>>> You can now EXECUTE TRAINING PIPELINE.')

    def _start_training(self):
        if self._is_training:
            return
        self._is_training = True
        self._train_btn.text = 'TRAINING...'
        self._train_btn.disabled = True
        self._demo_btn.disabled = True
        self._compat_btn.disabled = True

        self._clear_log()
        self._add_log('>>> INITIATING RANDOM FOREST TRAINING PIPELINE...')
        self._simulate_training()

    def _simulate_training(self):
        """Simulate the 9-step Random Forest training pipeline."""
        self._unschedule_all()

        steps = [
            # Step 1 (0.2s)
            (0.2, [
                '[INFO] Loading calibration session data...',
                '[INFO] Found 3 classes: Calm, Stressed, Focused',
            ]),
            # Step 2 (0.5s)
            (0.5, [
                '[INFO] Feature extraction complete.',
                '[INFO] Samples: Calm=480  Stressed=480  Focused=480',
                '[INFO] Feature vector size: 12 features per sample',
                '[INFO] Total dataset: 1440 samples',
            ]),
            # Step 3 (0.9s)
            (0.9, [
                '[INFO] Applying StandardScaler normalization...',
                '[INFO] Mean per feature: [0.82, 1.14, 2.31, ...]',
                '[INFO] Std  per feature: [0.21, 0.33, 0.58, ...]',
            ]),
            # Step 4 (1.3s)
            (1.3, [
                '[INFO] Splitting dataset: 80% train / 20% test',
                '[INFO] Train: 1152 samples | Test: 288 samples',
                '[INFO] Stratified split \u2014 class balance preserved',
            ]),
            # Step 5 (1.8s)
            (1.8, [
                '[INFO] Running 5-fold cross-validation...',
                '[INFO] Fold 1/5 \u2014 CV Accuracy: 87.3%',
                '[INFO] Fold 2/5 \u2014 CV Accuracy: 89.1%',
                '[INFO] Fold 3/5 \u2014 CV Accuracy: 86.8%',
                '[INFO] Fold 4/5 \u2014 CV Accuracy: 90.2%',
                '[INFO] Fold 5/5 \u2014 CV Accuracy: 88.6%',
                '[INFO] Mean CV Accuracy: 88.4% \u00b1 1.2%',
            ]),
            # Step 6 (2.5s)
            (2.5, [
                '[INFO] Training RandomForestClassifier...',
                '[INFO] n_estimators=200  max_depth=None',
                '[INFO] min_samples_split=2  n_jobs=-1',
                '[INFO] Building tree   1/200...',
                '[INFO] Building tree  50/200...',
                '[INFO] Building tree 100/200...',
                '[INFO] Building tree 150/200...',
                '[INFO] Building tree 200/200...',
            ]),
            # Step 7 (3.5s)
            (3.5, [
                '[INFO] Training complete.',
                '[INFO] OOB Score: 91.2%',
                '[INFO] Test Accuracy: 90.6%',
                '[INFO] ',
                '[INFO] Classification Report:',
                '[INFO]              precision  recall  f1-score',
                '[INFO] Calm           0.93      0.91    0.92',
                '[INFO] Stressed       0.89      0.90    0.89',
                '[INFO] Focused        0.92      0.93    0.92',
            ]),
            # Step 8 (4.2s)
            (4.2, [
                '[INFO] Feature Importances (top 5):',
                '[INFO] 1. focus_index        0.187',
                '[INFO] 2. alpha_power        0.163',
                '[INFO] 3. stress_index       0.141',
                '[INFO] 4. beta_power         0.128',
                '[INFO] 5. spectral_entropy   0.097',
            ]),
            # Step 9 (4.8s)
            (4.8, [
                '[INFO] Saving model to user profile...',
                '[INFO] Model size: 2.3 MB',
                '[SUCCESS] Random Forest model saved. \u2713',
                '[SUCCESS] User profile updated. \u2713',
                '[SUCCESS] Ready for live classification. \u2713',
            ]),
        ]

        for delay, lines in steps:
            ev = Clock.schedule_once(
                lambda dt, msgs=lines: self._add_log_batch(msgs),
                delay,
            )
            self._scheduled_events.append(ev)

        # Re-enable buttons after all steps
        ev = Clock.schedule_once(lambda dt: self._finish_training(), 5.3)
        self._scheduled_events.append(ev)

    def _add_log_batch(self, lines):
        for line in lines:
            self._add_log(line)

    def _finish_training(self):
        self._is_training = False
        self._train_btn.text = 'EXECUTE TRAINING PIPELINE'
        self._train_btn.disabled = False
        self._demo_btn.disabled = False
        self._compat_btn.disabled = False

    # ──────────────────────────────────────────────────────────
    # Compatibility Check
    # ──────────────────────────────────────────────────────────

    def _run_compat_check(self):
        """Run the compatibility diagnostic in a background-friendly way."""
        if self._is_training or self._is_checking:
            return

        self._is_checking = True
        self._compat_btn.text = 'RUNNING...'
        self._compat_btn.disabled = True
        self._train_btn.disabled = True
        self._demo_btn.disabled = True
        self._compat_label.text = 'Running diagnostic...\n'

        # Schedule the actual work for the next frame so the UI updates first
        Clock.schedule_once(lambda dt: self._do_compat_check(), 0.1)

    def _do_compat_check(self):
        """Execute the actual compatibility check."""
        try:
            output = run_compatibility_check()
        except Exception as e:
            import traceback
            output = f'ERROR running compatibility check:\n{traceback.format_exc()}'

        self._compat_label.text = output
        self._is_checking = False
        self._compat_btn.text = 'RUN COMPATIBILITY CHECK'
        self._compat_btn.disabled = False
        self._train_btn.disabled = False
        self._demo_btn.disabled = False

    def _unschedule_all(self):
        for ev in self._scheduled_events:
            ev.cancel()
        self._scheduled_events.clear()

    def cleanup(self):
        """Cleanup scheduled events."""
        self._unschedule_all()


# ============================================================
# File: screens/monitoring_screen.py
# ============================================================

"""
Monitoring screen matching Flutter's MonitoringScreen.
Live EEG monitoring with Neuro-Game (mind visualizer) and Technical Data tabs.
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics import Color, RoundedRectangle, Rectangle


class MonitoringScreen(BoxLayout):
    """Live monitoring with tab-like view: Neuro-Game and Technical Data."""

    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)
        self._app_state = app_state
        self._is_monitoring = False
        self._state_label = 'IDLE'
        self._confidence = 0.0
        self._showing_game = True

        # Tab bar
        tab_row = BoxLayout(size_hint_y=None, height=45, spacing=5)
        with tab_row.canvas.before:
            Color(*theme.BG_DARK)
            tab_row._bg = RoundedRectangle(
                pos=tab_row.pos, size=tab_row.size, radius=[8]
            )
        tab_row.bind(
            pos=lambda inst, val: setattr(inst._bg, 'pos', val),
            size=lambda inst, val: setattr(inst._bg, 'size', val),
        )

        self._tab_game = ToggleButton(
            text='NEURO-GAME',
            group='monitor_tab',
            state='down',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.GOLD,
        )
        self._tab_game.bind(on_press=lambda *a: self._switch_tab(True))

        self._tab_tech = ToggleButton(
            text='TECHNICAL DATA',
            group='monitor_tab',
            state='normal',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEXT_MUTED,
        )
        self._tab_tech.bind(on_press=lambda *a: self._switch_tab(False))

        tab_row.add_widget(self._tab_game)
        tab_row.add_widget(self._tab_tech)
        self.add_widget(tab_row)

        # Content area
        self._content_area = BoxLayout()
        self.add_widget(self._content_area)

        # Build both views
        self._game_view = self._build_game_view()
        self._tech_view = self._build_tech_view()

        # Show game view by default
        self._content_area.add_widget(self._game_view)

        # Control button
        self._control_btn = ShadowButton(
            text='INITIATE LIVE STREAM',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=44,
            background_color=theme.BUTTON_BG,
            color=theme.GOLD,
        )
        self._control_btn.bind(on_press=lambda *a: self._toggle_monitoring())
        self.add_widget(self._control_btn)

    def _build_game_view(self):
        view = MindVisualizer(
            is_active=False,
            state_label='IDLE',
        )
        return view

    def _build_tech_view(self):
        view = BoxLayout(orientation='vertical', spacing=10)

        # State display
        state_box = GradientCard(
            size_hint_y=None,
            height=120,
            padding=[20, 20]
        )

        self._state_display = Label(
            text='IDLE',
            font_size=theme.FONT_DISPLAY_LARGE,
            bold=True,
            color=theme.TEXT_MUTED,
        )
        state_box.add_widget(self._state_display)

        conf_row = BoxLayout(size_hint_y=None, height=20)
        self._conf_lbl = Label(
            text='CONF: 0%',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
        )
        ver_lbl = Label(
            text='VER: ---',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
        )
        conf_row.add_widget(self._conf_lbl)
        conf_row.add_widget(ver_lbl)
        state_box.add_widget(conf_row)
        view.add_widget(state_box)

        # EEG Graph
        self._tech_eeg = EegGraph()
        view.add_widget(self._tech_eeg)

        # Band power bars
        self._band_bars = BandPowerBars(size_hint_y=None, height=160)
        view.add_widget(self._band_bars)

        return view

    def _switch_tab(self, show_game):
        if self._showing_game == show_game:
            return
        self._showing_game = show_game
        self._content_area.clear_widgets()
        if show_game:
            self._content_area.add_widget(self._game_view)
        else:
            self._content_area.add_widget(self._tech_view)

    def _toggle_monitoring(self):
        self._is_monitoring = not self._is_monitoring
        self._game_view.is_active = self._is_monitoring

        if self._is_monitoring:
            self._control_btn.text = 'TERMINATE STREAM'
            self._control_btn.color = theme.RED
            self._control_btn.background_color = theme.DANGER_BUTTON_BG
        else:
            self._control_btn.text = 'INITIATE LIVE STREAM'
            self._control_btn.color = theme.GOLD
            self._control_btn.background_color = theme.BUTTON_BG
            self._state_label = 'IDLE'
            self._confidence = 0.0
            self._state_display.text = 'IDLE'
            self._state_display.color = theme.TEXT_MUTED
            self._conf_lbl.text = 'CONF: 0%'


# ============================================================
# File: screens/login_screen.py
# ============================================================

"""
Login screen with previous user quick-select tiles and new user input.
"""
import re
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, RoundedRectangle, Rectangle, Ellipse
from kivy.metrics import dp, sp


# Avatar color palette for user tiles
_AVATAR_COLORS = [
    '#2563eb', '#16a34a', '#ea0c0c',
    '#f59e0b', '#7c3aed', '#db2777',
]


class UserTile(ButtonBehavior, Widget):
    """Clickable user avatar tile for quick login."""

    def __init__(self, username, on_select=None, **kwargs):
        self.username = username
        self._on_select = on_select
        kwargs['size_hint'] = (None, None)
        kwargs['size'] = (dp(80), dp(90))
        super().__init__(**kwargs)
        self._pressed = False

        self.bind(pos=self._update_canvas, size=self._update_canvas)
        self._draw()

    def _get_avatar_color(self):
        idx = hash(self.username) % len(_AVATAR_COLORS)
        return theme.rgba_hex(_AVATAR_COLORS[idx], 1.0)

    def _draw(self):
        self._update_canvas()

    def _update_canvas(self, *args):
        self.canvas.before.clear()
        self.canvas.after.clear()

        with self.canvas.before:
            # Background rounded rect
            if self._pressed:
                Color(*theme.rgba_hex('#2563eb', 0.4))
            else:
                Color(*theme.rgba_hex('#1a2535', 1.0))
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[12, 12, 12, 12],
            )

            # Avatar circle
            avatar_color = self._get_avatar_color()
            Color(*avatar_color)
            avatar_d = dp(44)
            avatar_x = self.x + (self.width - avatar_d) / 2
            avatar_y = self.y + self.height - avatar_d - dp(8)
            Ellipse(
                pos=(avatar_x, avatar_y),
                size=(avatar_d, avatar_d),
            )

        with self.canvas.after:
            pass  # Labels added as children below

        # Remove old children labels
        self.clear_widgets()

        # Initial letter label (centered on avatar)
        letter = self.username[0].upper() if self.username else '?'
        avatar_d = dp(44)
        avatar_x = self.x + (self.width - avatar_d) / 2
        avatar_y = self.y + self.height - avatar_d - dp(8)

        initial_lbl = Label(
            text=letter,
            font_size=sp(20),
            bold=True,
            color=(1, 1, 1, 1),
            size_hint=(None, None),
            size=(avatar_d, avatar_d),
            pos=(avatar_x, avatar_y),
            halign='center',
            valign='middle',
        )
        initial_lbl.bind(size=initial_lbl.setter('text_size'))
        self.add_widget(initial_lbl)

        # Username label below avatar
        display_name = self.username
        if len(display_name) > 9:
            display_name = display_name[:9] + '\u2026'

        name_lbl = Label(
            text=display_name,
            font_size=sp(11),
            color=theme.rgba_hex('#cbd5e1', 1.0),
            size_hint=(None, None),
            size=(self.width, dp(16)),
            pos=(self.x, self.y + dp(4)),
            halign='center',
            valign='middle',
        )
        name_lbl.bind(size=name_lbl.setter('text_size'))
        self.add_widget(name_lbl)

    def on_press(self):
        self._pressed = True
        self._update_canvas()

    def on_release(self):
        self._pressed = False
        self._update_canvas()
        if self._on_select:
            self._on_select(self.username)


class LoginScreen(FloatLayout):
    """Login screen with previous user tiles and username input."""

    def __init__(self, app_state, on_login=None, **kwargs):
        super().__init__(**kwargs)
        self._app_state = app_state
        self._on_login = on_login
        self._error = None

        # Background
        with self.canvas.before:
            Color(*theme.BG_DARK)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

        # Check for existing users
        saved_users = app_state.get_sorted_users()
        has_users = len(saved_users) > 0

        # Calculate card height based on whether we have users
        card_height = 540 if has_users else 420

        # Center card container
        card = GradientCard(
            padding=[30, 30],
            size_hint=(None, None),
            size=(400, card_height),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
        )

        # Title
        title = Label(
            text='NEURO-MENTOR',
            font_size=32,
            bold=True,
            color=theme.GOLD,
            size_hint_y=None,
            height=50,
        )
        card.add_widget(title)

        # Subtitle
        subtitle = Label(
            text='Multi-User Brain Computer Interface System',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
            italic=True,
            size_hint_y=None,
            height=25,
        )
        card.add_widget(subtitle)

        # Spacer
        card.add_widget(Label(size_hint_y=None, height=12))

        # ============================================================
        # PREVIOUS USERS SECTION
        # ============================================================
        if has_users:
            # Section label
            prev_label = Label(
                text='PREVIOUS USERS',
                font_size=theme.FONT_HEADING_LARGE,
                bold=True,
                color=theme.TEXT_PRIMARY,
                size_hint_y=None,
                height=28,
                halign='left',
                valign='middle',
            )
            prev_label.bind(size=prev_label.setter('text_size'))
            card.add_widget(prev_label)

            # Spacer
            card.add_widget(Label(size_hint_y=None, height=8))

            # Horizontal scroll of user tiles
            tile_scroll = ScrollView(
                size_hint_y=None,
                height=dp(95),
                do_scroll_x=True,
                do_scroll_y=False,
            )
            tile_row = BoxLayout(
                orientation='horizontal',
                spacing=12,
                size_hint_x=None,
            )
            tile_row.bind(minimum_width=tile_row.setter('width'))

            for user in saved_users:
                tile = UserTile(
                    username=user.username,
                    on_select=self._quick_login,
                )
                tile_row.add_widget(tile)

            tile_scroll.add_widget(tile_row)
            card.add_widget(tile_scroll)

            # Spacer
            card.add_widget(Label(size_hint_y=None, height=8))

            # Divider line
            divider = Widget(size_hint_y=None, height=dp(1))
            with divider.canvas:
                Color(*theme.rgba_hex('#334155', 1.0))
                divider._line = Rectangle(pos=divider.pos, size=divider.size)
            divider.bind(
                pos=lambda inst, val: setattr(inst._line, 'pos', val),
                size=lambda inst, val: setattr(inst._line, 'size', val),
            )
            card.add_widget(divider)

            # "OR SIGN IN AS NEW USER" label
            or_label = Label(
                text='OR SIGN IN AS NEW USER',
                font_size=sp(11),
                color=theme.rgba_hex('#64748b', 1.0),
                size_hint_y=None,
                height=24,
                halign='center',
                valign='middle',
            )
            or_label.bind(size=or_label.setter('text_size'))
            card.add_widget(or_label)
        else:
            # Info text when no previous users
            info = Label(
                text='Enter your username to login or create a new profile.\n'
                     'Each user has isolated data and trained models.',
                font_size=theme.FONT_BODY_SMALL,
                color=theme.TEXT_SECONDARY,
                halign='center',
                valign='middle',
                size_hint_y=None,
                height=50,
            )
            info.bind(size=info.setter('text_size'))
            card.add_widget(info)

        # ============================================================
        # NEW USER INPUT SECTION
        # ============================================================

        # Username label
        usr_label = Label(
            text='USERNAME',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.GOLD,
            bold=True,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        usr_label.bind(size=usr_label.setter('text_size'))
        card.add_widget(usr_label)

        # Username input
        self._username_input = TextInput(
            hint_text='Enter your username (alphanumeric)',
            font_size=theme.FONT_BODY_REGULAR,
            multiline=False,
            size_hint_y=None,
            height=40,
            background_color=theme.INPUT_BG,
            foreground_color=theme.TEXT_PRIMARY,
            hint_text_color=theme.TEXT_MUTED,
            cursor_color=theme.TEAL,
            padding=[12, 10],
        )
        self._username_input.bind(on_text_validate=lambda *a: self._login())
        card.add_widget(self._username_input)

        # Help text
        help_lbl = Label(
            text='Use letters, numbers, and underscores only',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        help_lbl.bind(size=help_lbl.setter('text_size'))
        card.add_widget(help_lbl)

        # Error label
        self._error_lbl = Label(
            text='',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.RED,
            size_hint_y=None,
            height=20,
            halign='left',
            valign='middle',
        )
        self._error_lbl.bind(size=self._error_lbl.setter('text_size'))
        card.add_widget(self._error_lbl)

        # Login button
        login_btn = ShadowButton(
            text='ENTER SYSTEM',
            font_size=theme.FONT_BODY_REGULAR,
            bold=True,
            size_hint_y=None,
            height=44,
            background_color=theme.GOLD,
            color=(0, 0, 0, 1),
        )
        login_btn.bind(on_press=lambda *a: self._login())
        card.add_widget(login_btn)

        self.add_widget(card)

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def _quick_login(self, username):
        """Login via user tile tap."""
        self._app_state.login(username)
        if self._on_login:
            self._on_login()

    def _login(self):
        username = self._username_input.text.strip()

        if not username:
            self._error_lbl.text = 'Username cannot be empty'
            return

        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            self._error_lbl.text = 'Username: letters, numbers, underscores only'
            return

        self._error_lbl.text = ''
        self._app_state.login(username)
        if self._on_login:
            self._on_login()


# ============================================================
# File: screens/main_shell.py
# ============================================================

"""
Main shell with collapsible sidebar navigation.
Matches Flutter's MainShell with animated sidebar overlay.
"""
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.animation import Animation
from kivy.properties import BooleanProperty, NumericProperty
from kivy.clock import Clock


PAGE_NAMES = ['DASHBOARD', 'PROFILE', 'CALIBRATE', 'RANDOM FOREST', 'LIVE FEED']
SIDEBAR_WIDTH = 240


class MainShell(FloatLayout):
    """Main app shell with top bar, sidebar overlay, and screen switching."""

    sidebar_open = BooleanProperty(False)

    def __init__(self, app_state, on_logout=None, **kwargs):
        super().__init__(**kwargs)
        self._app_state = app_state
        self._on_logout = on_logout
        self._screens = {}
        self._current_screen = None

        # Background
        with self.canvas.before:
            Color(*theme.BG_DARK)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

        # ============================================================
        # MAIN CONTENT (top bar + screen area) - full width always
        # ============================================================
        self._main_column = BoxLayout(orientation='vertical')

        # Top bar
        top_bar = BoxLayout(
            size_hint_y=None, height=50,
            padding=[8, 0], spacing=10,
        )
        with top_bar.canvas.before:
            Color(*theme.SIDEBAR_BG)
            top_bar._bg = Rectangle(pos=top_bar.pos, size=top_bar.size)
            Color(*theme.SIDEBAR_BORDER)
            top_bar._border = Rectangle(pos=top_bar.pos, size=(1, 1))
        def _update_top_bar(inst, val):
            inst._bg.pos = inst.pos
            inst._bg.size = inst.size
            inst._border.pos = (inst.x, inst.y)
            inst._border.size = (inst.width, 1)
        top_bar.bind(pos=_update_top_bar, size=_update_top_bar)

        # Menu toggle button
        self._menu_btn = MenuBurgerButton(
            size_hint=(None, None),
            size=(45, 40),
            pos_hint={'center_y': 0.5},
            color=theme.GOLD,
        )
        self._menu_btn.bind(on_press=lambda *a: self._toggle_sidebar())
        top_bar.add_widget(self._menu_btn)

        # App title
        title = Label(
            text='NEUROMENTOR',
            font_size=theme.FONT_HEADING_MEDIUM,
            bold=True,
            color=theme.GOLD,
            size_hint_x=None,
            width=160,
            halign='left',
            valign='middle',
        )
        title.bind(size=title.setter('text_size'))
        top_bar.add_widget(title)

        # Spacer
        top_bar.add_widget(Label())

        # Page indicator
        self._page_indicator = Label(
            text='DASHBOARD',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
            size_hint_x=None,
            width=100,
            halign='right',
            valign='middle',
        )
        self._page_indicator.bind(size=self._page_indicator.setter('text_size'))
        top_bar.add_widget(self._page_indicator)

        self._main_column.add_widget(top_bar)

        # Screen content area
        self._screen_area = BoxLayout()
        self._main_column.add_widget(self._screen_area)

        self.add_widget(self._main_column)

        # ============================================================
        # BACKDROP (semi-transparent overlay, dismisses sidebar on tap)
        # ============================================================
        self._backdrop = _Backdrop(on_tap=self._close_sidebar)
        self._backdrop.opacity = 0
        self.add_widget(self._backdrop)

        # ============================================================
        # SIDEBAR (slides in from left over content)
        # ============================================================
        self._sidebar = SidebarNavigation(
            app_state=app_state,
            on_item_selected=self._close_sidebar,
            on_switch_user=self._handle_switch_user,
            size_hint=(None, 1),
            width=SIDEBAR_WIDTH,
        )
        self._sidebar.x = -SIDEBAR_WIDTH  # Start off-screen
        self.add_widget(self._sidebar)

        # Build screens
        self._build_screens()

        # Listen for page changes
        app_state.bind(selected_page_index=self._on_page_change)
        self._on_page_change()

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def _build_screens(self):

        self._screens = {
            0: DashboardScreen(app_state=self._app_state),
            1: ProfileScreen(app_state=self._app_state),
            2: CalibrationScreen(app_state=self._app_state),
            3: RandomForestScreen(app_state=self._app_state),
            4: MonitoringScreen(app_state=self._app_state),
        }

    def _on_page_change(self, *args):
        idx = self._app_state.selected_page_index
        self._page_indicator.text = PAGE_NAMES[idx] if idx < len(PAGE_NAMES) else ''

        self._screen_area.clear_widgets()
        screen = self._screens.get(idx)
        if screen:
            self._screen_area.add_widget(screen)
            self._current_screen = screen

    def _toggle_sidebar(self):
        if self.sidebar_open:
            self._close_sidebar()
        else:
            self._open_sidebar()

    def _open_sidebar(self):
        if self.sidebar_open:
            return
        self.sidebar_open = True

        # Show backdrop with fade-in
        anim_backdrop = Animation(opacity=1, duration=0.25, t='out_quad')
        anim_backdrop.start(self._backdrop)
        self._backdrop.active = True

        # Slide sidebar in
        anim_sidebar = Animation(x=self.x, duration=0.25, t='out_quad')
        anim_sidebar.start(self._sidebar)

        # Change menu icon to X
        self._menu_btn.set_open(True)

    def _close_sidebar(self, *args):
        if not self.sidebar_open:
            return
        self.sidebar_open = False

        # Fade out backdrop
        anim_backdrop = Animation(opacity=0, duration=0.25, t='out_quad')
        anim_backdrop.start(self._backdrop)
        self._backdrop.active = False

        # Slide sidebar out
        anim_sidebar = Animation(x=self.x - SIDEBAR_WIDTH, duration=0.25, t='out_quad')
        anim_sidebar.start(self._sidebar)

        # Change menu icon back to hamburger
        self._menu_btn.set_open(False)

    def _handle_switch_user(self):
        self._close_sidebar()
        self._app_state.logout()
        if self._on_logout:
            self._on_logout()


class _Backdrop(Widget):
    """Semi-transparent overlay that dismisses sidebar on tap."""

    active = BooleanProperty(False)

    def __init__(self, on_tap=None, **kwargs):
        super().__init__(**kwargs)
        self._on_tap = on_tap

        with self.canvas:
            Color(0, 0, 0, 0.5)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(
            pos=lambda inst, val: setattr(inst._rect, 'pos', val),
            size=lambda inst, val: setattr(inst._rect, 'size', val),
        )

    def on_touch_down(self, touch):
        if self.active and self.collide_point(*touch.pos):
            if self._on_tap:
                self._on_tap()
            return True
        return False


class SidebarNavigation(BoxLayout):
    """Sidebar navigation panel with nav buttons, signal indicator, switch user."""

    def __init__(self, app_state, on_item_selected=None,
                 on_switch_user=None, **kwargs):
        super().__init__(orientation='vertical', padding=[0, 15], **kwargs)
        self._app_state = app_state
        self._on_item_selected = on_item_selected
        self._on_switch_user = on_switch_user
        self._nav_buttons = []

        # Background
        with self.canvas.before:
            Color(*theme.SIDEBAR_BG)
            self._bg = Rectangle(pos=self.pos, size=self.size)
            Color(*theme.SIDEBAR_BORDER)
            self._right_border = Rectangle(pos=self.pos, size=(1, 1))

        def _update_sidebar_bg(inst, val):
            inst._bg.pos = inst.pos
            inst._bg.size = inst.size
            inst._right_border.pos = (inst.x + inst.width - 1, inst.y)
            inst._right_border.size = (1, inst.height)
        self.bind(pos=_update_sidebar_bg, size=_update_sidebar_bg)

        # Logo / Title
        logo = Label(
            text='NEURO\nMENTOR',
            font_size=theme.FONT_TITLE_LARGE,
            bold=True,
            color=theme.GOLD,
            size_hint_y=None,
            height=80,
            halign='center',
        )
        self.add_widget(logo)

        # Spacer
        self.add_widget(Label(size_hint_y=None, height=20))

        # Navigation buttons
        for idx, name in enumerate(PAGE_NAMES):
            btn = NavShadowButton(
                text=name,
                nav_index=idx,
                app_state=app_state,
                on_selected=self._on_nav_select,
            )
            self._nav_buttons.append(btn)
            self.add_widget(btn)

        # Spacer (takes remaining space)
        self.add_widget(Label())

        # Signal indicator
        signal_row = BoxLayout(
            size_hint_y=None, height=25, padding=[15, 0], spacing=8,
        )
        sig_label = Label(
            text='SIGNAL:',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
            size_hint_x=None,
            width=60,
            halign='left',
            valign='middle',
        )
        sig_label.bind(size=sig_label.setter('text_size'))
        signal_row.add_widget(sig_label)

        # LED circle
        led = Label(
            text='\u25cf',
            font_size=16,
            color=(0.2, 0.2, 0.2, 1),
            size_hint_x=None,
            width=20,
        )
        signal_row.add_widget(led)

        idle_label = Label(
            text='IDLE',
            font_size=theme.FONT_BODY_SMALL,
            color=theme.TEXT_MUTED,
            halign='left',
            valign='middle',
        )
        idle_label.bind(size=idle_label.setter('text_size'))
        signal_row.add_widget(idle_label)
        self.add_widget(signal_row)

        # Spacer
        self.add_widget(Label(size_hint_y=None, height=10))

        # Switch user button
        switch_btn = ShadowButton(
            text='\U0001f504 SWITCH USER',
            font_size=theme.FONT_BODY_REGULAR,
            size_hint_y=None,
            height=40,
            background_color=(0.133, 0.133, 0.133, 1),
            color=theme.TEAL,
        )
        switch_btn.bind(on_press=lambda *a: self._show_switch_dialog())
        switch_row = BoxLayout(size_hint_y=None, height=55, padding=[15, 8])
        switch_row.add_widget(switch_btn)
        self.add_widget(switch_row)

    def _on_nav_select(self, index):
        self._app_state.set_selected_page(index)
        if self._on_item_selected:
            self._on_item_selected()

    def _show_switch_dialog(self):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        msg = Label(
            text='This will end the current session\nand return to login. Continue?',
            font_size=theme.FONT_BODY_REGULAR,
            color=theme.TEXT_SECONDARY,
            halign='center',
        )
        msg.bind(size=msg.setter('text_size'))
        content.add_widget(msg)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        no_btn = ShadowButton(
            text='No',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.PANEL_BG,
            color=theme.TEXT_MUTED,
        )
        yes_btn = ShadowButton(
            text='Yes',
            font_size=theme.FONT_BODY_REGULAR,
            background_color=theme.PANEL_BG,
            color=theme.GOLD,
        )
        btn_row.add_widget(no_btn)
        btn_row.add_widget(yes_btn)
        content.add_widget(btn_row)

        popup = Popup(
            title='SWITCH USER',
            title_color=theme.TEXT_PRIMARY,
            content=content,
            size_hint=(None, None),
            size=(320, 200),
            background_color=theme.PANEL_BG,
            auto_dismiss=True,
        )

        no_btn.bind(on_press=lambda *a: popup.dismiss())

        def _yes_pressed(*a):
            popup.dismiss()
            if self._on_switch_user:
                self._on_switch_user()
        yes_btn.bind(on_press=_yes_pressed)
        popup.open()


class NavShadowButton(BoxLayout):
    """
    Navigation button with selection highlight matching Flutter's style:
    - Gold left border accent when selected
    - Gold text when selected, muted text otherwise
    - Subtle background tint when selected
    """

    def __init__(self, text, nav_index, app_state, on_selected=None, **kwargs):
        super().__init__(
            size_hint_y=None, height=48,
            padding=[0, 2],
            **kwargs,
        )
        self._nav_index = nav_index
        self._app_state = app_state
        self._on_selected = on_selected
        self._text = text

        # Selection accent (left border, 4px wide)
        with self.canvas.before:
            self._sel_bg_color = Color(
                theme.GOLD[0], theme.GOLD[1], theme.GOLD[2], 0
            )
            self._sel_bg = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[8]
            )
            self._accent_color = Color(*theme.GOLD, 0)
            self._accent_rect = Rectangle(
                pos=self.pos, size=(4, 1)
            )

        self._btn = ShadowButton(
            text=text,
            font_size=theme.FONT_BODY_REGULAR,
            halign='left',
            valign='middle',
            background_color=theme.TRANSPARENT,
            color=theme.TEXT_MUTED,
            padding=[14, 0],
        )
        self._btn.bind(on_press=lambda *a: self._select())
        self._btn.bind(size=self._btn.setter('text_size'))
        self.add_widget(self._btn)

        # Update styling when selection changes
        app_state.bind(selected_page_index=self._update_style)
        self.bind(pos=self._update_canvas, size=self._update_canvas)
        self._update_style()

    def _select(self):
        if self._on_selected:
            self._on_selected(self._nav_index)

    def _update_canvas(self, *args):
        self._sel_bg.pos = (self.x + 8, self.y + 2)
        self._sel_bg.size = (self.width - 16, self.height - 4)
        self._accent_rect.pos = (self.x + 8, self.y + 2)
        self._accent_rect.size = (4, self.height - 4)

    def _update_style(self, *args):
        is_selected = (self._app_state.selected_page_index == self._nav_index)
        if is_selected:
            self._btn.color = theme.GOLD
            self._btn.bold = True
            self._sel_bg_color.rgba = (
                theme.GOLD[0], theme.GOLD[1], theme.GOLD[2], 0.1
            )
            self._accent_color.a = 1
        else:
            self._btn.color = theme.TEXT_MUTED
            self._btn.bold = False
            self._sel_bg_color.a = 0
            self._accent_color.a = 0
        self._update_canvas()


# ============================================================
# File: main.py
# ============================================================

"""
NeuroMentor - EEG Brain-Computer Interface Application
Kivy version for Android APK generation via Buildozer.

Entry point: run this file to start the application.
"""
import os
import sys

# Add project root to path for imports

from kivy.config import Config
# Disable multi-touch emulation (red dots) on desktop
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# Lock orientation to portrait on desktop
Config.set('graphics', 'rotation', '0')

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout

# Portrait mobile phone aspect ratio (like iPhone 14: 390x844)
Window.size = (390, 844)



class NeuroMentorApp(App):
    """Main NeuroMentor Kivy Application."""

    def build(self):
        # Lock orientation to portrait on Android via jnius
        try:
            from jnius import autoclass
            activity = autoclass('org.kivy.android.PythonActivity').mActivity
            activity.setRequestedOrientation(1)  # 1 = SCREEN_ORIENTATION_PORTRAIT
        except ImportError:
            pass  # Not on Android, skip

        self.title = 'NeuroMentor'

        # Set dark background
        Window.clearcolor = theme.BG_DARK

        # Create app state
        self.app_state = AppState()

        # Root container
        self.root_container = FloatLayout()

        # Show login screen initially
        self._show_login()

        # Listen for user changes
        self.app_state.bind(current_user=self._on_user_change)

        return self.root_container

    def _on_user_change(self, *args):
        """Handle user login/logout."""
        if self.app_state.current_user is None:
            self._show_login()
        else:
            self._show_main()

    def _show_login(self):
        """Show the login screen."""
        self.root_container.clear_widgets()
        login = LoginScreen(
            app_state=self.app_state,
            on_login=self._show_main,
        )
        self.root_container.add_widget(login)

    def _show_main(self, *args):
        """Show the main app shell."""
        self.root_container.clear_widgets()
        main_shell = MainShell(
            app_state=self.app_state,
            on_logout=self._show_login,
        )
        self.root_container.add_widget(main_shell)


if __name__ == '__main__':
    NeuroMentorApp().run()
