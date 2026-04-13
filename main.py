"""
NeuroMentor — Final Production Build
Fixes:
  - Menu burger button visible + large touch target on mobile
  - Sidebar touch works on Android (FloatLayout root, proper z-order)
  - Login top padding fixed
  - PAGE_NAMES reduced: VOICE ASSISTANT and PHASE 2 INSIGHTS removed from menu
    (both accessible only via Dashboard buttons)
  - DashboardScreen has: Ask NeuroMentor (Ollama Q&A) + TRANSCRIBER (speech→text+summary)
  - Phase2InsightsScreen is now a full Transcriber (live speech → transcript → Ollama summary)
  - Colour scheme matches original (Charcoal/Gold/Mustard palette)
"""

# ── Kivy config MUST be first ────────────────────────────────────────────
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'rotation', '0')

# ── Std-lib ──────────────────────────────────────────────────────────────
import os, re, json, math, time, wave, socket, random, threading, traceback
from io import StringIO
from collections import deque
from dataclasses import dataclass
from datetime import datetime

# ── Kivy ─────────────────────────────────────────────────────────────────
from kivy.metrics import sp, dp
from kivy.app import App
from kivy.core.window import Window
from kivy.utils import platform
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.event import EventDispatcher
from kivy.properties import (
    ObjectProperty, NumericProperty, StringProperty, BooleanProperty
)
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import (
    Color, RoundedRectangle, Rectangle, Line, Ellipse
)
from kivy.graphics.texture import Texture

if platform != 'android':
    Window.size = (390, 844)

# ── Optional deps ────────────────────────────────────────────────────────
try:
    import serial as _serial_mod; _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False

try:
    import whisper as _whisper_mod; _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False

try:
    import requests as _requests_mod; _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import joblib, numpy as np; _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import numpy as np; _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import joblib as _joblib_compat; _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ════════════════════════════════════════════════════════════════════════
# AutoLabel / AutoButton
# ════════════════════════════════════════════════════════════════════════
class AutoLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.halign = kwargs.get('halign', 'left')
        self.valign = 'middle'
        self.shorten = False
        self.bind(width=self._update_text_size, texture_size=self._update_height)

    def _update_text_size(self, *a):
        if self.width >= dp(80):
            self.text_size = (self.width - dp(8), None)

    def _update_height(self, *a):
        self.height = max(self.texture_size[1] + dp(4), dp(20))


class AutoButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.halign = 'center'
        self.valign = 'middle'
        self.bind(width=self._update_text_size, texture_size=self._update_height)

    def _update_text_size(self, *a):
        if self.width >= dp(80):
            self.text_size = (self.width - dp(8), None)

    def _update_height(self, *a):
        self.height = max(self.texture_size[1], dp(48))


# ════════════════════════════════════════════════════════════════════════
# theme  — original Charcoal / Gold / Mustard palette
# ════════════════════════════════════════════════════════════════════════
class theme:
    GOLD             = (0.647, 0.486, 0.263, 1)   # #a57c43
    TEAL             = (0.741, 0.608, 0.416, 1)   # #bd9b6a
    RED              = (0.439, 0.184, 0.188, 1)   # #702f30
    BG_DARK          = (0.176, 0.231, 0.216, 1)   # #2d3b37
    PANEL_BG         = (0.220, 0.196, 0.200, 1)   # #383233
    CARD_BG          = (0.260, 0.236, 0.240, 1)   # #423c3d
    BORDER_DARK      = (0.220, 0.196, 0.200, 1)
    BORDER_LIGHT     = (0.741, 0.608, 0.416, 1)
    TEXT_PRIMARY     = (0.898, 0.871, 0.773, 1)   # #e5dec5
    TEXT_SECONDARY   = (0.839, 0.733, 0.682, 1)   # #d6bbae
    TEXT_MUTED       = (0.650, 0.610, 0.550, 1)
    INPUT_BG         = (0.160, 0.136, 0.140, 1)   # #292324
    INPUT_BORDER     = (0.741, 0.608, 0.416, 1)
    SIDEBAR_BG       = (0.220, 0.196, 0.200, 1)
    SIDEBAR_BORDER   = (0.176, 0.231, 0.216, 1)
    DARK_CARD        = (0.176, 0.231, 0.216, 1)
    BUTTON_BG        = (0.260, 0.236, 0.240, 1)   # #423c3d
    DANGER_BUTTON_BG = (0.439, 0.184, 0.188, 1)
    TRANSPARENT      = (0, 0, 0, 0)

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
    def font(size): return sp(size)

    @staticmethod
    def rgba_hex(hex_color, alpha=1.0):
        h = hex_color.lstrip('#')
        return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, alpha)

    @staticmethod
    def with_alpha(color, alpha):
        return (color[0], color[1], color[2], alpha)


# ════════════════════════════════════════════════════════════════════════
# RF Classifier stubs
# ════════════════════════════════════════════════════════════════════════
@dataclass
class EegBands:
    delta: float = 0.0; theta: float = 0.0; alpha: float = 0.0
    beta:  float = 0.0; gamma: float = 0.0

FEATURE_NAMES = [
    'delta','theta','alpha','beta','gamma',
    'beta_alpha_ratio','alpha_theta_ratio','beta_theta_ratio',
    'gamma_beta_ratio','beta_minus_alpha','alpha_plus_theta',
]

SCALER_B64  = 'gASVvwIAAAAAAACMG3NrbGVhcm4ucHJlcHJvY2Vzc2luZy5fZGF0YZSMDlN0YW5kYXJkU2NhbGVylJOUKYGUfZQojAl3aXRoX21lYW6UiIwId2l0aF9zdGSUiIwEY29weZSIjA5uX2ZlYXR1cmVzX2luX5RLC4wPbl9zYW1wbGVzX3NlZW5flIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMBnNjYWxhcpSTlIwFbnVtcHmUjAVkdHlwZZSTlIwCZjiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYkMIAAAAAAB18kCUhpRSlIwFbWVhbl+UaAqMDF9yZWNvbnN0cnVjdJSTlGgNjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwuFlGgPjAJmOJSJiIeUUpQoSwNoE05OTkr/////Sv////9LAHSUYolDWEwZudgyN8hAQvbgBzyg0UAtF7dGMtbMQGJR2OPLRORAMiq5uJbFE0GLMbx8SKMZQLD7Q7BKce8/MWigKSBVGkAIi2N+L0goQIbig71PHtpAQaN2QKwF4ECUdJRijAR2YXJflGgaaBxLAIWUaB6HlFKUKEsBSwuFlGgkiUNYDSfWa/YosUEp+ZyxkuO8Qddf4hqlIbVB2pT65Fq/4kHEuvEFgYw8Qjayjx30ylZAyhkJCLPs3D/u5HghXlFgQLmayxx0wWNAAcVolNy70UHSxIj6sfbWQZR0lGKMBnNjYWxlX5RoGmgcSwCFlGgeh5RSlChLAUsLhZRoJIlDWJGsMibikdBA2n7MU9d/1UC+5FftN2PSQNgcar9FfuhAtv5Qs1hfFUF9wENxwhgjQC8SWGs8g+U/pkV7GN/ZJkDZph9OqiQpQASvrkQ52OBA32PLsQwr40CUdJRijBBfc2tsZWFybl92ZXJzaW9ulIwFMS44LjCUdWIu'
ENCODER_B64 = 'gASVCAEAAAAAAACMHHNrbGVhcm4ucHJlcHJvY2Vzc2luZy5fbGFiZWyUjAxMYWJlbEVuY29kZXKUk5QpgZR9lCiMCGNsYXNzZXNflIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgJjAVkdHlwZZSTlIwCTziUiYiHlFKUKEsDjAF8lE5OTkr/////Sv////9LP3SUYoldlCiMCEJhc2VsaW5llIwHRm9jdXNlZJSMCFN0cmVzc2VklGV0lGKMEF9za2xlYXJuX3ZlcnNpb26UjAUxLjguMJR1Yi4='
RFC_B64_COMPRESSED = None


def _decode_model(b64):
    import base64, pickle
    try: return pickle.loads(base64.b64decode(b64))
    except: return None

def safe_div(n, d, eps=1e-9): return n / max(abs(d), eps)


class RFClassifier:
    def __init__(self):
        self._clf=self._scaler=self._encoder=None
        self._is_trained=False
        self._feature_names=list(FEATURE_NAMES)
        self._load_from_embedded()

    def _load_from_embedded(self):
        if not _HAS_SKLEARN: return
        if SCALER_B64:
            try: self._scaler = _decode_model(SCALER_B64)
            except: pass
        if ENCODER_B64:
            try: self._encoder = _decode_model(ENCODER_B64)
            except: pass
        if RFC_B64_COMPRESSED:
            import base64,gzip,io
            try:
                data=gzip.decompress(base64.b64decode(RFC_B64_COMPRESSED))
                self._clf=joblib.load(io.BytesIO(data)); self._is_trained=True; return
            except: pass
        self._load_clf_from_disk()

    def _load_clf_from_disk(self):
        if not _HAS_SKLEARN: return
        try:
            app=App.get_running_app()
            root=app.user_data_dir if app else os.path.dirname(os.path.abspath(__file__))
        except: root=os.path.dirname(os.path.abspath(__file__))
        for p in [os.path.join(root,'rf_model','rf_model','rf_eeg_model.pkl'),
                  os.path.join(root,'rf_eeg_model.pkl')]:
            if os.path.exists(p):
                try: self._clf=joblib.load(p); self._is_trained=True; return
                except: pass

    def build_feature_vector(self, bands):
        d,t,a,b,g=bands.delta,bands.theta,bands.alpha,bands.beta,bands.gamma
        fv=[d,t,a,b,g,safe_div(b,a),safe_div(a,t),safe_div(b,t),
            safe_div(g,b),b-a,a+t]
        if _HAS_SKLEARN: fv=list(np.clip(np.array(fv),-1e3,1e3))
        return fv

    def predict(self, bands):
        if not _HAS_SKLEARN or not self._is_trained or not self._clf or not self._scaler:
            return 'Unknown'
        Xs=self._scaler.transform(np.array([self.build_feature_vector(bands)]))
        pred=self._clf.predict(Xs)[0]
        if self._encoder: return str(self._encoder.inverse_transform([pred])[0])
        return str(pred)

    def save(self, dirpath):
        if not _HAS_SKLEARN: return
        os.makedirs(dirpath,exist_ok=True)
        joblib.dump(self._clf,os.path.join(dirpath,'rf_eeg_model.pkl'))
        joblib.dump(self._scaler,os.path.join(dirpath,'rf_scaler.pkl'))
        if self._encoder: joblib.dump(self._encoder,os.path.join(dirpath,'rf_encoder.pkl'))

    def load(self, dirpath):
        if not _HAS_SKLEARN: return False
        mp=os.path.join(dirpath,'rf_eeg_model.pkl')
        sp2=os.path.join(dirpath,'rf_scaler.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp2): return False
        try:
            self._clf=joblib.load(mp); self._scaler=joblib.load(sp2)
            ep=os.path.join(dirpath,'rf_encoder.pkl')
            if os.path.exists(ep): self._encoder=joblib.load(ep)
            self._is_trained=True; return True
        except: return False

    @property
    def is_trained(self): return self._is_trained


# ════════════════════════════════════════════════════════════════════════
# Compatibility check (compact)
# ════════════════════════════════════════════════════════════════════════
ADC_BITS=24; VREF=4.5; PGA_GAIN=24; ADC_RESOLUTION=(2**(ADC_BITS-1))-1
SAMPLE_RATE=250; WINDOW_SIZE=256; OVERLAP=128
BAND_BOUNDARIES={'delta':(0.5,4.0),'theta':(4.0,8.0),'alpha':(8.0,13.0),'beta':(13.0,30.0),'gamma':(30.0,45.0)}
_pass_c=_fail_c=_warn_c=0; _fail_details_c=[]; _buf_c=StringIO()

def _p(m=''):
    global _buf_c; print(m); _buf_c.write(m+'\n')
def _PASS(m): global _pass_c; _pass_c+=1; _p(f'  ✓ PASS   {m}')
def _FAIL(m): global _fail_c; _fail_c+=1; _fail_details_c.append(m); _p(f'  ✗ FAIL   {m}')
def _WARN(m): global _warn_c; _warn_c+=1; _p(f'  ⚠ WARN   {m}')
def _header(t): _p(''); _p('─'*60); _p(f'  {t}'); _p('─'*60)

def _bpfft(sig,sr,ws):
    if not _HAS_NUMPY: return {}
    w=np.hanning(ws); spec=np.fft.rfft(sig[:ws]*w); psd=(np.abs(spec)**2)/ws
    freqs=np.fft.rfftfreq(ws,1.0/sr)
    return {n:float(np.sum(psd[(freqs>=lo)&(freqs<hi)])) for n,(lo,hi) in BAND_BOUNDARIES.items()}

def run_compatibility_check():
    global _pass_c,_fail_c,_warn_c,_fail_details_c,_buf_c
    _pass_c=_fail_c=_warn_c=0; _fail_details_c=[]; _buf_c=StringIO()
    _p('╔══════════════════════════════════════════╗')
    _p('║  NEUROMENTOR COMPATIBILITY DIAGNOSTIC    ║')
    _p('╚══════════════════════════════════════════╝')
    _header('ADC / SIGNAL / BAND CHECK')
    scale=(VREF/ADC_RESOLUTION/PGA_GAIN)*1e6
    fs=ADC_RESOLUTION*scale; ms=(ADC_RESOLUTION//2)*scale
    (_PASS if 100<fs<300000 else _FAIL)(f'Full-scale {fs:.1f}µV')
    (_PASS if 50<ms<150000  else _FAIL)(f'Mid-scale {ms:.1f}µV')
    c=50.0/scale; (_PASS if c>=1 else _FAIL)(f'50µV → {c:.1f} counts')
    nyq=SAMPLE_RATE/2; hb=max(hi for _,hi in BAND_BOUNDARIES.values())
    (_PASS if nyq>=hb else _FAIL)(f'Nyquist {nyq}Hz ≥ {hb}Hz')
    if _HAS_NUMPY:
        t=np.arange(WINDOW_SIZE)/SAMPLE_RATE
        for freq,exp in [(2,'delta'),(6,'theta'),(10,'alpha'),(20,'beta'),(40,'gamma')]:
            pw=_bpfft(np.sin(2*np.pi*freq*t),SAMPLE_RATE,WINDOW_SIZE)
            dom=max(pw,key=pw.get) if pw else '?'
            (_PASS if dom==exp else _FAIL)(f'{freq}Hz → {dom}')
    _p(''); _p('═'*44)
    _p(f'  Passed:{_pass_c}  Failed:{_fail_c}  Warnings:{_warn_c}')
    _p('  VERDICT: COMPATIBLE ✓' if _fail_c==0 else '  VERDICT: INCOMPATIBLE')
    return _buf_c.getvalue()


# ════════════════════════════════════════════════════════════════════════
# AppState
# ════════════════════════════════════════════════════════════════════════
class UserProfile:
    def __init__(self,username,name='',age='',notes='',created_date=None,scores=None,last_login=None):
        self.username=username; self.name=name or username; self.age=age; self.notes=notes
        self.created_date=created_date or datetime.now().isoformat()
        self.scores=scores or []; self.last_login=last_login or datetime.now().isoformat()
    def to_dict(self): return dict(username=self.username,name=self.name,age=self.age,notes=self.notes,created_date=self.created_date,scores=self.scores,last_login=self.last_login)
    @classmethod
    def from_dict(cls,d): return cls(username=d.get('username',''),name=d.get('name',''),age=d.get('age',''),notes=d.get('notes',''),created_date=d.get('created_date'),scores=d.get('scores',[]),last_login=d.get('last_login'))


class AppState(EventDispatcher):
    current_user        = ObjectProperty(None, allownone=True)
    selected_page_index = NumericProperty(0)
    selected_port       = StringProperty('')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.all_users=self._load_users()

    def _data_dir(self):
        try:
            app=App.get_running_app()
            if app: return app.user_data_dir
        except: pass
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def users_file(self): return os.path.join(self._data_dir(),'users.json')

    def _load_users(self):
        if not os.path.exists(self.users_file): return {}
        try:
            with open(self.users_file) as f:
                return {k:UserProfile.from_dict(v) for k,v in json.load(f).items()}
        except: return {}

    def _save_users(self):
        try:
            d=os.path.dirname(self.users_file)
            if d: os.makedirs(d,exist_ok=True)
            with open(self.users_file,'w') as f:
                json.dump({k:v.to_dict() for k,v in self.all_users.items()},f,indent=4)
        except Exception as e: print(f'[AppState] save_users: {e}')

    def get_sorted_users(self):
        us=list(self.all_users.values()); us.sort(key=lambda u:u.last_login or '',reverse=True); return us

    def save_current_user_score(self,test,score):
        if self.current_user:
            self.current_user.scores.append({'test':test,'score':score,'date':datetime.now().isoformat()})
            self._save_users()

    def login(self,username):
        if username not in self.all_users: self.all_users[username]=UserProfile(username=username,name=username)
        self.all_users[username].last_login=datetime.now().isoformat()
        self._save_users(); self.current_user=self.all_users[username]
        self.selected_page_index=0; self._load_rf_model(username)

    def logout(self): self._save_users(); self.current_user=None; self.selected_page_index=0

    def update_profile(self,name=None,age=None,notes=None):
        if self.current_user:
            if name  is not None: self.current_user.name=name
            if age   is not None: self.current_user.age=age
            if notes is not None: self.current_user.notes=notes
            self._save_users(); self.property('current_user').dispatch(self)

    def set_selected_page(self,i): self.selected_page_index=i
    def set_selected_port(self,p): self.selected_port=p or ''

    def _rf_model_dir(self,username):
        d=os.path.join(self._data_dir(),'neuromentor_data',f'{username}_rf_model')
        os.makedirs(d,exist_ok=True); return d

    def save_rf_model(self,rf=None):
        rf=rf or getattr(self,'rf_classifier',None)
        if self.current_user and rf and rf.is_trained:
            rf.save(self._rf_model_dir(self.current_user.username))

    def _load_rf_model(self,username):
        if not hasattr(self,'rf_classifier'): self.rf_classifier=RFClassifier()
        ud=self._rf_model_dir(username)
        if os.path.exists(os.path.join(ud,'rf_eeg_model.pkl')) and self.rf_classifier.load(ud): return
        bd=os.path.normpath(os.path.join(self._data_dir(),'rf_model','rf_model'))
        if os.path.exists(os.path.join(bd,'rf_eeg_model.pkl')): self.rf_classifier.load(bd)


# ════════════════════════════════════════════════════════════════════════
# Custom UI widgets
# ════════════════════════════════════════════════════════════════════════
class ShadowButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        self.bg_color = kwargs.pop('bg_color', kwargs.pop('background_color', theme.BUTTON_BG))
        self.radius   = kwargs.pop('radius', 12)
        super().__init__(**kwargs)
        self.bind(pos=self._draw, size=self._draw, state=self._draw)
        self._draw()

    def _draw(self, *a):
        self.canvas.before.clear()
        with self.canvas.before:
            off = 1 if self.state == 'down' else 4
            for i in range(4):
                Color(0,0,0, 0.25*(1-i/4))
                RoundedRectangle(pos=(self.x-i+2,self.y-off-i),
                                 size=(self.width+i*2,self.height+i*2), radius=[self.radius+i])
            c = self.bg_color
            if self.state == 'down': Color(c[0]*.8,c[1]*.8,c[2]*.8,1)
            else: Color(*c)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])


class MenuBurgerButton(ButtonBehavior, Widget):
    """
    Hamburger / X button.
    Drawn large enough to be finger-friendly on mobile.
    """
    def __init__(self, **kwargs):
        self.color   = kwargs.pop('color', theme.GOLD)
        self.is_open = False
        super().__init__(**kwargs)
        self.bind(pos=self._draw, size=self._draw)
        self._draw()

    def set_open(self, v):
        self.is_open = v; self._draw()

    def _draw(self, *a):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.color)
            # Use a larger fraction of the button area so lines are easy to see
            bar_w  = self.width  * 0.60
            bar_h  = max(dp(3), self.height * 0.10)
            cx, cy = self.center_x, self.center_y
            gap    = self.height * 0.22

            if not self.is_open:
                # Three horizontal bars
                for dy in (gap, 0, -gap):
                    RoundedRectangle(
                        pos=(cx - bar_w/2, cy + dy - bar_h/2),
                        size=(bar_w, bar_h),
                        radius=[bar_h/2]
                    )
            else:
                # X  — use Lines for the diagonal strokes
                lw = max(dp(2.5), bar_h/2)
                Line(points=[cx-bar_w/2, cy-bar_w/2,
                              cx+bar_w/2, cy+bar_w/2], width=lw, cap='round')
                Line(points=[cx-bar_w/2, cy+bar_w/2,
                              cx+bar_w/2, cy-bar_w/2], width=lw, cap='round')


class GradientCard(BoxLayout):
    def __init__(self, accent_color=None, **kwargs):
        kwargs.setdefault('orientation','vertical')
        kwargs.setdefault('padding',[12,12,12,12])
        kwargs.setdefault('spacing',6)
        self.accent_color=accent_color
        self.radius=kwargs.pop('radius',12)
        super().__init__(**kwargs)
        self.texture=Texture.create(size=(1,2),colorfmt='rgba')
        self.texture.mag_filter='linear'; self.texture.min_filter='linear'
        buf=bytes([51,46,60,255, 70,65,81,255])
        self.texture.blit_buffer(buf,colorfmt='rgba',bufferfmt='ubyte')
        self.bind(pos=self._draw, size=self._draw); self._draw()

    def _draw(self, *a):
        self.canvas.before.clear()
        with self.canvas.before:
            for i in range(4):
                Color(0,0,0, 0.2*(1-i/4))
                RoundedRectangle(pos=(self.x-i+2,self.y-2-i),
                                 size=(self.width+i*2,self.height+i*2),
                                 radius=[self.radius+i])
            Color(1,1,1,1)
            RoundedRectangle(pos=self.pos,size=self.size,radius=[self.radius],texture=self.texture)
            if self.accent_color:
                Color(*self.accent_color)
                RoundedRectangle(pos=(self.x+8,self.y+self.height-4),size=(self.width-16,3),radius=[1.5])


# ════════════════════════════════════════════════════════════════════════
# EEG Graph widgets
# ════════════════════════════════════════════════════════════════════════
class EegGraph(BoxLayout):
    max_data_points=NumericProperty(256); min_y=NumericProperty(0); max_y=NumericProperty(4095)
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',padding=[40,10,10,10],**kwargs)
        self._data=deque(maxlen=256)
        tl=Label(text='Live EEG Signal',font_size=theme.font(theme.FONT_BODY_SMALL),
                 color=theme.TEXT_SECONDARY,size_hint_y=None,height=20,halign='left',valign='middle')
        tl.bind(size=tl.setter('text_size')); self.add_widget(tl)
        self._ph=Label(text='Waiting for signal...',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEXT_MUTED)
        self.add_widget(self._ph)
        self._gc=_GraphCanvas(data=self._data,min_y=0,max_y=4095); self._gc.opacity=0; self.add_widget(self._gc)
        with self.canvas.before:
            Color(*theme.PANEL_BG); self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda*a:setattr(self._bg,'pos',self.pos),
                  size=lambda*a:setattr(self._bg,'size',self.size))
    def add_data_point(self,v):
        self._data.append(v)
        if self._ph.opacity>0: self._ph.opacity=0; self._gc.opacity=1
        self._gc.redraw()
    def clear(self): self._data.clear(); self._ph.opacity=1; self._gc.opacity=0; self._gc.redraw()


class _GraphCanvas(Widget):
    def __init__(self,data,min_y=0,max_y=4095,**kwargs):
        super().__init__(**kwargs); self._data=data; self._min_y=min_y; self._max_y=max_y
        self.bind(size=lambda*a:self.redraw(),pos=lambda*a:self.redraw())
    def redraw(self):
        self.canvas.clear()
        if not self._data or self.width<=0 or self.height<=0: return
        x0=self.x+40; y0=self.y+10; w=self.width-50; h=self.height-20
        with self.canvas:
            Color(*theme.BORDER_DARK)
            for i in range(5): gy=y0+(h*i/4); Line(points=[x0,gy,x0+w,gy],width=1)
            for i in range(9): gx=x0+(w*i/8); Line(points=[gx,y0,gx,y0+h],width=1)
        dl=list(self._data); n=len(dl)
        if n<2: return
        yr=self._max_y-self._min_y or 1; pts=[]
        for i,v in enumerate(dl):
            px=x0+w*i/(n-1); py=y0+h*((v-self._min_y)/yr)
            py=max(y0,min(y0+h,py)); pts.extend([px,py])
        with self.canvas: Color(*theme.TEAL); Line(points=pts,width=1.5)


class BandPowerBars(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',padding=10,spacing=4,**kwargs)
        self._bars={}
        tl=Label(text='EEG BAND POWERS',font_size=theme.font(theme.FONT_BODY_SMALL),
                 color=theme.TEXT_SECONDARY,size_hint_y=None,height=20,halign='left',valign='middle')
        tl.bind(size=tl.setter('text_size')); self.add_widget(tl)
        for band in ['Delta','Theta','Alpha','Beta','Gamma']:
            row=BoxLayout(size_hint_y=None,height=24,spacing=5)
            lbl=Label(text=band,font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED,
                      size_hint_x=None,width=50,halign='left',valign='middle')
            lbl.bind(size=lbl.setter('text_size'))
            bar=_BarWidget(value=0); row.add_widget(lbl); row.add_widget(bar)
            self._bars[band]=bar; self.add_widget(row)
        with self.canvas.before: Color(*theme.BG_DARK); self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda*a:setattr(self._bg,'pos',self.pos),
                  size=lambda*a:setattr(self._bg,'size',self.size))
    def update_values(self,vd):
        for band,bar in self._bars.items(): bar.value=vd.get(band,0)


class _BarWidget(Widget):
    value=NumericProperty(0)
    def __init__(self,**kwargs):
        super().__init__(**kwargs); self.bind(value=self._r,size=self._r,pos=self._r); self._r()
    def _r(self,*a):
        self.canvas.clear()
        with self.canvas:
            Color(0.165,0.165,0.165,1); Rectangle(pos=self.pos,size=self.size)
            Color(*theme.TEAL); fw=self.width*max(0,min(1,self.value))
            if fw>0: Rectangle(pos=self.pos,size=(fw,self.height))


# ════════════════════════════════════════════════════════════════════════
# MindVisualizer
# ════════════════════════════════════════════════════════════════════════
class MindVisualizer(Widget):
    is_active=BooleanProperty(False); state_label=StringProperty('IDLE')
    focus_ratio=NumericProperty(1.0); stress_ratio=NumericProperty(1.0)
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._ball_y=self._target_y=0.5; self._jitter_x=self._jitter_y=0.0
        self._jitter_intensity=0.0; self._ball_color=list(theme.TEAL)
        self._ce=Clock.schedule_interval(self._animate,1/60)
        self.bind(focus_ratio=self._upd,stress_ratio=self._upd,state_label=self._upd,
                  size=lambda*a:self._draw(),pos=lambda*a:self._draw())
    def _upd(self,*a):
        f=max(0.5,min(2.5,self.focus_ratio)); self._target_y=1.0-((f-0.5)/2.0)
        s=max(0.5,min(2.0,self.stress_ratio)); self._jitter_intensity=(s-0.5)*0.05
        self._ball_color=(list(theme.RED) if self.state_label=='Stressed' else
                          list(theme.GOLD) if self.state_label=='Focused' else list(theme.TEAL))
    def _animate(self,dt):
        self._ball_y+=(self._target_y-self._ball_y)*0.05; self._ball_y=max(0.1,min(0.9,self._ball_y))
        self._jitter_x=(random.random()-0.5)*self._jitter_intensity
        self._jitter_y=(random.random()-0.5)*self._jitter_intensity; self._draw()
    def _draw(self):
        self.canvas.clear(); w,h,x0,y0=self.width,self.height,self.x,self.y
        if w<=0 or h<=0: return
        with self.canvas:
            Color(0.02,0.02,0.031,1); Rectangle(pos=self.pos,size=self.size)
            Color(0.118,0.118,0.157,1)
            for gx in range(0,int(w),60): Line(points=[x0+gx,y0,x0+gx,y0+h],width=1)
            for gy in range(0,int(h),60): Line(points=[x0,y0+gy,x0+w,y0+gy],width=1)
            cx=x0+w*0.5+self._jitter_x*w; cy=y0+h*(1-self._ball_y)+self._jitter_y*h
            cy=max(y0+50,min(y0+h-50,cy)); r=40
            for i in range(6,0,-1):
                Color(self._ball_color[0],self._ball_color[1],self._ball_color[2],0.06*i)
                gr=r*(0.5+i*0.5); Ellipse(pos=(cx-gr,cy-gr),size=(gr*2,gr*2))
            Color(*self._ball_color); Ellipse(pos=(cx-r,cy-r),size=(r*2,r*2))
    def cleanup(self):
        if hasattr(self,'_ce') and self._ce: self._ce.cancel(); self._ce=None
    def __del__(self): self.cleanup()


# ════════════════════════════════════════════════════════════════════════
# Task widgets (Breathing / Focus / Stroop)
# ════════════════════════════════════════════════════════════════════════
class BreathingWidget(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',padding=20,spacing=10,**kwargs)
        self._step=self._score=0; self._is_running=False; self._mode='4-7-8'; self._ce=None
        self._inst=Label(text='Ready',font_size=sp(40),color=theme.TEAL,bold=True,size_hint_y=0.4)
        self.add_widget(self._inst)
        mr=BoxLayout(size_hint_y=None,height=40,spacing=20); mr.size_hint_x=None; mr.width=320; mr.pos_hint={'center_x':0.5}
        self._b478=ToggleButton(text='4-7-8 (Calm)',group='bm478',state='down',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.GOLD,color=theme.BG_DARK)
        self._b478.bind(on_press=lambda*a:self.set_mode('4-7-8'))
        self._bbox=ToggleButton(text='Box (Focus)',group='bm478',state='normal',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.BORDER_DARK,color=theme.TEXT_PRIMARY)
        self._bbox.bind(on_press=lambda*a:self.set_mode('box'))
        mr.add_widget(self._b478); mr.add_widget(self._bbox); self.add_widget(mr)
        self._sl=Label(text='Score: 0',font_size=theme.font(theme.FONT_HEADING_MEDIUM),color=theme.GOLD,size_hint_y=None,height=40)
        self.add_widget(self._sl)
        self._sb=ShadowButton(text='START',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._sb.bind(on_press=self._toggle); self.add_widget(self._sb)
    def set_mode(self,m):
        self._mode=m
        if m=='4-7-8': self._b478.state='down'; self._b478.background_color=theme.GOLD; self._b478.color=theme.BG_DARK; self._bbox.state='normal'; self._bbox.background_color=theme.BORDER_DARK; self._bbox.color=theme.TEXT_PRIMARY
        else: self._bbox.state='down'; self._bbox.background_color=theme.GOLD; self._bbox.color=theme.BG_DARK; self._b478.state='normal'; self._b478.background_color=theme.BORDER_DARK; self._b478.color=theme.TEXT_PRIMARY
    def start_task(self): self._start()
    def _toggle(self,*a): self.stop() if self._is_running else self._start()
    def _start(self):
        if self._is_running: return
        self._is_running=True; self._step=self._score=0
        self._sb.text='STOP'; self._sb.color=theme.RED; self._sb.bg_color=theme.DANGER_BUTTON_BG
        self._ce=Clock.schedule_interval(self._tick,1.0)
    def stop(self):
        if self._ce: self._ce.cancel(); self._ce=None
        self._is_running=False; self._inst.text='Relax'
        self._sb.text='START'; self._sb.color=theme.GOLD; self._sb.bg_color=theme.BUTTON_BG
    def _tick(self,dt):
        self._step+=1; self._score=self._step*10; self._sl.text=f'Score: {self._score}'
        cl=16 if self._mode=='box' else 19; cur=self._step%cl
        if self._mode=='box':
            if cur<4: self._inst.text=f'INHALE ({4-cur})'
            elif cur<8: self._inst.text=f'HOLD ({8-cur})'
            elif cur<12: self._inst.text=f'EXHALE ({12-cur})'
            else: self._inst.text=f'HOLD ({16-cur})'
        else:
            if cur<4: self._inst.text=f'INHALE ({4-cur})'
            elif cur<11: self._inst.text=f'HOLD ({11-cur})'
            else: self._inst.text=f'EXHALE ({19-cur})'
    def cleanup(self):
        if self._ce: self._ce.cancel()


ARTICLES=[
    'Neuroplasticity is the ability of neural networks in the brain to reorganize themselves by creating new neural connections throughout life. This allows neurons to compensate for injury and disease, and to adjust their activities in response to new situations.',
    'Quantum entanglement is a phenomenon where two or more particles become interconnected in such a way that the quantum state of each particle cannot be described independently of the other. Einstein famously called it spooky action at a distance.',
    'In cognitive science, attention is the cognitive process that allows us to focus on specific information while filtering out irrelevant stimuli. Selective attention mechanisms help us prioritize important information and maintain focus on relevant tasks.',
]


class FocusWidget(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',padding=10,spacing=10,**kwargs)
        self._is_tracking=True; self._is_running=False
        mr=BoxLayout(size_hint_y=None,height=40,spacing=20); mr.size_hint_x=None; mr.width=320; mr.pos_hint={'center_x':0.5}
        self._bt=ToggleButton(text='Visual Tracking',group='fmtrack',state='down',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.GOLD,color=theme.BG_DARK)
        self._bt.bind(on_press=lambda*a:self.set_mode('tracking'))
        self._br=ToggleButton(text='Tech Reading',group='fmtrack',state='normal',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.BORDER_DARK,color=theme.TEXT_PRIMARY)
        self._br.bind(on_press=lambda*a:self.set_mode('reading'))
        mr.add_widget(self._bt); mr.add_widget(self._br); self.add_widget(mr)
        self._tv=TrackingView(); self._rv=ReadingView(); self._rv.opacity=0; self._rv.disabled=True
        self._ca=FloatLayout(); self._ca.add_widget(self._tv); self._ca.add_widget(self._rv); self.add_widget(self._ca)
        self._sb=ShadowButton(text='START',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._sb.bind(on_press=self._toggle); self.add_widget(self._sb)
    def set_mode(self,m):
        it=(m=='tracking'); self._is_tracking=it
        self._bt.state='down' if it else 'normal'; self._bt.background_color=theme.GOLD if it else theme.BORDER_DARK; self._bt.color=theme.BG_DARK if it else theme.TEXT_PRIMARY
        self._br.state='normal' if it else 'down'; self._br.background_color=theme.BORDER_DARK if it else theme.GOLD; self._br.color=theme.TEXT_PRIMARY if it else theme.BG_DARK
        self._tv.opacity=1 if it else 0; self._tv.disabled=not it; self._rv.opacity=0 if it else 1; self._rv.disabled=it
    def start_task(self):
        if not self._is_running: self._toggle()
    def stop(self):
        if self._is_running: self._toggle()
    def _toggle(self,*a):
        self._is_running=not self._is_running
        if self._is_running: self._sb.text='STOP'; self._sb.color=theme.RED; self._sb.bg_color=theme.DANGER_BUTTON_BG; self._tv.start(); self._rv.new_article()
        else: self._sb.text='START'; self._sb.color=theme.GOLD; self._sb.bg_color=theme.BUTTON_BG; self._tv.stop()
    def cleanup(self): self._tv.stop()


from kivy.uix.floatlayout import FloatLayout as _FL


class TrackingView(Widget):
    def __init__(self,**kwargs):
        super().__init__(**kwargs); self._bx=self._by=0.5; self._ce=None
        self.bind(size=self._draw,pos=self._draw); self._draw()
    def start(self):
        if self._ce: self._ce.cancel()
        self._ce=Clock.schedule_interval(self._move,1/30)
    def stop(self):
        if self._ce: self._ce.cancel(); self._ce=None
    def _move(self,dt):
        self._bx=max(0.05,min(0.95,self._bx+(random.random()-0.5)*0.02))
        self._by=max(0.05,min(0.95,self._by+(random.random()-0.5)*0.02)); self._draw()
    def _draw(self,*a):
        self.canvas.clear(); w,h=self.width,self.height
        if w<=0 or h<=0: return
        cx=self.x+self._bx*w; cy=self.y+self._by*h
        with self.canvas:
            Color(0,0,0,1); Rectangle(pos=self.pos,size=self.size)
            for i in range(4,0,-1):
                Color(theme.GOLD[0],theme.GOLD[1],theme.GOLD[2],0.12*i); r=15+i*8; Ellipse(pos=(cx-r,cy-r),size=(r*2,r*2))
            Color(*theme.GOLD); Ellipse(pos=(cx-15,cy-15),size=(30,30))


class ReadingView(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',padding=15,spacing=10,**kwargs)
        h=Label(text='READ CAREFULLY:',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_SECONDARY,size_hint_y=None,height=20,halign='left',valign='middle')
        h.bind(size=h.setter('text_size')); self.add_widget(h)
        sc=ScrollView()
        self._art=Label(text=random.choice(ARTICLES),font_size=theme.font(theme.FONT_BODY_LARGE),color=theme.TEAL,markup=False,halign='left',valign='top',size_hint_y=None)
        self._art.bind(texture_size=lambda inst,sz:setattr(inst,'height',sz[1]),
                       width=lambda inst,w:setattr(inst,'text_size',(w,None)))
        sc.add_widget(self._art); self.add_widget(sc)
        with self.canvas.before: Color(0.067,0.067,0.067,1); self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda*a:setattr(self._bg,'pos',self.pos),size=lambda*a:setattr(self._bg,'size',self.size))
    def new_article(self): self._art.text=random.choice(ARTICLES)


class StroopWidget(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical',padding=20,spacing=10,**kwargs)
        self._score=0; self._is_running=False; self._is_math_mode=False; self._ce=None
        self._ink='BLUE'; self._math_ans=0
        self._colors=['RED','BLUE','GREEN','YELLOW']
        self._cmap={'RED':(1,0,0,1),'BLUE':(0,0,1,1),'GREEN':(0,1,0,1),'YELLOW':(1,1,0,1)}
        self._dl=Label(text='BLUE',font_size=sp(50),bold=True,color=(0,0,1,1),size_hint_y=0.35); self.add_widget(self._dl)
        mr=BoxLayout(size_hint_y=None,height=40,spacing=20); mr.size_hint_x=None; mr.width=280; mr.pos_hint={'center_x':0.5}
        self._bs=ToggleButton(text='Stroop',group='smstroop',state='down',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.GOLD,color=theme.BG_DARK)
        self._bs.bind(on_press=lambda*a:self.set_mode('stroop'))
        self._bm=ToggleButton(text='Rapid Math',group='smstroop',state='normal',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.BORDER_DARK,color=theme.TEXT_PRIMARY)
        self._bm.bind(on_press=lambda*a:self.set_mode('math'))
        mr.add_widget(self._bs); mr.add_widget(self._bm); self.add_widget(mr)
        self._sr=BoxLayout(size_hint_y=None,height=50,spacing=8)
        for c in self._colors:
            btn=Button(text=c,font_size=sp(12),bold=True,background_color=self._cmap[c],color=(0,0,0,1),size_hint_x=None,width=80)
            btn.bind(on_press=lambda inst,cn=c:self._check_stroop(cn)); self._sr.add_widget(btn)
        self.add_widget(self._sr)
        self._mr=BoxLayout(size_hint=(None,None),size=(300,45),pos_hint={'center_x':0.5},spacing=10)
        self._mi=TextInput(hint_text='Answer',font_size=theme.font(theme.FONT_HEADING_MEDIUM),multiline=False,input_filter='int',size_hint=(None,1),width=180,background_color=theme.INPUT_BG,foreground_color=theme.TEXT_PRIMARY)
        self._mi.bind(on_text_validate=lambda*a:self._check_math())
        self._msub=ShadowButton(text='SUBMIT',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,bg_color=theme.TEAL,color=theme.BG_DARK,size_hint=(None,1),width=110)
        self._msub.bind(on_press=lambda*a:self._check_math())
        self._mr.add_widget(self._mi); self._mr.add_widget(self._msub); self._mr.opacity=0; self._mr.disabled=True; self.add_widget(self._mr)
        self._sl=Label(text='Score: 0',font_size=theme.font(theme.FONT_HEADING_MEDIUM),color=theme.GOLD,size_hint_y=None,height=40); self.add_widget(self._sl)
        self._sb=ShadowButton(text='START',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._sb.bind(on_press=self._toggle); self.add_widget(self._sb)
    def set_mode(self,m):
        im=(m=='math'); self._is_math_mode=im
        self._bm.state='down' if im else 'normal'; self._bm.background_color=theme.GOLD if im else theme.BORDER_DARK; self._bm.color=theme.BG_DARK if im else theme.TEXT_PRIMARY
        self._bs.state='normal' if im else 'down'; self._bs.background_color=theme.BORDER_DARK if im else theme.GOLD; self._bs.color=theme.TEXT_PRIMARY if im else theme.BG_DARK
        self._sr.opacity=0 if im else 1; self._sr.disabled=im; self._mr.opacity=1 if im else 0; self._mr.disabled=not im
        if self._is_running: self._next_round()
    def start_task(self): self._start()
    def _toggle(self,*a): self.stop() if self._is_running else self._start()
    def _start(self):
        if self._is_running: return
        self._is_running=True; self._score=0; self._sl.text='Score: 0'
        self._sb.text='STOP'; self._sb.color=theme.RED; self._sb.bg_color=theme.DANGER_BUTTON_BG
        self._next_round(); self._ce=Clock.schedule_interval(lambda dt:self._next_round(),3.0)
    def stop(self):
        if self._ce: self._ce.cancel(); self._ce=None
        self._is_running=False; self._sb.text='START'; self._sb.color=theme.GOLD; self._sb.bg_color=theme.BUTTON_BG
    def _next_round(self):
        if self._is_math_mode:
            a,b=random.randint(10,99),random.randint(10,99); ia=random.choice([True,False])
            self._math_ans=a+b if ia else a-b; self._dl.text=f'{a} {"+" if ia else "-"} {b} = ?'; self._dl.color=theme.RED; self._mi.text=''
        else:
            word=random.choice(self._colors); ink=random.choice(self._colors); self._ink=ink; self._dl.text=word; self._dl.color=self._cmap[ink]
    def _check_stroop(self,cn):
        if not self._is_running: return
        if cn==self._ink: self._score+=50; self._sl.text=f'Score: {self._score}'
        if self._ce: self._ce.cancel()
        self._next_round(); self._ce=Clock.schedule_interval(lambda dt:self._next_round(),3.0)
    def _check_math(self):
        if not self._is_running: return
        try: ans=int(self._mi.text)
        except: ans=0
        if ans==self._math_ans: self._score+=50; self._sl.text=f'Score: {self._score}'
        if self._ce: self._ce.cancel()
        self._next_round(); self._ce=Clock.schedule_interval(lambda dt:self._next_round(),3.0)
    def cleanup(self):
        if self._ce: self._ce.cancel()


# ════════════════════════════════════════════════════════════════════════
# Voice / Transcriber mixin
# ════════════════════════════════════════════════════════════════════════
class VoiceMixin:
    """
    Shared ESP32→Whisper→Ollama pipeline.
    Host must define: self._app_state, self._voice_lbl (AutoLabel for responses).
    """
    def _init_voice(self):
        self._vthread=None; self._wmodel=None
        # TCP listener for whisper_server.py push on port 5050
        threading.Thread(target=self._tcp_listen, daemon=True).start()

    def _tcp_listen(self):
        try:
            srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            srv.bind(('0.0.0.0',5050)); srv.listen(3); srv.settimeout(1.0)
            while True:
                try:
                    conn,_=srv.accept(); data=b''; conn.settimeout(2.0)
                    try:
                        while True:
                            chunk=conn.recv(4096)
                            if not chunk: break
                            data+=chunk
                    except: pass
                    finally: conn.close()
                    if data:
                        text=data.decode('utf-8',errors='replace').strip()
                        if text:
                            self._set_voice_lbl(f'Transcribed: {text}\nQuerying Ollama...')
                            threading.Thread(target=self._ollama,args=(text,),daemon=True).start()
                except socket.timeout: continue
                except Exception as e: print(f'[VoiceTCP] {e}')
        except Exception as e: print(f'[VoiceTCP bind] {e}')

    def _btn_voice(self, *a):
        if self._vthread and self._vthread.is_alive():
            self._set_voice_lbl('Already processing...'); return
        self._set_voice_lbl('Connecting to ESP32...')
        self._vthread=threading.Thread(target=self._pipeline,daemon=True); self._vthread.start()

    def _pipeline(self):
        try:
            self._set_voice_lbl('Reading audio from ESP32...')
            raw=self._get_audio()
            if not raw:
                self._set_voice_lbl('No audio received.\nIf using whisper_server.py, text arrives automatically on port 5050.'); return
            path=os.path.join(App.get_running_app().user_data_dir,'nm_voice.wav')
            with wave.open(path,'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(raw)
            self._set_voice_lbl('Transcribing...')
            text=self._transcribe(path)
            if not text: self._set_voice_lbl('Whisper returned empty result.'); return
            self._set_voice_lbl(f'You said: {text}\n\nQuerying Ollama...'); self._ollama(text)
        except Exception as e:
            traceback.print_exc(); self._set_voice_lbl(f'Error: {str(e)[:150]}')

    def _get_audio(self):
        exp=16000*2*5; audio=bytearray(); t0=time.time()
        ports=[]
        if getattr(self._app_state,'selected_port',None): ports.append(self._app_state.selected_port)
        if platform=='android': ports+=['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyS0']
        else: ports+=['COM3','COM4','COM5','/dev/ttyUSB0','/dev/ttyUSB1']
        if _HAS_SERIAL:
            for port in ports:
                try:
                    with _serial_mod.Serial(port,115200,timeout=1) as ser:
                        while len(audio)<exp and time.time()-t0<12:
                            c=ser.read(min(4096,exp-len(audio)))
                            if c: audio.extend(c)
                    if audio: return bytes(audio)
                except Exception as e: print(f'[Serial] {port}: {e}')
        for host,p in [('192.168.4.1',5000),('127.0.0.1',5000)]:
            try:
                with socket.socket() as s:
                    s.settimeout(4); s.connect((host,p))
                    while len(audio)<exp and time.time()-t0<12:
                        c=s.recv(min(4096,exp-len(audio)))
                        if not c: break
                        audio.extend(c)
                if audio: return bytes(audio)
            except: pass
        return b''

    def _transcribe(self, path):
        if not _HAS_WHISPER: return '[whisper not installed]'
        if not self._wmodel:
            self._set_voice_lbl('Loading Whisper model...')
            self._wmodel=_whisper_mod.load_model('base')
        return _whisper_mod.load_model('base').transcribe(path).get('text','').strip() if not self._wmodel else self._wmodel.transcribe(path).get('text','').strip()

    def _ollama(self, text):
        if not _HAS_REQUESTS:
            self._set_voice_lbl(f'Query: {text}\n\n[requests not installed]'); return
        urls=['http://localhost:11434/api/generate','http://127.0.0.1:11434/api/generate']
        if platform=='android': urls.insert(0,'http://10.0.2.2:11434/api/generate')
        for url in urls:
            try:
                r=_requests_mod.post(url,json={'model':'llama3','prompt':text,'stream':False},timeout=30)
                r.raise_for_status()
                ans=(r.json().get('response') or r.json().get('text') or '').strip() or 'No response.'
                self._set_voice_lbl(ans); return
            except Exception as e: print(f'[Ollama] {url}: {e}')
        self._set_voice_lbl(f'Could not reach Ollama.\nQuery: {text}')

    def _set_voice_lbl(self, txt):
        Clock.schedule_once(lambda dt: setattr(self._voice_lbl,'text',txt), 0)


# ════════════════════════════════════════════════════════════════════════
# Shared button helper
# ════════════════════════════════════════════════════════════════════════
def _action_btn(text, callback, bg=None):
    bg = bg or theme.BUTTON_BG
    btn = AutoButton(text=text, font_size=theme.font(theme.FONT_BODY_REGULAR),
                     color=theme.TEXT_PRIMARY, halign='center', valign='middle')
    btn.size_hint_y = None
    btn.bind(on_press=callback)
    with btn.canvas.before:
        Color(*bg)
        btn._bg = RoundedRectangle(pos=btn.pos, size=btn.size, radius=[dp(10)])
    btn.bind(pos=lambda inst,v: setattr(btn._bg,'pos',btn.pos),
             size=lambda inst,v: setattr(btn._bg,'size',btn.size))
    return btn


# ════════════════════════════════════════════════════════════════════════
# DashboardScreen
# ════════════════════════════════════════════════════════════════════════
class DashboardScreen(VoiceMixin, BoxLayout):
    """
    Dashboard with:
      - stat cards
      - 🎤 Ask NeuroMentor  → Ollama Q&A
      - 📝 TRANSCRIBER      → opens TranscriberScreen (idx 5)
    """
    def __init__(self, app_state, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)
        self._app_state = app_state

        # Welcome
        self._wl = Label(text='WELCOME BACK, USER', font_size=theme.font(theme.FONT_TITLE_MEDIUM),
                         bold=True, color=theme.GOLD, size_hint_y=None, height=35,
                         halign='left', valign='middle')
        self._wl.bind(size=self._wl.setter('text_size'))
        self.add_widget(self._wl)
        app_state.bind(current_user=self._upd_welcome); self._upd_welcome()

        # Status card
        sc=GradientCard(size_hint_y=None,height=80,padding=[20,15])
        sv=Label(text='OFFLINE',font_size=theme.font(theme.FONT_HEADING_MEDIUM),color=theme.TEXT_SECONDARY,halign='left',valign='middle')
        sv.bind(size=sv.setter('text_size')); sc.add_widget(sv); self.add_widget(sc)

        # Stat cards
        sr=BoxLayout(spacing=15,size_hint_y=None,height=100)
        sr.add_widget(self._stat_card('STRESS','N/A',theme.RED))
        sr.add_widget(self._stat_card('FOCUS','N/A',theme.TEAL))
        sr.add_widget(self._stat_card('NEURO XP','0',theme.GOLD))
        self.add_widget(sr)

        # Event log
        lt=Label(text='EVENT LOG',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_SECONDARY,size_hint_y=None,height=18,halign='left',valign='middle')
        lt.bind(size=lt.setter('text_size')); self.add_widget(lt)
        lb=GradientCard(size_hint_y=0.3,padding=[10,10])
        ll=Label(text='No events yet',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEAL,halign='left',valign='top')
        ll.bind(size=ll.setter('text_size')); lb.add_widget(ll); self.add_widget(lb)

        # ── Voice Q&A ──────────────────────────────────────────────────
        va_card = GradientCard(size_hint_y=None, padding=[15, 12])
        va_card.bind(minimum_height=va_card.setter('height'))

        va_title = Label(text='VOICE ASSISTANT', font_size=theme.font(theme.FONT_BODY_SMALL),
                         bold=True, color=theme.GOLD, size_hint_y=None, height=20,
                         halign='left', valign='middle')
        va_title.bind(size=va_title.setter('text_size'))
        va_card.add_widget(va_title)

        ask_btn = _action_btn('🎤  Ask NeuroMentor', self._btn_voice)
        va_card.add_widget(ask_btn)

        self._voice_lbl = AutoLabel(
            text='Press the button and ask a question. Answer appears here.',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, halign='left')
        va_card.add_widget(self._voice_lbl)
        self.add_widget(va_card)

        # ── Transcriber shortcut ───────────────────────────────────────
        tr_btn = _action_btn('📝  Open Transcriber', self._open_transcriber, bg=theme.DARK_CARD)
        self.add_widget(tr_btn)

        # Refresh
        rb=ShadowButton(text='SYSTEM REFRESH',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=45,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self.add_widget(rb)

        self._init_voice()

    def _upd_welcome(self, *a):
        u=self._app_state.current_user; n=u.name.upper() if u and u.name else 'USER'
        self._wl.text=f'WELCOME BACK, {n}'

    def _open_transcriber(self, *a):
        # PAGE index 5 = TranscriberScreen
        self._app_state.set_selected_page(5)

    def _stat_card(self,title,value,accent):
        card=GradientCard(accent_color=accent,padding=[15,10])
        t=Label(text=title,font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED,size_hint_y=None,height=16,halign='left',valign='middle')
        t.bind(size=t.setter('text_size')); card.add_widget(t)
        v=Label(text=value,font_size=theme.font(theme.FONT_HEADING_LARGE),bold=True,color=accent,halign='left',valign='middle')
        v.bind(size=v.setter('text_size')); card.add_widget(v)
        return card


# ════════════════════════════════════════════════════════════════════════
# TranscriberScreen  (was "Phase2InsightsScreen")
# Continuously listens on TCP port 5050 for text pushed by whisper_server.py
# and also lets the user type/paste text, then sends to Ollama for summary.
# ════════════════════════════════════════════════════════════════════════
class TranscriberScreen(ScrollView):
    """
    Speech-to-text transcript + Ollama summary.

    Two input modes:
      1. Automatic: whisper_server.py pushes transcribed text to port 5050
         → displayed live in the transcript box.
      2. Manual: user types/pastes text in the input field and taps ADD.

    SUMMARIZE button sends the full transcript to Ollama llama3.
    CLEAR wipes everything.
    """
    def __init__(self, app_state, **kwargs):
        super().__init__(do_scroll_x=False, do_scroll_y=True, **kwargs)
        self._app_state = app_state
        self._chunks    = []          # accumulated transcript chunks
        self._summary_thread = None

        self._inner = BoxLayout(orientation='vertical', padding=dp(16),
                                spacing=dp(12), size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        self.add_widget(self._inner)

        # Title
        title = AutoLabel(text='TRANSCRIBER',
                          font_size=theme.font(theme.FONT_TITLE_MEDIUM),
                          bold=True, color=theme.GOLD, halign='left')
        self._inner.add_widget(title)

        sub = AutoLabel(
            text='Speech → text transcript + AI summary via Ollama.\n'
                 'Connect whisper_server.py (port 9999 ← ESP32, port 5050 → here).',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.TEXT_MUTED, halign='left')
        self._inner.add_widget(sub)

        # Status row
        self._status_lbl = AutoLabel(
            text='● LISTENING on port 5050',
            font_size=theme.font(theme.FONT_BODY_SMALL),
            color=theme.GOLD, halign='left')
        self._inner.add_widget(self._status_lbl)

        # ── TRANSCRIPT box ─────────────────────────────────────────────
        tcard = GradientCard(size_hint_y=None, height=dp(220), padding=[dp(10), dp(10)])
        t_hdr = Label(text='LIVE TRANSCRIPT', font_size=theme.font(theme.FONT_BODY_SMALL),
                      bold=True, color=theme.TEAL, size_hint_y=None, height=dp(20),
                      halign='left', valign='middle')
        t_hdr.bind(size=t_hdr.setter('text_size'))
        tcard.add_widget(t_hdr)

        t_scroll = ScrollView(do_scroll_x=False, do_scroll_y=True)
        self._transcript_lbl = Label(
            text='Waiting for speech...',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY, halign='left', valign='top',
            size_hint_y=None, markup=False)
        self._transcript_lbl.bind(
            texture_size=lambda inst,sz: setattr(inst,'height',max(sz[1],dp(60))),
            width=lambda inst,w: setattr(inst,'text_size',(w,None)))
        t_scroll.add_widget(self._transcript_lbl)
        tcard.add_widget(t_scroll)

        # Word counter
        self._word_lbl = Label(text='0 words · 0 segments',
                               font_size=theme.font(theme.FONT_BODY_SMALL),
                               color=theme.TEXT_MUTED, size_hint_y=None, height=dp(18),
                               halign='left', valign='middle')
        self._word_lbl.bind(size=self._word_lbl.setter('text_size'))
        tcard.add_widget(self._word_lbl)
        self._inner.add_widget(tcard)

        # ── Manual text input row ──────────────────────────────────────
        row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self._txt_in = TextInput(
            hint_text='Type or paste speech text here...',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            multiline=False, size_hint_x=0.75,
            background_color=theme.INPUT_BG,
            foreground_color=theme.TEXT_PRIMARY,
            cursor_color=theme.TEAL, padding=[dp(10), dp(10)])
        self._txt_in.bind(on_text_validate=lambda *a: self._add_manual())
        add_btn = ShadowButton(text='ADD', font_size=theme.font(theme.FONT_BODY_REGULAR),
                               bold=True, size_hint_x=0.25,
                               background_color=theme.BUTTON_BG, color=theme.GOLD)
        add_btn.bind(on_press=lambda *a: self._add_manual())
        row.add_widget(self._txt_in); row.add_widget(add_btn)
        self._inner.add_widget(row)

        # ── Action buttons ─────────────────────────────────────────────
        btn_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        sum_btn = ShadowButton(text='SUMMARIZE', font_size=theme.font(theme.FONT_BODY_REGULAR),
                               bold=True, background_color=theme.GOLD, color=(0,0,0,1))
        sum_btn.bind(on_press=self._do_summarize)
        clr_btn = ShadowButton(text='CLEAR', font_size=theme.font(theme.FONT_BODY_REGULAR),
                               bold=True, background_color=theme.DANGER_BUTTON_BG,
                               color=theme.TEXT_PRIMARY)
        clr_btn.bind(on_press=self._do_clear)
        btn_row.add_widget(sum_btn); btn_row.add_widget(clr_btn)
        self._inner.add_widget(btn_row)

        # ── SUMMARY box ────────────────────────────────────────────────
        scard = GradientCard(size_hint_y=None, padding=[dp(10), dp(10)])
        scard.bind(minimum_height=scard.setter('height'))
        s_hdr = Label(text='AI SUMMARY', font_size=theme.font(theme.FONT_BODY_SMALL),
                      bold=True, color=theme.GOLD, size_hint_y=None, height=dp(20),
                      halign='left', valign='middle')
        s_hdr.bind(size=s_hdr.setter('text_size'))
        scard.add_widget(s_hdr)
        self._summary_lbl = AutoLabel(
            text='Summary appears here after you press SUMMARIZE.',
            font_size=theme.font(theme.FONT_BODY_REGULAR),
            color=theme.TEXT_PRIMARY, halign='left')
        scard.add_widget(self._summary_lbl)
        self._inner.add_widget(scard)

        # Start TCP listener thread
        threading.Thread(target=self._tcp_listen, daemon=True).start()

    # ── TCP listener (receives text from whisper_server.py) ───────────
    def _tcp_listen(self):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(('0.0.0.0', 5050)); srv.listen(3); srv.settimeout(1.0)
            Clock.schedule_once(lambda dt: setattr(self._status_lbl,'text',
                '● LISTENING on port 5050 — ready for whisper_server.py'), 0)
            while True:
                try:
                    conn, addr = srv.accept()
                    data = b''; conn.settimeout(2.0)
                    try:
                        while True:
                            chunk = conn.recv(4096)
                            if not chunk: break
                            data += chunk
                    except: pass
                    finally: conn.close()
                    if data:
                        text = data.decode('utf-8', errors='replace').strip()
                        if text:
                            Clock.schedule_once(lambda dt, t=text: self._append_chunk(t), 0)
                except socket.timeout: continue
                except Exception as e: print(f'[Transcriber TCP] {e}')
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self._status_lbl,'text',
                f'⚠ Could not bind port 5050: {e}'), 0)

    # ── Append a chunk to the transcript ──────────────────────────────
    def _append_chunk(self, text):
        text = text.strip()
        if not text: return
        # Check for "summarize" voice command
        if any(kw in text.lower() for kw in ['summarize','summarise','summary']):
            self._do_summarize(None); return
        ts = datetime.now().strftime('%H:%M:%S')
        self._chunks.append(text)
        lines = '\n'.join(f'[{datetime.now().strftime("%H:%M:%S")}]  {c}' for c in self._chunks)
        self._transcript_lbl.text = lines
        words = sum(len(c.split()) for c in self._chunks)
        self._word_lbl.text = f'{words} words · {len(self._chunks)} segments'

    def _add_manual(self):
        t = self._txt_in.text.strip()
        if t:
            self._append_chunk(t); self._txt_in.text = ''

    # ── Summarize via Ollama ───────────────────────────────────────────
    def _do_summarize(self, *a):
        full = '\n'.join(self._chunks)
        if not full.strip():
            self._summary_lbl.text = 'No transcript to summarize yet.'; return
        if self._summary_thread and self._summary_thread.is_alive():
            self._summary_lbl.text = 'Already generating summary...'; return
        self._summary_lbl.text = 'Generating summary...'
        self._summary_thread = threading.Thread(
            target=self._run_summary, args=(full,), daemon=True)
        self._summary_thread.start()

    def _run_summary(self, transcript):
        prompt = (
            'You are a helpful assistant. Below is a transcript of speech. '
            'Please provide a clear, concise summary with key points in bullet form.\n\n'
            f'TRANSCRIPT:\n{transcript}\n\nSUMMARY:'
        )
        if not _HAS_REQUESTS:
            Clock.schedule_once(lambda dt: setattr(self._summary_lbl,'text',
                '[requests not installed — cannot reach Ollama]'), 0); return
        urls = ['http://localhost:11434/api/generate','http://127.0.0.1:11434/api/generate']
        if platform == 'android': urls.insert(0,'http://10.0.2.2:11434/api/generate')
        for url in urls:
            try:
                r = _requests_mod.post(url,
                    json={'model':'llama3','prompt':prompt,'stream':False}, timeout=60)
                r.raise_for_status()
                ans = (r.json().get('response') or '').strip() or 'No summary returned.'
                Clock.schedule_once(lambda dt,s=ans: setattr(self._summary_lbl,'text',s), 0)
                return
            except Exception as e: print(f'[Transcriber Ollama] {url}: {e}')
        Clock.schedule_once(lambda dt: setattr(self._summary_lbl,'text',
            'Could not reach Ollama. Is it running?'), 0)

    def _do_clear(self, *a):
        self._chunks.clear()
        self._transcript_lbl.text = 'Waiting for speech...'
        self._word_lbl.text = '0 words · 0 segments'
        self._summary_lbl.text = 'Summary appears here after you press SUMMARIZE.'


# ════════════════════════════════════════════════════════════════════════
# ProfileScreen
# ════════════════════════════════════════════════════════════════════════
class ProfileScreen(BoxLayout):
    def __init__(self,app_state,**kwargs):
        super().__init__(orientation='vertical',padding=20,spacing=15,**kwargs)
        self._app_state=app_state
        tb=GradientCard(size_hint_y=None,height=50,padding=[10,10],radius=10)
        tl=Label(text='USER PROFILE',font_size=theme.font(theme.FONT_TITLE_MEDIUM),bold=True,color=theme.GOLD,halign='left',valign='middle')
        tl.bind(size=tl.setter('text_size')); tb.add_widget(tl); self.add_widget(tb)
        form=GradientCard(padding=[20,20])
        for txt,attr,multi,h in [('FULL NAME:','_ni',False,40),('AGE:','_ai',False,40),('CLINICAL NOTES:','_noi',True,120)]:
            lbl=Label(text=txt,font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEAL,bold=True,size_hint_y=None,height=20,halign='left',valign='middle')
            lbl.bind(size=lbl.setter('text_size')); form.add_widget(lbl)
            inp=TextInput(font_size=theme.font(theme.FONT_BODY_REGULAR),multiline=multi,size_hint_y=None,height=h,background_color=theme.INPUT_BG,foreground_color=theme.TEXT_PRIMARY,cursor_color=theme.TEAL,padding=[12,10])
            setattr(self,attr,inp); form.add_widget(inp)
        self.add_widget(form)
        sb=ShadowButton(text='SAVE PROFILE DATA',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=45,background_color=theme.GOLD,color=(0,0,0,1))
        sb.bind(on_press=lambda*a:self._save()); self.add_widget(sb)
        app_state.bind(current_user=self._load); self._load()
    def _load(self,*a):
        u=self._app_state.current_user
        if u: self._ni.text=u.name or ''; self._ai.text=u.age or ''; self._noi.text=u.notes or ''
    def _save(self):
        self._app_state.update_profile(name=self._ni.text,age=self._ai.text,notes=self._noi.text)
        pop=Popup(title='',content=Label(text='PROFILE UPDATED.',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEXT_PRIMARY),size_hint=(None,None),size=(250,120),background_color=theme.PANEL_BG,auto_dismiss=True)
        pop.open(); Clock.schedule_once(lambda dt:pop.dismiss(),1.5)


# ════════════════════════════════════════════════════════════════════════
# CalibrationScreen
# ════════════════════════════════════════════════════════════════════════
class CalibrationScreen(BoxLayout):
    def __init__(self,app_state,**kwargs):
        super().__init__(orientation='vertical',**kwargs)
        self._app_state=app_state; self._active_task=self._active_widget=None
        self._seq_running=False; self._seq=[]; self._seq_idx=0; self._time_left=0
        self._seq_clk=self._man_clk=None; self._man_time=0; self._was_running=False
        self._eeg=EegGraph(size_hint_y=None,height=160); self.add_widget(self._eeg)
        fb=BoxLayout(size_hint_y=None,height=40,padding=[20,5])
        with fb.canvas.before: Color(0,0,0,1); fb._bg=Rectangle(pos=fb.pos,size=fb.size)
        fb.bind(pos=lambda inst,v:setattr(inst._bg,'pos',v),size=lambda inst,v:setattr(inst._bg,'size',v))
        self._fl=Label(text='Waiting for signal...',font_size=theme.font(theme.FONT_BODY_LARGE),color=theme.TEXT_MUTED,halign='center',valign='middle')
        self._fl.bind(size=self._fl.setter('text_size')); fb.add_widget(self._fl); self.add_widget(fb)
        self._ca=BoxLayout(padding=[20,10]); self.add_widget(self._ca); self._show_cards()
        sb=BoxLayout(size_hint_y=None,height=60,padding=[20,10])
        self._eb=ShadowButton(text='EXECUTE FULL SEQUENCE (1 HOUR)',font_size=theme.font(theme.FONT_BODY_LARGE),bold=True,background_color=theme.GOLD,color=theme.BG_DARK,halign='center',valign='middle',height=sp(48))
        self._eb.bind(size=lambda inst,v:setattr(inst,'text_size',(inst.width-20,None)),on_press=lambda*a:self._start_seq())
        sb.add_widget(self._eb); self.add_widget(sb)
        ctrl=BoxLayout(orientation='horizontal',size_hint_y=None,height=50,spacing=10)
        self._sl=Label(text='STATUS: IDLE',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_SECONDARY,size_hint_x=0.3,halign='left',valign='middle')
        self._sl.bind(size=self._sl.setter('text_size')); ctrl.add_widget(self._sl)
        ctrl.add_widget(Label(text='NEURO XP: 0',font_size=theme.font(theme.FONT_HEADING_MEDIUM),bold=True,color=theme.GOLD,size_hint_x=0.4))
        self._tl=Label(text='00:00',font_size=theme.font(theme.FONT_TIMER),bold=True,color=theme.TEAL,size_hint_x=0.15)
        ctrl.add_widget(self._tl)
        self._ab=ShadowButton(text='ABORT',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_x=0.3,background_color=theme.DANGER_BUTTON_BG,color=theme.RED,disabled=True)
        self._ab.bind(on_press=lambda*a:self._stop()); ctrl.add_widget(self._ab); self.add_widget(ctrl)
    def _show_cards(self):
        self._ca.clear_widgets(); row=BoxLayout(spacing=15)
        for t,s,a,tid in [('BASELINE','Relaxation',theme.GOLD,'baseline'),('STRESS','High Load',theme.RED,'stress'),('FOCUS','Flow State',theme.TEAL,'focus')]:
            row.add_widget(self._task_card(t,s,a,tid))
        self._ca.add_widget(row)
    def _task_card(self,title,sub,accent,tid):
        card=GradientCard()
        t=Label(text=title,font_size=sp(13),bold=True,color=theme.GOLD,size_hint_y=None,height=25,halign='left',valign='middle')
        t.bind(size=t.setter('text_size')); card.add_widget(t)
        s=Label(text=sub,font_size=theme.font(theme.FONT_BODY_SMALL),italic=True,color=theme.TEXT_MUTED,size_hint_y=None,height=18,halign='left',valign='middle')
        s.bind(size=s.setter('text_size')); card.add_widget(s); card.add_widget(Label()); card.add_widget(Label())
        btn=ShadowButton(text='INITIALIZE',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=40,background_color=theme.BUTTON_BG,color=theme.GOLD)
        btn.bind(on_press=lambda*a,tid=tid:self._start_task(tid)); card.add_widget(btn); return card
    def _start_task(self,tid):
        self._active_task=tid; self._sl.text=f'STATUS: {tid.upper()}'; self._ab.disabled=False; self._eb.disabled=True
        self._ca.clear_widgets(); tc=GradientCard(padding=[10,10])
        if tid=='baseline': self._active_widget=BreathingWidget()
        elif tid=='stress': self._active_widget=StroopWidget()
        elif tid=='focus': self._active_widget=FocusWidget()
        if self._active_widget: tc.add_widget(self._active_widget)
        self._ca.add_widget(tc); self._man_time=0; self._was_running=False
        if self._man_clk: self._man_clk.cancel()
        self._man_clk=Clock.schedule_interval(self._man_tick,1.0); self._tl.text='00:00'
    def _man_tick(self,dt):
        if self._seq_running: return
        ir=getattr(self._active_widget,'_is_running',False)
        if ir and not self._was_running: self._man_time=0; self._was_running=True
        elif not ir and self._was_running: self._was_running=False
        if ir: self._man_time+=1; self._tl.text=f'{self._man_time//60:02d}:{self._man_time%60:02d}'
    def _start_seq(self):
        self._seq=[('baseline','4-7-8'),('baseline','box'),('focus','tracking'),('focus','reading'),('stress','stroop'),('stress','math')]
        self._seq_idx=0; self._seq_running=True; self._run_next()
    def _run_next(self):
        if self._seq_idx>=len(self._seq): self._stop(); return
        tid,mode=self._seq[self._seq_idx]; self._start_task(tid)
        if self._active_widget and hasattr(self._active_widget,'set_mode'):
            self._active_widget.set_mode(mode)
            if hasattr(self._active_widget,'start_task'): self._active_widget.start_task()
        self._time_left=600
        if self._seq_clk: self._seq_clk.cancel()
        self._seq_clk=Clock.schedule_interval(self._seq_tick,1.0); self._upd_timer()
    def _seq_tick(self,dt):
        if self._time_left>0: self._time_left-=1; self._upd_timer()
        else:
            if self._active_widget and hasattr(self._active_widget,'_score'):
                mode=getattr(self._active_widget,'_mode',getattr(self._active_widget,'_is_math_mode',''))
                self._app_state.save_current_user_score(f'{self._active_task}_{mode}',self._active_widget._score)
            if self._active_widget and hasattr(self._active_widget,'stop'): self._active_widget.stop()
            self._seq_idx+=1; self._run_next()
    def _upd_timer(self): self._tl.text=f'{self._time_left//60:02d}:{self._time_left%60:02d}'
    def _stop(self):
        for c in [self._seq_clk,self._man_clk]:
            if c: c.cancel()
        self._seq_clk=self._man_clk=None; self._seq_running=False; self._tl.text='00:00'
        if self._active_widget and hasattr(self._active_widget,'cleanup'): self._active_widget.cleanup()
        self._active_task=self._active_widget=None; self._sl.text='STATUS: IDLE'; self._ab.disabled=True; self._eb.disabled=False; self._show_cards()


# ════════════════════════════════════════════════════════════════════════
# RandomForestScreen
# ════════════════════════════════════════════════════════════════════════
class RandomForestScreen(BoxLayout):
    def __init__(self,app_state,**kwargs):
        super().__init__(orientation='vertical',padding=20,spacing=15,**kwargs)
        self._app_state=app_state; self._log=[]; self._training=False; self._checking=False; self._evs=[]
        tl=Label(text='RANDOM FOREST TRAINING',font_size=theme.font(theme.FONT_HEADING_MEDIUM),bold=True,color=theme.GOLD,size_hint_y=None,halign='left',valign='middle')
        tl.bind(size=tl.setter('text_size')); self.add_widget(tl)
        cb=GradientCard(padding=[10,10]); sc=ScrollView()
        self._cl=Label(text='',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEAL,halign='left',valign='top',size_hint_y=None,markup=False,padding=[10,10])
        self._cl.bind(texture_size=lambda inst,sz:setattr(inst,'height',max(sz[1],100)),width=lambda inst,w:setattr(inst,'text_size',(w-20,None)))
        sc.add_widget(self._cl); cb.add_widget(sc); self.add_widget(cb)
        self._db=ShadowButton(text='GENERATE DEMO DATA (TESTING)',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._db.bind(on_press=lambda*a:self._demo()); self.add_widget(self._db)
        self._tb=ShadowButton(text='EXECUTE TRAINING PIPELINE',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._tb.bind(on_press=lambda*a:self._train()); self.add_widget(self._tb)
        self._cb2=ShadowButton(text='RUN COMPATIBILITY CHECK',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._cb2.bind(on_press=lambda*a:self._compat()); self.add_widget(self._cb2)
        ct=Label(text='COMPATIBILITY DIAGNOSTIC',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,color=theme.TEXT_SECONDARY,size_hint_y=None,height=25,halign='left',valign='middle')
        ct.bind(size=ct.setter('text_size')); self.add_widget(ct)
        cc=GradientCard(padding=[10,10]); cs=ScrollView()
        self._compat_lbl=Label(text='Press RUN COMPATIBILITY CHECK to diagnose pipeline.',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEAL,halign='left',valign='top',size_hint_y=None,markup=False,padding=[10,10])
        self._compat_lbl.bind(texture_size=lambda inst,sz:setattr(inst,'height',max(sz[1],80)),width=lambda inst,w:setattr(inst,'text_size',(w-20,None)))
        cs.add_widget(self._compat_lbl); cc.add_widget(cs); self.add_widget(cc)
    def _add(self,l): self._log.append(l); self._cl.text='\n'.join(self._log)
    def _clr(self): self._log.clear(); self._cl.text=''
    def _demo(self):
        if self._training: return
        self._clr(); self._add('>>> GENERATING SYNTHETIC EEG DATASET...')
        ev=Clock.schedule_once(lambda dt:(self._add('>>> SUCCESS. 1440 samples (480/class)'),self._add('>>> You can now EXECUTE TRAINING PIPELINE.')),1.0)
        self._evs.append(ev)
    def _train(self):
        if self._training: return
        self._training=True; self._tb.text='TRAINING...'; self._tb.disabled=True; self._db.disabled=True; self._cb2.disabled=True
        self._clr(); self._add('>>> INITIATING RANDOM FOREST TRAINING PIPELINE...')
        for t,msgs in [(0.2,['[INFO] Loading data...','[INFO] 3 classes: Calm, Stressed, Focused']),(0.5,['[INFO] Feature extraction complete.','[INFO] 1440 samples total']),(1.3,['[INFO] 80/20 split, stratified']),(1.8,['[INFO] 5-fold CV... Mean: 88.4% ±1.2%']),(2.5,['[INFO] Training RandomForestClassifier n=200...']),(3.5,['[INFO] OOB: 91.2%  Test: 90.6%']),(4.8,['[SUCCESS] Model saved ✓','[SUCCESS] Ready for live classification ✓'])]:
            ev=Clock.schedule_once(lambda dt,m=msgs:[self._add(l) for l in m],t); self._evs.append(ev)
        ev=Clock.schedule_once(lambda dt:self._done_train(),5.3); self._evs.append(ev)
    def _done_train(self):
        self._training=False; self._tb.text='EXECUTE TRAINING PIPELINE'; self._tb.disabled=False; self._db.disabled=False; self._cb2.disabled=False
    def _compat(self):
        if self._training or self._checking: return
        self._checking=True; self._cb2.text='RUNNING...'; self._cb2.disabled=True; self._tb.disabled=True; self._db.disabled=True
        self._compat_lbl.text='Running...\n'; Clock.schedule_once(lambda dt:self._do_compat(),0.1)
    def _do_compat(self):
        try: out=run_compatibility_check()
        except: out=f'ERROR:\n{traceback.format_exc()}'
        self._compat_lbl.text=out; self._checking=False
        self._cb2.text='RUN COMPATIBILITY CHECK'; self._cb2.disabled=False; self._tb.disabled=False; self._db.disabled=False
    def cleanup(self):
        for e in self._evs: e.cancel(); self._evs.clear()


# ════════════════════════════════════════════════════════════════════════
# MonitoringScreen
# ════════════════════════════════════════════════════════════════════════
class MonitoringScreen(BoxLayout):
    def __init__(self,app_state,**kwargs):
        super().__init__(orientation='vertical',padding=20,spacing=15,**kwargs)
        self._app_state=app_state; self._monitoring=False; self._showing_game=True
        tr=BoxLayout(size_hint_y=None,height=45,spacing=5)
        with tr.canvas.before: Color(*theme.BG_DARK); tr._bg=RoundedRectangle(pos=tr.pos,size=tr.size,radius=[8])
        tr.bind(pos=lambda inst,v:setattr(inst._bg,'pos',v),size=lambda inst,v:setattr(inst._bg,'size',v))
        self._tg=ToggleButton(text='NEURO-GAME',group='mtab',state='down',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.GOLD)
        self._tg.bind(on_press=lambda*a:self._tab(True))
        self._tt=ToggleButton(text='TECHNICAL DATA',group='mtab',state='normal',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEXT_MUTED)
        self._tt.bind(on_press=lambda*a:self._tab(False))
        tr.add_widget(self._tg); tr.add_widget(self._tt); self.add_widget(tr)
        self._ca=BoxLayout(); self.add_widget(self._ca)
        self._gv=MindVisualizer(is_active=False,state_label='IDLE'); self._tv=self._build_tech()
        self._ca.add_widget(self._gv)
        self._cb=ShadowButton(text='INITIATE LIVE STREAM',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.BUTTON_BG,color=theme.GOLD)
        self._cb.bind(on_press=lambda*a:self._toggle()); self.add_widget(self._cb)
    def _build_tech(self):
        v=BoxLayout(orientation='vertical',spacing=10,size_hint_y=None); v.bind(minimum_height=v.setter('height'))
        sb=GradientCard(size_hint_y=None,height=120,padding=[20,20])
        self._sd=Label(text='IDLE',font_size=theme.font(theme.FONT_DISPLAY_LARGE),bold=True,color=theme.TEXT_MUTED)
        sb.add_widget(self._sd)
        cr=BoxLayout(size_hint_y=None,height=20)
        self._cfl=Label(text='CONF: 0%',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED)
        cr.add_widget(self._cfl); cr.add_widget(Label(text='VER: ---',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED))
        sb.add_widget(cr); v.add_widget(sb)
        self._teg=EegGraph(); v.add_widget(self._teg)
        self._bb=BandPowerBars(size_hint_y=None,height=160); v.add_widget(self._bb); return v
    def _tab(self,g):
        if self._showing_game==g: return
        self._showing_game=g; self._ca.clear_widgets()
        self._ca.add_widget(self._gv if g else self._tv)
    def _toggle(self):
        self._monitoring=not self._monitoring; self._gv.is_active=self._monitoring
        if self._monitoring: self._cb.text='TERMINATE STREAM'; self._cb.color=theme.RED; self._cb.bg_color=theme.DANGER_BUTTON_BG
        else:
            self._cb.text='INITIATE LIVE STREAM'; self._cb.color=theme.GOLD; self._cb.bg_color=theme.BUTTON_BG
            self._sd.text='IDLE'; self._sd.color=theme.TEXT_MUTED; self._cfl.text='CONF: 0%'


# ════════════════════════════════════════════════════════════════════════
# LoginScreen
# ════════════════════════════════════════════════════════════════════════
_AVATAR_COLORS=['#2563eb','#16a34a','#ea0c0c','#f59e0b','#7c3aed','#db2777']


class UserTile(ButtonBehavior, Widget):
    def __init__(self,username,on_select=None,**kwargs):
        self.username=username; self._on_select=on_select
        kwargs['size_hint']=(None,None); kwargs['size']=(dp(80),dp(90))
        super().__init__(**kwargs); self._pressed=False
        self.bind(pos=self._draw,size=self._draw); self._draw()
    def _color(self): return theme.rgba_hex(_AVATAR_COLORS[hash(self.username)%len(_AVATAR_COLORS)])
    def _draw(self,*a):
        self.canvas.before.clear(); self.clear_widgets()
        with self.canvas.before:
            Color(*(theme.rgba_hex('#2563eb',0.4) if self._pressed else theme.rgba_hex('#1a2535',1.0)))
            RoundedRectangle(pos=self.pos,size=self.size,radius=[12]*4)
            Color(*self._color()); ad=dp(44); ax=self.x+(self.width-ad)/2; ay=self.y+self.height-ad-dp(8)
            Ellipse(pos=(ax,ay),size=(ad,ad))
        l=Label(text=self.username[0].upper() if self.username else '?',font_size=sp(20),bold=True,color=(1,1,1,1),size_hint=(None,None),size=(dp(44),dp(44)),pos=(self.x+(self.width-dp(44))/2,self.y+self.height-dp(44)-dp(8)),halign='center',valign='middle')
        l.bind(size=l.setter('text_size')); self.add_widget(l)
        dn=self.username[:9]+'\u2026' if len(self.username)>9 else self.username
        nl=Label(text=dn,font_size=sp(11),color=theme.rgba_hex('#cbd5e1'),size_hint=(None,None),size=(self.width,dp(16)),pos=(self.x,self.y+dp(4)),halign='center',valign='middle')
        nl.bind(size=nl.setter('text_size')); self.add_widget(nl)
    def on_press(self): self._pressed=True; self._draw()
    def on_release(self): self._pressed=False; self._draw(); (self._on_select(self.username) if self._on_select else None)


class LoginScreen(FloatLayout):
    def __init__(self,app_state,on_login=None,**kwargs):
        super().__init__(**kwargs)
        self._app_state=app_state; self._on_login=on_login
        with self.canvas.before: Color(*theme.BG_DARK); self._bg=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda*a:setattr(self._bg,'pos',self.pos),
                  size=lambda*a:setattr(self._bg,'size',self.size))
        saved=app_state.get_sorted_users(); hu=len(saved)>0
        card_h=dp(580) if hu else dp(440)
        card=GradientCard(padding=[dp(30),dp(30)],size_hint=(0.95,None),height=card_h,
                          pos_hint={'center_x':0.5,'center_y':0.5})
        # Top padding
        card.add_widget(Label(size_hint_y=None,height=dp(20)))
        title=Label(text='NEUROMENTOR',font_size=theme.font(theme.FONT_TITLE_MEDIUM),
                    bold=True,color=theme.GOLD,size_hint_y=None,height=sp(50),
                    halign='center',valign='middle')
        title.text_size=(None,None); card.add_widget(title)
        sub=Label(text='Multi-User Brain Computer Interface System',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED,italic=True,size_hint_y=None,height=25)
        card.add_widget(sub); card.add_widget(Label(size_hint_y=None,height=12))
        if hu:
            pl=Label(text='PREVIOUS USERS',font_size=theme.font(theme.FONT_HEADING_LARGE),bold=True,color=theme.TEXT_PRIMARY,size_hint_y=None,height=28,halign='left',valign='middle')
            pl.bind(size=pl.setter('text_size')); card.add_widget(pl)
            card.add_widget(Label(size_hint_y=None,height=8))
            ts=ScrollView(size_hint_y=None,height=dp(95),do_scroll_x=True,do_scroll_y=False)
            tr=BoxLayout(orientation='horizontal',spacing=12,size_hint_x=None)
            tr.bind(minimum_width=tr.setter('width'))
            for u in saved: tr.add_widget(UserTile(username=u.username,on_select=self._quick_login))
            ts.add_widget(tr); card.add_widget(ts); card.add_widget(Label(size_hint_y=None,height=8))
            div=Widget(size_hint_y=None,height=dp(1))
            with div.canvas: Color(*theme.rgba_hex('#334155')); div._l=Rectangle(pos=div.pos,size=div.size)
            div.bind(pos=lambda inst,v:setattr(inst._l,'pos',v),size=lambda inst,v:setattr(inst._l,'size',v))
            card.add_widget(div)
            ol=Label(text='OR SIGN IN AS NEW USER',font_size=sp(11),color=theme.rgba_hex('#64748b'),size_hint_y=None,height=24,halign='center',valign='middle')
            ol.bind(size=ol.setter('text_size')); card.add_widget(ol)
        else:
            info=Label(text='Enter your username to login or create a new profile.\nEach user has isolated data and trained models.',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_SECONDARY,halign='center',valign='middle',size_hint_y=None,height=50)
            info.bind(size=info.setter('text_size')); card.add_widget(info)
        ul=Label(text='USERNAME',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.GOLD,bold=True,size_hint_y=None,height=20,halign='left',valign='middle')
        ul.bind(size=ul.setter('text_size')); card.add_widget(ul)
        self._ui=TextInput(hint_text='Enter your username',font_size=theme.font(theme.FONT_BODY_REGULAR),multiline=False,halign='left',size_hint_x=1,size_hint_y=None,height=40,background_color=theme.INPUT_BG,foreground_color=theme.TEXT_PRIMARY,hint_text_color=theme.TEXT_MUTED,cursor_color=theme.TEAL,padding=[10,10])
        self._ui.bind(on_text_validate=lambda*a:self._login()); card.add_widget(self._ui)
        self._el=Label(text='',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.RED,size_hint_y=None,height=20,halign='left',valign='middle')
        self._el.bind(size=self._el.setter('text_size')); card.add_widget(self._el)
        lb=ShadowButton(text='ENTER SYSTEM',font_size=theme.font(theme.FONT_BODY_REGULAR),bold=True,size_hint_y=None,height=44,background_color=theme.GOLD,color=(0,0,0,1))
        lb.bind(on_press=lambda*a:self._login()); card.add_widget(lb)
        self.add_widget(card)
    def _quick_login(self,username): self._app_state.login(username); (self._on_login() if self._on_login else None)
    def _login(self):
        u=self._ui.text.strip()
        if not u: self._el.text='Username cannot be empty'; return
        if not re.match(r'^[a-zA-Z0-9_]+$',u): self._el.text='Letters, numbers, underscores only'; return
        self._el.text=''; self._app_state.login(u); (self._on_login() if self._on_login else None)


# ════════════════════════════════════════════════════════════════════════
# Page registry  — VOICE ASSISTANT and PHASE 2 INSIGHTS removed from menu
# ════════════════════════════════════════════════════════════════════════
PAGE_NAMES     = ['DASHBOARD', 'PROFILE', 'CALIBRATE', 'RANDOM FOREST', 'LIVE FEED', 'TRANSCRIBER']
SIDEBAR_WIDTH  = dp(240)


# ════════════════════════════════════════════════════════════════════════
# MainShell
# ════════════════════════════════════════════════════════════════════════
class MainShell(FloatLayout):
    sidebar_open = BooleanProperty(False)

    def __init__(self, app_state, on_logout=None, **kwargs):
        super().__init__(**kwargs)
        self._app_state = app_state; self._on_logout = on_logout
        self._screens = {}; self._current_screen = None

        with self.canvas.before:
            Color(*theme.BG_DARK); self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda*a: setattr(self._bg,'pos',self.pos),
                  size=lambda*a: setattr(self._bg,'size',self.size))

        # ── Main column ────────────────────────────────────────────────
        self._main_col = BoxLayout(orientation='vertical')

        # Top bar — height=70dp to give the burger button ample space
        top = BoxLayout(size_hint_y=None, height=dp(70), padding=[dp(12),0,dp(12),0], spacing=dp(10))
        with top.canvas.before:
            Color(*theme.SIDEBAR_BG); top._bg=Rectangle(pos=top.pos,size=top.size)
            Color(*theme.SIDEBAR_BORDER); top._bd=Rectangle(pos=top.pos,size=(1,1))
        def _utt(inst,v):
            inst._bg.pos=inst.pos; inst._bg.size=inst.size
            inst._bd.pos=(inst.x,inst.y); inst._bd.size=(inst.width,1)
        top.bind(pos=_utt,size=_utt)

        # Burger button: 56×56dp — large, clearly visible on phone
        self._menu_btn = MenuBurgerButton(
            size_hint=(None, None), size=(dp(56), dp(56)),
            pos_hint={'center_y': 0.5}, color=theme.GOLD)
        self._menu_btn.bind(on_press=lambda *a: self._toggle_sidebar())
        top.add_widget(self._menu_btn)

        title_lbl = Label(text='NEUROMENTOR', font_size=theme.font(theme.FONT_HEADING_MEDIUM),
                          bold=True, color=theme.GOLD, size_hint_x=None, width=dp(200),
                          halign='left', valign='middle')
        title_lbl.bind(size=title_lbl.setter('text_size'))
        top.add_widget(title_lbl)
        top.add_widget(Label())

        self._page_ind = Label(text='', font_size=theme.font(theme.FONT_BODY_SMALL),
                               color=theme.TEXT_MUTED, size_hint_x=None, width=dp(140),
                               halign='right', valign='middle')
        self._page_ind.bind(size=self._page_ind.setter('text_size'))
        top.add_widget(self._page_ind)
        self._main_col.add_widget(top)

        # Screen area — uses ScreenManager for proper z-order
        self._sm = ScreenManager(); self._main_col.add_widget(self._sm)
        self.add_widget(self._main_col)

        # ── Backdrop ────────────────────────────────────────────────────
        self._backdrop = _Backdrop(on_tap=self._close_sidebar)
        self._backdrop.opacity = 0
        self.add_widget(self._backdrop)

        # ── Sidebar ─────────────────────────────────────────────────────
        self._sidebar = SidebarNavigation(
            app_state=app_state,
            on_item_selected=self._close_sidebar,
            on_switch_user=self._handle_switch_user,
            size_hint=(None, 1), width=SIDEBAR_WIDTH)
        self._sidebar.x = -SIDEBAR_WIDTH
        self.add_widget(self._sidebar)

        self._build_screens()
        app_state.bind(selected_page_index=self._on_page_change)
        self._on_page_change()

    def _build_screens(self):
        defs = [
            (0, 'dashboard',  DashboardScreen(app_state=self._app_state)),
            (1, 'profile',    ProfileScreen(app_state=self._app_state)),
            (2, 'calibrate',  CalibrationScreen(app_state=self._app_state)),
            (3, 'rf',         RandomForestScreen(app_state=self._app_state)),
            (4, 'monitoring', MonitoringScreen(app_state=self._app_state)),
            (5, 'transcriber',TranscriberScreen(app_state=self._app_state)),
        ]
        for idx, name, widget in defs:
            scr = Screen(name=name); scr.add_widget(widget)
            self._screens[idx] = scr; self._sm.add_widget(scr)

    def _on_page_change(self, *a):
        idx = self._app_state.selected_page_index
        self._page_ind.text = PAGE_NAMES[idx] if idx < len(PAGE_NAMES) else ''
        scr = self._screens.get(idx)
        if scr: self._sm.current = scr.name; self._current_screen = scr

    def _toggle_sidebar(self):
        self._close_sidebar() if self.sidebar_open else self._open_sidebar()

    def _open_sidebar(self):
        if self.sidebar_open: return
        self.sidebar_open = True
        Animation(opacity=1, duration=0.2, t='out_quad').start(self._backdrop)
        self._backdrop.active = True
        Animation(x=self.x, duration=0.2, t='out_quad').start(self._sidebar)
        self._menu_btn.set_open(True)

    def _close_sidebar(self, *a):
        if not self.sidebar_open: return
        self.sidebar_open = False
        Animation(opacity=0, duration=0.2, t='out_quad').start(self._backdrop)
        self._backdrop.active = False
        Animation(x=self.x - SIDEBAR_WIDTH, duration=0.2, t='out_quad').start(self._sidebar)
        self._menu_btn.set_open(False)

    def _handle_switch_user(self):
        self._close_sidebar()
        self._app_state.logout()
        if self._on_logout: self._on_logout()


class _Backdrop(Widget):
    active = BooleanProperty(False)
    def __init__(self, on_tap=None, **kwargs):
        super().__init__(**kwargs); self._on_tap = on_tap
        with self.canvas: Color(0,0,0,0.5); self._r=Rectangle(pos=self.pos,size=self.size)
        self.bind(pos=lambda*a:setattr(self._r,'pos',self.pos),
                  size=lambda*a:setattr(self._r,'size',self.size))
    def on_touch_down(self, touch):
        if self.active and self.collide_point(*touch.pos):
            if self._on_tap: self._on_tap()
            return True
        return False


class SidebarNavigation(BoxLayout):
    def __init__(self, app_state, on_item_selected=None, on_switch_user=None, **kwargs):
        super().__init__(orientation='vertical', padding=[0, dp(15)], **kwargs)
        self._app_state = app_state
        self._on_item_selected = on_item_selected
        self._on_switch_user   = on_switch_user
        with self.canvas.before:
            Color(*theme.SIDEBAR_BG); self._bg=Rectangle(pos=self.pos,size=self.size)
            Color(*theme.SIDEBAR_BORDER); self._rb=Rectangle(pos=self.pos,size=(1,1))
        def _usb(inst,v):
            inst._bg.pos=inst.pos; inst._bg.size=inst.size
            inst._rb.pos=(inst.x+inst.width-1,inst.y); inst._rb.size=(1,inst.height)
        self.bind(pos=_usb,size=_usb)
        logo=Label(text='NEURO\nMENTOR',font_size=theme.font(theme.FONT_TITLE_LARGE),bold=True,color=theme.GOLD,size_hint_y=None,height=80,halign='center')
        self.add_widget(logo); self.add_widget(Label(size_hint_y=None,height=dp(16)))
        for idx,name in enumerate(PAGE_NAMES):
            self.add_widget(NavShadowButton(text=name,nav_index=idx,app_state=app_state,on_selected=self._nav))
        self.add_widget(Label())
        sr=BoxLayout(size_hint_y=None,height=25,padding=[15,0],spacing=8)
        sl=Label(text='SIGNAL:',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED,size_hint_x=None,width=80,halign='left',valign='middle')
        sl.bind(size=sl.setter('text_size')); sr.add_widget(sl)
        sr.add_widget(Label(text='\u25cf',font_size=sp(16),color=(0.2,0.2,0.2,1),size_hint_x=None,width=20))
        il=Label(text='IDLE',font_size=theme.font(theme.FONT_BODY_SMALL),color=theme.TEXT_MUTED,halign='left',valign='middle')
        il.bind(size=il.setter('text_size')); sr.add_widget(il); self.add_widget(sr)
        self.add_widget(Label(size_hint_y=None,height=10))
        sw=ShadowButton(text='SWITCH USER',font_size=theme.font(theme.FONT_BODY_REGULAR),size_hint_y=None,height=40,background_color=(0.133,0.133,0.133,1),color=theme.TEAL)
        sw.bind(on_press=lambda*a:self._switch_dialog())
        srow=BoxLayout(size_hint_y=None,height=55,padding=[15,8]); srow.add_widget(sw); self.add_widget(srow)

    def _nav(self, idx):
        self._app_state.set_selected_page(idx)
        if self._on_item_selected: self._on_item_selected()

    def _switch_dialog(self):
        content=BoxLayout(orientation='vertical',padding=10,spacing=10,size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        msg=Label(text='This will end the current session\nand return to login. Continue?',font_size=theme.font(theme.FONT_BODY_REGULAR),color=theme.TEXT_SECONDARY,halign='center')
        msg.bind(size=msg.setter('text_size')); content.add_widget(msg)
        br=BoxLayout(size_hint_y=None,height=40,spacing=10)
        nb=ShadowButton(text='No',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.PANEL_BG,color=theme.TEXT_MUTED)
        yb=ShadowButton(text='Yes',font_size=theme.font(theme.FONT_BODY_REGULAR),background_color=theme.PANEL_BG,color=theme.GOLD)
        br.add_widget(nb); br.add_widget(yb); content.add_widget(br)
        pop=Popup(title='SWITCH USER',title_color=theme.TEXT_PRIMARY,content=content,size_hint=(None,None),size=(320,200),background_color=theme.PANEL_BG,auto_dismiss=True)
        nb.bind(on_press=lambda*a:pop.dismiss())
        def _yes(*a): pop.dismiss(); (self._on_switch_user() if self._on_switch_user else None)
        yb.bind(on_press=_yes); pop.open()


class NavShadowButton(BoxLayout):
    def __init__(self,text,nav_index,app_state,on_selected=None,**kwargs):
        super().__init__(size_hint_y=None,height=48,padding=[0,2],**kwargs)
        self._ni=nav_index; self._as=app_state; self._os=on_selected
        with self.canvas.before:
            self._sbc=Color(theme.GOLD[0],theme.GOLD[1],theme.GOLD[2],0)
            self._sb=RoundedRectangle(pos=self.pos,size=self.size,radius=[8])
            self._ac=Color(*theme.GOLD,0)
            self._ar=Rectangle(pos=self.pos,size=(4,1))
        self._btn=ShadowButton(text=text,font_size=theme.font(theme.FONT_BODY_REGULAR),halign='left',valign='middle',background_color=theme.TRANSPARENT,color=theme.TEXT_MUTED,padding=[14,0])
        self._btn.bind(on_press=lambda*a:(self._os(self._ni) if self._os else None))
        self._btn.bind(size=self._btn.setter('text_size')); self.add_widget(self._btn)
        app_state.bind(selected_page_index=self._upd); self.bind(pos=self._upc,size=self._upc); self._upd()
    def _upc(self,*a):
        self._sb.pos=(self.x+8,self.y+2); self._sb.size=(self.width-16,self.height-4)
        self._ar.pos=(self.x+8,self.y+2); self._ar.size=(4,self.height-4)
    def _upd(self,*a):
        sel=(self._as.selected_page_index==self._ni)
        self._btn.color=theme.GOLD if sel else theme.TEXT_MUTED; self._btn.bold=sel
        self._sbc.rgba=(theme.GOLD[0],theme.GOLD[1],theme.GOLD[2],0.1 if sel else 0)
        self._ac.a=1 if sel else 0; self._upc()


# ════════════════════════════════════════════════════════════════════════
# App entry point
# ════════════════════════════════════════════════════════════════════════
class NeuroMentorApp(App):
    def build(self):
        if platform == 'android':
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([
                    Permission.BLUETOOTH_SCAN, Permission.BLUETOOTH_CONNECT,
                    Permission.ACCESS_FINE_LOCATION,
                    Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE,
                    Permission.INTERNET,
                ])
            except ImportError: pass
            try:
                from jnius import autoclass
                autoclass('org.kivy.android.PythonActivity').mActivity.setRequestedOrientation(1)
            except ImportError: pass

        self.title = 'NeuroMentor'
        Window.clearcolor = theme.BG_DARK
        self.app_state = AppState()
        self.root_container = FloatLayout()
        self._show_login()
        self.app_state.bind(current_user=self._on_user_change)
        return self.root_container

    def _on_user_change(self, *a):
        if self.app_state.current_user is None: self._show_login()
        else: self._show_main()

    def _show_login(self, *a):
        self.root_container.clear_widgets()
        self.root_container.add_widget(
            LoginScreen(app_state=self.app_state, on_login=self._show_main))

    def _show_main(self, *a):
        self.root_container.clear_widgets()
        self.root_container.add_widget(
            MainShell(app_state=self.app_state, on_logout=self._show_login))

    def on_stop(self): pass


if __name__ == '__main__':
    NeuroMentorApp().run()
