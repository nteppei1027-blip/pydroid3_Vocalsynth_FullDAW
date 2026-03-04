# ==========================================
# HyperNekoProduct v22.0 "HyperMusicRagnarok" 
# Part 1: Imports, Constants, DSP & Vocal Engine
# ==========================================

import numpy as np
import random
import threading
import traceback
import re
import wave
import os
import logging
import time
import json
import datetime
import copy
import math
import shutil

# --- Audio Backend ---
import pygame

# --- Scipy (EQ/Filter/Formant/WAV) ---
try:
    import scipy.signal as signal
    from scipy.io import wavfile as scipy_wav  # <--- 名前を変えて衝突回避！
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: Scipy not found.")

# --- MIDI ---
try:
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False
    print("Warning: 'mido' library not found. MIDI export will be disabled.")

# --- Kivy UI Imports ---
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.core.text import Label as CoreLabel
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission
    def ask_permissions():
        request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])

from kivy.core.audio import SoundLoader

# --- Logging Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HyperNekoVocaloid")

# --- System Config ---
FS = 48000
P_NYAN = 0.102  # Chaos Constant
RHYTHM_STEPS = 16

# Pygame Mixer Init
pygame.mixer.pre_init(FS, -16, 2, 1024)
pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(64)

# --- Global Data Definitions ---

DEFAULT_ENGINE_CONFIG = {
    "master_vol": 1.0,
    "vel_humanize": 0.2,
    "reverb_mix": 0.3,
    "reverb_decay": 0.3,
    "vibrato_depth": 0.05,
    "swing_amount": 0.0,
    "sc_strength": 0.8,
    "sc_release": 0.15,
    "sc_curve": 2.0,
    "bitcrush_rate": 0.5,
    "bitcrush_depth": 8,
    "vocal_formant_shift": 1.0,
    "vocal_mix": 1.0,
    "breath_vol": 0.02,
    "breath_chance": 0.5,
    "cons_attack": 1.0
}

def simple_lp(x, strength=0.5):
    if strength <= 0: return x
    # 修正: strengthが小さい時にウィンドウサイズが0にならないように保護
    window_size = max(2, int(15 * strength))
    w = np.ones(window_size) / window_size
    # mode='same' でサイズが変わらないようにする
    if len(x) == 0: return x
    return np.convolve(x, w, mode='same')

def simple_hp(x, strength=0.5):
    return x - simple_lp(x, strength)

def three_band_eq(x, low=1.0, mid=1.0, high=1.0):
    if len(x) == 0: return x
    l_f = simple_lp(x, 0.8)
    h_f = simple_hp(x, 0.8)
    m_f = x - l_f - h_f
    return l_f * low + m_f * mid + h_f * high

# Formant Data with Glottal Physics Parameters
# oq (Open Quotient): 0.5=Tense/Buzzy, 0.8=Breathy/Soft
# alpha: Higher = More spectral tilt (Softer/Darker), Lower = Brighter
# ==========================================
# 1. FORMANT_DATA の定義 (F4追加版: High-Definition)
# ==========================================
# 超人仕様：F5拡張版 FORMANT_DATA
# [F1, F2, F3, F4, F5] の5帯域構成
# ==========================================
FORMANT_DATA = {
    # 基本母音
    # F5(4500Hz以上)を追加。超人レベルの「抜け感」と「エアー」を担当。
    'a': {'freq': [730, 1090, 2440, 3500, 4800], 'bw': [80, 90, 120, 150, 200], 'gain': 1.0, 'oq': 0.70, 'alpha': 2.0},
    'i': {'freq': [270, 2290, 3010, 3700, 5200], 'bw': [60, 90, 140, 200, 250], 'gain': 0.85, 'oq': 0.55, 'alpha': 2.5},
    'u': {'freq': [300, 870, 2240, 3300, 4500], 'bw': [70, 100, 120, 150, 180], 'gain': 0.80, 'oq': 0.60, 'alpha': 2.3},
    'e': {'freq': [530, 1840, 2480, 3600, 4900], 'bw': [70, 90, 110, 150, 220], 'gain': 0.90, 'oq': 0.60, 'alpha': 2.2},
    'o': {'freq': [460, 880, 2800, 3400, 4600], 'bw': [80, 90, 120, 150, 190], 'gain': 0.95, 'oq': 0.65, 'alpha': 2.1},
# 鼻音 (F5追加で高域の鼻腔共鳴を強調、抜け感向上)
'n': {'freq': [250, 1000, 2000, 3200, 4500], 'bw': [70, 100, 150, 200, 250], 'gain': 0.70, 'oq': 0.75, 'alpha': 3.0},
'm': {'freq': [250, 900, 1800, 3000, 4300], 'bw': [60, 100, 120, 200, 220], 'gain': 0.65, 'oq': 0.75, 'alpha': 3.2},
'ng': {'freq': [200, 800, 1600, 2800, 4200], 'bw': [80, 150, 200, 250, 300], 'gain': 0.60, 'oq': 0.80, 'alpha': 3.5},

# デボイスト母音 (F5追加で息漏れの高域ノイズを強化)
'i_dv': {'freq': [270, 2290, 3010, 4000, 5300], 'bw': [200, 300, 400, 500, 600], 'gain': 0.0, 'oq': 0.90, 'alpha': 4.0},
'u_dv': {'freq': [300, 870, 2240, 3800, 5000], 'bw': [200, 300, 400, 500, 550], 'gain': 0.0, 'oq': 0.90, 'alpha': 4.0},
'a_dv': {'freq': [730, 1090, 2440, 3800, 5100], 'bw': [200, 300, 400, 500, 600], 'gain': 0.0, 'oq': 0.90, 'alpha': 4.0},

# 半母音 (F5追加で滑らかな移行時の高域輝きを追加)
'y': {'freq': [270, 2290, 3010, 3800, 5100], 'bw': [60, 90, 140, 180, 230], 'gain': 0.50, 'oq': 0.55, 'alpha': 2.5},
'w': {'freq': [300, 870, 2240, 3200, 4500], 'bw': [70, 100, 120, 150, 200], 'gain': 0.50, 'oq': 0.60, 'alpha': 2.3},
'r': {'freq': [530, 1840, 2480, 3500, 4800], 'bw': [70, 90, 110, 150, 220], 'gain': 0.60, 'oq': 0.65, 'alpha': 2.4},

# 特殊母音拡張 (歌唱用強調形: 大文字で区別。基本形を基にF1-F3を強調調整、F4/F5で輝き追加)
'A': {'freq': [750, 1150, 2500, 3600, 4900], 'bw': [90, 100, 130, 160, 210], 'gain': 1.1, 'oq': 0.68, 'alpha': 2.1},  # 'a'の強調版
'I': {'freq': [280, 2350, 3100, 3800, 5300], 'bw': [65, 95, 150, 210, 260], 'gain': 0.88, 'oq': 0.53, 'alpha': 2.6},  # 'i'の強調版
'U': {'freq': [310, 900, 2300, 3400, 4600], 'bw': [75, 110, 130, 160, 190], 'gain': 0.82, 'oq': 0.58, 'alpha': 2.4},  # 'u'の強調版
'E': {'freq': [550, 1900, 2550, 3700, 5000], 'bw': [75, 95, 120, 160, 230], 'gain': 0.92, 'oq': 0.58, 'alpha': 2.3},  # 'e'の強調版
'O': {'freq': [380, 670, 2500, 3500, 4700], 'bw': [75, 85, 110, 130, 200], 'gain': 0.92, 'oq': 0.63, 'alpha': 2.2},  # 'o'の強調版
'N': {'freq': [250, 1000, 2000, 3200, 4500], 'bw': [70, 100, 150, 200, 250], 'gain': 0.70, 'oq': 0.75, 'alpha': 3.0},  # 'n'の歌唱版 (案から)
'R': {'freq': [550, 1900, 2550, 3600, 4900], 'bw': [75, 95, 120, 160, 230], 'gain': 0.62, 'oq': 0.63, 'alpha': 2.5},  # 'r'の強調版
}
# ==========================================
# 2. 子音データ（別物として定義）
# ==========================================
CONSONANT_SPECTRAL = {
    # 摩擦音 (Fricatives: noise_level追加で息の強さを制御、高周波強調)
    's': {'band': (4500, 10000), 'gain': 0.95, 'type': 'fricative', 'noise_level': 1.0},
    'sh': {'band': (3500, 7500), 'gain': 0.85, 'type': 'fricative', 'noise_level': 0.9},
    'h': {'band': (1200, 4500), 'gain': 0.75, 'type': 'fricative', 'noise_level': 0.8},
    'f': {'band': (800, 3500), 'gain': 0.45, 'type': 'fricative', 'noise_level': 0.6},
    'z': {'band': (3000, 6500), 'gain': 0.85, 'type': 'fricative', 'noise_level': 0.9},  # 有声: noise_level高め
    'v': {'band': (1000, 3500), 'gain': 0.55, 'type': 'fricative', 'noise_level': 0.7},
    'hy': {'band': (2000, 5000), 'gain': 0.65, 'type': 'fricative', 'noise_level': 0.7},  # 拗音
    
    # 破裂音・破擦音 (Plosives/Affricates: burst_lenを微調整、bandを狭く精密に)
    'k': {'band': (1500, 3500), 'gain': 1.2, 'type': 'plosive', 'burst_len': 0.015, 'noise_level': 1.0},
    't': {'band': (3000, 7000), 'gain': 1.2, 'type': 'plosive', 'burst_len': 0.010, 'noise_level': 1.0},
    'p': {'band': (500, 1500), 'gain': 1.0, 'type': 'plosive', 'burst_len': 0.012, 'noise_level': 0.8},
    'ts': {'band': (4000, 9000), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.020, 'noise_level': 0.9},
    'ch': {'band': (3000, 6500), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.025, 'noise_level': 0.9},
    'j': {'band': (2500, 5500), 'gain': 0.75, 'type': 'plosive', 'burst_len': 0.020, 'noise_level': 0.8},  # 有声
    'dz': {'band': (3500, 7000), 'gain': 0.80, 'type': 'plosive', 'burst_len': 0.015, 'noise_level': 0.8},  # 新規: 有声ts
    
    # 有声破裂音 (Voiced Plosives: noise_level低めで声帯振動強調)
    'g': {'band': (1000, 2500), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.015, 'noise_level': 0.6},
    'd': {'band': (2000, 4500), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.010, 'noise_level': 0.6},
    'b': {'band': (200, 1000), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.012, 'noise_level': 0.5},
    
    # 鼻音・流音 (Nasals/Liquids: FORMANT_DATAと連携、低周波中心)
    'm': {'band': (200, 800), 'gain': 0.80, 'type': 'nasal', 'noise_level': 0.2},
    'n': {'band': (200, 1200), 'gain': 0.80, 'type': 'nasal', 'noise_level': 0.2},
    'r': {'band': (1000, 2500), 'gain': 0.65, 'type': 'liquid', 'noise_level': 0.3},  # 日本語rはflap寄り
    
    # 拗音拡張 (Palatalized: bandを高めにシフト、noise_level調整)
    'ky': {'band': (2000, 4000), 'gain': 1.1, 'type': 'plosive', 'burst_len': 0.015, 'noise_level': 1.0},
    'gy': {'band': (1500, 3000), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.015, 'noise_level': 0.6},
    'py': {'band': (800, 2000), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.012, 'noise_level': 0.8},
    'by': {'band': (400, 1200), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.012, 'noise_level': 0.5},
    'ny': {'band': (500, 1500), 'gain': 0.80, 'type': 'nasal', 'noise_level': 0.2},
    'my': {'band': (400, 1000), 'gain': 0.80, 'type': 'nasal', 'noise_level': 0.2},
    'ry': {'band': (1500, 3000), 'gain': 0.65, 'type': 'liquid', 'noise_level': 0.3},
    'ty': {'band': (3500, 7500), 'gain': 1.1, 'type': 'plosive', 'burst_len': 0.010, 'noise_level': 1.0},  # 新規: ちゅ系
    'dy': {'band': (2500, 5000), 'gain': 0.95, 'type': 'plosive', 'burst_len': 0.010, 'noise_level': 0.6},  # 新規
}
CONSONANT_SPECTRAL.update({
    # English / Extra Consonants
    'th': {'band': (2000, 6000), 'gain': 0.6, 'type': 'fricative'},
    'dh': {'band': (1000, 4000), 'gain': 0.7, 'type': 'fricative'}, # The
    'l':  {'band': (300, 1000),  'gain': 0.8, 'type': 'liquid'},
    'br': {'band': (200, 800),   'gain': 0.9, 'type': 'plosive', 'burst_len': 0.02}, # Brrr
    'clt': {'band': (0, 1),      'gain': 0.0, 'type': 'silence'}, # Sokuon (Pause)
})

# ==========================================
# 3.音韻処理用のセット (拡張版)
# ==========================================
# 無声子音 (拗音追加)
VOICELESS_CONS = {'k', 's', 't', 'h', 'p', 'ch', 'ts', 'sh', 'f', 'ky', 'py', 'ty', 'hy'}

# 唇音 (f/v追加で唇摩擦をカバー)
LABIAL_CONS = {'p', 'b', 'm', 'f', 'v'}

# 喉奥音 (ng追加、拗音対応)
VELAR_CONS = {'k', 'g', 'ng', 'ky', 'gy'}

# ユーザーの解析データを保持する辞書
USER_VOICE_DATA = {} 
# 例: {'a': [800, 1200, 2500, 3500], 'i': ...}

#Note, Chord

NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_INDEX = {n: i for i, n in enumerate(NOTE_ORDER)}
FLAT_MAP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# --- CHORD RULES (Extended) ---
CHORD_RULES = {

    # --- Triads ---
    "":      [0, 4, 7],
    "maj":   [0, 4, 7],
    "m":     [0, 3, 7],
    "min":   [0, 3, 7],
    "-":     [0, 3, 7],

    "dim":   [0, 3, 6],
    "aug":   [0, 4, 8],

    "5":     [0, 7],

    # --- Seventh ---
    "7":     [0, 4, 7, 10],
    "maj7":  [0, 4, 7, 11],
    "M7":    [0, 4, 7, 11],
    "m7":    [0, 3, 7, 10],
    "mM7":   [0, 3, 7, 11],

    "dim7":  [0, 3, 6, 9],
    "m7b5":  [0, 3, 6, 10],
    "halfdim":[0, 3, 6, 10],

    "6":     [0, 4, 7, 9],
    "m6":    [0, 3, 7, 9],

    # --- Suspended ---
    "sus2":      [0, 2, 7],
    "sus4":      [0, 5, 7],
    "7sus4":     [0, 5, 7, 10],
    "sus2add4":  [0, 2, 5, 7],
    "sus4add9":  [0, 5, 7, 14],

    # --- Add ---
    "add2":  [2],
    "add4":  [5],
    "add6":  [9],
    "add9":  [14],
    "add11": [17],
    "add13": [21],

    # --- Extended ---
    "9":     [0, 4, 7, 10, 14],
    "m9":    [0, 3, 7, 10, 14],
    "11":    [0, 4, 7, 10, 14, 17],
    "13":    [0, 4, 7, 10, 14, 17, 21],

    # --- Altered Tensions ---
    "b5":   [6],
    "#5":   [8],
    "b9":   [13],
    "#9":   [15],
    "#11":  [18],
    "b13":  [20],

    # --- Dominant Altered Shortcut ---
    "alt":  [10, 13, 15, 18, 20],

    # --- Omit / No ---
    "no3":   [0, 7],
    "omit3": [0, 7],
    "no5":   [0, 4],
    "omit5": [0, 4],
    "no7":   [0, 4, 7],

    # --- Quartal / Modern ---
    "quartal":  [0, 5, 10],
    "cluster2": [0, 1, 2],
    "cluster3": [0, 1, 3],
}

# Sort tokens by length (descending) is crucial for the parser
QUALITY_TOKENS = sorted(CHORD_RULES.keys(), key=len, reverse=True)
# ==========================================
# 1.5 Data Structures (New Architecture)
# ==========================================
class NoteEvent:
    """
    Represents a single musical event (Note).
    Separates 'Composition' from 'Rendering'.
    """
    def __init__(self, start_time, duration, pitch, velocity=1.0, lyric="", params=None):
        self.start_time = start_time  # In seconds
        self.duration = duration      # In seconds
        self.pitch = pitch            # Frequency (Hz)
        self.velocity = velocity      # 0.0 - 1.0
        self.lyric = lyric            # For Vocal Tracks
        self.params = params or {}    # Extra data (vib depth, timbre, etc.)

    def to_dict(self):
        return {
            "start_time": self.start_time,
            "duration": self.duration,
            "pitch": self.pitch,
            "velocity": self.velocity,
            "lyric": self.lyric,
            "params": self.params
        }

    @staticmethod
    def from_dict(d):
        return NoteEvent(
            d["start_time"], d["duration"], d["pitch"], 
            d.get("velocity", 1.0), d.get("lyric", ""), d.get("params", {})
        )

# ==========================================
# 2. KanaLogic (Language Processing)
# ==========================================
class KanaLogic:
    """
    Handles conversion from Hiragana/Katakana to Phonemes (Consonant + Vowel).
    """
    def __init__(self):
        self.mapping = [
            ('きゃ', 'kya'), ('きゅ', 'kyu'), ('きょ', 'kyo'), ('しゃ', 'sha'), ('しゅ', 'shu'), ('しょ', 'sho'),
            ('ちゃ', 'cha'), ('ちゅ', 'chu'), ('ちょ', 'cho'), ('ふぁ', 'fa'), ('ふぃ', 'fi'), ('ふぇ', 'fe'), ('ふぉ', 'fo'),
            ('てぃ', 'ti'), ('でぃ', 'di'), ('うぃ', 'wi'),
            ('にゃ', 'nya'), ('にゅ', 'nyu'), ('にょ', 'nyo'),
            ('ひゃ', 'hya'), ('ひゅ', 'hyu'), ('ひょ', 'hyo'),
            ('みゃ', 'mya'), ('みゅ', 'myu'), ('みょ', 'myo'),
            ('りゃ', 'rya'), ('りゅ', 'ryu'), ('りょ', 'ryo'),
            ('ぎゃ', 'gya'), ('ぎゅ', 'gyu'), ('ぎょ', 'gyo'),
            ('じゃ', 'ja'), ('じゅ', 'ju'), ('じょ', 'jo'),
            ('びゃ', 'bya'), ('びゅ', 'byu'), ('びょ', 'byo'),
            ('ぴゃ', 'pya'), ('ぴゅ', 'pyu'), ('ぴょ', 'pyo'),
            ('あ', 'a'), ('い', 'i'), ('う', 'u'), ('え', 'e'), ('お', 'o'), ('ん', 'n'),
            ('か', 'ka'), ('き', 'ki'), ('く', 'ku'), ('け', 'ke'), ('こ', 'ko'),
            ('さ', 'sa'), ('し', 'shi'), ('す', 'su'), ('せ', 'se'), ('そ', 'so'),
            ('た', 'ta'), ('ち', 'chi'), ('つ', 'tsu'), ('て', 'te'), ('と', 'to'),
            ('な', 'na'), ('に', 'ni'), ('ぬ', 'nu'), ('ね', 'ne'), ('の', 'no'),
            ('は', 'ha'), ('ひ', 'hi'), ('ふ', 'fu'), ('へ', 'he'), ('ほ', 'ho'),
            ('ま', 'ma'), ('み', 'mi'), ('む', 'mu'), ('め', 'me'), ('も', 'mo'),
            ('や', 'ya'), ('ゆ', 'yu'), ('よ', 'yo'),
            ('ら', 'ra'), ('り', 'ri'), ('る', 'ru'), ('れ', 're'), ('ろ', 'ro'),
            ('わ', 'wa'), ('を', 'o'),
            ('が', 'ga'), ('ぎ', 'gi'), ('ぐ', 'gu'), ('げ', 'ge'), ('ご', 'go'),
            ('ざ', 'za'), ('じ', 'ji'), ('ず', 'zu'), ('ぜ', 'ze'), ('ぞ', 'zo'),
            ('だ', 'da'), ('ぢ', 'ji'), ('づ', 'zu'), ('で', 'de'), ('ど', 'do'),
            ('ば', 'ba'), ('び', 'bi'), ('ぶ', 'bu'), ('べ', 'be'), ('ぼ', 'bo'),
            ('ぱ', 'pa'), ('ぴ', 'pi'), ('ぷ', 'pu'), ('ぺ', 'pe'), ('ぽ', 'po'),
            ('ー', '-'), ('っ', 'clt'),
            ('ぁ', 'a'), ('ぃ', 'i'), ('ぅ', 'u'), ('ぇ', 'e'), ('ぉ', 'o'), ('ゃ', 'ya'), ('ゅ', 'yu'), ('ょ', 'yo')
        ]
        self.accent_dict = {'ki': 1.05, 'to': 0.95, 'a': 1.0, 'i': 1.0, 'u': 1.0, 'e': 1.0, 'o': 1.0, 'n': 0.98}

    def to_romaji(self, text):
        """Converts Japanese text to Romaji."""
        hira = ""
        for char in text:
            code = ord(char)
            # Katakana to Hiragana conversion
            if 0x30A1 <= code <= 0x30F6:
                hira += chr(code - 0x60)
            else:
                hira += char
        res = hira
        for k, v in self.mapping:
            res = res.replace(k, v)
        # Remove non-alphabetic chars except hyphen
        res = re.sub(r'[^a-zA-Z\-]', '', res)
        return res if res else 'a'

    def parse_phonemes(self, kana):
        """Splits Kana into (Consonant, Vowel). e.g., 'ka' -> ('k', 'a')"""
        romaji = self.to_romaji(kana)
        if romaji in ('clt', '-'): return None, romaji
        if len(romaji) == 1 and romaji in 'aiueon': return None, romaji
        if len(romaji) >= 2: return romaji[:-1], romaji[-1]
        return None, 'a'
    # KanaLogic クラス内に追加（既存のクラス定義内）
    def get_cons_type(self, kana):
        """Kanaから子音の種類を判定する"""
        romaji = self.to_romaji(kana)
        if romaji == 'clt': return 'sokuon'
        if romaji in ('-', 'a', 'i', 'u', 'e', 'o', 'n'): return 'vowel_or_n'
        
        cons = ''
        if len(romaji) >= 2 and romaji[:2] in CONSONANT_SPECTRAL: cons = romaji[:2]
        elif len(romaji) >= 1 and romaji[0] in CONSONANT_SPECTRAL: cons = romaji[0]
        
        return cons


# ==========================================
# 3. Nekotopy Logic (The Chaos Engine)
# ==========================================
class NekotopyLogic:
    """
    Math-based Chaos Engine for Humanizing Vocal Output.
    """
    def __init__(self):
        self.history = []

    def operator_evolve(self, x0, lam, N):
        x = x0
        traj = []
        for k in range(N):
            x = x + lam * np.tanh(x) / (k + 1)
            traj.append(x)
        return traj

    # 曲率密度収束判定版
    def curvature_witness(self, vals, lam, threshold=0.204):
        if len(vals) < 3:
            return True
        vals = np.array(vals)
        kappa = lam * (1.0 / np.cosh(vals))**2
        kappa_eff = np.mean(kappa)
        return kappa_eff <= threshold

    def observe(self, lam_input=1.0):
        N = int(np.ceil(lam_input * np.log(1 / P_NYAN) * 5)) + 5
        x0 = np.random.randn()
        traj = self.operator_evolve(x0, lam_input, N)

        # ここを変更
        is_observed = self.curvature_witness(traj[-5:], lam_input)

        final_val = traj[-1]
        d_eff = abs(final_val * lam_input) * 2.0
        R_I = -0.5 * np.tanh(final_val) if final_val < 0 else 0.2
        state_sign = 1 if final_val > 0 else -1

        return d_eff, R_I, state_sign, is_observed
()
# ==========================================
# 4. DSP & Music Theory Logic
# ==========================================
class VocalDSP:
    """
    Advanced DSP methods for the Vocal Engine.
    Modified to include Nasal Resonance EQ and LF-model source.
    """
    @staticmethod
    def design_formant_filter(freq, bw, fs=48000):
        if not HAS_SCIPY: return [1.0], [1.0]
        freq = min(freq, fs * 0.45)
        bw = max(10.0, bw)
        q = freq / max(1.0, bw)
        b, a = signal.iirpeak(freq, q, fs=fs)
        return b, a

    @staticmethod
    def design_notch_filter(freq, q=5.0, fs=48000):
        """Creates a notch filter to remove specific frequencies (anti-resonance)."""
        if not HAS_SCIPY: return [1.0], [1.0]
        b, a = signal.iirnotch(freq, q, fs=fs)
        return b, a

    @staticmethod
    def design_peak_filter(freq, gain_db, q=1.0, fs=48000):
        """Creates a peaking EQ filter for boosting/cutting."""
        if not HAS_SCIPY: return [1.0], [1.0]
        # Using iirpeak as a simplified boost implementation
        # Real peaking EQ is more complex, but this suffices for formant/body boost
        # Gain is approximated by mixing dry/wet or chaining
        b, a = signal.iirpeak(freq, q, fs=fs)
        return b, a

    @staticmethod
    def apply_nasal_eq(wave_data, fs=48000):
        """
        Applies 'Nasal' characteristic EQ:
        1. Boost Lows (~200Hz) for body.
        2. Notch Mids (~250-400Hz) to remove 'mud' or anti-resonance.
        """
        if len(wave_data) == 0 or not HAS_SCIPY: return wave_data
        
        # 1. Low Body Boost (approx 200Hz)
        b1, a1 = VocalDSP.design_formant_filter(200, 100, fs)
        # Apply gently (mix 50%)
        boosted = signal.lfilter(b1, a1, wave_data)
        wave_data = wave_data * 0.6 + boosted * 1.5
        
        # 2. Nasal Notch (Anti-resonance around 300-400Hz)
        b2, a2 = VocalDSP.design_notch_filter(350, q=2.0, fs=fs)
        wave_data = signal.lfilter(b2, a2, wave_data)
        
        return wave_data.astype(np.float32)

    @staticmethod
    def generate_glottal_source_lf(freq, dur, open_quotient=0.65, alpha=2.0, jitter_amount=0.005, oq_slope=0.1, fs=48000):
        samples = int(fs * dur)
        if samples <= 0: return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Frequency Array
        if np.isscalar(freq):
            freqs = np.full(samples, freq, dtype=np.float32)
        else:
            freqs = freq # Assumed array

        # --- Dynamic OQ Calculation (Modified) ---
        # oq_slope: Pop=0.1, Utada=0.04 (息を暴れさせない)
        dynamic_oq = open_quotient + oq_slope * (freqs / 400.0)
        dynamic_oq = np.clip(dynamic_oq, 0.4, 0.95)

        # Jitter & Phase (Jitter is heavily reduced for Utada mode)
        jitter = np.random.normal(0, jitter_amount, samples)
        instant_freq = freqs * (1.0 + jitter)
        phase = np.cumsum(instant_freq / fs)
        phase = phase - np.floor(phase)

        # Source & Gate Generation
        source = np.zeros(samples, dtype=np.float32)
        noise_gate = np.zeros(samples, dtype=np.float32)
        
        mask_open = phase < dynamic_oq
        if np.any(mask_open):
            t_open = phase[mask_open] / dynamic_oq[mask_open]
            pulse = np.sin(np.pi * t_open) * np.exp(alpha * t_open)
            source[mask_open] = pulse
            noise_gate[mask_open] = 1.0
            noise_gate[~mask_open] = 0.1

        if len(source) > 0:
            peak = np.max(np.abs(source))
            if peak > 1e-6: source /= peak

        if samples > 1:
            source = 0.3 * source + 0.7 * np.roll(source, 1)
            
        return source.astype(np.float32), noise_gate.astype(np.float32)

    @staticmethod
    def poly_blep(t, dt):
        """
        PolyBLEP Smoothing for Band-Limited Step.
        [Fixed] Handles array broadcasting correctly for `dt`.
        """
        t = t - np.floor(t)
        blep = np.zeros_like(t)
        
        # 0付近の不連続緩和
        mask_start = t < dt
        if np.any(mask_start):
            # dtが配列の場合、maskを使って対応する要素だけを取り出す必要がある
            dt_s = dt[mask_start] if np.ndim(dt) > 0 else dt
            t_s = t[mask_start] / dt_s
            blep[mask_start] = 2 * t_s - t_s**2 - 1.0
            
        # 1付近の不連続緩和
        mask_end = t > (1.0 - dt)
        if np.any(mask_end):
            # 同様にdtをマスク処理
            dt_e = dt[mask_end] if np.ndim(dt) > 0 else dt
            t_e = (t[mask_end] - 1.0) / dt_e
            blep[mask_end] = 2 * t_e + t_e**2 + 1.0
            
        return blep

    @staticmethod
    def generate_bandlimited_pulse(phase, freq, fs, width=0.5):
        """
        PolyBLEPを使った帯域制限パルス波を生成します。
        """
        dt = freq / fs
        # Naive Pulse
        naive = np.where(phase < width, 1.0, -1.0)
        
        # PolyBLEP Correction (Rising Edge at 0)
        blep1 = VocalDSP.poly_blep(phase, dt)
        
        # PolyBLEP Correction (Falling Edge at width)
        phase_shifted = (phase - width) 
        phase_shifted = phase_shifted - np.floor(phase_shifted)
        blep2 = VocalDSP.poly_blep(phase_shifted, dt)
        
        return naive + blep1 - blep2

    # ★追加: 宇多田ヒカル専用マスタリングEQ (Distance期)
    @staticmethod
    def apply_utada_eq(wave_data, fs=48000):
        if len(wave_data) == 0 or not HAS_SCIPY: return wave_data
        
        # 1. Low Cut (100Hz Shelf) - 不要な超低域カット
        sos_lc = signal.butter(1, 100/(fs/2), btype='high', output='sos')
        out = signal.sosfilt(sos_lc, wave_data)

        # 2. Body Boost (200Hz, +1.2x approx +2dB) - 太さ
        b_body, a_body = signal.iirpeak(200, 1.5, fs=fs) # Q=1.5 Wide
        body_part = signal.lfilter(b_body, a_body, out)
        out = out + body_part * 0.4 

        # 3. Mud Cut (350Hz, Notch Q=3.0) - 鼻詰まり解消
        b_notch, a_notch = signal.iirnotch(350, 3.0, fs=fs)
        out = signal.lfilter(b_notch, a_notch, out)
        
        # 4. Boxiness Cut (500Hz, -1dB) - 篭り解消
        b_box, a_box = signal.iirpeak(500, 2.0, fs=fs)
        box_part = signal.lfilter(b_box, a_box, out)
        out = out - box_part * 0.3

        # 5. Core Presence (1.8kHz, +1dB) - 歌詞の輪郭
        b_pres, a_pres = signal.iirpeak(1800, 2.0, fs=fs)
        pres_part = signal.lfilter(b_pres, a_pres, out)
        out = out + pres_part * 0.3

        # 6. High Shelf Cut (8kHz, -2dB) - 刺さりを抑える
        sos_high = signal.butter(1, 8000/(fs/2), btype='low', output='sos')
        high_part = signal.sosfilt(sos_high, out)
        # Mix slightly towards low pass result
        out = out * 0.7 + high_part * 0.3
        
        return out.astype(np.float32)

    @staticmethod
    def pink_noise(n, intensity=1.0):
        if n <= 0: return np.array([], dtype=np.float32)
        white = np.random.randn(n)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        if HAS_SCIPY:
             pink = signal.lfilter(b, a, white)
        else:
             pink = np.cumsum(white) * 0.02
        std = np.std(pink)
        if std == 0: return pink.astype(np.float32)
        return ((pink / std) * intensity).astype(np.float32)

    @staticmethod
    def bandpass(wave_data, low, high, fs=48000):
        if len(wave_data) == 0 or not HAS_SCIPY: return wave_data
        nyq = 0.5 * fs
        low = np.clip(low, 1, nyq - 100)
        high = np.clip(high, low + 100, nyq - 1)
        b, a = signal.butter(2, [low / nyq, high / nyq], btype='band')
        return signal.lfilter(b, a, wave_data).astype(np.float32)

    @staticmethod
    def get_exp_curve(length, start=0.0, end=1.0, curve=2.0):
        if length <= 0: return np.array([], dtype=np.float32)
        t = np.linspace(0, 1, length, dtype=np.float32)
        if curve == 0: return t * (end - start) + start
        exp_env = (np.exp(curve * t) - 1) / (np.exp(curve) - 1)
        return start + exp_env * (end - start)
        
    @staticmethod
    def generate_cymbal_wave(dur, type="crash", fs=48000):
        samples = int(fs * dur)
        t = np.linspace(0, dur, samples, False, dtype=np.float32)
        if type == "crash": bases = [300, 420, 600, 800]; decay_rate = 3.0
        else: bases = [500, 780, 1100, 1400]; decay_rate = 6.0
        metal = np.zeros(samples, dtype=np.float32)
        for bf in bases: metal += np.sin(2 * np.pi * bf * t + np.sin(2 * np.pi * (bf * 1.45) * t) * 2.0)
        noise = np.random.uniform(-1, 1, samples).astype(np.float32)
        metal = metal * 0.4 + noise * 0.6
        env = np.exp(-t * decay_rate)
        return (metal * env).astype(np.float32)

    @staticmethod
    def generate_string_wave(freq, dur, fs=48000):
        samples = int(fs * dur)
        t = np.linspace(0, dur, samples, False, dtype=np.float32)
        detune = 1.002
        saw1 = 2 * (t * freq - np.floor(t * freq + 0.5))
        saw2 = 2 * (t * freq * detune - np.floor(t * freq * detune + 0.5))
        mix = (saw1 + saw2) * 0.5
        if samples > 1: mix = 0.5 * mix + 0.5 * np.roll(mix, 1)
        return mix.astype(np.float32)
        
    @staticmethod
    def apply_3band_eq(wave_data, low_g, mid_g, high_g, fs=48000):
        """
        3-band Equalizer (LowShelf, Peaking, HighShelf)
        Gains are in dB (e.g., -6.0 to +6.0). 0.0 is flat.
        """
        if not HAS_SCIPY or len(wave_data) == 0: return wave_data * (1.0 + (mid_g * 0.1)) # Fallback gain

        # Convert dB to linear gain
        def db_to_linear(db): return 10.0 ** (db / 20.0)
        
        lg = db_to_linear(low_g)
        mg = db_to_linear(mid_g)
        hg = db_to_linear(high_g)

        # 1. Low Shelf (150Hz)
        sos_low = signal.butter(1, 150/(fs/2), btype='low', output='sos')
        low_comp = signal.sosfilt(sos_low, wave_data, axis=0)
        high_pass_part = wave_data - low_comp
        out = (low_comp * lg) + high_pass_part
        
        # 2. Mid Peaking (1000Hz, Wide Q)
        # Simplified implementation using Bandpass split
        b_mid, a_mid = signal.butter(1, [600/(fs/2), 2500/(fs/2)], btype='band')
        mid_comp = signal.lfilter(b_mid, a_mid, out, axis=0)
        out = out + (mid_comp * (mg - 1.0)) # Boost/Cut existing mid content

        # 3. High Shelf (5000Hz)
        sos_high = signal.butter(1, 5000/(fs/2), btype='high', output='sos')
        high_comp = signal.sosfilt(sos_high, out, axis=0)
        low_pass_part = out - high_comp
        out = low_pass_part + (high_comp * hg)

        return out.astype(np.float32)
        
        # 新規追加: ビブラート適用
    @staticmethod
    def apply_vibrato(wave_data, rate=5.0, depth=0.005, fs=48000):
        if len(wave_data) == 0: return wave_data
        t = np.arange(len(wave_data)) / fs
        mod = depth * np.sin(2 * np.pi * rate * t)
        indices = np.arange(len(wave_data)) * (1 + mod)
        indices = np.clip(indices, 0, len(wave_data) - 1)
        return np.interp(indices, np.arange(len(wave_data)), wave_data).astype(np.float32)

    @staticmethod
    def get_env_array(dur, atk, dec, sus, rel, fs=48000):
        """
        波形に掛け合わせるためではなく、VCF制御用のエンベロープカーブ(0.0〜1.0)を生成する
        (既存の apply_env のロジックを流用)
        """
        samples = int(fs * dur)
        if samples <= 0: return np.array([], dtype=np.float32)
        
        a_s, d_s, r_s = int(atk * fs), int(dec * fs), int(rel * fs)
        total_env = a_s + d_s + r_s
        if total_env > samples:
            factor = samples / total_env if total_env > 0 else 1
            a_s, d_s, r_s = int(a_s * factor), int(d_s * factor), int(r_s * factor)
        
        s_len = max(0, samples - a_s - d_s - r_s)
        
        env_parts = []
        if a_s > 0: env_parts.append(VocalDSP.get_exp_curve(a_s, 0.0, 1.0, 4.0))
        if d_s > 0: env_parts.append(VocalDSP.get_exp_curve(d_s, 1.0, sus, 5.0))
        if s_len > 0: env_parts.append(np.full(s_len, sus, dtype=np.float32))
        if r_s > 0: env_parts.append(VocalDSP.get_exp_curve(r_s, sus, 0.0, 5.0))
        
        if not env_parts: return np.zeros(samples, dtype=np.float32)
        env = np.concatenate(env_parts)
        
        if len(env) < samples: 
            env = np.pad(env, (0, samples - len(env)), 'constant')
        else: 
            env = env[:samples]
            
        return env.astype(np.float32)

    @staticmethod
    def apply_vcf(wave_data, env_curve, base_freq=200.0, env_amount=4000.0, q=2.0, fs=48000):
        """
        レゾナンス付きの動的ローパスフィルター (Biquad LPF)
        env_curve: 0.0 ~ 1.0 に変化する配列
        base_freq: フィルターが一番閉じた時の周波数
        env_amount: エンベロープが最大(1.0)の時に加算される周波数
        """
        if not HAS_SCIPY or len(wave_data) == 0: return wave_data

        block_size = 128
        num_blocks = (len(wave_data) + block_size - 1) // block_size
        out = np.zeros_like(wave_data)
        
        # フィルターの初期状態 (2nd order)
        zi = np.zeros(2)
        nyq = fs * 0.45 # ナイキスト周波数（安全マージン）

        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(wave_data))
            
            # ブロック先頭のエンベロープ値を取得してカットオフを計算
            current_env = env_curve[start]
            cutoff = base_freq + (env_amount * current_env)
            cutoff = np.clip(cutoff, 20.0, nyq)
            
            # オーディオEQクックブックに基づくBiquad LPFの係数計算
            w0 = 2 * np.pi * cutoff / fs
            alpha = np.sin(w0) / (2 * q)
            
            b0 = (1 - np.cos(w0)) / 2
            b1 = 1 - np.cos(w0)
            b2 = b0
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
            
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1/a0, a2/a0])
            
            block_in = wave_data[start:end]
            filtered_block, zi = signal.lfilter(b, a, block_in, zi=zi)
            out[start:end] = filtered_block
            
        return out.astype(np.float32)

# ==========================================
# Voice Clone Analyzer (Fixed for Log-Domain)
# ==========================================
class VoiceCloneAnalyzer:
    @staticmethod
    def analyze_spectral_envelope(filepath, num_frames=5):  # num_frames増やして安定（新: 5に）
        if not HAS_SCIPY or not os.path.exists(filepath):
            return None

        try:
            fs, data = scipy_wav.read(filepath)
            
            # 正規化・ステレオ・無音チェック: 閾値下げ (新: 0.02に下げて弱音対応)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            if len(data.shape) > 1:
                data = data[:, 0]
            if np.max(np.abs(data)) < 0.02:  # 閾値下げ
                return None 

            # Pre-emphasis: 係数弱め (新: 0.95にで過剰強調回避)
            data = np.append(data[0], data[1:] - 0.95 * data[:-1])
            
            # 複数フレーム分析
            frame_length = max(1, len(data) // num_frames)  # 短データ対応
            all_formants = []
            
            for i in range(num_frames):
                start = i * frame_length
                segment = data[start:start + frame_length]
                
                # Windowing: N動的、短データで小さく (新: min 1024に下げ)
                N = min(8192, max(1024, int(fs / 10)))
                if len(segment) < N:
                    padding = np.zeros(N - len(segment))
                    segment = np.concatenate((segment, padding))
                else:
                    segment = segment[:N]
                
                segment = segment * np.hanning(N)
                
                # FFT -> Log Magnitude: epsilon下げ (新: 1e-12で負値対応)
                spectrum = np.fft.rfft(segment)
                log_mag = np.log(np.abs(spectrum) + 1e-12)
                
                # Cepstrum
                cepstrum = np.fft.irfft(log_mag)
                
                # Liftering: cutoff広め (新: fs/1000に下げて詳細保持)
                lifter_cutoff = max(10, min(80, int(fs / 1000)))
                cepstrum[lifter_cutoff: -lifter_cutoff] = 0
                
                # Envelope
                envelope = np.real(np.fft.rfft(cepstrum))
                freqs_scale = np.fft.rfftfreq(N, d=1/fs)
                
                # ピーク検出: prominenceさらに動的&下げ (新: 0.05 * std)
                env_std = np.std(envelope)
                prom = max(0.05, 0.1 * env_std)  # 下げて拾いやすく
                dist = max(5, int(800 / (fs / N)))  # dist少し狭く
                peaks, props = signal.find_peaks(envelope, prominence=prom, distance=dist)
                
                if len(peaks) > 0:
                    strengths = props['prominences']
                    sorted_idx = np.argsort(strengths)[::-1]
                    peaks = peaks[sorted_idx][:12]  # 上位増やし
                
                extracted_formants = [freqs_scale[p] for p in peaks if 150 < freqs_scale[p] < 5500]
                extracted_formants.sort()
                
                # fallback強化: ピークなし時、スムージング (新: window広げ、prom=0.03)
                if len(extracted_formants) < 2:
                    window_size = max(30, min(150, int(N / 30)))  # 広げてスムーズ
                    smooth_env = np.convolve(log_mag, np.ones(window_size)/window_size, mode='same')
                    peaks2, props2 = signal.find_peaks(smooth_env, prominence=0.03, distance=dist * 2)  # prom下げ/dist広げ
                    if len(peaks2) > 0:
                        strengths2 = props2['prominences']
                        sorted_idx2 = np.argsort(strengths2)[::-1]
                        peaks2 = peaks2[sorted_idx2][:12]
                    extracted_formants = [freqs_scale[p] for p in peaks2 if 150 < freqs_scale[p] < 5500]
                    extracted_formants.sort()
                
                # 追加fallback: まだ少ない場合、envelopeの最大値から推定 (新)
                if len(extracted_formants) < 2:
                    # 粗いピーク: prom=0.01, no dist
                    peaks3, _ = signal.find_peaks(envelope, prominence=0.01)
                    extracted_formants = [freqs_scale[p] for p in peaks3[:4] if 150 < freqs_scale[p] < 5500]
                    extracted_formants.sort()
                
                if extracted_formants:  # 空回避
                    all_formants.append(extracted_formants)
            
            if not all_formants:
                return [500, 1100, 2500, 3500]  # 最終デフォルト
            
            # 平均計算: 少ない場合も使う (新: len(f) >=2 に下げ)
            valid_formants = [f for f in all_formants if len(f) >= 2]
            if not valid_formants:
                return [500, 1100, 2500, 3500]
            # パッドして平均 (新: 短いリストをデフォで埋め)
            padded = [f + [2500, 3500][:4-len(f)] if len(f) < 4 else f[:4] for f in valid_formants]
            avg_formants = np.mean(padded, axis=0)
            
            # フォルマント整理: 間隔最小200Hzに下げ (新)
            final_freqs = [avg_formants[0]]
            for i in range(1, len(avg_formants)):
                if avg_formants[i] > final_freqs[-1] + 200:
                    final_freqs.append(avg_formants[i])
                else:
                    final_freqs.append(final_freqs[-1] * 1.4)  # 比率下げ
            while len(final_freqs) < 4:
                final_freqs.append(final_freqs[-1] * 1.3)
            
            return final_freqs[:4]

        except Exception as e:
            print(f"Analyzer Crash: {e}")
            traceback.print_exc()
            return None


# --- Utility Functions (Updated) ---
def norm_note(n):
    return FLAT_MAP.get(n, n)

def transpose_note_name(note_name, semitones):
    n = norm_note(note_name)
    if n not in NOTE_INDEX: return note_name
    current_idx = NOTE_INDEX[n]
    new_idx = (current_idx + semitones) % 12
    return NOTE_ORDER[new_idx]

def note_freq(n, o):
    n = norm_note(n)
    if n not in NOTE_INDEX: return 440.0
    return 440.0 * 2 ** ((((o + 1) * 12 + NOTE_INDEX[n]) - 69) / 12)

#new
def parse_bar(chord_str, key_offset=0, beats_per_bar=4):
    """
    Handles intra-bar split like:
    C
    C_D
    C_D_E_F
    """
    if "_" not in chord_str:
        chord = parse_complex_chord(chord_str, key_offset)
        chord["beats"] = beats_per_bar
        return [chord]

    parts = chord_str.split("_")
    split_count = len(parts)
    beats_each = beats_per_bar / split_count

    result = []
    for p in parts:
        chord = parse_complex_chord(p.strip(), key_offset)
        if chord:
            chord["beats"] = beats_each
            result.append(chord)

    return result

def parse_complex_chord(chord_str, key_offset=0):
    """
    Parses complex chord strings (e.g., 'C#m7', 'G/B', 'Fadd9').
    Returns dictionary with root, type, intervals, and bass.
    """
    if not chord_str: return None
    chord_str = chord_str.replace('"', '').strip()

    # 1. Handle Slash Chords (Bass Split)
    bass = None
    if "/" in chord_str:
        chord_str, bass = chord_str.split("/", 1)
        bass = bass.strip()

    # 2. Extract Root Note
    match = re.match(r'^([A-Ga-g][#b]?)(.*)$', chord_str)
    if not match:
        return {"root": "C", "type": "maj", "intervals": [0, 4, 7], "bass": "C"}

    root_raw, qual_raw = match.groups()
    root = root_raw[0].upper() + root_raw[1:] if len(root_raw) > 1 else root_raw.upper()
    root = norm_note(root)

    if key_offset != 0:
        root = transpose_note_name(root, key_offset)

    # 3. Parse Chord Quality
    qual_raw = qual_raw.strip()
    intervals = []
    chord_type_parts = []
    remaining = qual_raw

    for token in QUALITY_TOKENS:
        if token and token in remaining:
            intervals += CHORD_RULES[token]
            chord_type_parts.append(token)
            remaining = remaining.replace(token, "")

    if not intervals: intervals = CHORD_RULES[""]
    
    # Sort unique intervals
    intervals = sorted(list(set(intervals)))
    chord_type = "".join(chord_type_parts) if chord_type_parts else "maj"

    # 4. Handle Bass Transposition
    final_bass = root
    if bass:
        bass_clean = bass[0].upper() + bass[1:] if len(bass) > 1 else bass.upper()
        bass_norm = norm_note(bass_clean)
        if key_offset != 0:
            final_bass = transpose_note_name(bass_norm, key_offset)
        else:
            final_bass = bass_norm

    return {"root": root, "type": chord_type, "intervals": intervals, "bass": final_bass}

# ==========================================
# Missing Global DSP Functions (Fixed by Grok)
# ==========================================

def stereo_pan(wave_data, pan):
    """Pans a mono wave to stereo."""
    p = max(-1.0, min(1.0, pan))
    theta = (p + 1.0) * (np.pi / 4.0)
    left = wave_data * np.cos(theta)
    right = wave_data * np.sin(theta)
    return np.column_stack((left, right)).astype(np.float32)

def apply_simple_delay(wave_data, delay_time=0.2, mix=0.4, fs=48000):
    """Applies a simple echo/delay."""
    d_s = int(fs * delay_time)
    if d_s >= len(wave_data): return wave_data
    out = wave_data.copy()
    # Handle both mono and stereo
    if len(wave_data.shape) > 1:
        delayed_part = wave_data[:-d_s, :]
        out[d_s:, :] = (out[d_s:, :] * (1-mix)) + (delayed_part * mix)
    else:
        delayed_part = wave_data[:-d_s]
        out[d_s:] = (out[d_s:] * (1-mix)) + (delayed_part * mix)
    return out
    
# ==========================================
# ▼▼▼ ここから追加 ▼▼▼
# ==========================================

def apply_tape_delay(wave_data, delay_time=0.375, feedback=0.5, mix=0.4, high_cut=0.5, fs=48000):
    """
    アナログテープ風ピンポンディレイ (New!)
    """
    if len(wave_data) == 0: return wave_data
    
    d_samples = int(fs * delay_time)
    if d_samples >= len(wave_data): return wave_data
    
    out = wave_data.copy()
    current_l = wave_data[:, 0].copy()
    current_r = wave_data[:, 1].copy()
    
    fb_gain = feedback
    
    # 簡易LPF (High Cut)
    lpf_tap = int(5 * high_cut) + 1 if high_cut > 0 else 0
    
    # 5回反射させる (ループ処理)
    for i in range(1, 6):
        shift = d_samples * i
        if shift >= len(wave_data): break
        
        # LPF適用 (反射するたびに籠もらせる)
        if lpf_tap > 1:
            current_l = np.convolve(current_l, np.ones(lpf_tap)/lpf_tap, mode='same')
            current_r = np.convolve(current_r, np.ones(lpf_tap)/lpf_tap, mode='same')
        
        # ピンポン処理 (LとRを入れ替える)
        delayed_l = np.zeros_like(current_l)
        delayed_r = np.zeros_like(current_r)
        
        # クロスフィードバック
        delayed_l[shift:] = current_r[:-shift] * fb_gain
        delayed_r[shift:] = current_l[:-shift] * fb_gain
        
        current_l = delayed_l
        current_r = delayed_r
        
        # ミックス
        out[:, 0] += current_l * mix
        out[:, 1] += current_r * mix
        
        fb_gain *= feedback # 減衰
        
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def apply_plate_reverb(wave_data, decay=0.5, mix=0.3, fs=48000):
    """
    プレートリバーブ (New!)
    Scipyの畳み込み積分を使ってリッチな響きを作る
    """
    if not HAS_SCIPY or len(wave_data) == 0: 
        # Scipyがない場合は古いDelayで代用
        return apply_simple_delay(wave_data, 0.1, mix)
    
    # インパルス応答 (IR) 作成
    ir_len = int(fs * (0.5 + decay * 1.5))
    t = np.linspace(0, 1, ir_len)
    
    # ノイズバースト + 減衰エンベロープ
    noise = np.random.randn(ir_len)
    envelope = np.exp(-t * (8.0 - decay * 4.0))
    ir = noise * envelope
    
    # IRの音作り (バンドパス)
    if HAS_SCIPY:
         sos = signal.butter(2, [500/(fs/2), 6000/(fs/2)], btype='band', output='sos')
         ir = signal.sosfilt(sos, ir)
    
    # 畳み込み (L/R別々に)
    wet_l = signal.fftconvolve(wave_data[:, 0], ir, mode='full')[:len(wave_data)]
    wet_r = signal.fftconvolve(wave_data[:, 1], ir, mode='full')[:len(wave_data)]
    
    # 音量調整 & ミックス
    wet_l *= 0.05 * mix
    wet_r *= 0.05 * mix
    
    out = wave_data.copy()
    out[:, 0] += wet_l
    out[:, 1] += wet_r
    
    return np.tanh(out)

# ==========================================
# ▲▲▲ ここまで追加 ▲▲▲
# ==========================================
    

def apply_haas_widener(stereo_wave, delay_ms=15.0, fs=48000):
    """Creates a stereo widening effect using the Haas effect."""
    samples = int(fs * (delay_ms / 1000.0))
    if samples >= len(stereo_wave): return stereo_wave
    left = stereo_wave[:, 0]
    right = np.roll(stereo_wave[:, 1], samples)
    right[:samples] = 0
    return np.column_stack((left, right))

def apply_stereo_detune(stereo_wave, detune_amount=0.002):
    """
    ステレオ波形の片方のピッチをわずかにずらして、音に厚みを出す (Numpy高速版)
    detune_amount: 0.002 = 0.2% くらいのズレが自然です
    """
    # 左チャンネル (L) はそのまま
    left = stereo_wave[:, 0]
    
    # 右チャンネル (R) をリサンプリングしてピッチを変える
    original_indices = np.arange(len(stereo_wave))
    # インデックスを「少し速く」進めることでピッチを上げる
    new_indices = original_indices * (1.0 + detune_amount)
    
    # 配列の長さを超えないようにクリップ（端の処理）
    new_indices = np.clip(new_indices, 0, len(stereo_wave) - 1)
    
    # 線形補間 (Linear Interpolation) で波形を再構築
    right = np.interp(new_indices, original_indices, stereo_wave[:, 1])
    
    # ステレオに戻して返す
    return np.column_stack((left, right)).astype(np.float32)

def apply_bitcrush(stereo_wave, rate_reduce=0.0, bit_depth=16):
    """Reduces bit depth for lo-fi effect."""
    if bit_depth < 16:
        steps = 2 ** bit_depth
        stereo_wave = np.round(stereo_wave * steps) / steps
    return stereo_wave

def apply_distortion(stereo_wave, drive=2.0, threshold=0.8):
    """Applies tanh distortion."""
    if drive <= 0: return stereo_wave
    return np.tanh(stereo_wave * drive) * threshold

def soft_limit(wave_data, threshold=0.95, drive=1.0):
    """Soft limiter to prevent clipping (This was the missing function)."""
    return np.tanh(wave_data * drive) * threshold

# ==========================================
# (追加) S/Eによるゲートマスク生成関数
# ==========================================
def create_gate_mask(gate_str, total_steps):
    if not gate_str:
        return [1] * total_steps
        
    flat_str = gate_str.replace(" ", "").replace("_", "")
    mask = []
    is_playing = False if 'S' in flat_str else True
    
    for char in flat_str:
        if char.upper() == 'S':
            is_playing = True
        elif char.upper() == 'E':
            is_playing = False
            
        mask.append(1 if is_playing else 0)
        
    while len(mask) < total_steps:
        mask.append(1 if is_playing else 0)
        
    return mask[:total_steps]

# ==========================================
# 5. Vocal Synthesis Engine (v19.8 Realism Enhanced)
# ==========================================
class VocalSynth:
    def __init__(self, mode='pop'):
        self.mode = mode
        self.base_transition_time = 0.08
        # [FIX] Parallel Filter用のゲイン設定
        self.makeup_gain = 4.0 if self.mode == 'utada' else 3.5

    def get_formant_targets(self, vowel):
        """母音ごとのフォルマント周波数等を取得（ユーザーデータ対応版）"""
        
        # ★変更: ユーザー優先、BWを現実的にclip & スケール
        if vowel in USER_VOICE_DATA:
            user_freqs = USER_VOICE_DATA[vowel]
            base_data = FORMANT_DATA.get(vowel, FORMANT_DATA.get('a'))
            
            freqs = np.array(user_freqs, dtype=np.float32)
            
            # BW調整: freqs * 0.08 \~ 0.12 で柔軟、clip to 50-250Hz (新: ブザー回避)
            bws = freqs * 0.1
            bws = np.clip(bws, 50.0, 250.0)  # 狭すぎ/広すぎ防止
            
            # モード依存スケール (新: utadaなど高め)
            if self.mode == 'utada':
                bws *= 1.15
            
            oq = base_data.get('oq', 0.6)
            alpha = base_data.get('alpha', 2.0)
            
            return freqs, bws, oq, alpha
        
        # 標準データ部 (変更少: mode調整のみ)
        data = FORMANT_DATA.get(vowel, FORMANT_DATA.get('a'))
        freqs = np.array(data['freq'], dtype=np.float32)
        bws = np.array(data['bw'], dtype=np.float32)
        oq = data.get('oq', 0.6)
        alpha = data.get('alpha', 2.0)

        # --- Mode Specific Adjustments --- (変更なし)
        if self.mode == 'pop':
            alpha = max(0.5, alpha - 0.8)
        elif self.mode == 'utada':
            if vowel == 'a':
                freqs[0] = 820
                freqs[1] = 1150
            elif vowel == 'i':
                freqs[1] = 2200
            bws *= 1.2
            base_oq = data.get('oq', 0.65)
            oq = np.clip(base_oq * 0.9 + 0.15, 0.72, 0.78)
            alpha = np.clip(alpha + 0.2, 2.4, 2.8)
        else:
            freqs *= 0.9
            alpha = max(0.5, alpha - 0.8)

        return freqs, bws, oq, alpha

    def _get_render_params(self, custom_params=None):
        """Merged default params with track-specific custom params"""
        params = {
            'oq_slope': 0.1,
            'jitter': 0.01,  # ★変更: 強めで自然さ向上 (ブザー回避)
            'cons_attack_scale': 1.0,
            'vib_rate_override': None,
            'vib_depth_override': None,
            'glide_duration': 0.06,
            'dip_amount': 0.985,
            'drop_depth': 0.03,
            'vib_start_delay': 0.15,
            'gain_mult': 1.0,
            'len_mult_plosive': 1.0,
            'nasal_mix': 0.0,
            'comp_drive': 0.8,
            'harmonic_mix': 0.0, 
            'phase_mode': 'continuous',
            'detune_amount': 0.0
        }
        
        if self.mode == 'utada':
            params.update({
                'oq_slope': 0.04, 'jitter': 0.005,  # utadaは弱め維持
                'comp_drive': 1.1,
                'vib_rate_override': 5.2, 'nasal_mix': 0.2
            })
            
        if custom_params:
            params.update(custom_params)
            
        return params

    def _generate_pitch_curve(self, total_samples, freq, prev_pitch, duration, params, cons, is_phrase_end, vib_depth, vib_rate, fs):
        """ピッチカーブ生成 (Improved with 1/f Jitter)"""
        freq_curve = np.full(total_samples, freq, dtype=np.float32)
        t_vec = np.linspace(0, duration, total_samples, dtype=np.float32)

        # A. Note Transition (Glide) (変更なし)

        if abs(freq - prev_pitch) > 1.0:
            glide_samples = int(fs * params['glide_duration'])
            if glide_samples < total_samples:
                t_glide = np.linspace(0, 1, glide_samples)
                curve = 1.0 - np.exp(-t_glide * 8.0)
                freq_curve[:glide_samples] = prev_pitch + (freq - prev_pitch) * curve

        # B. Micro-prosody (変更なし)

        if cons and duration > 0.1:
            dip_len = int(fs * 0.05)
            if dip_len < total_samples:
                start_p = freq * params['dip_amount']
                t_dip = np.linspace(0, 1, dip_len)
                curve = 1.0 - np.exp(-t_dip * 12.0)
                freq_curve[:dip_len] = start_p + (freq - start_p) * curve

        # C. Vibrato (変更なし)

        eff_vib_depth = params['vib_depth_override'] if params['vib_depth_override'] is not None else vib_depth
        eff_vib_rate = params['vib_rate_override'] if params['vib_rate_override'] is not None else vib_rate

        if eff_vib_depth > 0.001 and duration > 0.25:
            if duration > params['vib_start_delay']:
                vib_mask = np.clip((t_vec - params['vib_start_delay']) * 4.0, 0.0, 1.0)
                vib_osc = np.sin(2 * np.pi * eff_vib_rate * t_vec)
                freq_curve += freq * eff_vib_depth * vib_osc * vib_mask
        
        # D. 1/f Jitter (Pink Noise) - ★変更: intensityをfreqスケール
        if params['jitter'] > 0:
            jitter_sig = VocalDSP.pink_noise(total_samples, intensity=params['jitter'] * (freq / 440.0))  # 高音で強め
            freq_curve *= (1.0 + jitter_sig)
        
        return freq_curve

    def _apply_formant_filter(self, source, start_freqs, start_bws, tgt_freqs, tgt_bws, total_samples, duration, romaji, fs):
        """フォルマントフィルタ (Exponential Smoothing & F4 Support)"""
        block_size = 128
        num_blocks = (total_samples + block_size - 1) // block_size
        output_buffer = np.zeros(total_samples, dtype=np.float32)

        # 指数遷移のための時定数 - ★変更: duration依存で長め
        trans_time = min(0.15, duration * 0.6)  # 緩やか遷移でブザー減
        
        zi = [None] * 4
        
        current_freqs = start_freqs.copy()
        current_bws = start_bws.copy()

        # バンドゲイン追加 (新: 低F強く [1.0, 0.8, 0.6, 0.4])
        band_gains = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)

        for b_idx in range(num_blocks):
            start_pos = b_idx * block_size
            end_pos = min(start_pos + block_size, total_samples)
            
            t_now = (b_idx * block_size) / fs
            
            # --- Exponential Interpolation --- ★変更: kを4.0に緩和
            if romaji != '-':
                if t_now < trans_time:
                    progress = t_now / trans_time
                    k = 4.0  # 緩やかカーブで自然遷移
                    alpha = (1.0 - np.exp(-k * progress)) / (1.0 - np.exp(-k))
                else:
                    alpha = 1.0
                
                current_freqs = start_freqs + (tgt_freqs - start_freqs) * alpha
                current_bws = start_bws + (tgt_bws - start_bws) * alpha
            else:
                current_freqs = tgt_freqs
                current_bws = tgt_bws

            block_source = source[start_pos:end_pos]
            block_accum = np.zeros_like(block_source)

            num_bands = min(len(current_freqs), 4)
            
            for i in range(num_bands):
                b, a = VocalDSP.design_formant_filter(current_freqs[i], current_bws[i], fs)
                if zi[i] is None: zi[i] = signal.lfilter_zi(b, a) * 0.0
                filtered, zi[i] = signal.lfilter(b, a, block_source, zi=zi[i])
                
                # ★変更: ゲイン適用
                filtered *= band_gains[i]
                
                block_accum += filtered

            # ★変更: ブロック正規化 (クリップ/ブザー回避)
            block_accum /= max(1.0, np.max(np.abs(block_accum)) + 1e-6)  # ピーク1.0に
            
            output_buffer[start_pos:end_pos] = block_accum
        
        # 全体正規化 (新)
        output_buffer /= max(1.0, np.max(np.abs(output_buffer)) + 1e-6)
        
        return output_buffer, current_freqs, current_bws

    def render(self, kana, freq, length, velocity, prev_vowel_state, ag_on,
               initial_phase=0.0, vocal_params=None, pitch_curve_data=None, start_time=0.0,
               breath_vol=0.05, breath_chance=0.5, cons_attack=1.0, 
               scoop_amount=0.0, whisper_amount=0.0, vib_depth=0.05, 
               vib_rate=5.0, formant_shift=1.0, is_devoiced=False,
               is_sokuon_next=False, after_sokuon=False, next_cons_type=None,
               is_phrase_end=False, fs=48000):

        # 1. Phoneme Parsing
        romaji = kana_engine.to_romaji(kana)
        if romaji == 'clt': 
            return np.random.uniform(-0.01, 0.01, int(fs * length)).astype(np.float32), prev_vowel_state

        target_vowel = 'a'
        cons = ''
        if romaji in ('-', '', '\n', ' '):
            target_vowel = prev_vowel_state.get('vowel', 'a')
        else:
            for length_check in [2, 1]:
                if len(romaji) >= length_check and romaji[:length_check] in CONSONANT_SPECTRAL:
                    cons, target_vowel = romaji[:length_check], romaji[length_check:]
                    break
            else:
                target_vowel = romaji
            if target_vowel == '': target_vowel = prev_vowel_state.get('vowel', 'a')

        # --- Load Render Parameters ---
        params = self._get_render_params(vocal_params)
        cons_attack *= params['cons_attack_scale']

        # Devoicing Setup
        eff_velocity = velocity * (0.3 if is_devoiced and target_vowel in ['i', 'u'] else 1.0)
        eff_whisper = 0.8 if is_devoiced else whisper_amount
        eff_breath = 0.2 if is_devoiced else breath_vol

        # 2. Source Setup
        duration = max(0.05, length)
        total_samples = int(fs * duration)
        
        prev_pitch = prev_vowel_state.get('pitch', freq)

        # ---------------------------------------------------------
        # ★【重要変更】手描きのピッチカーブ（Pitch編集）を反映させる処理
        # ---------------------------------------------------------
        if pitch_curve_data and isinstance(pitch_curve_data, dict) and len(pitch_curve_data) > 0:
            # 1. 時間軸をサンプリングレートに合わせて作成
            time_axis = np.linspace(start_time, start_time + duration, total_samples)
            
            # 2. pitch_curve_data (dict) のキー(時間)と値(MIDIノート)を配列に変換
            # ※ 前提として、Pitch編集UIで保存されたデータの時間単位が秒(またはステップ)であること
            # ここではUIの x座標(ステップ) * 0.125秒 / 4分割 = 実際の秒数 に変換していると仮定
            # （※データ構造に合わせて微調整が必要な場合があります）
            step_to_sec = 0.125 / 4.0 # pitch_resolution=4 の場合
            curve_times = np.array(sorted(pitch_curve_data.keys())) * step_to_sec
            curve_midis = np.array([pitch_curve_data[k] for k in sorted(pitch_curve_data.keys())])
            
            # 3. numpyの線形補間(interp)を使って、全サンプルの滑らかなMIDIカーブを生成
            midi_curve = np.interp(time_axis, curve_times, curve_midis)
            
            # 4. MIDIノート番号を周波数(Hz)に変換
            freq_curve = 440.0 * (2.0 ** ((midi_curve - 69.0) / 12.0))
            
            # 5. ピン(📍)の効果（しゃくり等のパラメータ）を追加
            # vocal_params に "pin_state" が渡されていれば適用
            pin = vocal_params.get("pin_state", 0) if vocal_params else 0
            if pin == 1: # 赤ピン: しゃくり (音の出だしを少し低くして素早く戻す)
                scoop_env = np.linspace(-1.0, 0.0, min(total_samples, int(fs * 0.1))) # 0.1秒かけて戻る
                scoop_full = np.zeros(total_samples)
                scoop_full[:len(scoop_env)] = scoop_env
                freq_curve *= (2.0 ** (scoop_full / 12.0)) # 最大1半音下からしゃくる
            elif pin == 2: # 青ピン: フォール (音の終わりをダウンスケール)
                fall_env = np.linspace(0.0, -2.0, min(total_samples, int(fs * 0.15)))
                fall_full = np.zeros(total_samples)
                fall_full[-len(fall_env):] = fall_env
                freq_curve *= (2.0 ** (fall_full / 12.0)) # 最大2半音下がる
            
        else:
            # PITCHデータが無い場合は、元々の自動生成ピッチ（まっすぐな音＋自動ビブラート等）を使う
            freq_curve = self._generate_pitch_curve(total_samples, freq, prev_pitch, duration, params, cons, is_phrase_end, vib_depth, vib_rate, fs)
        # ---------------------------------------------------------


        if params['detune_amount'] > 0:
            detune_lfo = np.sin(np.linspace(0, duration * 2 * np.pi * 3.0, total_samples)) 
            freq_curve *= (1.0 + params['detune_amount'] * 0.005 * detune_lfo)

        # [NEW] Jitter (Realism: Frequency Micro-fluctuation)
        if params['jitter'] > 0:
            jitter_sig = np.random.normal(0, params['jitter'], total_samples)
            freq_curve *= (1.0 + jitter_sig)

        # Global Phase Accumulation
        phase_increment = freq_curve / fs
        phase_array = np.cumsum(phase_increment) + initial_phase
        phase_array = phase_array % 1.0 
        
        final_phase_out = phase_array[-1]

        # --- Hybrid Source Generation ---
        tgt_freqs, tgt_bws, tgt_oq, tgt_alpha = self.get_formant_targets(target_vowel)
        
        if next_cons_type in VELAR_CONS: tgt_freqs[1] *= 1.1
        elif next_cons_type in LABIAL_CONS: tgt_freqs[0] *= 0.9

        tgt_freqs *= formant_shift
        if is_devoiced: tgt_oq = 0.95

        # [NEW] Dynamic OQ Calculation (Realism: Breathier highs)
        dynamic_oq = tgt_oq + params['oq_slope'] * (freq_curve / 400.0)
        dynamic_oq = np.clip(dynamic_oq, 0.4, 0.95)

        # LF Model Source Generation using Dynamic OQ
        # Use dynamic_oq array for t_open calculation
        t_open = phase_array / np.clip(dynamic_oq, 0.1, 0.9)
        lf_source = np.zeros_like(phase_array)
        
        # Mask using Dynamic OQ
        mask_open = phase_array < dynamic_oq
        
        if np.any(mask_open):
            # Note: tgt_alpha is scalar, t_open is array.
            lf_source[mask_open] = np.sin(np.pi * t_open[mask_open]) * np.exp(tgt_alpha * t_open[mask_open])
        
        if params['harmonic_mix'] > 0.01:
            blep_source = VocalDSP.generate_bandlimited_pulse(phase_array, freq_curve, fs)
            mix = params['harmonic_mix']
            source = lf_source * (1.0 - mix) + blep_source * mix * 0.5
        else:
            source = lf_source

        noise_gate = np.zeros_like(phase_array)
        noise_gate[mask_open] = 1.0
        noise_gate[~mask_open] = 0.1

        if eff_breath > 0.0 or eff_whisper > 0.0:
            raw_noise = VocalDSP.pink_noise(total_samples, intensity=1.0)
            air_noise = VocalDSP.bandpass(raw_noise, 3000, 8000, fs)
            mod_noise = air_noise * noise_gate
            mix_ratio = 1.0 - (eff_whisper * 0.9)
            source = (source * mix_ratio) + (mod_noise * (eff_breath + eff_whisper*0.8) * eff_velocity * 4.0)

        # 3. Formant Filtering (Parallel)
        start_freqs = np.array(prev_vowel_state.get('freqs', tgt_freqs), dtype=np.float32)
        start_bws = np.array(prev_vowel_state.get('bws', tgt_bws), dtype=np.float32)
        
        output_buffer, final_freqs, final_bws = self._apply_formant_filter(
            source, start_freqs, start_bws, tgt_freqs, tgt_bws, total_samples, duration, romaji, fs
        )

        # 4. Consonant Burst (Dynamic Bandwidth / Coarticulation)
        if cons and cons in CONSONANT_SPECTRAL and not is_devoiced:
            p = CONSONANT_SPECTRAL[cons].copy() # 辞書を書き換えないようにコピー
            cons_type = p.get('type', 'fricative')
            
            # --- Coarticulation Logic ---
            # 母音のF2（tgt_freqs[1]）に応じて子音の強調帯域をシフトさせる
            # F2が高い（'i', 'e'） -> 前舌 -> 子音も高域寄りになる
            # F2が低い（'u', 'o'） -> 後舌/円唇 -> 子音も低域寄りになる
            
            vowel_f2 = tgt_freqs[1]
            base_f2 = 1500.0 # 基準F2
            
            # シフト量を計算（係数0.4は適度な追従性）
            f2_shift = (vowel_f2 - base_f2) * 0.4
            
            # 特に軟口蓋音(k, g)はF2の影響を強く受ける
            if next_cons_type in VELAR_CONS or cons in ['k', 'g']:
                f2_shift *= 1.5 
            
            # バンド帯域をシフト
            orig_low, orig_high = p['band']
            new_low = max(100, orig_low + f2_shift)
            new_high = min(fs/2 - 100, orig_high + f2_shift)
            
            # --- 以下、生成処理 ---
            len_mult = params['len_mult_plosive'] if cons_type == 'plosive' else 1.0
            c_dur = (p.get('burst_len', 0.015) if cons_type == 'plosive' else 0.05) * cons_attack * len_mult
                
            c_len = int(fs * c_dur)
            if c_len > 0:
                raw_noise = np.random.uniform(-1, 1, c_len).astype(np.float32)
                # シフトした帯域でフィルタリング
                c_noise = VocalDSP.bandpass(raw_noise, new_low, new_high, fs) * p['gain'] * params['gain_mult']
                mix_len = min(c_len, len(output_buffer))
                
                cons_gain = 0.6 
                if cons_type == 'plosive':
                    output_buffer[:mix_len] *= 0.1
                    output_buffer[:mix_len] += c_noise[:mix_len] * cons_gain * 1.5
                else:
                    ducking = np.linspace(0.2, 1.0, mix_len)
                    output_buffer[:mix_len] *= ducking
                    output_buffer[:mix_len] += c_noise[:mix_len] * cons_gain

        # 5. Post Processing
        if target_vowel in ['n', 'm', 'ng'] or cons in ['n', 'm']:
            nasal_sig = VocalDSP.apply_nasal_eq(output_buffer, fs)
            if params['nasal_mix'] > 0.0:
                 output_buffer = nasal_sig * (1.0 - params['nasal_mix']) + output_buffer * params['nasal_mix']
            else:
                 output_buffer = nasal_sig

        output_buffer *= self.makeup_gain
        output_buffer = np.tanh(output_buffer * params['comp_drive'])

        # Envelope
        fade_in = min(200, len(output_buffer)//4)
        output_buffer[:fade_in] *= np.linspace(0, 1, fade_in)
        
        if is_sokuon_next:
            cut_len = int(fs * 0.02)
            if cut_len < len(output_buffer):
                output_buffer[-cut_len:] *= np.linspace(1, 0, cut_len)
        else:
            fade_out = min(200, len(output_buffer)//4)
            output_buffer[-fade_out:] *= np.linspace(1, 0, fade_out)

        output_buffer *= eff_velocity

        new_state = {
            'vowel': target_vowel,
            'freqs': final_freqs,
            'bws': final_bws,
            'pitch': freq,
            'oq': tgt_oq,
            'alpha': tgt_alpha,
            'phase': final_phase_out 
        }

        return output_buffer.astype(np.float32), new_state
# ==========================================
# 5. Vocal Synthesis Engine (v20.0 LF-Model Refined)
# ==========================================
class VocalSynth:
    def __init__(self, mode='pop'):
        self.mode = mode
        self.base_transition_time = 0.08
        # [Adjust] Soft Clipに変更したため、入力ゲインを少し調整
        self.makeup_gain = 5.0 if self.mode == 'utada' else 4.5

    def get_formant_targets(self, vowel):
        """
        母音ごとのフォルマント周波数等を取得。
        ★Self-Vocaloid機能: ユーザー解析データがあれば優先使用する。
        """
        # 1. デフォルトデータのロード
        base_data = FORMANT_DATA.get(vowel, FORMANT_DATA.get('a'))
        
        # 2. ユーザーデータのチェック
        # USER_VOICE_DATA = {'a': [F1, F2, F3, F4], 'i': ...}
        if vowel in USER_VOICE_DATA:
            user_freqs = USER_VOICE_DATA[vowel]
            
            # Numpy配列化
            freqs = np.array(user_freqs, dtype=np.float32)
            
            # 帯域幅(BW)の推定
            # 録音データから正確なBWを出すのは難しいので、周波数に比例させる経験則を使用
            # Freqが高いほど共鳴は広くなる
            bws = freqs * 0.08 + 50.0 
            
            # Gain補正（高域減衰の補償）
            # 解析値は正規化されているため、合成用に強調
            
            # OQ (Open Quotient) と Alpha (Spectral Tilt)
            # 解析データからこれらを出すのは高度な逆フィルタが必要なため、
            # ベースデータの「声質キャラ」を引き継ぐ
            oq = base_data.get('oq', 0.6)
            alpha = base_data.get('alpha', 2.0)
            
        else:
            # 既存ロジック (Fallback)
            freqs = np.array(base_data['freq'], dtype=np.float32)
            bws = np.array(base_data['bw'], dtype=np.float32)
            oq = base_data.get('oq', 0.6)
            alpha = base_data.get('alpha', 2.0)

        # --- Mode Specific Adjustments (既存処理) ---
        if self.mode == 'pop':
            alpha = max(0.5, alpha - 0.8)
        elif self.mode == 'utada':
            # Utadaモードでもユーザーデータがある場合は、ユーザーの骨格(Freq)にUtadaの息遣い(OQ/Alpha)を乗せる
            if vowel == 'a' and vowel not in USER_VOICE_DATA:
                freqs[0] = 820; freqs[1] = 1150
            bws *= 1.2
            base_oq = base_data.get('oq', 0.65)
            oq = np.clip(base_oq * 0.9 + 0.15, 0.72, 0.78)
            alpha = np.clip(alpha + 0.2, 2.4, 2.8)
        else:
            freqs *= 0.9

        return freqs, bws, oq, alpha

    def _get_render_params(self, custom_params=None):
        params = {
            'oq_slope': 0.15,       # [Update] 高音でより息っぽくするため増加
            'jitter': 0.005,
            'cons_attack_scale': 1.0,
            'vib_rate_override': None,
            'vib_depth_override': None,
            'glide_duration': 0.06,
            'dip_amount': 0.985,
            'drop_depth': 0.03,
            'vib_start_delay': 0.15,
            'gain_mult': 1.0,
            'len_mult_plosive': 1.0,
            'nasal_mix': 0.0,
            'comp_drive': 0.8,
            'harmonic_mix': 0.0, 
            'phase_mode': 'continuous',
            'detune_amount': 0.0,
            'ta_factor': 0.04       # Return Phase Time (小さいほどBuzz感が強い)
        }
        
        if self.mode == 'utada':
            params.update({
                'oq_slope': 0.08, 'jitter': 0.003, 'comp_drive': 1.2,
                'vib_rate_override': 5.2, 'nasal_mix': 0.2, 'ta_factor': 0.05
            })
            
        if custom_params:
            params.update(custom_params)
        return params

    def _generate_pitch_curve(self, total_samples, freq, prev_pitch, duration, params, cons, is_phrase_end, vib_depth, vib_rate, fs):
        """ピッチカーブ生成 (変更なし)"""
        freq_curve = np.full(total_samples, freq, dtype=np.float32)
        t_vec = np.linspace(0, duration, total_samples, dtype=np.float32)

        if abs(freq - prev_pitch) > 1.0:
            glide_samples = int(fs * params['glide_duration'])
            if glide_samples < total_samples:
                t_glide = np.linspace(0, 1, glide_samples)
                curve = t_glide * t_glide * (3 - 2 * t_glide)
                freq_curve[:glide_samples] = prev_pitch + (freq - prev_pitch) * curve

        if cons and duration > 0.1:
            dip_len = int(fs * 0.05)
            if dip_len < total_samples:
                start_p = freq * params['dip_amount']
                t_dip = np.linspace(0, 1, dip_len)
                curve = 1.0 - np.exp(-t_dip * 12.0)
                freq_curve[:dip_len] = start_p + (freq - start_p) * curve

        if is_phrase_end and duration > 0.15:
            tail_len = int(fs * 0.15)
            if tail_len < total_samples:
                t_tail = np.linspace(0, 1, tail_len)
                freq_curve[-tail_len:] -= freq * params['drop_depth'] * (t_tail * t_tail)

        eff_vib_depth = params['vib_depth_override'] if params['vib_depth_override'] is not None else vib_depth
        eff_vib_rate = params['vib_rate_override'] if params['vib_rate_override'] is not None else vib_rate

        if eff_vib_depth > 0.001 and duration > 0.25:
            if duration > params['vib_start_delay']:
                vib_mask = np.clip((t_vec - params['vib_start_delay']) * 4.0, 0.0, 1.0)
                vib_osc = np.sin(2 * np.pi * eff_vib_rate * t_vec)
                freq_curve += freq * eff_vib_depth * vib_osc * vib_mask
        
        return freq_curve

    def _apply_formant_filter(self, source, start_freqs, start_bws, tgt_freqs, tgt_bws, total_samples, duration, romaji, fs):
        """フォルマントフィルタ (F5拡張 & 超人仕様オーバーシュート)"""
        import math
        
        block_size = 128
        num_blocks = (total_samples + block_size - 1) // block_size
        output_buffer = np.zeros(total_samples, dtype=np.float32)

        # 1. 限界突破の遷移時間 (人間の限界約15msを無視し、5〜25msの超速で切り替える)
        trans_time = max(0.005, min(0.025, duration * 0.15)) 
        
        zi = [None] * 5  # F5に対応するため配列を5に拡張
        current_freqs = start_freqs.copy()
        current_bws = start_bws.copy()

        # バンドゲイン: F5用に追加 (基本は高域ほど弱くなる自然法則に従う)
        base_band_gains = np.array([1.0, 0.8, 0.6, 0.4, 0.25], dtype=np.float32)

        for b_idx in range(num_blocks):
            start_pos = b_idx * block_size
            end_pos = min(start_pos + block_size, total_samples)
            t_now = (b_idx * block_size) / fs
            
            # --- 超人ロジックの計算 ---
            active_band_gains = base_band_gains.copy()
            
            if romaji != '-' and t_now < trans_time * 1.5:  # 遷移完了後も少し余韻を残すため1.5倍
                # 正規化された進行度 (0.0 〜 1.0)
                progress = min(1.0, t_now / trans_time)
                
                # 魔法1: 超速のEase-Outカーブ (cubic-out)
                # 最初はワープのように動き、最後でフワッと止まる
                curve_t = 1.0 - math.pow(1.0 - progress, 3)
                
                # 魔法2: オーバーシュート現象
                # 遷移の後半(70%以降)で目標値を一瞬だけ最大5%通り越して戻る
                if progress > 0.7:
                    overshoot = math.sin((progress - 0.7) * math.pi / 0.3) * 0.03
                    curve_t += overshoot
                
                # 魔法3: 遷移中のQ値（帯域幅）の鋭利化
                # 口の形が変化している最中（progress=0.5付近）だけ帯域幅を最大50%絞ってシンセっぽく尖らせる
                sharpness = 1.0 - (0.3 * math.sin(progress * math.pi))
                
                # 魔法4: F4・F5スパーク (高域アタックのブースト)
                # 遷移が完了する直前の「パツッ」というタイミングで、F4とF5のゲインを1.8倍に跳ね上げる
                spark_boost = 1.0 + (0.8 * math.sin(progress * math.pi))
                active_band_gains[3] *= spark_boost # F4
                active_band_gains[4] *= spark_boost # F5

                # 周波数と帯域幅の適用
                current_freqs = start_freqs + (tgt_freqs - start_freqs) * curve_t
                current_bws = (start_bws + (tgt_bws - start_bws) * curve_t) * sharpness
                
            else:
                # 遷移完了後はターゲット値に完全固定
                current_freqs = tgt_freqs.copy()
                current_bws = tgt_bws.copy()

            # --- フィルタ適用 (5バンドに拡張) ---
            block_source = source[start_pos:end_pos]
            block_out = np.zeros_like(block_source)
            
            for i in range(5): # 4から5へ変更
                # scipyのiirpeak等を使ってフィルタ係数を計算する（既存の実装に合わせる）
                # ※ここではv21.9にあるIIRフィルタ設計と想定
                b, a = signal.iirpeak(current_freqs[i], current_freqs[i] / current_bws[i], fs)
                
                if zi[i] is None:
                    zi[i] = signal.lfilter_zi(b, a) * block_source[0]
                    
                filtered_block, zi[i] = signal.lfilter(b, a, block_source, zi=zi[i])
                block_out += filtered_block * active_band_gains[i]
                
            output_buffer[start_pos:end_pos] = block_out

        # F5の追加により全体の音量が上がるため、出力ゲインを少し調整
        output_buffer *= 0.95 
        return output_buffer, current_freqs, current_bws

    def render(self, kana, freq, length, velocity, prev_vowel_state, ag_on,
               initial_phase=0.0, vocal_params=None, pitch_curve_data=None, start_time=0.0,
               breath_vol=0.05, breath_chance=0.5, cons_attack=1.0, 
               scoop_amount=0.0, whisper_amount=0.0, vib_depth=0.05, 
               vib_rate=5.0, formant_shift=1.0, is_devoiced=False,
               is_sokuon_next=False, after_sokuon=False, next_cons_type=None,
               is_phrase_end=False, fs=48000):

        # 1. Phoneme Parsing
        romaji = kana_engine.to_romaji(kana)
        if romaji == 'clt': 
            return np.random.uniform(-0.01, 0.01, int(fs * length)).astype(np.float32), prev_vowel_state

        target_vowel = 'a'
        cons = ''
        if romaji in ('-', '', '\n', ' '):
            target_vowel = prev_vowel_state.get('vowel', 'a')
        else:
            for length_check in [2, 1]:
                if len(romaji) >= length_check and romaji[:length_check] in CONSONANT_SPECTRAL:
                    cons, target_vowel = romaji[:length_check], romaji[length_check:]
                    break
            else:
                target_vowel = romaji
            if target_vowel == '': target_vowel = prev_vowel_state.get('vowel', 'a')

        # --- Load Render Parameters ---
        params = self._get_render_params(vocal_params)
        cons_attack *= params['cons_attack_scale']

        # Devoicing Setup
        eff_velocity = velocity * (0.3 if is_devoiced and target_vowel in ['i', 'u'] else 1.0)
        eff_whisper = 0.8 if is_devoiced else whisper_amount
        eff_breath = 0.2 if is_devoiced else breath_vol

        # 2. Source Setup
        duration = max(0.05, length)
        total_samples = int(fs * duration)
        
        prev_pitch = prev_vowel_state.get('pitch', freq)
        freq_curve = self._generate_pitch_curve(total_samples, freq, prev_pitch, duration, params, cons, is_phrase_end, vib_depth, vib_rate, fs)

        if params['detune_amount'] > 0:
            detune_lfo = np.sin(np.linspace(0, duration * 2 * np.pi * 3.0, total_samples)) 
            freq_curve *= (1.0 + params['detune_amount'] * 0.005 * detune_lfo)

        # Jitter
        if params['jitter'] > 0:
            jitter_sig = np.random.normal(0, params['jitter'], total_samples)
            freq_curve *= (1.0 + jitter_sig)

        # Global Phase Accumulation
        phase_increment = freq_curve / fs
        phase_array = np.cumsum(phase_increment) + initial_phase
        phase_array = phase_array % 1.0 
        
        final_phase_out = phase_array[-1]

        # --- LF-Model Source Generation (Refined) ---
        tgt_freqs, tgt_bws, tgt_oq, tgt_alpha = self.get_formant_targets(target_vowel)
        
        if next_cons_type in VELAR_CONS: tgt_freqs[1] *= 1.1
        elif next_cons_type in LABIAL_CONS: tgt_freqs[0] *= 0.9

        tgt_freqs *= formant_shift
        if is_devoiced: tgt_oq = 0.95

        # Dynamic Parameters
        # Te (Closure Instant): ピッチが高いと短くなる（息っぽくなる）
        dynamic_te = tgt_oq + params['oq_slope'] * (freq_curve / 400.0)
        dynamic_te = np.clip(dynamic_te, 0.45, 0.96)
        
        # Tp (Peak Flow Time): Teに対する比率を少し動かす（低OQで少し尖らせる）
        # 元のLFモデルに近づけるためのヒューリスティック
        tp_ratio = 0.65 + (0.1 * (1.0 - dynamic_te)) 
        dynamic_tp = dynamic_te * tp_ratio
        
        # Ta (Return Phase Time): 固定値に近いが少し揺らす
        ta = params['ta_factor']

        lf_source = np.zeros_like(phase_array)
        mask_open = phase_array < dynamic_te
        
        # [Corrected Logic: Open Phase]
        # E(t) = sin(omega * t) * exp(alpha * t)
        # ここでの alpha は tgt_alpha をスケーリング係数として少し弱めて使う
        alpha_eff = tgt_alpha * 0.5 
        omega = np.pi / dynamic_tp

        if np.any(mask_open):
            t = phase_array[mask_open]
            val = np.sin(omega[mask_open] * t) * np.exp(alpha_eff * t)
            lf_source[mask_open] = val

        # [Corrected Logic: Return Phase]
        # 以前の「再計算」をやめ、理論的な接続点(Ee)を計算して連続させる
        mask_return = ~mask_open
        if np.any(mask_return):
            t_ret = phase_array[mask_return]
            te_ret = dynamic_te[mask_return]
            
            # Ee (Value at Te) の計算
            # Note: ここではmask_return内の各点におけるTeを使って、その瞬間のEe理論値を出す
            # これにより「Te時点での値」から指数減衰を開始できる
            omega_ret = np.pi / (te_ret * tp_ratio[mask_return])
            ee_est = np.sin(omega_ret * te_ret) * np.exp(alpha_eff * te_ret)
            
            # Decay
            # leakage: 閉鎖不全（息漏れ）。OQが高いほど漏れやすい
            leakage = (1.0 - dynamic_te[mask_return]) * 0.25
            
            decay = np.exp(-(t_ret - te_ret) / ta)
            
            # [Fix] エイリアシング対策: 急激な不連続を避けるため、減衰後の値をleakageにソフトランディングさせる
            lf_source[mask_return] = (ee_est * decay) * (1.0 - leakage) + (ee_est * 0.01 + leakage * 0.1)

        source = lf_source

        # Harmonic Mix
        if params['harmonic_mix'] > 0.01:
            blep_source = VocalDSP.generate_bandlimited_pulse(phase_array, freq_curve, fs)
            mix = params['harmonic_mix']
            source = source * (1.0 - mix) + blep_source * mix * 0.5

        # Breath Noise (Dynamic Gating)
        # 閉鎖時のゲートレベルをOQ依存に変更
        noise_gate = np.zeros_like(phase_array)
        noise_gate[mask_open] = 1.0
        # 閉鎖時も完全に0にせず、OQに応じて少し通す（声門閉鎖不全の再現）
        noise_gate[mask_return] = (1.0 - dynamic_te[mask_return]) * 0.5

        if eff_breath > 0.0 or eff_whisper > 0.0:
            raw_noise = VocalDSP.pink_noise(total_samples, intensity=1.0)
            air_noise = VocalDSP.bandpass(raw_noise, 3000, 8000, fs)
            mod_noise = air_noise * noise_gate
            mix_ratio = 1.0 - (eff_whisper * 0.9)
            source = (source * mix_ratio) + (mod_noise * (eff_breath + eff_whisper*0.8) * eff_velocity * 4.0)

        # 3. Formant Filtering
        start_freqs = np.array(prev_vowel_state.get('freqs', tgt_freqs), dtype=np.float32)
        start_bws = np.array(prev_vowel_state.get('bws', tgt_bws), dtype=np.float32)
        
        output_buffer, final_freqs, final_bws = self._apply_formant_filter(
            source, start_freqs, start_bws, tgt_freqs, tgt_bws, total_samples, duration, romaji, fs
        )

        # 4. Consonant Burst
        if cons and cons in CONSONANT_SPECTRAL and not is_devoiced:
            p = CONSONANT_SPECTRAL[cons]
            cons_type = p.get('type', 'fricative')
            
            len_mult = params['len_mult_plosive'] if cons_type == 'plosive' else 1.0
            c_dur = (p.get('burst_len', 0.015) if cons_type == 'plosive' else 0.05) * cons_attack * len_mult
                
            c_len = int(fs * c_dur)
            if c_len > 0:
                raw_noise = np.random.uniform(-1, 1, c_len).astype(np.float32)
                c_noise = VocalDSP.bandpass(raw_noise, p['band'][0], p['band'][1], fs) * p['gain'] * params['gain_mult']
                mix_len = min(c_len, len(output_buffer))
                
                cons_gain = 0.6 
                if cons_type == 'plosive':
                    output_buffer[:mix_len] *= 0.1
                    output_buffer[:mix_len] += c_noise[:mix_len] * cons_gain * 1.5
                else:
                    ducking = np.linspace(0.2, 1.0, mix_len)
                    output_buffer[:mix_len] *= ducking
                    output_buffer[:mix_len] += c_noise[:mix_len] * cons_gain

        # 5. Post Processing
        if target_vowel in ['n', 'm', 'ng'] or cons in ['n', 'm']:
            nasal_sig = VocalDSP.apply_nasal_eq(output_buffer, fs)
            if params['nasal_mix'] > 0.0:
                 output_buffer = nasal_sig * (1.0 - params['nasal_mix']) + output_buffer * params['nasal_mix']
            else:
                 output_buffer = nasal_sig

        output_buffer *= self.makeup_gain
        
        # [FIX] Soft Clipper (tanh -> x / (1+|x|))
        # 金属的な歪みを抑え、アナログライクな飽和感にする
        drive_sig = output_buffer * params['comp_drive']
        output_buffer = drive_sig / (1.0 + np.abs(drive_sig))

        # Envelope
        fade_in = min(200, len(output_buffer)//4)
        output_buffer[:fade_in] *= np.linspace(0, 1, fade_in)
        
        if is_sokuon_next:
            cut_len = int(fs * 0.02)
            if cut_len < len(output_buffer):
                output_buffer[-cut_len:] *= np.linspace(1, 0, cut_len)
        else:
            fade_out = min(200, len(output_buffer)//4)
            output_buffer[-fade_out:] *= np.linspace(1, 0, fade_out)

        output_buffer *= eff_velocity

        new_state = {
            'vowel': target_vowel,
            'freqs': final_freqs,
            'bws': final_bws,
            'pitch': freq,
            'oq': tgt_oq,
            'alpha': tgt_alpha,
            'phase': final_phase_out 
        }

        return output_buffer.astype(np.float32), new_state


# --- Initialization of Instances ---
kana_engine = KanaLogic()
neko_brain = NekotopyLogic()
vocal_engine = VocalSynth()

# ==========================================
# HyperNekoProduct v16.1 (Fixed Version)
# Part 2: Genre Data, Spec Builder, and Core Audio Generation
# ==========================================

# ==========================================
# 6. Genre Style Definitions & Presets
# ==========================================
RHYTHM_TEMPLATES = {

# =========================
# リズム系（刻み・ハット）
# =========================
    "Sixteen":   [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    "Eight":     [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    "Offbeat":   [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
    "HouseHat":  [0,1,0,1, 0,1,0,1, 0,1,0,1, 0,1,0,1],
    "HatGroove": [3,1,2,1, 3,1,2,1, 3,1,2,1, 3,1,2,1],
    "RideSwing": [1,0,1,1, 1,0,1,0, 1,0,1,1, 1,0,1,0],
    "TrapHat":   [1,0,1,1, 1,0,1,0, 1,1,0,1, 1,0,1,0],

# =========================
# ビート系（ドラム骨格）
# =========================
    "4onFloor":  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
    "Backbeat":  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
    "TrapKick":  [1,0,0,0, 0,0,0,1, 0,0,1,0, 0,0,0,0],
    "TrapSnare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
    "TechnoKick":[1,0,0,0, 1,0,1,0, 1,0,0,0, 1,0,1,0],
    "BreakKick": [1,0,0,1, 0,0,1,0, 1,0,1,0, 0,0,0,1],
    "SnareRoll": [0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1],
    "HalfTime":  [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
    "OneDrop":   [0,0,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,0],
    "DembowKick":[1,0,0,1, 0,0,1,0, 1,0,0,1, 0,0,1,0],
    "332Clave":  [1,0,0,1, 0,0,1,0, 1,0,0,1, 0,0,1,0],
    "Clave":     [1,0,0,1, 0,0,1,0, 0,0,1,0, 1,0,0,0],
    "CrashOne":  [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    "EndFill":   [0,0,0,0, 0,0,0,0, 0,0,1,1, 1,1,1,1],
    "Empty":     [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],

# =========================
# メロディー系（リフ・ベース・ボーカル）
# =========================
    "RockRiff":      [1,0,1,1, 0,0,1,0, 1,0,1,0, 0,1,0,0],
    "VocalMain":     [1,0,1,0, 1,1,0,0, 1,0,1,0, 1,1,1,0],
    "Motown":        [1,0,1,0, 1,1,0,1, 1,0,1,0, 1,1,0,0],
    "PalmMuteDrive": [1,0,1,0, 1,0,1,1, 1,0,1,0, 1,0,1,0],
    "AltRockChug":   [1,1,0,1, 1,0,1,0, 1,1,0,1, 1,0,0,1],
    "SyncRockGt":    [1,0,0,1, 0,1,0,0, 1,0,1,0, 0,1,0,0],
    "OpenLift":      [1,0,0,0, 1,0,1,0, 1,0,0,0, 1,0,1,0],
    "RootEight":     [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    "RockWalk":      [1,0,1,1, 0,0,1,0, 1,0,1,1, 0,0,1,0],
    "SyncBass":      [1,0,0,1, 0,1,0,0, 1,0,0,1, 0,1,0,0],
    "PedalDrive":    [1,1,1,0, 1,1,0,1, 1,1,1,0, 1,1,0,0],
}

GENRE_STYLES = {
    "Lofi": {
        "engine": {"sc_strength": 0.3, "bpm": 80, "swing_amount": 0.15, "bitcrush_rate": 0.2, "bitcrush_depth": 12},
        "common": {"dsp": ["lpf", "bitcrush"]},
        "kick": {"vol": 0.9, "pattern": "TrapKick"},
        "snare": {"vol": 0.7, "pattern": "Backbeat"},
        "hihat": {"vol": 0.5, "pattern": "Eight"},
        "cymbal_ride": {"active": True, "vol": 0.4, "pattern": "Eight", "dsp": ["reverb_short"]},
        "bass_sub": {"vol": 1.0, "pattern": "Sixteen", "style": "Normal"},
        "vocal_lead": {"active": True, "vol": 0.9, "pattern": "VocalMain", "dsp": ["reverb_short", "lpf"]},
        "vocal_harmony": {"active": False},
        "strings_high": {"active": True, "vol": 0.6, "pattern": "LongNote", "dsp": ["reverb_long", "wide"]},
        "pad_L": {"vol": 0.8, "pattern": "Sixteen"},
        "lead_main": {"active": False}
    },
    "Hyperpop": {
        "engine": {"sc_strength": 0.9, "bpm": 160, "bitcrush_rate": 0.6, "bitcrush_depth": 6, "vocal_formant_shift": 1.2},
        "common": {},
        "kick": {"vol": 1.4, "pattern": "4onFloor", "dsp": ["dist", "comp"]},
        "snare": {"vol": 1.1, "pattern": "Backbeat", "dsp": ["dist", "bitcrush"]},
        "cymbal_crash": {"active": True, "pattern": "CrashOne"},
        "bass_sub": {"vol": 1.1, "pattern": "Sixteen", "style": "Normal"},
        "vocal_lead": {"active": True, "vol": 1.1, "wave": "pop", "pattern": "VocalMain", "style": "Rand", "dsp": ["dist", "wide", "delay"]},
        "vocal_harmony": {"active": True, "vol": 0.7, "wave": "pop", "pattern": "VocalMain", "dsp": ["wide", "dist"]},
        "lead_main": {"active": True, "pattern": "Sixteen", "style": "Rand"},
        "guitar_riff": {"active": False}
    },
    "EDM": {
        "engine": {"sc_strength": 0.95, "bpm": 128},
        "common": {"dsp": ["comp"]},
        "kick": {"vol": 1.5, "pattern": "4onFloor"},
        "snare": {"vol": 1.0, "pattern": "Backbeat"},
        "hihat": {"pattern": "Offbeat"},
        "cymbal_ride": {"active": True, "pattern": "Offbeat", "vol": 0.6},
        "cymbal_crash": {"active": True, "pattern": "CrashOne", "vol": 0.8},
        "bass_sub": {"pattern": "Sixteen", "style": "Normal"},
        "vocal_lead": {"active": True, "vol": 1.0, "wave": "adult", "pattern": "VocalMain", "style": "Up", "dsp": ["wide", "reverb_long"]},
        "vocal_harmony": {"active": True, "vol": 0.6, "pattern": "VocalMain", "dsp": ["reverb_long", "wide"]},
        "strings_high": {"active": True, "pattern": "LongNote", "vol": 0.7, "dsp": ["reverb_long", "wide"]},
        "lead_main": {"pattern": "Sixteen", "style": "Rand"},
        "pad_L": {"pattern": "Sixteen"}
    },
    "Rock": {
        "engine": {"sc_strength": 0.0, "bpm": 140},
        "common": {},
        "kick": {"vol": 1.1, "pattern": "4onFloor"},
        "snare": {"vol": 1.2, "pattern": "Backbeat"},
        "hihat": {"pattern": "Eight"},
        "cymbal_crash": {"active": True, "pattern": "CrashOne", "vol": 0.9},
        "cymbal_ride": {"active": True, "pattern": "Eight", "vol": 0.6},
        "bass_sub": {"pattern": "Sixteen"},
        "guitar_riff": {"active": True, "vol": 1.0, "pan": 0.25, "dsp": ["dist", "reverb_short"], "pattern": "RockRiff"},
        "vocal_lead": {"active": True, "vol": 1.1, "wave": "pop", "pattern": "VocalMain", "style": "Normal", "dsp": ["comp", "reverb_short"]},
        "vocal_harmony": {"active": True, "vol": 0.8},
        "lead_main": {"active": False}
    },
    "Vocaloid-ish": {
        "engine": {"sc_strength": 0.3, "bpm": 160, "vocal_formant_shift": 1.0},
        "common": {},
        "kick": {"vol": 1.2, "pattern": "4onFloor"},
        "snare": {"vol": 1.0, "pattern": "Backbeat"},
        "cymbal_crash": {"active": True, "pattern": "CrashOne"},
        "bass_sub": {"vol": 0.9, "pattern": "Sixteen"},
        "guitar_riff": {"active": True, "pattern": "Eight", "dsp": ["dist", "wide"]},
        "vocal_lead": {"active": True, "vol": 1.3, "wave": "pop", "pattern": "Sixteen", "style": "Rand", "dsp": ["comp", "delay"]},
        "vocal_harmony": {"active": True, "vol": 0.7, "wave": "pop", "pattern": "Sixteen", "dsp": ["wide"]},
        "strings_high": {"active": True, "pattern": "LongNote", "vol": 0.5},
        "lead_main": {"active": True, "pattern": "Sixteen", "style": "Up"}
    },
    "Ambient": {
        "engine": {"sc_strength": 0.4, "bpm": 90},
        "common": {"dsp": ["reverb_long"]},
        "kick": {"vol": 0.5, "pattern": "Clave"},
        "vocal_lead": {"active": True, "vol": 0.8, "wave": "adult", "pattern": "Offbeat", "style": "Normal", "dsp": ["reverb_long", "wide"]},
        "vocal_harmony": {"active": True, "vol": 0.5, "wave": "adult", "pattern": "Offbeat"},
        "pad_L": {"pattern": "Sixteen"},
        "strings_high": {"active": True, "pattern": "LongNote", "vol": 0.6, "dsp": ["reverb_long"]},
        "guitar_riff": {"active": True, "pattern": "Sixteen"}
    },
    "HipHop": {
        "engine": {"sc_strength": 0.5, "bpm": 95, "swing_amount": 0.25},
        "kick": {"vol": 1.4, "pattern": "TrapKick"},
        "snare": {"vol": 1.0, "pattern": "Backbeat"},
        "hihat": {"pattern": "Sixteen", "style": "Rand"},
        "bass_sub": {"vol": 1.2, "pattern": "Sixteen"},
        "vocal_lead": {"active": True, "vol": 1.1, "wave": "pop", "pattern": "VocalMain", "style": "Normal", "dsp": ["comp", "bitcrush"]},
        "vocal_harmony": {"active": False},
        "strings_high": {"active": True, "pattern": "LongNote", "vol": 0.4}
    },
    "KiraKira": {
        "engine": {"sc_strength": 0.2, "bpm": 170},
        "kick": {"vol": 0.8, "pattern": "4onFloor"},
        "snare": {"vol": 0.6, "pattern": "Backbeat"},
        "vocal_lead": {"active": True, "wave": "pop", "pattern": "Sixteen", "style": "Rand", "dsp": ["wide", "delay"]},
        "vocal_harmony": {"active": True, "vol": 0.7},
        "lead_main": {"active": True, "pattern": "Sixteen", "style": "Up"},
        "glitch": {"active": True},
        "strings_high": {"active": True, "vol": 0.5}
    }
}

GENRE_STYLES["Sick Love R&B"] = {
    "engine": {
        "bpm": 96,                  # ゆったり揺れるミッドテンポ
        "swing_amount": 0.15,       # R&B特有の少し跳ねたグルーヴ
        "sc_strength": 0.05,        # サイドチェインはほぼ掛けない（自然に）
        "vocal_formant_shift": 1.0, # Ne-Yoのような中性的なテノール
        "reverb_mix": 0.25          # ボーカルを綺麗に響かせる
    },
    "common": {
        "dsp": []
    },
    
    # リズム：指パッチン（Snare）を強調したシンプルビート
    "kick": {
        "active": True,
        "vol": 1.2,
        "pattern": "TrapKick",      # キックは少しトラップ寄りでもOK
        "dsp": ["lpf", "comp"],
        "eq_params": {"low": 2.0, "mid": 0.0, "high": 0.0}
    },
    "snare": {
        "active": True,
        "role": "snare",
        "vol": 1.0,
        "pattern": "Backbeat",      # 2拍4拍で確実に鳴らす
        "dsp": ["high_pass"]        # 低音を削って「パチン」という音に近づける
    },
    "hihat": {
        "active": True,
        "vol": 0.6,
        "pattern": "Eight",         # 細かく刻みすぎない
        "dsp": []
    },

    # 上モノ：あの「ハープ」を再現
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 1.0,
        "wave": "harp",             # 【重要】ハープ音色
        "pattern": "Sixteen",       # 16分音符で埋める
        "style": "Arp",             # 【重要】アルペジオで「テロレロ」鳴らす
        "div": 16,
        "dsp": ["delay", "reverb_short"] # 空間系で広げる
    },
    
    # コード感：エレピで甘く
    "epiano": {
        "active": True,
        "role": "epiano",
        "vol": 0.8,
        "wave": "organ",            # オルガンかエレピで代用
        "pattern": "Sixteen",
        "style": "Normal",
        "dsp": ["reverb_short", "wide"]
    },

    # ボーカル：甘いR&B
    "vocal_lead": {
        "active": True,
        "vol": 1.1,
        "wave": "pop",
        "pattern": "VocalMain",
        "style": "Legato",          # 滑らかに歌う
        "dsp": ["comp", "reverb_short", "delay"],
        "vocal_params": {
            "attack": 0.05,
            "release": 0.1,
            "vibrato_depth": 0.04,  # ビブラートは浅く細かく
            "vibrato_rate": 5.0,
            "scoop_amount": 0.1,    # 少しだけしゃくる
            "breath_vol": 0.05
        }
    },
    
    # ベース：主張しすぎないサブベース
    "bass_sub": {
        "active": True,
        "vol": 0.9,
        "wave": "sin",              # 丸い音
        "pattern": "Sixteen",
        "style": "Normal"
    }
}
GENRE_STYLES["Wagakki Rock"] = {
    "engine": {
        "bpm": 160,                 # ボカロ・和ロック特有の疾走感
        "swing_amount": 0.0,        # ストレートな縦ノリ
        "reverb_mix": 0.3,          # ライブハウス感
        "vocal_formant_shift": 1.05,# 少し鼻にかかったような艶のある声
        "sc_strength": 0.0          # サイドチェインは不要
    },
    "common": {
        "dsp": []
    },
    
    # リズム：手数の多い和太鼓×ロックドラム
    "kick": {
        "active": True,
        "vol": 1.2,
        "pattern": "RockRiff",      # ドコドコしたツーバス気味のフレーズ
        "dsp": ["comp", "eq"],
        "eq_params": {"low": 3.0, "mid": -1.0, "high": 1.0} # ドンシャリ
    },
    "snare": {
        "active": True,
        "vol": 1.1,
        "pattern": "Backbeat",      # 鋭いスネア
        "dsp": ["reverb_short"]
    },
    "impact": {
        "active": True,
        "role": "impact",
        "vol": 1.0,
        "pattern": "CrashOne",      # 締め太鼓や拍子木のようなアクセント
        "dsp": ["high_pass"]
    },

    # 旋律楽器：和楽器のシミュレーション
    "lead_main": {
        "active": True,
        "role": "lead_main", # 津軽三味線パート
        "vol": 0.9,
        "wave": "heavy_guitar",     # 歪んだギター波形を流用して激しく叩く三味線を表現
        "pattern": "Sixteen",       # 高速バチさばき
        "style": "Rand",            # 乱れ弾き
        "div": 16,
        "dsp": ["dist", "wide", "eq"],
        "eq_params": {"low": -2.0, "mid": 3.0, "high": 2.0} # 中高域を突き刺すようにブースト
    },
    "guitar_riff": {
        "active": True,
        "role": "guitar_riff", # 箏（琴）パート
        "vol": 0.8,
        "wave": "pluck",            # 減衰音（琴に近い）
        "pattern": "Sixteen",
        "style": "Arp",             # 流れるようなアルペジオ
        "dsp": ["delay", "wide"]
    },
    "pad_L": {
        "active": True,
        "role": "pad_L",     # 尺八パート
        "vol": 0.7,
        "wave": "celtic_flute",     # フルート波形で代用
        "pattern": "LongNote",      # 息の長いフレーズ
        "dsp": ["reverb_long"],
        # ビブラートで尺八特有の「首振り」を再現
    },

    # ボーカル：こぶしと艶
    "vocal_lead": {
        "active": True,
        "vol": 1.2,
        "wave": "pop",
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["comp", "delay", "reverb_short"],
        "vocal_params": {
            "attack": 0.05,
            "release": 0.1,
            "vibrato_depth": 0.08,  # 深めのビブラート（ちりめん）
            "vibrato_rate": 6.0,    # 速く揺らす
            "scoop_amount": 0.4,    # 【重要】強くしゃくり上げて色気を出す
            "breath_vol": 0.05
        }
    },
    "bass_sub": {
        "active": True,
        "vol": 1.0,
        "wave": "saw",
        "pattern": "Eight",
        "style": "Root",
        "dsp": ["dist"]
    }
}

GENRE_STYLES["SNES 16-bit"] = {
    "engine": {
        "bpm": 140,                 # アクションゲームやRPGの戦闘曲風
        "swing_amount": 0.0,
        "bitcrush_rate": 0.4,       # 【重要】全体を32kHz/16bit程度に落とす
        "bitcrush_depth": 12,       # 少し荒い解像度
        "reverb_mix": 0.0,          # 全体リバーブは切る（個別のDelayでエコーを作る）
        "vocal_formant_shift": 1.5  # PCM音源のような少し人工的な声
    },
    "common": {
        "dsp": ["bitcrush"]         # 全トラックを強制的にレトロ化
    },
    
    # リズム：PCMドラム
    "kick": {
        "active": True,
        "vol": 1.2,
        "pattern": "RockRiff",
        "dsp": ["comp", "eq"],      # コンプでパツパツにする
        "eq_params": {"low": 2.0, "mid": 0.0, "high": 0.0}
    },
    "snare": {
        "active": True,
        "vol": 1.0,
        "pattern": "Backbeat",
        "dsp": ["bitcrush"]         # ザラついたスネア
    },
    "impact": {
        "active": True,
        "role": "impact",
        "vol": 0.8,
        "pattern": "CrashOne",      # オケヒ（Orchestra Hit）的な役割
        "dsp": ["bitcrush", "delay"]
    },

    # メロディ：矩形波とノコギリ波
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.9,
        "wave": "sqr",              # 矩形波（ファミコン〜スーファミの定番）
        "pattern": "Sixteen",
        "style": "Arp",             # 高速アルペジオ
        "div": 16,
        "dsp": ["delay"]            # 【重要】スーファミ特有の「疑似エコー」
    },
    "vocal_lead": {
        "active": True,
        "vol": 1.0,
        "wave": "pop",              # PCMボーカル
        "pattern": "VocalMain",
        "style": "Normal",
        "dsp": ["bitcrush", "delay"], # 容量削減のために劣化した声を再現
        "vocal_params": {
            "attack": 0.01,         # アタック最速（サンプル再生っぽく）
            "vibrato_depth": 0.0,   # ビブラートなし（打ち込みっぽさ）
            "breath_vol": 0.0       # ノイズはカット
        }
    },

    # 伴奏：ストリングスカーテン
    "strings_high": {
        "active": True,
        "role": "strings_high",
        "vol": 0.7,
        "wave": "saw",              # ノコギリ波で作るストリングス
        "pattern": "LongNote",
        "dsp": ["lpf", "delay"]     # フィルターでこもらせてエコーをかける
    },
    
    # ベース：ゴリゴリのシンセベース
    "bass_sub": {
        "active": True,
        "vol": 1.0,
        "wave": "saw",
        "pattern": "Eight",
        "style": "Octave",          # オクターブで動き回る（F-ZEROとかの感じ）
        "dsp": ["bitcrush"]
    }
}


GENRE_STYLES["Golden Lilium"] = {
    "engine": {
        "bpm": 64,                  # 重厚でゆっくりとしたテンポ
        "swing_amount": 0.0,        # 厳格なクラシック調
        "reverb_mix": 0.65,         # 大聖堂のような深い残響
        "reverb_decay": 0.85,       # 長い余韻
        "vocal_formant_shift": 0.9, # 少し太く、大人びたソプラノの声色
        "vel_humanize": 0.15        # 機械的すぎない揺らぎ
    },
    "common": {
        "dsp": ["reverb_long"]      # 全体的に深いリバーブをかける
    },
    # リズム隊は排除し、鐘の音だけを残す
    "kick": {"active": False},
    "snare": {"active": False},
    "hihat": {"active": False},
    "cymbal_ride": {"active": False},
    "impact": {
        "active": True,
        "role": "impact", 
        "vol": 0.8,
        "pattern": "Clave",    # 厳かなタイミングで鳴る
        "dsp": ["reverb_long", "lpf"]
    },
    
    # メイン楽器：パイプオルガンとハープ（クリムトの金色の装飾イメージ）
    "organ_pad": {
        "active": True,
        "role": "organ_pad",
        "vol": 0.9,
        "wave": "organ",
        "pattern": "LongNote",
        "style": "Normal",
        "dsp": ["reverb_long", "wide"]
    },
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.7,
        "wave": "harp",         # 繊細な金箔のようなアルペジオ
        "pattern": "Sixteen",
        "style": "Arp",         # アルペジオで装飾音を奏でる
        "div": 12,              # 3連符（6/8拍子的な揺らぎ）
        "dsp": ["delay", "wide"]
    },

    # ボーカル：オペラ調（Liliumのような独唱と聖歌隊）
    "vocal_lead": {
        "active": True,
        "vol": 1.2,
        "wave": "adult",        # 太めの波形を使用
        "pattern": "VocalMain",
        "style": "Legato",      # 滑らかに繋ぐ
        "dsp": ["reverb_long", "delay"],
        "vocal_params": {
            "attack": 0.1,          # ゆっくりとした立ち上がり
            "release": 0.3,         # 長いリリース
            "vibrato_depth": 0.1,   # オペラのような深いビブラート
            "vibrato_rate": 5.5,    # 速めの揺れ
            "breath_vol": 0.1,      # 生々しい息遣い
            "scoop_amount": 0.1     # 厳格なピッチ
        }
    },
    "vocal_harmony": {
        "active": True,
        "vol": 0.8,
        "wave": "adult",
        "pattern": "VocalMain",
        "dsp": ["wide", "reverb_long", "detune"], # コーラス効果
        "vocal_params": {
            "formant_shift": 1.2    # 少し細い声で聖歌隊っぽく
        }
    },
    
    # 背景：不穏な弦楽器
    "strings_high": {
        "active": True,
        "vol": 0.6,
        "wave": "strings",
        "pattern": "LongNote",
        "dsp": ["reverb_long"]
    },
    "bass_sub": {
        "active": True,
        "vol": 0.8,
        "wave": "saw",
        "pattern": "Root",     # ルート音で重厚な低音を支える
        "dsp": ["lpf"]         # 高域をカットして暗くする
    }
}

GENRE_STYLES["Isekai Folk"] = {
    "engine": {
        "bpm": 110,                 # 軽快な旅のテンポ
        "swing_amount": 0.2,        # 6/8拍子風の跳ね感（ジグ/リール）
        "reverb_mix": 0.35,         # 広大な草原感
        "reverb_decay": 0.7
    },
    "common": {
        "dsp": []
    },
    # リズム隊：バウロン（Bodhran）をKick/Tomで代用
    "kick": {
        "active": True,
        "vol": 0.9, 
        "pattern": "Clave", # ドン、タ、ドン...のような民族的なリズム
        "dsp": ["lpf"]      # こもらせて皮の太鼓っぽく
    },
    "snare": {
        "active": True,
        "vol": 0.6,
        "pattern": "Offbeat", # 弱拍に軽く
        "dsp": ["reverb_short"]
    },
    # メインメロディ：フルート
    "vocal_lead": {
        "active": True,
        "vol": 1.0,
        "wave": "celtic_flute",
        "pattern": "VocalMain",
        "style": "Legato",      # 滑らかに繋ぐ
        "dsp": ["delay", "reverb_long"]
    },
    # 伴奏：ハープのアルペジオ
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.8,
        "wave": "harp",
        "pattern": "Sixteen",
        "style": "Up",          # ポロロンと上がるアルペジオ
        "dsp": ["wide"],
        "div": 16
    },
    # 裏メロ・ドローン：バグパイプやフィドル
    "pad_L": {
        "active": True,
        "vol": 0.5,
        "wave": "bagpipe",      # 持続音でドローン効果
        "pattern": "LongNote",
        "dsp": ["reverb_long"]
    },
    "guitar_riff": {
        "active": True,
        "vol": 0.7,
        "wave": "fiddle",
        "pattern": "Eight",     # 刻むフィドル
        "style": "Rand",
        "dsp": ["reverb_short"]
    },
    # ベースは控えめに
    "bass_sub": {
        "active": True,
        "vol": 0.8,
        "wave": "sin",          # アコースティックベースっぽくシンプルに
        "pattern": "Root"
    }
}

GENRE_STYLES["Insane V-Core"] = {
    "engine": {
        "bpm": 180,                 # 激しい疾走曲（「残-ZAN-」や「Filth」イメージ）
                                    # 重い曲なら140くらいに下げてください
        "swing_amount": 0.0,
        "sc_strength": 0.6,         # ベースとキックを馴染ませる
        "vocal_formant_shift": 0.9, # 京(Dir)やルキ(GazettE)のような、低めの色気ある声
        "reverb_mix": 0.35,         # ダークな空間
        "comp_drive": 1.2           # 全体を潰して音圧を稼ぐ
    },
    "common": {
        "dsp": ["comp"]             # マスターコンプ必須
    },
    
    # リズム隊：メタル仕様
    "kick": {
        "active": True,
        "vol": 1.4,                 # キックは最強に
        "pattern": "RockRiff",      # ドコドコドコドコ
        "dsp": ["comp", "eq"],
        "eq_params": {"low": 4.0, "mid": -2.0, "high": 2.0} # ドンシャリ（低音とアタック強調）
    },
    "snare": {
        "active": True,
        "vol": 1.2,
        "pattern": "Backbeat",      # 抜けるスネア
        "dsp": ["high_pass", "comp", "reverb_short"] # カン！という硬い音
    },
    "cymbal_crash": {
        "active": True,
        "vol": 0.9,
        "pattern": "CrashOne",
        "dsp": ["wide"]
    },

    # 弦楽器隊：轟音の壁
    "bass_sub": {
        "active": True,
        "role": "bass_sub",
        "vol": 1.3,
        "wave": "saw",              # 輪郭のあるSaw波形
        "pattern": "Sixteen",       # 高速ルート弾き
        "style": "Normal",
        "dsp": ["dist", "lpf", "comp"], # 歪ませて太くする
        "eq_params": {"low": 2.0, "mid": 1.0, "high": 0.0}
    },
    "guitar_riff": {
        "active": True,
        "role": "guitar_riff",
        "vol": 1.1,
        "wave": "heavy_guitar",     # 【必須】ヘヴィギター
        "pattern": "Sixteen",       # 高速カッティング
        "style": "Normal",
        "dsp": ["dist", "wide", "eq"],
        "eq_params": {"low": 1.5, "mid": 0.5, "high": 1.5}, # 壁のような音圧
        "pan": 0.0                  # センターで存在感
    },
    
    # 上モノ：不穏なシンセ・同期音
    "pad_L": {
        "active": True,
        "role": "pad_L",
        "vol": 0.6,
        "wave": "saw",
        "pattern": "LongNote",
        "dsp": ["lpf", "sidechain", "reverb_long"] # 唸るような背景音
    },
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.7,
        "wave": "sqr",              # 攻撃的な矩形波
        "pattern": "Sixteen",
        "style": "Rand",            # カオスなフレーズ
        "dsp": ["dist", "delay"]
    },

    # ボーカル：感情的・スクリーム
    "vocal_lead": {
        "active": True,
        "vol": 1.2,
        "wave": "adult",            # 太い声
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["comp", "dist", "delay"], # 歪み(dist)を入れてシャウト感を出す
        "vocal_params": {
            "attack": 0.02,         # アタック速め
            "release": 0.1,
            "vibrato_depth": 0.08,  # ビブラート強め
            "scoop_amount": 0.3,    # しゃくり多用（V系歌唱）
            "breath_vol": 0.2,      # 吐息多め
            "harmonic_mix": 0.3     # 倍音を足してざらつかせる
        }
    },
    "vocal_harmony": {
        "active": True,
        "vol": 0.8,
        "dsp": ["wide", "detune", "pitch_shift_low"] # 低音ハモリで厚みを出す
    }
}

GENRE_STYLES["Janne Rock"] = {
    "engine": {
        "bpm": 175,                 # 「Lunatic Gate」や「Vampire」のような疾走感
        "swing_amount": 0.0,        # 完全なストレートロック
        "sc_strength": 0.1,         # ロックなのでサイドチェインは弱め
        "vocal_formant_shift": 0.92, # yasu風：少し細くて高い、少年のような成分を残した声
        "reverb_mix": 0.3           # ライブハウス〜ホール感
    },
    "common": {
        "dsp": []
    },
    # ドラム：手数多めのツーバス・ロックスタイル
    "kick": {
        "active": True,
        "vol": 1.1,
        "pattern": "RockRiff",      # ドコドコ・ドコドコ
        "dsp": ["comp", "eq"],
        "eq_params": {"low": 2.0, "mid": -1.0, "high": 1.0}
    },
    "snare": {
        "active": True,
        "vol": 1.2,
        "pattern": "Backbeat",      # パァン！と抜けるスネア
        "dsp": ["reverb_short"]
    },
    "hihat": {
        "active": True,
        "pattern": "Eight",         # 8ビートで疾走
        "dsp": []
    },
    "cymbal_crash": {
        "active": True,
        "pattern": "CrashOne",      # 小節頭のアクセント
        "vol": 1.0
    },

    # ベース：ka-yu風に動き回る
    "bass_sub": {
        "active": True,
        "vol": 1.1,
        "wave": "saw",              # 輪郭のくっきりしたソリッドベース
        "pattern": "Sixteen",       # 16分音符でルート弾き＋オカズ
        "style": "Normal",          # ランダムではなくルートを刻みつつ動く
        "dsp": ["dist", "eq"],      # 歪ませてドライブ感を出す
        "eq_params": {"low": 1.0, "mid": 2.0, "high": 0.5} # 中域を持ち上げてラインを聞かせる
    },

    # ギター：you風のテクニカルなリフ
    "guitar_riff": {
        "active": True,
        "role": "guitar_riff",
        "vol": 1.0,
        "wave": "heavy_guitar",     # 歪みギター
        "pattern": "Sixteen",       # 高速カッティング
        "style": "Normal",
        "dsp": ["dist", "reverb_short"]
    },

    # キーボード：kiyo風のチェンバロ/オルガン速弾き
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.9,
        "wave": "organ",            # チェンバロ的な役割
        "pattern": "Sixteen",       # 16分音符のアルペジオ
        "style": "Arp",             # ピロピロと分散和音を奏でる
        "div": 16,
        "dsp": ["delay", "wide"]    # 左右に広げて存在感を出す
    },

    # ボーカル：yasu風のエロティックかつパワフルなハイトーン
    "vocal_lead": {
        "active": True,
        "vol": 1.3,
        "wave": "pop",              # クリアなPop波形を使用
        "pattern": "VocalMain",
        "style": "Legato",          # 滑らかに
        "dsp": ["comp", "delay", "reverb_short"],
        "vocal_params": {
            "attack": 0.02,         # アタックは速く（リズムに遅れない）
            "release": 0.1,
            "vibrato_depth": 0.08,  # ビブラートは強め
            "vibrato_rate": 6.2,    # かなり速いちりめんビブラート
            "scoop_amount": 0.35,   # 下からしゃくり上げる（これがyasu節の肝！）
            "breath_vol": 0.05      # ブレスは控えめに、声の芯を強調
        }
    },
    
    # ハモリ：V系特有の厚み
    "vocal_harmony": {
        "active": True,
        "vol": 0.7,
        "wave": "pop",
        "dsp": ["wide", "detune"]
    }
}
GENRE_STYLES["Crimson Requiem"] = {
    "engine": {
        "bpm": 72,                  # 重く、引きずるようなスローテンポ
        "swing_amount": 0.0,
        "reverb_mix": 0.55,         # 深海にいるような深いリバーブ
        "reverb_decay": 0.8,
        "sc_strength": 0.7,         # 波のうねりのような強いサイドチェイン
        "vocal_formant_shift": 0.88 # 低めの、祈るような太い声
    },
    "common": {
        "dsp": ["reverb_long"]      # 全トラックを深い霧で包む
    },
    
    # リズム：重い一撃と、遠くで鳴る金属音
    "kick": {
        "active": True,
        "vol": 1.3,
        "pattern": "Clave",         # ドン……ドン…（心臓の鼓動）
        "dsp": ["lpf", "comp"]      # こもらせて重低音重視
    },
    "snare": {
        "active": True,
        "vol": 1.0,
        "pattern": "Backbeat",
        "dsp": ["reverb_long", "dist"] # 爆発音のようなスネア
    },
    "impact": {
        "active": True,
        "role": "impact",
        "pattern": "CrashOne",
        "vol": 0.9,
        "dsp": ["delay", "wide"]    # 波しぶきのような衝撃音
    },

    # 旋律：月光（ピアノ）とサイレン（リード）
    "piano_main": {
        "active": True,
        "role": "piano_main",
        "vol": 0.9,
        "wave": "legacy_piano",     # 繊細なピアノ
        "pattern": "Sixteen",
        "style": "Arp",             # 水面が光るようなアルペジオ
        "dsp": ["reverb_short"]
    },
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.7,
        "wave": "saw",              # 鋭い音
        "pattern": "LongNote",      # ずっと鳴り続ける
        "style": "Glides",          # 【重要】音程をずり上げて「サイレン」を表現
        "dsp": ["dist", "wide", "delay"]
    },
    
    # 激情：紅蓮の炎（歪んだギターとベース）
    "bass_sub": {
        "active": True,
        "vol": 1.1,
        "wave": "heavy_guitar",     # ベース域でギターを鳴らす轟音
        "pattern": "Eight",
        "style": "Root",
        "dsp": ["dist", "lpf"]
    },
    "guitar_riff": {
        "active": True,
        "vol": 0.8,
        "wave": "heavy_guitar",
        "pattern": "RockRiff",
        "dsp": ["dist", "bitcrush"] # 叫びのようなノイズギター
    },

    # ボーカル：鎮魂の祈りから叫びへ
    "vocal_lead": {
        "active": True,
        "vol": 1.2,
        "wave": "adult",            # 大人びた声
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["reverb_long", "delay", "comp"],
        "vocal_params": {
            "attack": 0.1,          # ゆったり入る
            "release": 0.3,         # 余韻を残す
            "vibrato_depth": 0.12,  # 深い悲しみのビブラート
            "vibrato_rate": 4.5,    # ゆっくり揺らす
            "breath_vol": 0.2,      # 吐息多め
            "scoop_amount": 0.2
        }
    }
}

# ==========================================
# New Presets: Kawaii Future & Electro Pop
# ==========================================
GENRE_STYLES["KawaiiFuture"] = {
    "engine": {
        "bpm": 170,
        "sc_strength": 0.85, # Heavy Sidechain
        "vocal_formant_shift": 1.35, # Chipmunk Voice
        "bitcrush_rate": 0.2,
        "reverb_mix": 0.35
    },
    "common": {"dsp": []},
    "kick": {"pattern": "4onFloor", "vol": 1.0},
    "snare": {"pattern": "Backbeat", "vol": 0.9},
    "hihat": {"pattern": "Sixteen", "vol": 0.7},
    "bass_sub": {"pattern": "Sixteen", "vol": 1.2, "wave": "bass"},
    "vocal_lead": {
        "active": True,
        "dsp": ["wide", "delay"],
        "vol": 1.1,
        "vocal_params": {
            "breath_vol": 0.08, # Airy
            "whisper_amount": 0.2,
            "vib_depth": 0.03
        }
    },
    "strings_high": {"active": True, "dsp": ["reverb_long", "wide"], "vol": 0.6},
    "arp_quint": {"active": True, "style": "Up", "vol": 0.7},
    "pad_L": {"active": True, "vol": 0.6},
    "pad_R": {"active": True, "vol": 0.6}
}

GENRE_STYLES["Sugarless Future"] = {
    "engine": {
        "bpm": 128,                 # エレクトロ・ハウスの黄金テンポ
        "sc_strength": 0.85,        # 【重要】サイドチェインを強烈にかけて「うねり」を作る
        "sc_release": 0.1,          # 戻りを速くしてキビキビさせる
        "bitcrush_rate": 0.3,       # 全体に薄くデジタルノイズを乗せる
        "bitcrush_depth": 12,       # 少し粗い解像度
        "vocal_formant_shift": 1.1, # 少し無機質でキュートな声質
        "swing_amount": 0.05        # ほぼジャスト、わずかにハネる
    },
    "common": {
        "dsp": ["comp"]             # 全体をコンプでパツパツに潰す
    },
    # リズム：硬くて重い「ドン・チー・ドン・チー」
    "kick": {
        "active": True,
        "vol": 1.4,                 # キックが主役
        "pattern": "4onFloor",
        "dsp": ["dist", "comp", "eq"],
        "eq_params": {"low": 3.0, "mid": 0.0, "high": 1.0} # 低音ブースト
    },
    "hihat": {
        "active": True,
        "vol": 0.8,
        "pattern": "Offbeat",       # 裏打ち（チー、チー）を強調
        "dsp": ["high_pass"]
    },
    "snare": {
        "active": True,
        "pattern": "Backbeat",      # 2拍4拍のクラップ/スネア
        "dsp": ["wide", "bitcrush"] # 左右に広げてザラつかせる
    },
    
    # ベース：FM音源で「ブリブリ」言わせる
    "bass_sub": {
        "active": True,
        "role": "bass_sub",
        "vol": 1.1,
        "wave": "fm_bass",          # 【重要】FMベースを選択
        "pattern": "Sixteen",       # 16分音符で刻む
        "style": "Octave",          # 【重要】オクターブ奏法（ド・ド↑・ド・ド↑）
        "div": 16,
        "dsp": ["dist", "lpf"]      # 歪ませてフィルターを通す
    },
    
    # リード：ピコピコしたプラック音
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.8,
        "wave": "pluck",            # 減衰音
        "pattern": "Sixteen",
        "style": "Rand",            # ランダムなフレーズでテクノ感
        "div": 16,
        "dsp": ["delay", "bitcrush"]
    },
    
    # ボーカル：チョップされた素材のように
    "vocal_lead": {
        "active": True,
        "vol": 1.0,
        "wave": "pop",
        "pattern": "VocalMain",
        "style": "Rand",            # メロディを歌うより、断片的に鳴らす
        "div": 8,                   # 細かく刻む
        "dsp": ["dist", "delay", "bitcrush"], # 歪みとビットクラッシュで加工
        "vocal_params": {
            "attack": 0.01,         # アタック最速（楽器的に）
            "release": 0.1,         # 余韻短く
            "vibrato_depth": 0.0,   # ビブラートなし（ロボット的）
            "breath_vol": 0.0       # ブレスなし
        }
    },
    
    # 装飾：高音のキラキラ
    "arp_quint": {
        "active": True,
        "vol": 0.6,
        "wave": "sqr",              # ファミコンっぽい矩形波
        "pattern": "Sixteen",
        "style": "Up",
        "dsp": ["wide", "delay"]
    }
}

GENRE_STYLES["ElectroPopJP"] = {
    "engine": {
        "bpm": 132,
        "sc_strength": 0.4,
        "vocal_formant_shift": 1.05, # Slight Brightness
        "reverb_mix": 0.15
    },
    "common": {"dsp": []},
    "kick": {"pattern": "4onFloor", "vol": 1.0},
    "snare": {"pattern": "Backbeat", "vol": 0.9},
    "hihat": {"pattern": "Offbeat", "vol": 0.8},
    "bass_sub": {"pattern": "Sixteen", "vol": 0.9, "wave": "sqr"},
    "lead_main": {"active": True, "style": "Up", "vol": 0.8},
    "vocal_lead": {
        "dsp": ["comp"],
        "vol": 1.0,
        "vocal_params": {
            "breath_vol": 0.02,
            "whisper_amount": 0.0,
            "vib_depth": 0.01
        }
    },
    "strings_high": {"active": False}
}

GENRE_STYLES["4s4ki"] = {
    "engine": {
        "bpm": 160,
        "sc_strength": 0.8,         # キックとベースのぶつかり合いを強調
        "bitcrush_rate": 0.35,      # 全体にデジタルな荒れ（グリッチ感）を足す
        "bitcrush_depth": 10,       # 解像度を落としてディストピア感を演出
        "vocal_formant_shift": 1.15,# 子供っぽさと無機質さの中間
        "reverb_mix": 0.25          # 空間は広めに
    },
    "common": {
        "dsp": ["comp", "lpf_mod"]  # LPFでフィルタが開閉するような動きがあるとベスト
    },
    "kick": {
        "vol": 1.4,                 # 爆音
        "pattern": "TrapKick",      # 4つ打ちではなくトラップノリでハズす
        "dsp": ["dist", "comp"]     # 歪ませて「割れた」音にする
    },
    "snare": {
        "vol": 1.1,
        "pattern": "Backbeat",
        "dsp": ["bitcrush"]         # スネアもザラつかせる
    },
    "hihat": {
        "vol": 0.8,
        "pattern": "Rand",          # チチチ…という規則性の中にランダムなロールを入れる
        "dsp": ["wide"]
    },
    "bass_sub": {
        "active": True,
        "vol": 1.3,
        "pattern": "Sixteen",
        "style": "Glides",          # ピッチベンド（グリス）を多用する
        "dsp": ["dist", "width"]    # 歪み＋広がりで空間を埋める
    },
    "guitar_riff": {
        "active": True,             # エモ要素：サビ裏で鳴る轟音ギター
        "vol": 0.9,
        "pattern": "PowerChord",
        "dsp": ["dist", "reverb_short"]
    },
    "vocal_lead": {
        "active": True,
        "vol": 1.2,
        "wave": "whisper",          # ウィスパーボイスベース
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["dist", "delay", "autotune_hard"], # 歪み＋強烈なオートチューン
        "vocal_params": {
            "breath_vol": 0.15,     # 吐息多め（切なさ）
            "whisper_amount": 0.4,  # ささやき成分強め
            "vib_depth": 0.0        # ビブラートは機械的に0にする
        }
    },
    "vocal_harmony": {
        "active": True,
        "vol": 0.6,
        "dsp": ["wide", "pitch_shift_low"] # 低音のダブりを入れて「悪魔的」な雰囲気に
    },
    "pad_L": {
        "active": True,
        "vol": 0.7,
        "pattern": "LongNote",
        "dsp": ["reverb_long", "sidechain"] # 浮遊感のあるパッド
    },
    "lead_main": {
        "active": True,
        "vol": 0.8,
        "pattern": "Arp",           # ピコピコしたアルペジオ
        "style": "Rand",
        "dsp": ["bitcrush"]
    },
    "glitch": {
        "active": True,             # ランダムなスタッター効果
        "intensity": 0.8
    }
}

GENRE_STYLES["J-R&B (Utada)"] = {
    "engine": {
        "bpm": 92, # Travel/Distance era typical
        "sc_strength": 0.3, # Light SC
        "swing_amount": 0.1,
        "reverb_mix": 0.22,
        "reverb_decay": 0.6
    },
    "vocal_lead": {
        "active": True,
        "vol": 1.1,
        "wave": "utada",
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["comp", "delay", "reverb_short"],
        "vocal_params": {
            "breath_vol": 0.08, # Moderate breath
            "vib_depth": 0.015,
            "vib_rate": 5.2,
            "whisper_amount": 0.0
        }
    },
    "vocal_harmony": {
        "active": True,
        "vol": 0.6, # -8dB relative to lead
        "wave": "utada",
        "pan": 0.3, # Pan 30%
        "dsp": ["wide", "reverb_long"],
        "vocal_params": {
            "formant_shift": 1.02 # +2% Formant
        }
    }
}
GENRE_STYLES["Psychedelico Folk"] = {
    "engine": {
        "bpm": 98,                  # 代表曲「Last Smile」や「Your Song」のようなミッドテンポ
        "swing_amount": 0.15,       # 独特のグルーヴ感（タメ）
        "sc_strength": 0.2,         # サイドチェインは弱め（生演奏のダイナミクス重視）
        "vocal_formant_shift": 0.92,# Kumi風：ハスキーで中性的な低い声
        "reverb_mix": 0.25          # 乾いたヴィンテージ・ルーム感
    },
    "common": {
        "dsp": ["comp"]             # 全体をアナログコンプでまとめる
    },
    
    # リズム隊：ヴィンテージ・ブレイクビーツ
    "kick": {
        "active": True,
        "vol": 1.1,
        "pattern": "TrapKick",      # あえてヒップホップ的なパターンでループ感を出す
        "dsp": ["lpf", "comp", "bitcrush"], # 少し汚してサンプリングっぽくする
        "bitcrush_params": {"rate": 0.1, "depth": 12},
        "eq_params": {"low": 2.0, "mid": 0.5, "high": 0.0}
    },
    "snare": {
        "active": True,
        "vol": 1.0,
        "pattern": "Backbeat",
        "dsp": ["high_pass", "comp"] # タイトで乾いたスネア（カン！という音）
    },
    "hihat": {
        "active": True,
        "pattern": "Sixteen",       # シェイカーのように細かく刻む
        "vol": 0.6,
        "dsp": ["wide"]
    },

    # 楽器隊：リフ主体のフォークロック
    "guitar_riff": {
        "active": True,
        "role": "guitar_riff",
        "vol": 1.0,
        "wave": "heavy_guitar",     # 歪みギター
        "pattern": "RockRiff",      # カントリー/フォーク調のリフ
        "style": "Normal",
        "dsp": ["dist", "reverb_short"],
        "eq_params": {"low": 0.0, "mid": 2.0, "high": 0.5} # 中域を持ち上げてヴィンテージアンプ風に
    },
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 0.8,
        "wave": "organ",            # オルガンでレトロな厚みを足す
        "pattern": "LongNote",
        "style": "Normal",
        "dsp": ["reverb_long", "wide"]
    },
    "arp_quint": { 
        "active": True,
        "role": "arp_quint",    # アコギのストローク役
        "vol": 0.7,
        "wave": "karplus",          # 弦を弾く音
        "pattern": "Sixteen",
        "style": "Up",
        "div": 12,                  # 3連符っぽく混ぜて揺らぎを出す
        "dsp": ["wide"]
    },

    # ベース：うねるグルーヴ
    "bass_sub": {
        "active": True,
        "vol": 1.1,
        "wave": "saw",
        "pattern": "Sixteen",
        "style": "Normal",
        "dsp": ["lpf", "comp"],
        "eq_params": {"low": 1.5, "mid": 1.0, "high": 0.0}
    },

    # ボーカル：英語混じりの気だるげなスタイル
    "vocal_lead": {
        "active": True,
        "vol": 1.2,
        "wave": "adult",            # 大人びた声質
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["comp", "delay", "high_pass"], # ローカットしてオケに馴染ませる
        "vocal_params": {
            "attack": 0.05,
            "release": 0.1,
            "vibrato_depth": 0.03,  # ビブラートは浅く
            "scoop_amount": 0.15,   # 少ししゃくる
            "breath_vol": 0.1,      # ブレス感
            "whisper_amount": 0.1
        }
    },
    "vocal_harmony": { # ダブルトラッキング効果（声を重ねる）
        "active": True,
        "vol": 0.9,
        "wave": "adult",
        "pan": 0.3,
        "dsp": ["wide", "detune"],
        "vocal_params": {
            "formant_shift": 0.94   # メインと微妙に声質を変える
        }
    }
}
GENRE_STYLES["Soulful Diva"] = {
    "engine": {
        "bpm": 76,                  # 壮大なバラードテンポ（「366日」などを意識）
        "swing_amount": 0.1,        # 少しだけタメを作る
        "vocal_formant_shift": 0.85,# 【最重要】フォルマントを下げて「太く、胸に響く女性の声」にする
        "reverb_mix": 0.45,         # ホール鳴り
        "comp_drive": 1.1
    },
    "common": {
        "dsp": ["comp"]
    },
    "kick": {"active": True, "vol": 1.1, "pattern": "TrapKick", "dsp": ["lpf", "comp"], "note_len": 1.5},
    "snare": {"active": True, "vol": 1.0, "pattern": "Backbeat", "dsp": ["reverb_short"]},
    "hihat": {"active": True, "vol": 0.6, "pattern": "Eight", "note_len": 0.8},
    
    "piano_main": {
        "active": True, "role": "piano_main", "vol": 1.0, "wave": "legacy_piano", 
        "pattern": "Sixteen", "style": "Arp", "dsp": ["reverb_short", "wide"], "note_len": 2.0
    },
    "bass_sub": {
        "active": True, "vol": 1.0, "wave": "sin", "pattern": "Root", "dsp": ["lpf"], "note_len": 2.0
    },

    # 圧倒的ソウルボーカル
    "vocal_lead": {
        "active": True,
        "vol": 1.4,                 # オケをねじ伏せる音量
        "wave": "adult",            # 「pop」より太い波形
        "pattern": "VocalMain",
        "style": "Legato",          # 滑らかに繋ぐ
        "dsp": ["comp", "dist", "delay", "reverb_long"], # 【秘密兵器】軽くdist(歪み)を入れて声帯の鳴りを出す
        "note_len": 1.5,            # 声を限界まで張り上げる
        "vocal_params": {
            "attack": 0.15,         # 【重要】ゆっくり立ち上がり…
            "scoop_amount": 0.6,    # 【最重要】強烈に下からずり上げる！（これがソウルの魂）
            "vibrato_depth": 0.15,  # 震えるような深いビブラート
            "vibrato_rate": 4.5,    # ゆったりとした大きな波のビブラート
            "harmonic_mix": 0.4,    # 倍音を足して「声の張り」を強調
            "breath_vol": 0.25,     # 歌う前の大きなブレス
            "release": 0.3          # 歌い終わりの余韻
        }
    }
}

GENRE_STYLES["Cyber Rebellion"] = {
    "engine": {
        "bpm": 150,                 # 「踊 (Odo)」のような超ハイテンポなEDM
        "swing_amount": 0.0,        # 完全な縦ノリ
        "sc_strength": 0.95,        # 【最重要】TeddyLoid特有の強烈なEDMポンプ（サイドチェイン）
        "sc_release": 0.1,          # ダッキングの戻りを速くしてキレを出す
        "vocal_formant_shift": 1.05,# 少しだけ鋭く、突き刺さるような声色
        "comp_drive": 1.3,          # 全体的に音圧を限界まで突っ込む
        "reverb_mix": 0.2           # リバーブは少なめで「耳元で叫んでる感」を出す
    },
    "common": {
        "dsp": ["comp"]             # マスターコンプで暴力を束ねる
    },
    
    # リズム隊：治安の悪い重低音ダンスビート
    "kick": {
        "active": True,
        "vol": 1.5,                 # キックは全開
        "pattern": "4onFloor",      # 四つ打ち
        "dsp": ["dist", "comp", "eq"],
        "eq_params": {"low": 4.0, "mid": -1.0, "high": 2.0}, # ドンシャリの極み
        "note_len": 0.8             # 長すぎず短すぎず、パンチを最優先
    },
    "snare": {
        "active": True,
        "vol": 1.2,
        "pattern": "Backbeat",
        "dsp": ["bitcrush", "reverb_short"], # ザラッとした攻撃的なスネア
        "note_len": 0.6
    },
    "hihat": {
        "active": True,
        "vol": 0.8,
        "pattern": "Sixteen",
        "style": "Rand",            # トラップミュージック的な「チチチチ！」という細かいロール
        "dsp": ["wide", "high_pass"],
        "note_len": 0.3
    },
    "impact": {
        "active": True,
        "role": "impact",
        "vol": 1.0,
        "pattern": "CrashOne",
        "dsp": ["delay", "wide"]
    },

    # ベース：うねる極悪ウォブルベース
    "bass_sub": {
        "active": True,
        "role": "bass_sub",
        "vol": 1.3,
        "wave": "fm_bass",          # 金属的な響きのFMベース
        "pattern": "Sixteen",
        "style": "Glides",          # 【重要】ピッチを滑らせて「ギュワワ！」といううねりを作る
        "dsp": ["dist", "lpf", "comp"], 
        "note_len": 1.5             # 余韻を伸ばして隙間を埋め尽くす
    },

    # リード：ブラス/プラック系のスタッカートシンセ
    "lead_main": {
        "active": True,
        "role": "lead_main",
        "vol": 1.0,
        "wave": "supersaw",         # 分厚いシンセ
        "pattern": "Sixteen",
        "style": "Rand",            # 予測不能なフレーズ
        "dsp": ["dist", "delay", "bitcrush"],
        "note_len": 0.4             # 【重要】短く切って「パパパパン！」とキレ良く鳴らす
    },
    
    # サブウワモノ：不気味なアルペジオ
    "arp_quint": {
        "active": True,
        "vol": 0.7,
        "wave": "sqr",
        "pattern": "Sixteen",
        "style": "Up",
        "dsp": ["wide", "reverb_long"],
        "note_len": 0.5
    },

    # ボーカル：Ado風の「がなり」「しゃくり」「シャウト」
    "vocal_lead": {
        "active": True,
        "vol": 1.3,
        "wave": "pop",              # 芯のある声
        "pattern": "VocalMain",
        "style": "Legato",
        "dsp": ["comp", "dist", "delay"], # 【重要】ボーカルにも軽く歪み(dist)を乗せてエッジを立てる
        "note_len": 1.0,
        "vocal_params": {
            "attack": 0.01,         # 言葉の立ち上がりを最速に（噛み付くように）
            "release": 0.1,         # スパッと切る
            "vibrato_depth": 0.08,  # ビブラートは激しく
            "vibrato_rate": 6.5,    # 揺れも速く
            "scoop_amount": 0.45,   # 【最重要】強烈に下からしゃくり上げる（Ado特有の歌い回し）
            "harmonic_mix": 0.35,   # 【最重要】倍音を足して「がなり声（Growl）」を表現
            "whisper_amount": 0.1,  # たまに息っぽさを混ぜて不気味に
            "breath_vol": 0.15
        }
    },
    
    # コーラス：悪魔的な低音のダブリング
    "vocal_harmony": {
        "active": True,
        "vol": 0.8,
        "wave": "pop",
        "dsp": ["wide", "pitch_shift_low", "dist"], # 1オクターブ下の歪んだ声（Tot Musica風）
        "vocal_params": {
            "formant_shift": 0.8    # 極端に太い声にする
        },
        "note_len": 1.0
    }
}

SE_PRESETS = {

    # 1. 転調ブチ上げリード（半音階グリス＋FM暴れ）
    "Modulation_Riser": {
        "type": "fm",
        "base_freq": 440,
        "mod_freq": 12,
        "mod_depth": 1800,
        "dur": 1.8,
        "wave": "saw",
        "pitch_env": "chromatic_up",
        "dsp": ["dist", "delay", "wide", "comp"]
    },

    # 2. サビ直前・吸い込み逆再生ボイス
    "Reverse_Vox_Suck": {
        "type": "voice",
        "base_freq": 600,
        "dur": 1.2,
        "reverse": True,
        "formant": "oh",
        "dsp": ["reverb_long", "high_pass", "comp", "wide"]
    },

    # 3. メルトダウン・グリッチ落下（キーに合わせて半音落下）
    "Pitch_Fall_Glitch": {
        "type": "fm",
        "base_freq": 1200,
        "mod_freq": 30,
        "mod_depth": 2000,
        "dur": 0.9,
        "wave": "sqr",
        "pitch_env": "semitone_down",
        "dsp": ["bitcrush", "dist", "delay"]
    },

    # 4. キラキラ転調チャイム（長3度上昇アルペジオ）
    "KeyShift_Chime": {
        "type": "bell",
        "base_freq": 880,
        "dur": 1.4,
        "arp": "major_third_up",
        "dsp": ["reverb_long", "delay", "wide"]
    },

    # 5. ハイパーポンプビルド（サイドチェイン風ノイズ）
    "Pump_Noise_Build": {
        "type": "noise",
        "dur": 2.0,
        "filter": "bpf",
        "sidechain_rate": 4,
        "dsp": ["dist", "comp", "wide"]
    },

    # 6. イントロ掴み・超高域スパイク
    "Crystal_Impact": {
        "type": "fm",
        "base_freq": 4000,
        "mod_freq": 5,
        "mod_depth": 3500,
        "dur": 0.5,
        "wave": "saw",
        "dsp": ["dist", "reverb_long", "wide"]
    },

    # 7. 闇堕ちサブインパクト（マイナー感強調）
    "Dark_Sub_Hit": {
        "type": "saber",
        "base_freq": 48,
        "dur": 1.6,
        "pitch_env": "minor_fall",
        "dsp": ["dist", "comp", "sub_boost"]
    },

    # 8. アニメOP風スタートシグナル
    "Anime_Start_Signal": {
        "type": "fm",
        "base_freq": 1500,
        "mod_freq": 18,
        "mod_depth": 900,
        "dur": 0.6,
        "wave": "sqr",
        "arp": "octave_jump",
        "dsp": ["delay", "reverb_short", "wide"]
    }
}
DEFAULT_SAMPLER_MAPPING = ["Modulation_Riser", "Dark_Sub_Hit", "Pitch_Fall_Glitch", "KeyShift_Chime", "Crystal_Impact", "Anime_Start_Signal"]
# ==========================================
# 7. Preset Manager
# ==========================================
class PresetManager:
    @staticmethod
    def create_preset_data(app_instance):
        current_engine = getattr(app_instance, "engine_config", {})
        data = {
            "meta": {"version": "14.7", "created_at": datetime.datetime.now().isoformat(), "app": "HyperNekoVocaloid"},
            "global": {
                "bpm": app_instance.bpm,
                "structure_str": app_instance.structure_str,
                "chords_map": app_instance.chords_map,
                "seed": app_instance.seed_offset,
                "auto_fill": app_instance.auto_fill,
                "key_offset": getattr(app_instance, "key_offset", 0),
                "lyrics_map": current_engine.get("lyrics_map", {})
            },
            "engine": current_engine,
            "sampler": app_instance.sampler_mapping,
            "tracks": []
        }
        for t in app_instance.tracks:
            track_data = {
                "role": t.spec["role"],
                "vol": t.slider_vol.value,
                "pan": t.slider_pan.value,
                "active": t.is_active,
                "dsp": t.dsp_chain,
                "div": t.spec.get("div", 16),
                "div_map": t.spec.get("div_map", {}),
                "wave": t.spec.get("wave", "sin"),
                "style": t.spec.get("style", "Normal"),
                # ▼▼▼ 追加: スタイルマップを保存 ▼▼▼
                "style_map": t.spec.get("style_map", {}), 
                # ▲▲▲ 追加終わり ▲▲▲
                # ▼▼▼ 追加: セーブデータにカスタムコードを含める ▼▼▼
                "custom_chords": t.spec.get("custom_chords", ""),
                # ▲▲▲ 追加終わり ▲▲▲
                
                "active_sections": t.spec.get("active_sections", []),
                "xy_route": t.spec.get("xy_route", False),
                "pattern_map": t.spec.get("pattern_map", {}),
                "lfo_params": t.spec.get("lfo_params", {}),
                "bitcrush_params": t.spec.get("bitcrush_params", {}),
                "vocal_params": t.spec.get("vocal_params", {})
            }
            data["tracks"].append(track_data)
        return data

    @staticmethod
    def save_preset(filename, data):
        try:
            if not filename.endswith('.json'): filename += '.json'
            with open(filename, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
            return True, f"Saved: {filename}"
        except Exception as e: return False, str(e)

    @staticmethod
    def load_preset(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: return None


# ==========================================
# 8. Global Variables & Spec Builder
# ==========================================
BPM = 160
TRACK_SPECS = []
TOTAL_DURATION = 0
TOTAL_BARS = 0
SECTION_TIMINGS = []
DEFAULT_VERSE = "C#m9, AM7, B13, G#7"
DEFAULT_VERSE2 = "F#m7, B7, Emaj7, C#7"
DEFAULT_BRIDGE = "Dmaj7, E, C#m7, F#m7"

def rebuild_specs(bpm_val, structure_str, chords_map, auto_fill, preset_track_data=None, key_offset=0):
    global BPM, TRACK_SPECS, TOTAL_DURATION, TOTAL_BARS, SECTION_TIMINGS
    TRACK_SPECS.clear()
    SECTION_TIMINGS.clear()
    BPM = bpm_val
    structure_codes = [s.strip().upper() for s in structure_str.split(',') if s.strip()] or ["V"]

    # Auto-fill logic
    verse_prog = chords_map.get('V') or [c.strip() for c in DEFAULT_VERSE.split(',')]
    if not verse_prog or (len(verse_prog) == 1 and not verse_prog[0]): verse_prog = ["C"]
    verse2_prog = chords_map.get('V2') or verse_prog
    bridge_prog = chords_map.get('B') or verse_prog

    def get_section_chords(code):
        chords = chords_map.get(code, [])
        if chords and (len(chords) > 1 or (len(chords) == 1 and chords[0])):
            return chords[:]
        
        if auto_fill:
            if code == 'I': 
                root_chord = verse_prog[0].split('_')[0].replace('"', '').strip()
                info = parse_complex_chord(root_chord)
                return [f"{info['root']}m"] * 4
            elif code == 'C': return verse_prog[:] 
            elif code == 'V2': return verse2_prog[:]
            elif code == 'B': return bridge_prog[:]
            elif code == 'O': 
                return [verse_prog[0].split('_')[0].replace('"', '').strip()] * 4
            return verse_prog[:]

        fallback_len = len(verse_prog) if verse_prog else 4
        return [""] * fallback_len

    # Build Timeline
    current_bar = 0
    full_chords_sequence = []
    for code in structure_codes:
        this_sec_chords = get_section_chords(code)
        SECTION_TIMINGS.append((current_bar, code))
        full_chords_sequence.extend(this_sec_chords)
        current_bar += len(this_sec_chords)

    TOTAL_BARS = current_bar
    TOTAL_DURATION = TOTAL_BARS * 4 * (60 / BPM)
    ALL_SECTIONS = ["I", "V", "V2", "C", "B", "O"]

    # Role Definitions
    default_definitions = [
        # --- Drums ---
        {"role": "kick", "wave": "sin", "vol": 1.2, "pan": 0.0, "col": (1, 0.2, 0.2), "dsp": ["comp", "eq"], "def_pat": "4onFloor", "is_drum": True},
        {"role": "snare","wave": "noise","vol": 1.0,"pan": 0.0, "col": (1, 0.5, 0.2), "dsp": ["reverb_short"], "def_pat": "Backbeat", "is_drum": True},
        {"role": "hihat", "wave": "noise","vol": 0.7,"pan": 0.0, "col": (1, 0.4, 1), "dsp": ["wide"], "def_pat": "Eight", "is_drum": True},
        {"role": "cymbal_crash", "wave": "noise", "vol": 0.9, "pan": 0.1, "col": (1, 0.9, 0.2), "dsp": ["wide"], "def_pat": "CrashOne", "is_drum": True},
        {"role": "cymbal_ride", "wave": "noise", "vol": 0.6, "pan": 0.3, "col": (1, 0.9, 0.4), "dsp": ["reverb_short"], "def_pat": "RideSwing", "is_drum": True},
        {"role": "glitch", "wave": "saw", "vol": 0.5, "pan": 0.0, "col": (0.8, 0.8, 0.8), "dsp": ["wide", "dist", "bitcrush"], "def_pat": "Empty", "def_style": "Rand", "is_drum": True},
        {"role": "noise_swp", "wave": "noise", "vol": 0.3, "pan": 0.0, "col": (0.5, 0.5, 0.5), "dsp": ["wide"], "def_pat": "Sixteen", "def_style": "Normal", "is_drum": True},
        {"role": "impact", "wave": "noise", "vol": 0.9, "pan": 0.0, "col": (1, 1, 1), "dsp": ["reverb_long"], "def_pat": "Clave", "def_style": "Normal", "is_drum": True},
        {"role": "piano_main", "wave": "piano", "vol": 0.9, "pan": 0.0, "col": (0.2, 0.2, 0.8), "dsp": ["reverb_short"], "def_pat": "Sixteen", "def_style": "Normal"},
        {"role": "organ_pad", "wave": "organ", "vol": 0.8, "pan": 0.0, "col": (0.6, 0.4, 0.2), "dsp": ["reverb_long"], "def_pat": "LongNote", "def_style": "Normal"},
        {"role": "musicbox_arp", "div": 16, "wave": "musicbox", "vol": 0.7, "pan": 0.3, "col": (1.0, 0.8, 0.9), "dsp": ["delay"], "def_pat": "Sixteen", "def_style": "Rand"},

        # --- Vocal ---
        {"role": "vocal_lead", "wave": "pop", "vol": 1.1, "pan": 0.0, "col": (1, 0.8, 0.2), "dsp": ["comp", "delay"], "def_pat": "VocalMain", "def_style": "Rand", "xy_route": True},
        {"role": "vocal_harmony", "wave": "pop", "vol": 0.7, "pan": -0.2, "col": (1, 0.6, 0.4), "dsp": ["wide", "reverb_long"], "def_pat": "VocalMain", "def_style": "Normal"},

        # --- Bass ---
        {"role": "bass_sub", "wave": "saw", "vol": 0.9, "pan": 0.0, "col": (0.8, 0.2, 0.2), "dsp": ["lpf", "comp"], "def_pat": "Sixteen", "def_style": "Normal"}, 
        {"role": "bass_pedal", "div": 4, "wave": "saw", "vol": 0.7, "pan": 0.0, "col": (0.6, 0.1, 0.1), "dsp": ["lpf"], "def_pat": "Sixteen", "def_style": "Normal"},

        # --- Instruments ---
        {"role": "guitar_riff", "wave": "karplus", "vol": 1.0, "pan": 0.25, "col": (1, 0.6, 0.0), "dsp": ["dist", "reverb_short"], "def_pat": "RockRiff", "def_style": "Normal", "is_drum": False},
        {"role": "strings_high", "div": 4, "wave": "saw", "vol": 0.6, "pan": 0.4, "col": (0.8, 0.5, 1.0), "dsp": ["reverb_long", "wide"], "def_pat": "LongNote", "def_style": "Normal"},
        {"role": "pad_L", "wave": "saw", "vol": 0.5, "pan": -0.6,"col": (0.2, 0.4, 1), "dsp": ["reverb_long"], "def_pat": "Sixteen", "def_style": "Normal"}, 
        {"role": "pad_R", "wave": "saw", "vol": 0.5, "pan": 0.6, "col": (0.2, 0.4, 1), "dsp": ["reverb_long"], "def_pat": "Sixteen", "def_style": "Normal"},
        {"role": "epiano", "div": 8, "wave": "sin", "vol": 0.6, "pan": 0.2, "col": (0.5, 0.3, 0.8), "dsp": ["reverb_short"], "def_pat": "Sixteen", "def_style": "Normal"},
        {"role": "lead_main", "wave": "sqr", "vol": 0.7, "pan": 0.0, "col": (0.2, 1, 0.8), "dsp": ["delay", "dist"], "def_pat": "Sixteen", "def_style": "Rand"},
        {"role": "synth_lead", "wave": "saw", "vol": 1.1, "pan": 0.0, "col": (1.0, 0.3, 0.0), "dsp": ["delay", "reverb_short"], "def_pat": "Melody", "def_style": "Pentatonic"}, 
        {"role": "arp_quint", "div": 20, "wave": "sin", "vol": 0.5, "pan": 0.4, "col": (0.4, 1, 1), "dsp": ["delay"], "def_pat": "Sixteen", "def_style": "Up"},
        {"role": "arp_trip", "div": 12, "wave": "sin", "vol": 0.6, "pan": -0.4,"col": (1, 0.4, 1), "dsp": ["delay"], "def_pat": "Sixteen", "def_style": "Down"},
        {"role": "saw_arp", "div": 16, "wave": "saw", "vol": 0.55,"pan": -0.3,"col": (1, 0.8, 0.2), "dsp": ["delay"], "def_pat": "Sixteen", "def_style": "Up"},
    ]

    default_active_sections_map = {
        "kick": ["V", "V2", "C", "B"], "snare": ["V", "V2", "C", "B"], 
        "vocal_lead": ["V", "V2", "C", "B"], 
        "vocal_harmony": ["C", "O"],
        "lead_main": ["I", "C", "O"],
        # ▼▼▼ ここに追加！ ▼▼▼
        "synth_lead": ["I", "C", "B", "O"],
        # ▲▲▲ ここまで ▲▲▲
        "piano_main": ["I", "V", "C", "O"],
        "organ_pad": ["C", "B", "O"],
        "musicbox_arp": ["I", "B", "O"], 
        "pad_L": ALL_SECTIONS[:], "pad_R": ALL_SECTIONS[:], 
        "bass_sub": ["V", "C", "B"], "guitar_riff": ["V", "C", "B"],
        "saw_arp": ["C", "O"], "impact": ["I", "C", "B", "O"],
        "cymbal_crash": ["I", "C", "O", "B"],
        "cymbal_ride": ["C", "O"],
        "strings_high": ["C", "B", "O"],
        "noise_swp": ["B", "O"]
    }

    for def_spec in default_definitions:
        role = def_spec["role"]
        current_spec = def_spec.copy()
        
        loaded_settings = {}
        if preset_track_data:
            match = next((t for t in preset_track_data if t["role"] == role), None)
            if match: loaded_settings = match
        
        for k in ["vol", "pan", "div", "div_map", "wave", "style", "dsp", "active_sections", "xy_route", "pattern_map", "lfo_params", "bitcrush_params", "vocal_params"]:
            if k in loaded_settings: current_spec[k] = loaded_settings[k]
        
        current_spec["initial_active"] = loaded_settings.get("active", True)
        if "active_sections" not in current_spec:
            current_spec["active_sections"] = default_active_sections_map.get(role, ALL_SECTIONS[:])
        
        current_spec["key_offset"] = key_offset
        current_spec["lfo_params"] = current_spec.get("lfo_params", {"active": False, "rate": 1.0, "depth": 0.0})
        current_spec["bitcrush_params"] = current_spec.get("bitcrush_params", {"rate": 0.2, "depth": 12})
        if "style" not in current_spec: current_spec["style"] = def_spec.get("def_style", "Normal")
        
        if "vocal" in role and "vocal_params" not in current_spec:
             current_spec["vocal_params"] = {
                 "attack": 0.05, "release": 0.1, "vibrato_depth": 0.05, "vibrato_rate": 5.0, "reverb_send": 0.0
             }

        if "div_map" not in current_spec:
            default_div = current_spec.get("div", 16)
            current_spec["div_map"] = {sec: default_div for sec in ALL_SECTIONS}
            # ▼▼▼ 追加: style_map の初期化 ▼▼▼
        if "style_map" not in current_spec:
            default_style = current_spec.get("style", "Normal")
            # プリセットロード時などで style_map がない場合は、現在の style を全セクションに適用
            current_spec["style_map"] = {sec: default_style for sec in ALL_SECTIONS}
        # ▲▲▲ 追加終わり ▲▲▲

        if "pattern_map" not in current_spec:
            current_spec["pattern_map"] = {}
            def_pat_key = def_spec.get("def_pat", "Sixteen")
            base_pat = RHYTHM_TEMPLATES.get(def_pat_key, RHYTHM_TEMPLATES["Sixteen"])[:]
            for sec in ALL_SECTIONS:
                current_spec["pattern_map"][sec] = base_pat[:]

        current_spec["chords"] = full_chords_sequence[:]
        TRACK_SPECS.append(current_spec)

# ==========================================
# 9. Audio Generation Core (Mixing & Mastering Enhanced)
# ==========================================

# --- ADSR / Anti-aliasing Helpers ---
def _exp_curve(steps, start, end, curvature=5.0):
    if steps <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, 1, steps, dtype=np.float32)
    if curvature <= 0.1: return np.linspace(start, end, steps, dtype=np.float32)
    if start < end:
        raw = 1.0 - np.exp(-curvature * t)
        norm = raw / (1.0 - np.exp(-curvature))
        return start + norm * (end - start)
    else:
        raw = np.exp(-curvature * t)
        norm = (raw - np.exp(-curvature)) / (1.0 - np.exp(-curvature))
        return end + norm * (start - end)

def apply_env(wave_data, dur, atk, dec, sus, rel):
    l = len(wave_data)
    a_s, d_s, r_s = int(atk * FS), int(dec * FS), int(rel * FS)
    total_env = a_s + d_s + r_s
    if total_env > l:
        factor = l / total_env if total_env > 0 else 1
        a_s, d_s, r_s = int(a_s*factor), int(d_s*factor), int(r_s*factor)
    s_len = max(0, l - a_s - d_s - r_s)
    
    env_parts = []
    if a_s > 0: env_parts.append(_exp_curve(a_s, 0.0, 1.0, 4.0))
    if d_s > 0: env_parts.append(_exp_curve(d_s, 1.0, sus, 5.0))
    if s_len > 0: env_parts.append(np.full(s_len, sus, dtype=np.float32))
    if r_s > 0: env_parts.append(_exp_curve(r_s, sus, 0.0, 5.0))
    
    if not env_parts: return wave_data 
    env = np.concatenate(env_parts)
    if len(env) < l: env = np.pad(env, (0, l - len(env)), 'constant')
    else: env = env[:l]
    return wave_data * env

def poly_blep(t, dt):
    t = t - np.floor(t)
    blep = np.zeros_like(t)
    mask_start = t < dt
    if np.any(mask_start):
        t_s = t[mask_start] / dt
        blep[mask_start] = 2 * t_s - t_s**2 - 1.0
    mask_end = t > (1.0 - dt)
    if np.any(mask_end):
        t_e = (t[mask_end] - 1.0) / dt
        blep[mask_end] = 2 * t_e + t_e**2 + 1.0
    return blep

# 波形生成関数内のイメージ
def generate_synth_lead_wave(freq, duration, prev_freq=None, glide_time=0.15):
    # ポルタメント（グライド）の処理
    if prev_freq and glide_time > 0:
        # サンプル数に換算
        glide_samples = int(sample_rate * glide_time)
        total_samples = int(sample_rate * duration)
        
        # グライド部分は前の周波数から今の周波数へ線形補間（linspace等を使用）
        # ※np.linspace等で周波数の配列を作り、それに従って位相を進める
        freq_array = np.ones(total_samples) * freq
        if glide_samples > total_samples:
            glide_samples = total_samples
            
        freq_array[:glide_samples] = np.linspace(prev_freq, freq, glide_samples)
        
        # 周波数配列から波形を生成する処理へ...
    else:
        # 通常の単一ピッチ生成
        pass

def generate_supersaw(freq, dur, detune=0.004, voices=5):
    samples = int(FS * dur)
    if samples <= 0: return np.array([], dtype=np.float32)
    mix = np.zeros(samples, dtype=np.float32)
    t_base = np.arange(samples, dtype=np.float32) / FS
    for i in range(voices):
        delta = (i - (voices // 2)) * detune
        vol = 1.0 if delta == 0 else 0.7
        f_val = freq * (1.0 + delta)
        phase = t_base * f_val
        phase = phase - np.floor(phase)
        dt = f_val / FS
        naive = 2.0 * phase - 1.0
        blep = poly_blep(phase, dt)
        mix += (naive - blep) * vol
    return mix / (voices * 0.55)

def generate_karplus_custom(freq, dur, decay=0.996):
    if freq <= 0: freq = 440.0
    N = int(FS / freq)
    N = max(2, N)
    total = int(FS * dur)
    buf = np.random.uniform(-1, 1, N).astype(np.float32)
    out = np.zeros(total, dtype=np.float32)
    curr = buf.copy()
    w_len = min(total, N)
    out[:w_len] = curr[:w_len]
    c = w_len
    while c < total:
        prev = out[c-N:c]
        w_len = min(total-c, N)
        new_blk = 0.5 * (prev + np.roll(prev, 1)) * decay
        out[c:c+w_len] = new_blk[:w_len]
        c += w_len
    return out

def fast_osc_custom(freq, dur, type="sin"):
    samples = int(FS * dur)
    if samples <= 0: return np.array([], dtype=np.float32)
    if type == "noise": return np.random.uniform(-1, 1, samples).astype(np.float32)
    t = np.arange(samples, dtype=np.float32) / FS
    phase = t * freq
    phase = phase - np.floor(phase)
    dt = freq / FS
    if type == "sin": return np.sin(2 * np.pi * freq * t).astype(np.float32)
    elif type == "saw":
        naive = 2.0 * phase - 1.0
        blep = poly_blep(phase, dt)
        return (naive - blep).astype(np.float32)
    elif type == "sqr":
        naive = np.where(phase < 0.5, 1.0, -1.0)
        blep1 = poly_blep(phase, dt)
        phase_shifted = phase + 0.5
        phase_shifted = phase_shifted - np.floor(phase_shifted)
        blep2 = poly_blep(phase_shifted, dt)
        return (naive + blep1 - blep2).astype(np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

# --- Updated Generator (Fixed by Grok & Merged) ---
# ==========================================
# generate_poly_stem
# ==========================================
def compose_track_sequence(spec, seed_offset, engine_config, bpm):
    """
    Phase 1: Composition (AI Composer Evolved + Lyric Control Ultimate)
    Supported Syntax:
      Chars: Normal lyric
      .    : Rest (No sound, consumes time)
      ^    : Pitch Up (+1 interval, stackable)
      v    : Pitch Down (-1 interval, stackable)
      * : Force Root Pitch
      +    : Split/Connect (e.g. "a+b" = 2 notes in 1 step)
    """
    sequence = []
    
    # Unpack parameters
    role = spec["role"]
    chords = spec["chords"]
    pattern_map = spec.get("pattern_map", {})
    div_map = spec.get("div_map", {})
    fallback_div = int(spec.get("div", 16))
    # ▼▼▼ 修正: style_map を取得 (style単体変数はデフォルト値として使う) ▼▼▼
    style_map = spec.get("style_map", {})
    gate_map = spec.get("gate_map", {})
    fallback_style = spec.get("style", "Normal")
    # ▲▲▲ 修正終わり ▲▲▲
    key_offset = spec.get("key_offset", 0)
    is_drum = spec.get("is_drum", False)
    is_vocal_track = "vocal" in role
    is_harmony = "harmony" in role
    is_bass = "bass" in role
    # ▼▼▼ 新規追加: UIで設定した長さを取得（デフォルトは1.0倍） ▼▼▼
    note_len_mult = spec.get("note_len", 1.0)
    # ▲▲▲ 新規追加終わり ▲▲▲
    
    rng = random.Random(seed_offset)
    bar_dur = (60 / bpm) * 4
    
    # Lyrics Setup
    lyrics_map_raw = engine_config.get("lyrics_map", {})
    
    def parse_lyrics_text(text):
        if not text: return []
        text = text.replace('\n', '').replace(' ', '')
        
        parsed = []
        # 小さい文字のリスト（ひらがな・カタカナ両対応）
        small_kana = set('ぁぃぅぇぉゃゅょァィゥェォャュョ')
        
        for char in text:
            # 現在の文字が「小さいカナ」で、かつ記号(^, v, +など)ではない前の文字がある場合
            if char in small_kana and len(parsed) > 0 and parsed[-1] not in ['^', 'v', '*', '+', '.']:
                parsed[-1] += char # 前の文字と合体させる（'き' + 'ゃ' = 'きゃ'）
            else:
                parsed.append(char)
                
        return parsed
        
    parsed_lyrics_map = {k: parse_lyrics_text(v) for k, v in lyrics_map_raw.items()}
    fallback_lyrics = ["ら", "ら", "ら", "ら", "ら"]
    
    # Section Counters
    section_note_counters = {k: 0 for k in ["I", "V", "V2", "C", "B", "O"]}
    breath_counter = 0
    phrase_length = rng.randint(4, 12) 
    # ▼▼▼ 追加: そのトラック専用のカスタムコードを取得 ▼▼▼
    custom_chords_str = spec.get("custom_chords", "").strip()
    custom_chords_list = [c.strip() for c in custom_chords_str.split(',') if c.strip()]
    # ▲▲▲ 追加終わり ▲▲▲
    
    # ▼▼▼ ここから追記: セクションごとのステップ位置を記録する変数 ▼▼▼
    last_sec_code = None
    sec_step_idx = 0
    mask_array = []
    # ▲▲▲ ここまで追記 ▲▲▲
    
    # --- Composition Loop ---
    for bar_idx, chord_str in enumerate(chords):
        bar_start_time = bar_idx * bar_dur
        
        # Identify Section
        current_sec_code = "V"
        if 'SECTION_TIMINGS' in globals() and SECTION_TIMINGS:
            for s_bar, s_code in reversed(SECTION_TIMINGS):
                if bar_idx >= s_bar: 
                    current_sec_code = s_code
                    break
        
        if "active_sections" in spec and current_sec_code not in spec["active_sections"]:
            continue
            
            # ▼▼▼ ここから追記: セクションが切り替わったらマスクを作り直し、ステップを0に戻す ▼▼▼
        if current_sec_code != last_sec_code:
            last_sec_code = current_sec_code
            sec_step_idx = 0
            
            # gate_mapから文字列を取得。十分に長いステップ数(例: 2000)でマスクを作っておく
            gate_str = gate_map.get(current_sec_code, "")
            mask_array = create_gate_mask(gate_str, 2000)
        # ▲▲▲ ここまで追記 ▲▲▲

        current_div = int(div_map.get(current_sec_code, fallback_div))
        # ▼▼▼ 追加: 現在のセクションのスタイルを決定 ▼▼▼
        # ここでループごとの style を上書きします
        style = style_map.get(current_sec_code, fallback_style)
        # ▲▲▲ 追加終わり ▲▲▲
        
        # Fill Logic
        allow_fill_track = spec.get("allow_fill", True)
        fill_sections = spec.get("fill_sections", ["I", "V", "V2", "C", "B", "O"])
        is_fill_bar = False
        is_big_fill = False
        if is_drum and allow_fill_track and (current_sec_code in fill_sections):
            is_fill_bar = ((bar_idx + 1) % 4 == 0)
            is_big_fill = ((bar_idx + 1) % 8 == 0)
        
        if is_fill_bar and "kick" not in role: steps_in_bar = current_div * 2 
        else: steps_in_bar = max(1, current_div)
            
        step_dur = bar_dur / steps_in_bar
        base_pattern = pattern_map.get(current_sec_code, [0]*16)
        if not base_pattern: base_pattern = [1,0,0,0]*4
        # ▼▼▼ 修正: カスタムコードがあれば、全体のコードを上書きする！ ▼▼▼
        if custom_chords_list:
            # 入力されたコードリストをループさせる (例: 2個しか入力されてなくても小節数に合わせて繰り返す)
            chord_str = custom_chords_list[bar_idx % len(custom_chords_list)]
        # ▲▲▲ 修正終わり ▲▲▲
        
        chord_info = None
        if chord_str:
            root_c = chord_str.split('_')[0].replace('"', '').strip()
            chord_info = parse_complex_chord(root_c, key_offset=key_offset)
            
        swing_amt = engine_config.get("swing_amount", 0.0)

        for step_i in range(steps_in_bar):
            swing_offset = 0.0
            if swing_amt > 0 and (step_i % 2 == 1):
                swing_offset = step_dur * swing_amt * 0.3
            
            step_start_time = bar_start_time + (step_i * step_dur) + swing_offset
            
            # S/E ゲートマスクの確認とカウント
            gate_allow = True
            if sec_step_idx < len(mask_array):
                gate_allow = (mask_array[sec_step_idx] == 1)
            sec_step_idx += 1
            
            # ==========================================
            # ここで必ず初期化する！ (UnboundLocalError対策)
            # ==========================================
            hit = False
            dur_steps = 1  
            
            if is_fill_bar and step_i > steps_in_bar * 0.5:
                fill_density = 0.8 if is_big_fill else 0.5
                if rng.random() < fill_density: hit = True
            else:
                norm_step = int((step_i / steps_in_bar) * len(base_pattern))
                val = base_pattern[min(norm_step, len(base_pattern)-1)]
                
                if val == 1:
                    hit = True
                    # どこまで伸ばす(2)が続くか先読みチェック
                    for look_i in range(step_i + 1, steps_in_bar):
                        norm_look = int((look_i / steps_in_bar) * len(base_pattern))
                        look_val = base_pattern[min(norm_look, len(base_pattern)-1)]
                        
                        look_gate_allow = True
                        # sec_step_idx はすでに+1されているので、その分を補正して未来をチェック
                        look_sec_step = sec_step_idx + (look_i - step_i - 1)
                        if look_sec_step < len(mask_array):
                            look_gate_allow = (mask_array[look_sec_step] == 1)
                            
                        if look_val == 2 and look_gate_allow:
                            dur_steps += 1
                        else:
                            break
                elif val == 2:
                    hit = False 
                else:
                    hit = False

            # マスクやhit判定でミュート（音が鳴らないなら次のステップへ）
            if not hit or not gate_allow: continue

            # --- Note Data Collection ---
            notes_to_generate = [] 
            pitch_base = 0.0
            vel_base = 1.0 - (rng.random() * engine_config.get("vel_humanize", 0.2))
            
            # ここで先ほど計算した dur_steps を掛け算する！
            actual_step_dur = step_dur * dur_steps
            
            # --- Note Data Collection ---
            notes_to_generate = [] 
            pitch_base = 0.0
            vel_base = 1.0 - (rng.random() * engine_config.get("vel_humanize", 0.2))
            # ▼▼▼ 追加：実際のステップ長さを計算 ▼▼▼
            actual_step_dur = step_dur * dur_steps
            # ▲▲▲ 追加終わり ▲▲▲
            
            # 【歌詞制御ロジック：完全版】
            current_lyrics_list = parsed_lyrics_map.get(current_sec_code, [])
            if not current_lyrics_list: current_lyrics_list = fallback_lyrics
            
            if is_vocal_track or "lead" in role:
                lyrics_in_step = []
                
                def fetch_next_lyric_unit():
                    unit_shift = 0
                    unit_lyric = ""
                    is_rest = False
                    
                    # ▼ 新規フラグ
                    is_vib = False
                    is_cresc = False
                    is_decresc = False
                    
                    while True:
                        idx = section_note_counters[current_sec_code] % len(current_lyrics_list)
                        char = current_lyrics_list[idx]
                        
                        if char in ['.', '。', '_', '・']: 
                            is_rest = True
                            section_note_counters[current_sec_code] += 1
                            return None, 0, True, False, False, False
                        elif char in ['^', '↑']: 
                            unit_shift += 1
                            section_note_counters[current_sec_code] += 1
                        elif char in ['v', '↓']: 
                            unit_shift -= 1
                            section_note_counters[current_sec_code] += 1
                        elif char in ['*', '＊']: 
                            unit_shift = 99
                            section_note_counters[current_sec_code] += 1
                        elif char == '+': 
                            section_note_counters[current_sec_code] += 1
                            
                        # ▼ 文字の前に書いた場合（~あ）の対応
                        elif char == '~':
                            is_vib = True
                            section_note_counters[current_sec_code] += 1
                        elif char == '<':
                            is_cresc = True
                            section_note_counters[current_sec_code] += 1
                        elif char == '>':
                            is_decresc = True
                            section_note_counters[current_sec_code] += 1
                            
                        else: # 文字確定
                            unit_lyric = char
                            section_note_counters[current_sec_code] += 1
                            
                            # ▼▼▼ ポストフィックス（文字の後ろの記号）を先読みして回収 ▼▼▼
                            # これにより「あ~>」のように書いても正しくフラグが立つ！
                            while True:
                                next_idx = section_note_counters[current_sec_code] % len(current_lyrics_list)
                                next_char = current_lyrics_list[next_idx]
                                if next_char == '~':
                                    is_vib = True
                                    section_note_counters[current_sec_code] += 1
                                elif next_char == '<':
                                    is_cresc = True
                                    section_note_counters[current_sec_code] += 1
                                elif next_char == '>':
                                    is_decresc = True
                                    section_note_counters[current_sec_code] += 1
                                else:
                                    break # 記号以外が来たら先読み終了
                            # ▲▲▲ 先読み終わり ▲▲▲
                                    
                            return unit_lyric, unit_shift, False, is_vib, is_cresc, is_decresc

                # 1つ目のユニット取得
                l, s, r, vib, cresc, decresc = fetch_next_lyric_unit()
                lyrics_in_step.append({'lyric': l, 'shift': s, 'is_rest': r, 'vib': vib, 'cresc': cresc, 'decresc': decresc})
                
                # '+' が続く限りユニットを取得し続ける
                while True:
                    next_idx = section_note_counters[current_sec_code] % len(current_lyrics_list)
                    next_char = current_lyrics_list[next_idx]
                    
                    if next_char == '+':
                        section_note_counters[current_sec_code] += 1 # '+'を消費
                        l, s, r, vib, cresc, decresc = fetch_next_lyric_unit()
                        lyrics_in_step.append({'lyric': l, 'shift': s, 'is_rest': r, 'vib': vib, 'cresc': cresc, 'decresc': decresc})
                    else:
                        break 
                
                # ステップ時間を分割
                sub_count = len(lyrics_in_step)
                sub_dur = actual_step_dur / sub_count 
                
                for i, item in enumerate(lyrics_in_step):
                    notes_to_generate.append({
                        'start_offset': i * sub_dur,
                        'dur': sub_dur,
                        'lyric': item['lyric'],
                        'shift': item['shift'],
                        'is_rest': item['is_rest'],
                        'vib': item['vib'],        # 追加！
                        'cresc': item['cresc'],    # 追加！
                        'decresc': item['decresc'] # 追加！
                    })


            else:
                # 楽器トラック用（単純生成）
                notes_to_generate.append({
                    # ▼▼▼ 修正: step_dur の代わりに actual_step_dur を使う ▼▼▼
                    'start_offset': 0, 'dur': actual_step_dur, 
                    'lyric': "", 'shift': 0, 'is_rest': False
                })


            # --- Pitch Calculation & Event Creation ---
            for note_data in notes_to_generate:
                # ★休符判定：休符なら音を作らずスキップ（時間は消費される）
                if note_data['is_rest']:
                    continue

                final_pitch = 0.0
                forced_interval_shift = note_data['shift']
                
                if is_drum:
                    final_pitch = 0
                    note_params = {"drum_type": role}
                    this_vel = vel_base
                    if step_i % 4 == 0: this_vel *= 1.1
                    elif step_i % 2 == 1: this_vel *= 0.8
                    this_dur = note_data['dur']
                    # ▼▼▼ 修正: note_len_mult を掛け算する！ ▼▼▼
                    if "kick" in role: this_dur = 0.2 * note_len_mult
                    elif "snare" in role: this_dur = 0.15 * note_len_mult
                    elif "hihat" in role: this_dur = 0.04 * note_len_mult
                    else: this_dur = this_dur * note_len_mult
                    # ▲▲▲ 修正終わり ▲▲▲
                  
                    note = NoteEvent(
                        step_start_time + note_data['start_offset'], 
                        this_dur, 
                        final_pitch, 
                        vel_base, 
                        note_data['lyric'], 
                        # ▼▼▼ 空だった {} にパラメータを詰め込む！ ▼▼▼
                        {
                            'vib': note_data.get('vib', False),
                            'cresc': note_data.get('cresc', False),
                            'decresc': note_data.get('decresc', False)
                        }
                    )
                    sequence.append(note)


                elif chord_info:
                    intervals = chord_info["intervals"]
                    oct_shift = 4
                    if is_bass: oct_shift = 2
                    elif "pad" in role: oct_shift = 3
                    elif "guitar" in role: oct_shift = 2 
                    elif "lead" in role or is_vocal_track: 
                        oct_shift = 4
                        if current_sec_code == "C" and spec.get("chorus_octave", True): oct_shift = 5 
                    elif "strings" in role: oct_shift = 5
                    elif "arp" in role: oct_shift = 5
                    
                    available_intervals = intervals[:]
                    if current_sec_code in ["C", "B"]:
                        if len(available_intervals) < 4:
                            # 【修正】安全に拡張音を追加する
                            if len(intervals) > 0:
                                available_intervals.append(intervals[0] + 10) # 7th相当
                            
                            if len(intervals) > 1:
                                available_intervals.append(intervals[1] + 12) # 3rdのオクターブ上
                            elif len(intervals) > 0:
                                # 音が1つしかない場合は、ルートのオクターブ上を足す
                                available_intervals.append(intervals[0] + 12)
                    elif current_sec_code in ["I", "V"]:
                        available_intervals = intervals[:3]

                    # --- Pitch Selection Logic ---
                    if forced_interval_shift == 99: # Root
                        iv = intervals[0]
                    else:
                        if is_bass:
                            if step_i == 0: iv = intervals[0]
                            else: iv = rng.choice(intervals[:3])
                        else:
                            base_idx = 0
                            if style == "Up": 
                                base_idx = section_note_counters[current_sec_code] % len(available_intervals)
                                iv = available_intervals[base_idx]
                            elif style == "Down": 
                                base_idx = -(section_note_counters[current_sec_code] % len(available_intervals)) - 1
                                iv = available_intervals[base_idx]
                            elif style == "Rand": 
                                base_idx = rng.randint(0, len(available_intervals)-1)
                                iv = available_intervals[base_idx]
                            elif style == "Legato":
                                # 直前の音程(prev_iv)に最も近いものをavailable_intervalsから選ぶ
                                # prev_iv変数をループの外で管理する必要があるため、
                                # 関数内で保持するための簡易的なハックとして section_note_counters を利用するか、
                                # または単純にランダム要素を排除して前のインデックスを維持しようとします。
                                
                                # ※簡略化のため「前のカウンター位置」に近いインデックスを選びます
                                current_cnt = section_note_counters[current_sec_code]
                                # 音階の長さで割った余りが現在の位置。そこから大きく動かさない。
                                base_idx = current_cnt % len(available_intervals)
                                
                                # ランダムに「維持」か「隣へ移動」のみを許可（跳躍しない）
                                move = rng.choice([0, 0, 0, -1, 1]) # 維持する確率を高めに
                                base_idx = (base_idx + move) % len(available_intervals)
                                iv = available_intervals[base_idx]
                            elif style == "Arp":
                                # アルペジオ: 順番に上昇しつつ、2周目はオクターブ上に行く（広がり重視）
                                seq_count = section_note_counters[current_sec_code]
                                len_int = len(available_intervals)
                                base_idx = seq_count % len_int
                                iv = available_intervals[base_idx]
                                
                                # 周期ごとにオクターブを切り替える（0, +1, 0, +1...）
                                # これで琴やハープのような「広い」演奏になります
                                if (seq_count // len_int) % 2 == 1:
                                    oct_shift += 1 
                                    
                            elif style == "Glides":
                                # Glides: Legatoに近いけど、ピッチ変化を強調するために
                                # あえて「1つ飛ばし」で動いて、音を滑らせる隙間を作る
                                current_cnt = section_note_counters[current_sec_code]
                                base_idx = current_cnt % len(available_intervals)
                                # -2か+2動くことで、音程差を作ってグライドを聞こえやすくする
                                move = rng.choice([-2, 2]) 
                                base_idx = (base_idx + move) % len(available_intervals)
                                iv = available_intervals[base_idx]
                                
                                # ★重要: グライドさせるために、音の長さを強制的に少し伸ばして重ねる
                                # (sus_factorが後で掛かるけど、ここでも強調しておく)
                                note_data['dur'] *= 1.1 

                            elif style == "PingPong":
                                # PingPong: 行ったり来たり (0, 1, 2, 1, 0...)
                                length = len(available_intervals)
                                if length > 1:
                                    cycle = (length - 1) * 2
                                    step = section_note_counters[current_sec_code] % cycle
                                    if step >= length:
                                        idx = cycle - step
                                    else:
                                        idx = step
                                    iv = available_intervals[idx]
                                else:
                                    iv = available_intervals[0]

                            elif style == "Root":
                                # Root: 常にルート音（ベース用）
                                iv = available_intervals[0]

                            elif style == "Octave":
                                # Octave: ルート音とそのオクターブ上を交互に（ディスコベース風）
                                step = section_note_counters[current_sec_code] % 2
                                iv = available_intervals[0]
                                if step == 1:
                                    forced_interval_shift += 12 # 1オクターブ上げる処理として扱う
                                    # または iv に 12 を足してもいいが、共通処理に任せるなら shift
                                
                            elif style == "-1 Oct":
                                # Normalと同じ動きをするが、オクターブ変数を1下げる
                                current_cnt = section_note_counters[current_sec_code]
                                base_idx = current_cnt % len(available_intervals)
                                iv = available_intervals[base_idx]
                                oct_shift -= 1 # ★これでピッチが半分（1オクターブ下）になる！

                            elif style == "+1 Oct":
                                # こっちは1オクターブ上げる（女性ボーカルに重ねるキラキラコーラス用など）
                                current_cnt = section_note_counters[current_sec_code]
                                base_idx = current_cnt % len(available_intervals)
                                iv = available_intervals[base_idx]
                                oct_shift += 1 # ★倍のピッチになる！
                                
                              # ▼▼▼ 修正: PowerChord 系に Piano_Comping を追加 ▼▼▼
                            elif style in ["PowerChord", "FullChord", "PowerChord_Low", "FullChord_Low", "Piano_Comping"]:
                                iv = available_intervals[0]
                            # ▲▲▲ 修正終わり ▲▲▲
                                       
                            # ▼▼▼ 今回追加したペンタトニック ▼▼▼
                            elif style == "Pentatonic":
                                # ペンタトニックスケールの度数に制限するロジック
                                # メジャーコードならメジャーペンタ(0, 2, 4, 7, 9)、マイナーならマイナーペンタ(0, 3, 5, 7, 10)にマッピング
                                is_minor = (3 in intervals)  # マイナーサードが含まれているか判定
                                if is_minor:
                                    penta_intervals = [0, 3, 5, 7, 10] # マイナーペンタトニック
                                else:
                                    penta_intervals = [0, 2, 4, 7, 9]  # メジャーペンタトニック (ヨナ抜き)
                                
                                # 利用可能なインターバルをペンタトニックの構成音のみにフィルタリング
                                available_intervals = [iv for iv in penta_intervals]
                                
                                # フレーズが飛び飛びにならないよう、インデックスを順に辿る（あるいはランダムに選ぶ）
                                current_cnt = section_note_counters[current_sec_code]
                                base_idx = current_cnt % len(available_intervals)
                                iv = available_intervals[base_idx]
                            # ▲▲▲ ペンタトニック追加終わり ▲▲▲                                                         
                                                                                             
                            else: # Normal
                                last_cnt = section_note_counters[current_sec_code]
                                base_idx = (last_cnt % len(available_intervals))
                                move = rng.choice([-1, 0, 1])
                                base_idx = (base_idx + move) % len(available_intervals)
                                iv = available_intervals[base_idx]
                            
                            # Apply Shift (共通処理)
                            final_idx = (available_intervals.index(iv) + forced_interval_shift) % len(available_intervals)
                            iv = available_intervals[final_idx]
                            
                            

                    if is_harmony: 
                        try: iv = available_intervals[(available_intervals.index(iv) + 1) % len(available_intervals)]
                        except: pass
                    
                    f_base = note_freq(chord_info["root"], oct_shift)
                    final_pitch = f_base * (2**(iv/12.0))
                    
                    # ▼▼▼ 修正: 追加ピッチを「詳細データ(辞書)」で管理する最強の仕組み ▼▼▼
                    # p=ピッチ, v=ベロシティ倍率, d=発音タイミング(割合), dur_m=長さ倍率
                    extra_notes_data = [] 
                    
                    if style == "PowerChord":
                        extra_notes_data.append({'p': f_base * (2**((iv + 7)/12.0)), 'v': 0.85, 'd': 0.02})
                        extra_notes_data.append({'p': f_base * (2**((iv + 12)/12.0)), 'v': 0.85, 'd': 0.04})
                        
                    elif style == "PowerChord_Low":
                        extra_notes_data.append({'p': f_base * (2**((iv - 5)/12.0)), 'v': 0.9, 'd': 0.015})
                        extra_notes_data.append({'p': f_base * (2**((iv - 12)/12.0)), 'v': 1.0, 'd': 0.03})
                        
                    elif style == "FullChord":
                        for i, val in enumerate(available_intervals):
                            if val != iv:
                                extra_notes_data.append({'p': f_base * (2**(val/12.0)), 'v': 0.85, 'd': (i+1)*0.02})
                                
                    elif style == "FullChord_Low":
                        for i, val in enumerate(available_intervals):
                            extra_notes_data.append({'p': f_base * (2**((val - 12)/12.0)), 'v': 0.9, 'd': i*0.02})
                            
                    elif style == "Piano_Comping":
                        # === ピアニストの神ボイシング（ジャズ・R&B対応） ===
                        # 左手：Root(-12), 5th(-5) / 右手：3rd, 7th, 9th(+14)
                        third_val = available_intervals[1] if len(available_intervals) > 1 else 4
                        seventh_val = available_intervals[3] if len(available_intervals) > 3 else 10
                        ninth_val = 14 # おしゃれなナインスを強制追加
                        
                        # メインノート(単音)は鳴らさず、完全にカスタムボイシングで上書き！
                        final_pitch = None 
                        
                        # ① 左手 (Velocity: 弱め 0.75, Delay: ほぼ同時, NoteLen: 短め 0.8)
                        extra_notes_data.append({'p': f_base * (2**(-12/12.0)), 'v': 0.75, 'd': 0.0,   'dur_m': 0.8})
                        extra_notes_data.append({'p': f_base * (2**(-5/12.0)),  'v': 0.80, 'd': 0.01,  'dur_m': 0.8})
                        
                        # ② 右手内声 (Velocity: 普通 0.85, Delay: 少し遅らせる, NoteLen: 普通 1.0)
                        extra_notes_data.append({'p': f_base * (2**(third_val/12.0)),   'v': 0.85, 'd': 0.03, 'dur_m': 1.0})
                        extra_notes_data.append({'p': f_base * (2**(seventh_val/12.0)), 'v': 0.88, 'd': 0.05, 'dur_m': 1.0})
                        
                        # ③ トップノート (Velocity: 一番強く 1.05, Delay: タララーンと大きく遅らせる, NoteLen: 長め 1.2)
                        extra_notes_data.append({'p': f_base * (2**(ninth_val/12.0)), 'v': 1.05, 'd': 0.08, 'dur_m': 1.2})
                    # ▲▲▲ 修正終わり ▲▲▲
                    
                    sus_factor = 0.9
                    if "pad" in role: sus_factor = 2.0 
                    elif is_vocal_track: sus_factor = 0.95
                    elif "arp" in role: sus_factor = 0.5
                    
                    this_dur = note_data['dur'] * sus_factor * note_len_mult
                    
                    if (is_vocal_track or "lead" in role) and note_data['lyric'] != "":
                        breath_counter += 1
                        if breath_counter >= phrase_length:
                            breath_counter = 0
                            phrase_length = rng.randint(4, 12)
                            continue 

                    # 1. メインノートの登録（Piano_Compingの場合は final_pitch = None なのでスキップされる！）
                    if final_pitch is not None:
                        note = NoteEvent(
                            step_start_time + note_data['start_offset'], 
                            this_dur, 
                            final_pitch, 
                            vel_base, 
                            note_data['lyric'], 
                            # ボーカルの記号フラグも忘れずに引き継ぐ
                            {
                                'vib': note_data.get('vib', False), 
                                'cresc': note_data.get('cresc', False), 
                                'decresc': note_data.get('decresc', False)
                            }
                        )
                        sequence.append(note)

                    # 2. 追加ピッチ（ボイシング）の登録
                    for edata in extra_notes_data:
                        # 発音タイミング：音符の長さ(this_dur)に対する割合(d)で計算。BPMに完全連動！
                        step_delay = this_dur * edata.get('d', 0.0)
                        
                        # ベロシティ：指定の強さ(v) × わずかなヒューマナイズ揺らぎ
                        vel_human = vel_base * edata.get('v', 0.8) * rng.uniform(0.95, 1.05)
                        
                        # 音の長さ：低音は短く、トップノートは長く(dur_m)！
                        note_dur = this_dur * edata.get('dur_m', 1.0)
                        
                        # ピッチ：完璧なHzだと溶けちゃうので、ミリ単位でデチューンして響きをリッチに
                        detuned_p = edata['p'] * rng.uniform(0.998, 1.002)

                        extra_note = NoteEvent(
                            step_start_time + note_data['start_offset'] + step_delay, 
                            note_dur, 
                            detuned_p, 
                            vel_human, 
                            note_data['lyric'], 
                            {} # 追加ノートはボーカルではないので空でOK
                        )
                        sequence.append(extra_note)

    return sequence

def render_track_sequence(sequence, spec, engine_config, total_samples, seed_offset):
    """
    Phase 2: Rendering (Fixed: Velocity, Sidechain, High-Speed Articulation, and Phase Management)
    """
    track_buffer = np.zeros(total_samples, dtype=np.float32)
    
    # Setup Synths
    role = spec["role"]
    wave_type = spec.get("wave", "sin")
    is_vocal_track = "vocal" in role
    
    # Vocal Engines
    synth_pop = VocalSynth('pop')
    synth_adult = VocalSynth('adult')
    synth_utada = VocalSynth('utada')
    
    # Setup LFO
    lfo_params = spec.get("lfo_params", {})
    use_lfo = lfo_params.get("active", False)
    lfo_rate = lfo_params.get("rate", 1.0)
    lfo_depth = lfo_params.get("depth", 0.0)
    
    # Setup Sidechain
    sc_strength = engine_config.get("sc_strength", 0.0)
    sc_release = engine_config.get("sc_release", 0.15)
    
    # Sidechain Exclusion: Don't pump the vocals or kick
    should_sidechain = (sc_strength > 0.01) and ("kick" not in role) and (not spec.get("is_drum", False)) and (not is_vocal_track)

    vocal_params = spec.get("vocal_params", {})
    def get_v_param(key, default):
        return vocal_params.get(key, engine_config.get(key, default))
        
    # ★【追加①】UIで描画・保存されたピッチカーブのデータ(ピンクの線)を取得
    pitch_curve_data = spec.get("pitch_curve", {})

    current_phase = 0.0
    phase_mode = vocal_params.get("phase_mode", "continuous")
    
    prev_vowel_state = {}
    ag_on = False
    rng = random.Random(seed_offset) 

    prev_was_sokuon = False
    for i, note in enumerate(sequence):
        start_idx = int(note.start_time * FS)
        if start_idx >= total_samples: continue
        
        wave_out = np.array([], dtype=np.float32)
        
        # Gain Logic: Vocals handle velocity internally, others use gain
        gain = 1.0 
        
        if spec.get("is_drum", False):
            # Drum Track (Unchanged)
            gain = note.velocity
            d_role = note.params.get("drum_type", role)
            
            if "kick" in d_role:
                # ▼▼▼ ここから下のインデントを綺麗に揃えました ▼▼▼
                k_samples = int(FS * note.duration)
                t_k = np.linspace(0, note.duration, k_samples, False, dtype=np.float32)

                # =========================
                # ① BODY（ピッチ急降下）
                # =========================
                freq_env = 220 * np.exp(-t_k * 40) + 55
                phase = 2 * np.pi * np.cumsum(freq_env) / FS
                body = np.sin(phase)

                body_env = np.exp(-t_k * 5)
                body *= body_env

                # =========================
                # ② SUB（安定低域）
                # =========================
                sub_freq = 55.0
                sub = np.sin(2 * np.pi * sub_freq * t_k)
                sub *= np.exp(-t_k * 3)

                # =========================
                # ③ CLICK（超短トランジェント）
                # =========================
                click_len = int(FS * 0.004)
                click = np.random.randn(click_len).astype(np.float32)
                click *= np.exp(-np.linspace(0, 1, click_len, dtype=np.float32) * 70)

                click_layer = np.zeros_like(t_k)
                click_layer[:click_len] = click

                # =========================
                # ④ レイヤーミックス
                # =========================
                wave_out = body * 0.9 + sub * 1.1 + click_layer * 0.6

                # =========================
                # ⑤ サチュレーション（倍音生成）
                # =========================
                wave_out = np.tanh(wave_out * 4.5)

                # =========================
                # ⑥ 簡易コンプ風（潰して密度上げる）
                # =========================
                wave_out *= 1.5
                wave_out = np.tanh(wave_out)

                # 最終減衰
                final_env = np.exp(-t_k * 2.5)
                wave_out *= final_env
                # ▲▲▲ ここまで ▲▲▲
            elif "snare" in d_role:
                tone = fast_osc_custom(180, 0.1, "sin")
                tone *= np.exp(-np.linspace(0,0.1,len(tone),dtype=np.float32)*20)
                nz = fast_osc_custom(0, 0.15, "noise")
                nz *= np.exp(-np.linspace(0,0.15,len(nz),dtype=np.float32)*12)
                wave_out = nz * 0.8
                if len(tone) > 0: wave_out[:len(tone)] += tone * 0.5
            elif "hihat" in d_role:
                wave_out = fast_osc_custom(0, note.duration, "noise")
                if len(wave_out)>1: wave_out = wave_out[1:] - wave_out[:-1]
                gain *= 0.7
            elif "crash" in d_role:
                wave_out = VocalDSP.generate_cymbal_wave(2.5, "crash")
            elif "ride" in d_role:
                wave_out = VocalDSP.generate_cymbal_wave(note.duration*1.2, "ride")
                gain *= 0.8
            elif "guitar" in d_role:
                wave_out = generate_karplus_custom(note.pitch, note.duration, decay=0.99)
                wave_out = apply_env(wave_out, note.duration, 0.01, 0.05, 0.9, 0.1)
            elif "glitch" in d_role:
                g_freq = rng.uniform(200, 2000)
                wave_out = fast_osc_custom(g_freq, note.duration, wave_type)
                gain *= 0.6
            elif "impact" in d_role:
                wave_out = fast_osc_custom(60, 2.0, "noise")
                wave_out *= np.exp(-np.linspace(0,2,len(wave_out),dtype=np.float32)*4)
            elif "noise_swp" in d_role:
                swp_len = int(FS * 1.0)
                wave_out = np.random.uniform(-1, 1, swp_len).astype(np.float32)
                wave_out *= np.linspace(0, 1, swp_len)
                gain *= 0.5

        elif is_vocal_track:
            # Vocal Track
            if wave_type == "adult": synth = synth_adult
            elif wave_type == "utada": synth = synth_utada
            else: synth = synth_pop
            # ▼▼▼ note.params からスマートにフラグを取り出す ▼▼▼
            is_vib_heavy = note.params.get('vib', False)
            is_cresc = note.params.get('cresc', False)
            is_decresc = note.params.get('decresc', False)
            # ★【追加②】ノートごとの専用パラメータ（ピンの情報など）を結合
            current_vocal_params = vocal_params.copy()
            if hasattr(note, 'params') and isinstance(note.params, dict):
                current_vocal_params.update(note.params)            
            
            # --- 先読みロジック (Lookahead) ---
            num_notes = len(sequence)
            next_note = sequence[i+1] if i + 1 < num_notes else None
            next_kana = next_note.lyric if next_note else ""
            next_romaji = kana_engine.to_romaji(next_kana)
            next_cons = kana_engine.get_cons_type(next_kana)
            
            # フラグ初期化
            is_devoiced = False
            is_sokuon_next = (next_romaji == 'clt')
            
            # 文末判定
            is_end = (next_note is None or next_note.start_time > note.start_time + note.duration + 0.5)

            # カナ解析
            current_kana = note.lyric
            _, target_vowel = kana_engine.parse_phonemes(current_kana)
            if target_vowel is None: target_vowel = current_kana 
            
            # 撥音「ん」の同化
            if target_vowel == 'n':
                if next_cons in LABIAL_CONS:    # p, b, m -> 'm'
                    current_kana = "m"
                elif next_cons in VELAR_CONS:   # k, g -> 'ng'
                    current_kana = "ng"
            
            # 母音の無声化
            if target_vowel in ['i', 'u']:
                is_next_voiceless = (next_cons in VOICELESS_CONS)
                if is_next_voiceless or is_end:
                    if rng.random() < 0.9: 
                        is_devoiced = True

            # パラメータ取得
            b_vol = get_v_param("breath_vol", 0.05)
            b_chance = get_v_param("breath_chance", 0.5)
            c_atk = get_v_param("cons_attack", 1.0)
            whisper_val = get_v_param("whisper_amount", 0.0)
            scoop_val = get_v_param("scoop_amount", 0.2)
            vib_depth = get_v_param("vibrato_depth", 0.05)
            vib_rate = get_v_param("vibrato_rate", 5.0)
             # ▼▼▼ ビブラート上書き ▼▼▼
            if is_vib_heavy:
                vib_depth = 0.15
                vib_rate = 6.0
                
            
            # Formant Shift
            f_shift = engine_config.get("vocal_formant_shift", 1.0)
            
            # ピッチゆらぎ
            pitch = note.pitch * (1.0 + (rng.uniform(-1, 1) * 0.001)) 

            # [NEW] Determine Initial Phase for this note
            note_start_phase = 0.0
            if phase_mode == "continuous":
                note_start_phase = current_phase
            elif phase_mode == "random":
                note_start_phase = rng.random()
            else: # reset
                note_start_phase = 0.0
            
            # --- Render ---
            # ★【追加③】synth.render に pitch_curve_data と start_time などを渡す！
            wave_out, next_v = synth.render(
                current_kana, pitch, note.duration, note.velocity, prev_vowel_state, ag_on,
                initial_phase=note_start_phase,  
                vocal_params=current_vocal_params,       # ★ 変更 (ピン情報を渡すため)
                pitch_curve_data=pitch_curve_data,       # ★ 追加 (ピンクの線のデータ)
                start_time=note.start_time,              # ★ 追加 (現在のノートの開始時間)
                breath_vol=b_vol, breath_chance=b_chance, cons_attack=c_atk,
                scoop_amount=scoop_val, whisper_amount=whisper_val,
                vib_depth=vib_depth, vib_rate=vib_rate,
                formant_shift=f_shift,
                is_devoiced=is_devoiced,
                is_sokuon_next=is_sokuon_next,
                after_sokuon=prev_was_sokuon,
                next_cons_type=next_cons,
                is_phrase_end=is_end
            )
            
                        # ▼▼▼ クレッシェンド・デクレッシェンド処理 ▼▼▼
            if len(wave_out) > 0:
                if is_cresc:
                    dyn_env = np.linspace(0.3, 1.3, len(wave_out), dtype=np.float32)
                    wave_out *= dyn_env
                elif is_decresc:
                    dyn_env = np.linspace(1.2, 0.1, len(wave_out), dtype=np.float32)
                    wave_out *= dyn_env
                    
            # [NEW] Update Phase State for next note
            if phase_mode == "continuous":
                # renderの結果から最後の位相を取り出して保存
                current_phase = next_v.get('phase', 0.0)

            # ★UTADA SPECIFIC POST-PROCESSING
            if wave_type == "utada":
                wave_out = VocalDSP.apply_utada_eq(wave_out, FS)
                wave_out = np.tanh(wave_out * 2.0) * 0.6

            prev_vowel_state = next_v
            
            # 状態更新
            prev_was_sokuon = (kana_engine.to_romaji(current_kana) == 'clt')
            
            # Vocal Envelope
            v_atk = vocal_params.get("attack", 0.05)
            v_rel = vocal_params.get("release", 0.03)
            
            if note.duration < 0.15: 
                v_atk = min(v_atk, 0.01)
                v_rel = min(v_rel, 0.02)
            
            if prev_was_sokuon: 
                v_atk = 0.005 

            wave_out = apply_env(wave_out, len(wave_out)/FS, v_atk, 0.0, 1.0, v_rel)

        else:
            # Instrument Track (Unchanged... existing logic for piano, guitar etc.)
            gain = note.velocity
            samples = int(FS * note.duration)
            if samples <= 0:
                wave_out = np.array([], dtype=np.float32)


            else:
            # Instrument Track
                gain = note.velocity
            
            # ★修正: 先にサンプル数を計算
            samples = int(FS * note.duration)
            if samples <= 0:
                wave_out = np.array([], dtype=np.float32)
            else:
                # --- 楽器生成ロジック ---
                if wave_type == "piano":
                    # === Dance / Pop Piano (アタックが硬く、抜ける音) ===
                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    vel_factor = max(0.1, gain) # ベロシティ
                    
                    # ピッチによる減衰の変化（低音は太く、高音は細く）
                    decay_rate = 15.0 if note.pitch > 400 else 10.0
                    
                    # 強く弾くほどモジュレーション(倍音)が強くなり「カンッ！」と鳴る
                    mod_idx = (3.5 * vel_factor) * np.exp(-t * decay_rate) 
                    
                    w1 = np.sin(2 * np.pi * note.pitch * t + mod_idx * np.sin(2 * np.pi * note.pitch * t))
                    w2 = np.sin(2 * np.pi * (note.pitch * 1.002) * t) * (0.7 * vel_factor)
                    w3 = np.sin(2 * np.pi * (note.pitch * 0.5) * t) * 0.2
                    wave_out = (w1 + w2 + w3) * 0.5
                    
                    # リリースを少し短めにしてリズミカルに
                    wave_out = apply_env(wave_out, note.duration, 0.005, 0.4, 0.0, 0.1)

                elif wave_type == "legacy_piano":
                    # === Acoustic Grand Piano (リアルな物理モデリング) ===
                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch
                    vel = max(0.1, gain)

                    # --- Inharmonicity (弦の硬さによる倍音のズレ) ---
                    # 低音弦ほど太くて硬いためズレが大きく、うなりが生じる
                    B = 0.0001 + (100 / max(f, 50)) * 0.0004 
                    
                    # 高音ほど全体的にディケイ（減衰）が早い
                    pitch_decay_mult = math.sqrt(f / 220.0) 

                    partials = []
                    # ベロシティが高い(強く弾く)ほど、高次倍音の成分が増える
                    harm_amps = [
                        (1, 1.0), 
                        (2, 0.7 * vel), 
                        (3, 0.45 * vel), 
                        (4, 0.25 * (vel**1.5)), 
                        (5, 0.15 * (vel**2))
                    ]

                    for n, amp in harm_amps:
                        fn = f * n * math.sqrt(1 + B * n * n)
                        # 倍音が高いほど減衰が早い（基音だけが最後まで残る）
                        p_decay = np.exp(-t * (1.5 + n * 0.5) * pitch_decay_mult)
                        partial = amp * np.sin(2 * np.pi * fn * t) * p_decay
                        partials.append(partial)

                    harmonics = sum(partials)

                    # --- ハンマーアタックノイズ ---
                    # 強く弾いた時と、高音域で目立つ「カツッ」という打撃音
                    attack_len = int(FS * 0.015)
                    hammer = np.zeros(samples, dtype=np.float32)
                    if attack_len < samples:
                        noise = np.random.normal(0, 1, attack_len).astype(np.float32)
                        env = np.linspace(1, 0, attack_len, dtype=np.float32)
                        hammer[:attack_len] = noise * env * (vel * 0.6)
                        
                        if HAS_SCIPY:
                            b, a = signal.butter(2, min(7000, FS/2.1)/(FS/2), btype='high')
                            hammer = signal.lfilter(b, a, hammer)

                    wave_out = harmonics + hammer * 0.4

                    # --- サウンドボードの共鳴 (胴鳴り) ---
                    if HAS_SCIPY:
                        b1, a1 = signal.iirpeak(220, 2.0, fs=FS)
                        b2, a2 = signal.iirpeak(800, 3.0, fs=FS)
                        body1 = signal.lfilter(b1, a1, wave_out)
                        body2 = signal.lfilter(b2, a2, wave_out)
                        
                        wave_out = wave_out * 0.7 + body1 * 0.25 + body2 * 0.15

                    # アタックは最速、ペダルを踏んだように少し余韻(0.3s)を残す
                    wave_out = apply_env(wave_out, note.duration, 0.002, 0.0, 1.0, 0.3)
                    wave_out *= 1.5 # 音量補正

                    # --- Subtle stereo string spread (low notes feel wider) ---
                    if f < 200:
                        detune = np.sin(2*np.pi*(f*1.002)*t) * 0.3
                        wave_out += detune * 0.2

                    # --- Soft clip (real hammer nonlinearity) ---
                    wave_out = np.tanh(wave_out * 1.2)

                    wave_out = apply_env(wave_out, note.duration, 0.003, 0.2, 0.0, 0.4)

                elif wave_type == "heavy_guitar":
                    # === 最終兵器ギター (Power Chord Special) ===
                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch
                    
                    # ■ 1. パワーコード生成 (弦を増やします)
                    # ルート音 + 5度上(1.5倍) + オクターブ上(2倍) を混ぜて「壁」を作る
                    
                    # ヘルパー関数: ノコギリ波を2枚重ねて厚みを出す
                    def get_thick_saw(freq):
                        # 0.5% ピッチをずらした音を混ぜる(コーラス効果)
                        sa = (np.mod(t * freq, 1.0) * 2 - 1)
                        sb = (np.mod(t * freq * 1.005, 1.0) * 2 - 1)
                        return (sa + sb) * 0.5

                    # 3つの音を生成して混ぜる
                    w_root = get_thick_saw(f)          # メインの弦
                    w_5th  = get_thick_saw(f * 1.498)  # 5度上の弦 (パワーコードの要)
                    w_oct  = get_thick_saw(f * 2.0)    # オクターブ上の弦

                    # ミックス (ルートを一番強く)
                    w = w_root * 1.0 + w_5th * 0.8 + w_oct * 0.6

                    # ■ 2. パームミュート (ブリッジミュート) 判定
                    is_mute = (note.velocity < 0.9)
                    decay = 20.0 if is_mute else 1.0 # 通常時はサスティーンを長く(1.0)
                    env = np.exp(-t * decay)
                    
                    # ■ 3. Pre-Amp EQ (重低音ブースト)
                    # Lowをガツンと上げて重厚感を出す
                    w = three_band_eq(w, low=3.0, mid=1.2, high=0.5)
                    
                    # ■ 4. Hard Distortion (激歪み)
                    # 入力を大きくしてtanhで潰すことで倍音を爆発させる
                    w = np.tanh(w * 8.0) 
                    
                    # ■ 5. Cabinet Simulator
                    # ジリジリした高音を削って「アンプの箱鳴り」に近づける
                    w = simple_lp(w, 0.55) 
                    
                    wave_out = w * env * 0.8 # 最終音量調整

                elif wave_type == "organ":
                    # === Grand Pipe Organ (Improved) ===

                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch

                    # slight slow chorus drift
                    drift = 1.0 + 0.002 * np.sin(2 * np.pi * 0.4 * t)

                    # fundamental
                    w1 = np.sin(2 * np.pi * f * t * drift)

                    # octave
                    w2 = np.sin(2 * np.pi * f * 2 * t) * 0.5

                    # fifth (majestic character)
                    w3 = np.sin(2 * np.pi * f * 1.5 * t) * 0.35

                    # 3rd harmonic
                    w4 = np.sin(2 * np.pi * f * 3 * t) * 0.2

                    # sub reinforcement
                    sub = np.sin(2 * np.pi * (f * 0.5) * t) * 0.2

                    wave_out = (w1 + w2 + w3 + w4 + sub)

                    # body resonance (cathedral feel)
                    if HAS_SCIPY:
                        b1, a1 = signal.iirpeak(200, 1.5, fs=FS)
                        b2, a2 = signal.iirpeak(2500, 3.0, fs=FS)

                        body1 = signal.lfilter(b1, a1, wave_out)
                        body2 = signal.lfilter(b2, a2, wave_out)

                        wave_out = wave_out * 0.7 + body1 * 0.2 + body2 * 0.1

                    # gentle saturation
                    wave_out = np.tanh(wave_out * 1.3)

                    # slow attack, long release
                    wave_out = apply_env(wave_out, note.duration, 0.12, 0.0, 1.0, 0.25)



                elif wave_type == "musicbox":
                    # === Crystal Music Box (Improved) ===

                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch * np.random.uniform(0.998, 1.002)

                    # fundamental
                    tone = np.sin(2 * np.pi * f * t)

                    # bright upper partials (crystal feel)
                    h2 = np.sin(2 * np.pi * f * 2 * t) * 0.4
                    h3 = np.sin(2 * np.pi * f * 3 * t) * 0.25
                    h5 = np.sin(2 * np.pi * f * 5 * t) * 0.15

                    # slight inharmonic metal overtone
                    metal = np.sin(2 * np.pi * (f * 2.7) * t) * 0.2

                    wave_out = tone + h2 + h3 + h5 + metal

                    # brightness emphasis
                    if HAS_SCIPY:
                        sos = signal.butter(2, 3000/(FS/2), btype='high', output='sos')
                        bright = signal.sosfilt(sos, wave_out)
                        wave_out = wave_out * 0.6 + bright * 0.4

                    # high frequency sustain longer (sparkle tail)
                    brightness = np.exp(-t * 1.5)
                    wave_out *= (0.6 + 0.4 * brightness)

                    # tiny soft clip for sweetness
                    wave_out = np.tanh(wave_out * 1.2)

                    # fast attack, medium decay, no sustain
                    wave_out = apply_env(wave_out, note.duration, 0.002, 0.35, 0.0, 0.15)

                elif wave_type == "fm_bass":
                    # === FM Bass (Donk / Solid Bass) ===
                    # Carrier : Modulator = 1 : 1 (or 0.5)
                    # Index Envelope: Short Decay (Punchy!)
                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch
                    
                    # モジュレーター: ピッチと同じ周波数で揺らす (Ratio 1:1)
                    # Mod Indexエンベロープ: 一瞬で強く揺らして「ベチッ」というアタックを作る
                    mod_env = np.exp(-t * 18.0)  
                    mod_index = 3.5 * mod_env    # 最大変調度 3.5
                    
                    # FM計算: sin(Fc + I * sin(Fm))
                    # ※隠し味: わずかに矩形波っぽくするために feedback 的な歪みを tanh で足す
                    fm_body = np.sin(2 * np.pi * f * t + mod_index * np.sin(2 * np.pi * f * t))
                    
                    # サブベース成分を少し混ぜて低域を補強
                    sub = np.sin(2 * np.pi * (f * 0.5) * t) * 0.4
                    
                    wave_out = (fm_body + sub) * 0.8
                    # アンプエンベロープ: 短く歯切れよく
                    wave_out = apply_env(wave_out, note.duration, 0.005, 0.2, 0.6, 0.1)

                elif wave_type == "fm_bell":
                    # === FM Bell (Metallic / Glassy) ===
                    # Carrier : Modulator = 1 : 1.4 (Inharmonic)
                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch
                    
                    # Ratio 1:1.4 はベル音の黄金比
                    mod_freq = f * 1.4 
                    
                    # Index Envelope: 最初は金属的、だんだん丸くなる
                    mod_env = np.exp(-t * 2.5) 
                    mod_index = 5.0 * mod_env + 0.1 # 余韻にも少し金属感を残す
                    
                    # FM計算
                    wave_out = np.sin(2 * np.pi * f * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
                    
                    # 高域を強調するために少しGainを上げる
                    wave_out *= 1.2
                    # アンプエンベロープ: 長いリリース
                    wave_out = apply_env(wave_out, note.duration, 0.005, 0.5, 0.0, 1.5)

                elif wave_type == "pluck":
                    # === Pluck / Mallet Synth (Marimba, Kalimba style) ===
                    # User Spec: Sine + Small Square, Instant Attack, Short Decay
                    
                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    
                    # 1. Pitch Envelope (Transient "Knock")
                    # +10cents (approx 1.006x) with fast decay (50ms)
                    p_decay = 20.0 # 1/0.05s
                    p_env = 1.0 + 0.006 * np.exp(-t * p_decay)
                    
                    # Phase integration for pitch bending
                    phase = np.cumsum(note.pitch * p_env / FS)
                    
                    # 2. Oscillator Layering
                    # Body: Sine wave
                    w_sine = np.sin(2 * np.pi * phase)
                    
                    # Transient/Hardness: Square wave (Lower volume)
                    # Using naive square is fine here as filter will smooth it
                    w_sqr = np.sign(np.sin(2 * np.pi * phase)) * 0.15
                    
                    # Layering
                    raw = w_sine + w_sqr
                    
                    # 3. Filter (LPF ~5kHz)
                    # Simple 1-pole Lowpass approximation to retain attack but remove harsh fizz
                    if len(raw) > 1:
                        # coeff 0.6 keeps highs < 5-6kHz roughly
                        raw = 0.5 * raw + 0.5 * np.roll(raw, 1) 
                        
                        # If SciPy is available, use sharper filter
                        if HAS_SCIPY:
                            sos = signal.butter(1, 5000/(FS/2), btype='low', output='sos')
                            raw = signal.sosfilt(sos, raw)

                    # 4. Amp Envelope (Pluck Shape)
                    # Attack: 2ms, Decay: 250ms, Sustain: 0.0, Release: 100ms
                    wave_out = apply_env(raw, note.duration, 0.002, 0.25, 0.0, 0.1)
                    
                    # 5. Saturation & Transient Shaping (Emphasis)
                    # Drive it a bit to glue layers
                    wave_out = np.tanh(wave_out * 1.3)

                elif wave_type == "celtic_flute":
                    # === Celtic Flute / Tin Whistle (Improved) ===

                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)

                    # --- subtle pitch randomness ---
                    base_pitch = note.pitch * np.random.uniform(0.999, 1.001)

                    # --- Vibrato (non-linear fade in) ---
                    vib_start = 0.15
                    vib_freq = 5.0
                    vib_depth = 0.005

                    vib_env = 1.0 - np.exp(-(t - vib_start) * 8.0)
                    vib_env = np.clip(vib_env, 0.0, 1.0)

                    f_mod = base_pitch * (
                        1.0 + vib_depth *
                        np.sin(2 * np.pi * vib_freq * t) * vib_env
                    )

                    phase = np.cumsum(f_mod / FS)

                    # --- Add harmonics (critical) ---
                    tone = (
                        1.0 * np.sin(2 * np.pi * phase) +
                        0.15 * np.sin(2 * np.pi * 2 * phase) +
                        0.05 * np.sin(2 * np.pi * 3 * phase)
                    )

                    # --- Breath Noise (decaying over time) ---
                    noise = np.random.normal(0, 1, samples).astype(np.float32)

                    if HAS_SCIPY:
                        sos = signal.butter(1, 2000/(FS/2), btype='high', output='sos')
                        noise = signal.sosfilt(sos, noise)

                    breath_env = np.exp(-t * 4.0)
                    noise *= breath_env * 0.4

                    # --- Attack Chiff (band-shaped) ---
                    chiff = np.zeros(samples, dtype=np.float32)
                    chiff_dur = int(FS * 0.03)

                    if chiff_dur < samples:
                        chiff[:chiff_dur] = (
                            np.random.uniform(-1, 1, chiff_dur)
                            * np.linspace(1, 0, chiff_dur)
                        )

                        if HAS_SCIPY:
                            b, a = signal.iirpeak(4500, 3.0, fs=FS)
                            chiff = signal.lfilter(b, a, chiff)

                    raw = tone + noise + chiff * 0.3

                    # mild soft saturation
                    raw = np.tanh(raw * 1.2)

                    wave_out = apply_env(raw, note.duration, 0.04, 0.1, 0.9, 0.15)
                    wave_out = VocalDSP.apply_3band_eq(wave_out, 0.0, -2.0, 1.0, FS)



                elif wave_type == "fiddle":
                    # === Fiddle (Improved) ===

                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch * np.random.uniform(0.999, 1.001)

                    # Saw
                    raw = 2.0 * (t * f - np.floor(t * f + 0.5))

                    # soften edges
                    raw = 0.6 * raw + 0.4 * np.roll(raw, 1)

                    # bow amplitude wobble
                    bow_speed = 1.0 + 0.1 * np.sin(2 * np.pi * 4.0 * t)
                    raw *= bow_speed

                    # bow noise
                    bow_noise = np.random.normal(0, 0.1, samples)

                    if HAS_SCIPY:
                        b, a = signal.butter(1, 3000/(FS/2), btype='low')
                        bow_noise = signal.lfilter(b, a, bow_noise)

                    raw += bow_noise * 0.2

                    # dual body resonance (1k + 3k)
                    if HAS_SCIPY:
                        b1, a1 = signal.iirpeak(1000, 3.0, fs=FS)
                        b2, a2 = signal.iirpeak(3000, 4.0, fs=FS)

                        body1 = signal.lfilter(b1, a1, raw)
                        body2 = signal.lfilter(b2, a2, raw)

                        raw = raw * 0.6 + body1 * 0.2 + body2 * 0.2

                    raw = np.tanh(raw * 1.5)

                    wave_out = apply_env(raw, note.duration, 0.08, 0.1, 0.9, 0.15)



                elif wave_type == "bagpipe":
                    # === Bagpipe (Improved with Drones) ===

                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)
                    f = note.pitch

                    w1 = 2.0 * (t * f - np.floor(t * f + 0.5))
                    w2 = 2.0 * (t * f*1.002 - np.floor(t * f*1.002 + 0.5))
                    w3 = np.sign(np.sin(2*np.pi*f*t)) * 0.5

                    raw = (w1 + w2 + w3) * 0.33

                    # add drones (critical for realism)
                    drone1 = 0.4 * np.sin(2 * np.pi * (f/2) * t)
                    drone2 = 0.3 * np.sin(2 * np.pi * (f/4) * t)

                    raw += drone1 + drone2

                    if HAS_SCIPY:
                        b1, a1 = signal.iirpeak(1200, 3.0, fs=FS)
                        b2, a2 = signal.iirpeak(2500, 5.0, fs=FS)

                        body1 = signal.lfilter(b1, a1, raw)
                        body2 = signal.lfilter(b2, a2, raw)

                        raw = raw * 0.6 + body1 * 0.2 + body2 * 0.2

                    wave_out = apply_env(raw, note.duration, 0.005, 0.0, 1.0, 0.005)

                    wave_out = np.tanh(wave_out * 2.0)



                elif wave_type == "harp":
                    # === Celtic Harp (Improved Brightness Decay) ===

                    t = np.linspace(0, note.duration, samples, False, dtype=np.float32)

                    decay_factor = 0.998
                    wave_out = generate_karplus_custom(
                        note.pitch,
                        note.duration,
                        decay=decay_factor
                    )

                    # time-dependent brightness decay
                    brightness = np.exp(-t * 2.0)
                    wave_out *= (0.7 + 0.3 * brightness)

                    wave_out = VocalDSP.apply_3band_eq(wave_out, 1.0, 2.0, 0.0, FS)

                    wave_out = apply_env(wave_out, note.duration, 0.005, 0.1, 0.8, 0.5)

                
                elif "strings" in role:
                    # === Ensemble Strings (Improved) ===

                    # slight detune for ensemble width
                    detune = np.random.uniform(0.997, 1.003)

                    base = VocalDSP.generate_string_wave(
                        note.pitch * detune,
                        note.duration
                    )

                    # slow vibrato (section feel)
                    t = np.linspace(0, note.duration, len(base), False)
                    vib = 1.0 + 0.003 * np.sin(2 * np.pi * 5.5 * t)
                    base *= vib

                    # mild body resonance (warmth)
                    if HAS_SCIPY:
                        b, a = signal.iirpeak(600, 2.5, fs=FS)
                        body = signal.lfilter(b, a, base)
                        base = base * 0.7 + body * 0.3

                    # soft saturation
                    base = np.tanh(base * 1.2)

                    wave_out = apply_env(base, note.duration, 0.15, 0.15, 0.85, 0.4)



                elif wave_type == "supersaw":
                    # === Supersaw (VCF搭載でよりエレクトロに！) ===
                    
                    # 1. オシレーターから生の波形を生成
                    wave_out = generate_supersaw(note.pitch, note.duration)

                    # subtle slow drift (analog feel)
                    t = np.linspace(0, note.duration, len(wave_out), False)
                    drift = 1.0 + 0.002 * np.sin(2 * np.pi * 0.5 * t)
                    wave_out *= drift

                    # 2. フィルター用エンベロープを生成 (プラック感のある短い減衰)
                    # Attack: 0.01, Decay: 0.2, Sustain: 0.1, Release: 0.2
                    vcf_env = VocalDSP.get_env_array(note.duration, 0.01, 0.2, 0.1, 0.2, FS)
                    
                    # 3. VCFを適用！ (レゾナンス q=1.5 で少し「ミョン」と鳴らせる)
                    # base_freqを低くし、エンベロープで高域まで開く
                    wave_out = VocalDSP.apply_vcf(
                        wave_out, 
                        vcf_env, 
                        base_freq=300.0,      # 閉じた時の周波数
                        env_amount=6000.0,    # エンベロープ最大時の開き具合
                        q=1.5,                # レゾナンス（エグみ）
                        fs=FS
                    )

                    # 4. 最後に音量のエンベロープを適用
                    wave_out = apply_env(wave_out, note.duration, 0.03, 0.12, 0.75, 0.15)
                    
                    # gentle saturation for glue
                    wave_out = np.tanh(wave_out * 1.5)




                elif wave_type == "karplus":
                    # === Karplus-Strong (Improved Realism) ===

                    # longer, more natural decay variation
                    decay = 0.97 + np.random.uniform(-0.005, 0.005)

                    wave_out = generate_karplus_custom(
                        note.pitch,
                        note.duration,
                        decay=decay
                    )

                    t = np.linspace(0, note.duration, len(wave_out), False)

                    # brightness decay (important for realism)
                    brightness = np.exp(-t * 3.0)
                    wave_out *= (0.6 + 0.4 * brightness)

                    # subtle body resonance
                    if HAS_SCIPY:
                        b, a = signal.iirpeak(800, 2.0, fs=FS)
                        body = signal.lfilter(b, a, wave_out)
                        wave_out = wave_out * 0.75 + body * 0.25

                    # soft clip for attack realism
                    wave_out = np.tanh(wave_out * 1.3)

                    wave_out = apply_env(wave_out, note.duration, 0.015, 0.12, 0.85, 0.2)

                else:
                    # 汎用シンセ
                    wave_out = fast_osc_custom(note.pitch, note.duration, wave_type)
                    if "bass" in role: 
                        wave_out = apply_env(wave_out, note.duration, 0.01, 0.1, 0.6, 0.05)
                    elif "pad" in role: 
                        wave_out = apply_env(wave_out, note.duration, 0.5, 0.2, 0.8, 0.5)
                    elif "arp" in role or "lead" in role: 
                        wave_out = apply_env(wave_out, note.duration, 0.01, 0.1, 0.4, 0.1)
                    else: 
                        wave_out = apply_env(wave_out, note.duration, 0.02, 0.1, 0.7, 0.1)

        # Mix to Track Buffer (★重要: ここでインデントを戻す！)
        if len(wave_out) > 0:
            src_len = len(wave_out)
            mix_len = len(track_buffer)
            if start_idx < mix_len:
                end_idx = start_idx + src_len
                if end_idx <= mix_len:
                    track_buffer[start_idx:end_idx] += wave_out * gain
                else:
                    copy_len = mix_len - start_idx
                    track_buffer[start_idx:] += wave_out[:copy_len] * gain

    # --- Post-Processing Loop (ここからインデントをさらに1つ戻す) ---
    if use_lfo:
        t_vec = np.linspace(0, total_samples/FS, total_samples, False, dtype=np.float32)
        lfo_wave = np.sin(2 * np.pi * lfo_rate * t_vec)
        mod = 1.0 - (lfo_depth * 0.5 * (1.0 + lfo_wave))
        track_buffer *= mod

    # Sidechain (Pump)
    if should_sidechain:
        samples_per_beat = int((60/BPM) * FS)
        duck_env = np.ones(total_samples, dtype=np.float32)
        rel_s = int(FS * sc_release)
        
        sc_mult = 1.0
        if "pad" in role or "strings" in role: sc_mult = 0.6
        elif "bass" in role: sc_mult = 0.9 
        
        eff_strength = sc_strength * sc_mult
        
        if rel_s > 0 and eff_strength > 0.01:
            # Sidechain curve generation
            t_curve = np.linspace(0, 1, rel_s)
            curve = 1.0 - (np.exp(-t_curve * 5.0) * eff_strength)
            
            # Apply to beats
            beat_positions = np.arange(0, total_samples, samples_per_beat)
            for i in beat_positions:
                end = min(i + rel_s, total_samples)
                if end > i:
                    duck_env[i:end] = np.minimum(duck_env[i:end], curve[:end-i])
            track_buffer *= duck_env

    # DSP Chain & Panning
    stereo = stereo_pan(track_buffer, spec["pan"])
    dsp_chain = spec.get("dsp", [])
    if isinstance(dsp_chain, str): dsp_chain = [dsp_chain]

    if ("pad" in role or "strings" in role) and "wide" not in dsp_chain:
        stereo = apply_haas_widener(stereo, 15.0)
        stereo = apply_stereo_detune(stereo)

    for dsp_name in dsp_chain:
        if dsp_name == "wide": 
            stereo = apply_haas_widener(stereo)
            stereo = apply_stereo_detune(stereo)
            # ▼▼▼ ここに追加 ▼▼▼
        elif dsp_name == "high_pass":
            # 簡易ハイパスフィルター（低音カット）
            # スネアの「パチン」感を出すのに有効
            stereo[:,0] = stereo[:,0] - np.roll(stereo[:,0], 1)
            stereo[:,1] = stereo[:,1] - np.roll(stereo[:,1], 1)
            # 音量が下がるので少し補正
            stereo *= 1.2
        # ▲▲▲ 追加終わり ▲▲▲
        elif dsp_name == "detune":
             stereo = apply_stereo_detune(stereo)
        elif dsp_name == "reverb_short": 
            # 新しいプレートリバーブ（短め）
            stereo = apply_plate_reverb(stereo, decay=0.2, mix=0.25, fs=FS)

        elif dsp_name == "reverb_long": 
            # 新しいプレートリバーブ（長め・深め）
            stereo = apply_plate_reverb(stereo, decay=0.8, mix=0.4, fs=FS)
            stereo[:,1] = apply_simple_delay(stereo[:,1], 0.35, 0.3)
        elif dsp_name == "delay": 
            # 新しいテープディレイ
            stereo = apply_tape_delay(stereo, delay_time=0.375, feedback=0.5, mix=0.4, fs=FS)
        elif dsp_name == "dist": stereo = apply_distortion(stereo)
        elif dsp_name == "bitcrush": 
            bc_params = spec.get("bitcrush_params", {})
            stereo = apply_bitcrush(stereo, bit_depth=bc_params.get("depth", 12))
        elif dsp_name == "comp": stereo = np.tanh(stereo) 
        elif dsp_name == "lpf": 
            stereo[:,0] = (stereo[:,0] + np.roll(stereo[:,0], 1)) * 0.5
            stereo[:,1] = (stereo[:,1] + np.roll(stereo[:,1], 1)) * 0.5
            
    # Apply 3-Band EQ
    eq_params = spec.get("eq_params", {"low": 0.0, "mid": 0.0, "high": 0.0})
    if any(v != 0.0 for v in eq_params.values()):
        stereo[:, 0] = VocalDSP.apply_3band_eq(stereo[:, 0], eq_params["low"], eq_params["mid"], eq_params["high"], FS)
        stereo[:, 1] = VocalDSP.apply_3band_eq(stereo[:, 1], eq_params["low"], eq_params["mid"], eq_params["high"], FS)

    return stereo * spec["vol"]


def generate_poly_stem(spec, seed_offset, engine_config, cb):
    """
    Wrapper / Orchestrator Function.
    Modified to respect existing sequences if 'keep_sequence' flag is set.
    """
    global BPM
    
    # 1. Composition Phase
    # Check if we should preserve the existing sequence (Manual Edit Mode)
    if spec.get("keep_sequence", False) and "sequence" in spec:
        sequence = spec["sequence"]
        # Reset the flag so next "Regen" button press will overwrite it, 
        # unless we want "Lock" feature. For now, assume one-shot preservation.
        # spec["keep_sequence"] = False # Optional: Auto-reset
    else:
        # Generate new sequence
        sequence = compose_track_sequence(spec, seed_offset, engine_config, BPM)
        spec["sequence"] = sequence 
    
    # 2. Setup Render Buffer
    total_bars = len(spec["chords"])
    bar_dur = (60 / BPM) * 4
    total_samples = int(FS * bar_dur * total_bars)
    
    # 3. Rendering Phase
    audio = render_track_sequence(sequence, spec, engine_config, total_samples, seed_offset)
    
    return audio

def apply_ms_processing(stereo_wave, width_factor=1.3):
    """
    Mid/Side Processing for Stereo Widening.
    Mid = (L+R), Side = (L-R).
    Boosts Side component to create width without muddying the center.
    """
    # Mid (Center information: Kick, Bass, Vocal Center)
    mid = (stereo_wave[:, 0] + stereo_wave[:, 1]) * 0.5
    
    # Side (Stereo information: Reverb, Wide Synths)
    side = (stereo_wave[:, 0] - stereo_wave[:, 1]) * 0.5
    
    # Widen!
    side *= width_factor
    
    # Decode back to L/R
    l = mid + side
    r = mid - side
    return np.column_stack((l, r))

def apply_tilt_eq(wave_data, slope=0.2, fs=48000):
    """
    Tilt EQ: Shifts tonal balance.
    Positive slope = Brighter (High Boost, Low Cut).
    """
    if not HAS_SCIPY: return wave_data
    
    # Simple approach: High Shelf Boost + slight Low Cut
    # 1. High Shelf at 2.5kHz
    sos_high = signal.butter(1, 2500/(fs/2), btype='high', output='sos')
    highs = signal.sosfilt(sos_high, wave_data, axis=0)
    
    # Mix highs back in (Exciter effect)
    # slope 0.2 means +20% high presence
    brightened = wave_data + highs * slope
    
    # 2. Gentle Low Cut (Highpass) at 40Hz to remove mud/DC
    sos_low = signal.butter(1, 40/(fs/2), btype='high', output='sos')
    cleaned = signal.sosfilt(sos_low, brightened, axis=0)
    
    return cleaned

def mastering_limiter(mix_data, fs):
    """
    Advanced Mastering Chain:
    1. Glue Compression (Saturation)
    2. Tilt EQ (Brightening)
    3. Multiband Processing (Low/High split)
    4. M/S Widening
    5. Final Brickwall Limiter
    """
    # 1. Glue Compression (Soft Saturation)
    # Brings low-level details up and smooths peaks
    glue = np.tanh(mix_data * 1.1)
    
    # 2. Tilt EQ (Air/Brightness)
    # Add a bit of "shimmer"
    eq_mix = apply_tilt_eq(glue, slope=0.3, fs=fs)
    
    # 3. Multiband Split
    fc = 150.0
    if HAS_SCIPY:
        b, a = signal.butter(2, fc / (fs / 2), 'low')
        low_band = signal.lfilter(b, a, eq_mix, axis=0)
    else:
        # Fast Moving Average Fallback
        w_size = int(fs / (fc * 2.5)) 
        if w_size < 1: w_size = 1
        kernel = np.ones(w_size) / w_size
        low_band = np.zeros_like(eq_mix)
        low_band[:, 0] = np.convolve(eq_mix[:, 0], kernel, mode='same')
        low_band[:, 1] = np.convolve(eq_mix[:, 1], kernel, mode='same')

    high_band = eq_mix - low_band

    # Process Bands
    # Low: Mono-ize (Tighten bass) & Boost
    low_mono = (low_band[:, 0] + low_band[:, 1]) * 0.5
    low_band[:, 0] = low_mono * 1.2 # Boost Bass
    low_band[:, 1] = low_mono * 1.2
    low_band = np.tanh(low_band) # Saturate Bass

    # High: Widen (M/S Processing)
    high_band = apply_ms_processing(high_band, width_factor=1.3)
    
    # 4. Mix
    master = low_band + high_band
    
    # 5. Final Limiter (Maximizer)
    # Hard clip at -0.1dB after pushing gain
    drive_gain = 1.2
    master = master * drive_gain
    master = np.clip(master, -0.99, 0.99)
    
    return master.astype(np.float32)

# ==========================================
# End of Part 2
# ==========================================
# ==========================================
# HyperNekoProduct v16.1 (Fixed Version)
# Part 3: MIDI, SE, Live Synth, and UI Popups
# ==========================================

# ==========================================
# 10. MIDI Export Module
# ==========================================
def export_midi_file(filename="output.mid", seed_offset=0, engine_config=None):
    """
    Exports the current arrangement to a Standard MIDI File (SMF).
    """
    if not HAS_MIDO:
        print("MIDI export skipped (mido not installed).")
        return False

    # Use global TRACK_SPECS
    track_specs = TRACK_SPECS
    if not track_specs: return False
    
    try:
        mid = MidiFile()
        # Tempo Track
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_bpm = int(60000000 / BPM)
        tempo_track.append(MetaMessage('set_tempo', tempo=tempo_bpm))
        tempo_track.append(MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Process each track
        for i, spec in enumerate(track_specs):
            if not spec.get("initial_active", True): continue
            
            track = MidiTrack()
            mid.tracks.append(track)
            
            role = spec["role"]
            is_drum = spec.get("is_drum", False)
            chords = spec["chords"]
            div_map = spec.get("div_map", {})
            pattern_map = spec.get("pattern_map", {})
            fallback_div = spec.get("div", 16)
            key_offset = spec.get("key_offset", 0)
            style = spec.get("style", "Normal")
            
            channel = 9 if is_drum else (i % 15)
            if channel >= 9 and not is_drum: channel += 1 
            
            track.append(MetaMessage('track_name', name=role))
            track.append(Message('program_change', program=0, channel=channel, time=0))
            
            ticks_per_beat = mid.ticks_per_beat
            ticks_per_bar = ticks_per_beat * 4
            
            global_note_counter = 0
            current_time = 0
            last_event_time = 0

            for bar_idx, chord_str in enumerate(chords):
                current_sec_code = "V" 
                if 'SECTION_TIMINGS' in globals() and SECTION_TIMINGS:
                    for s_bar, s_code in reversed(SECTION_TIMINGS):
                        if bar_idx >= s_bar: 
                            current_sec_code = s_code
                            break

                if "active_sections" in spec and current_sec_code not in spec["active_sections"]:
                    current_time += ticks_per_bar
                    continue

                current_div = int(div_map.get(current_sec_code, fallback_div))
                steps = max(1, current_div)
                step_ticks = ticks_per_bar // steps
                base_pattern = pattern_map.get(current_sec_code, [0]*16)
                
                chord_info = None
                if chord_str:
                    root_c = chord_str.split('_')[0].replace('"', '').strip()
                    chord_info = parse_complex_chord(root_c, key_offset=key_offset)

                for step_i in range(steps):
                    pat_len = len(base_pattern)
                    hit = base_pattern[int((step_i / steps) * pat_len) % pat_len] if pat_len > 0 else 0
                    
                    if hit:
                        note_num = 60 
                        velocity = 100
                        if is_drum:
                            if "kick" in role: note_num = 36 
                            elif "snare" in role: note_num = 38
                            elif "hihat" in role: note_num = 42 
                            elif "crash" in role: note_num = 49 
                            elif "ride" in role: note_num = 51 
                        elif chord_info:
                            intervals = chord_info["intervals"]
                            oct_shift = 0 
                            if "bass" in role: oct_shift = -12
                            elif "pad" in role: oct_shift = 0
                            elif "lead" in role or "vocal" in role: oct_shift = 12
                            
                            iv_idx = 0
                            if style == "Up": iv_idx = global_note_counter % len(intervals)
                            elif style == "Down": iv_idx = (len(intervals) - 1 - (global_note_counter % len(intervals)))
                            elif style == "Rand": iv_idx = (global_note_counter * 7) % len(intervals)
                            
                            iv = intervals[iv_idx % len(intervals)]
                            root_name = chord_info["root"]
                            root_base = NOTE_INDEX.get(root_name, 0)
                            note_num = 48 + root_base + iv + oct_shift
                            
                        delta = current_time - last_event_time
                        track.append(Message('note_on', note=note_num, velocity=velocity, time=delta, channel=channel))
                        last_event_time = current_time
                        
                        dur_ticks = int(step_ticks * 0.9)
                        track.append(Message('note_off', note=note_num, velocity=0, time=dur_ticks, channel=channel))
                        last_event_time += dur_ticks
                        
                        current_time += (step_ticks - dur_ticks)
                        global_note_counter += 1
                    else:
                        current_time += step_ticks

        mid.save(filename)
        print(f"MIDI Exported: {filename}")
        return True
    except Exception as e:
        print(f"MIDI Export Error: {e}")
        return False

#=========================================
#midi import
def import_midi_to_sequence(filename, target_track_idx=0, base_lyric_text=""):
    """
    MIDIファイルを読み込み、HyperNekoエンジンの NoteEvent リストに変換する。
    歌詞は指定した文字列から1文字ずつ割り当てる（流し込み）。
    """
    if not HAS_MIDO:
        print("Error: mido not found.")
        return []

    try:
        mid = MidiFile(filename)
        sequence = []
        
        # 歌詞リストの準備（ひらがな/カタカナ抽出）
        import re
        lyrics_char_list = []
        if base_lyric_text:
            # 空白や改行を除去してリスト化
            cleaned_text = re.sub(r'\s+', '', base_lyric_text)
            lyrics_char_list = list(cleaned_text)
        
        # テンポ解析 (デフォルト120)
        tempo = 500000 # 120bpm in microseconds
        ticks_per_beat = mid.ticks_per_beat
        
        # トラック選択
        if target_track_idx >= len(mid.tracks):
            print(f"Track {target_track_idx} not found, using track 1")
            track = mid.tracks[1] if len(mid.tracks) > 1 else mid.tracks[0]
        else:
            track = mid.tracks[target_track_idx]

        # イベント解析
        abs_time = 0
        current_seconds = 0.0
        active_notes = {} # key: note_num, value: start_time_sec
        lyric_idx = 0
        
        for msg in track:
            # デルタタイムを秒に変換
            abs_time += msg.time
            # 秒 = (ticks * microseconds_per_beat) / (ticks_per_beat * 1,000,000)
            dt_seconds = (msg.time * tempo) / (ticks_per_beat * 1000000.0)
            current_seconds += dt_seconds

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            
            elif msg.type == 'note_on' and msg.velocity > 0:
                # ノート開始
                active_notes[msg.note] = {
                    'start': current_seconds,
                    'vel': msg.velocity / 127.0
                }
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # ノート終了
                if msg.note in active_notes:
                    start_data = active_notes.pop(msg.note)
                    duration = current_seconds - start_data['start']
                    
                    if duration > 0.01: # 極端に短いノイズは無視
                        # ピッチ計算 (MIDI Note -> Hz)
                        freq = 440.0 * (2 ** ((msg.note - 69) / 12.0))
                        
                        # 歌詞の割り当て（リストから順番に、尽きたら 'la'）
                        lyric = "la"
                        if lyric_idx < len(lyrics_char_list):
                            lyric = lyrics_char_list[lyric_idx]
                            lyric_idx += 1
                        
                        # NoteEvent作成
                        note = NoteEvent(
                            start_time=start_data['start'],
                            duration=duration,
                            pitch=freq,
                            velocity=start_data['vel'],
                            lyric=lyric
                        )
                        sequence.append(note)

        # 時間順にソート（必須）
        sequence.sort(key=lambda x: x.start_time)
        print(f"Imported {len(sequence)} notes from MIDI.")
        return sequence

    except Exception as e:
        print(f"MIDI Import Error: {e}")
        return []

# ==========================================
# 11. SE (Sound Effect) Generation
# ==========================================
def generate_se_wave(preset):
    type_ = preset.get("type", "fm")
    dur = preset.get("dur", 1.0)
    base_f = preset.get("base_freq", 440)
    samples = int(44100 * dur)
    t = np.linspace(0, dur, samples, False, dtype=np.float32)
    wave_out = np.zeros(samples, dtype=np.float32)

    # ==========================================
    # 1. あなたが発明した「神パラメータ」の解析
    # ==========================================
    freq_array = np.full(samples, base_f, dtype=np.float32)

    # --- ピッチエンベロープ（音程の急降下・上昇） ---
    pitch_env = preset.get("pitch_env")
    if pitch_env == "chromatic_up":
        freq_array *= (2.0 ** (t * 2.5)) # ギュイィィンと上がる！
    elif pitch_env == "semitone_down":
        freq_array *= (2.0 ** (-t * 1.5)) # ヒュ〜ンと下がる！
    elif pitch_env == "minor_fall":
        freq_array *= (2.0 ** (-3.0 * (t/dur) / 12.0)) # 闇落ち感（短3度ダウン）

    # --- アルペジオ（自動分散和音） ---
    arp = preset.get("arp")
    if arp == "major_third_up":
        # 0, +4半音, +7半音 を高速で繰り返すチャイム
        steps = np.floor((t * 12) % 3)
        intervals = np.array([0, 4, 7])
        shift = np.choose(steps.astype(int), intervals)
        freq_array *= (2.0 ** (shift / 12.0))
    elif arp == "octave_jump":
        # オクターブをピコピコ行き来する（アニメスタートシグナル）
        step = np.floor((t * 16) % 2) * 12
        freq_array *= (2.0 ** (step / 12.0))

    # 位相の計算
    phase = 2 * np.pi * np.cumsum(freq_array) / 44100

    # ==========================================
    # 2. ベース波形の生成
    # ==========================================
    if type_ == "fm":
        mod_f = preset.get("mod_freq", 2)
        mod_d = preset.get("mod_depth", 100)
        wave_str = preset.get("wave", "sin")

        mod_phase = 2 * np.pi * mod_f * t
        fm_phase = phase + np.sin(mod_phase) * mod_d

        if wave_str == "sqr": wave_out = np.sign(np.sin(fm_phase))
        elif wave_str == "saw": wave_out = 2.0 * (fm_phase / (2*np.pi) % 1.0) - 1.0
        else: wave_out = np.sin(fm_phase)
        wave_out *= np.exp(-t * (1.5/dur))

    elif type_ == "voice":
        wave_out = np.sign(np.sin(phase)) 
        wave_out *= np.exp(-t * (2.0/dur))
        if HAS_SCIPY: # 吸い込みボイスっぽくフィルターをかける
            b, a = signal.butter(2, [800/(44100/2), 2000/(44100/2)], btype='band')
            wave_out = signal.lfilter(b, a, wave_out)

    elif type_ == "bell":
        wave_out = np.sin(phase) + 0.5 * np.sin(phase * 2.78) + 0.2 * np.sin(phase * 4.1)
        wave_out *= np.exp(-t * 3.0)

    elif type_ == "saber":
        wave_out = 2.0 * (phase / (2*np.pi) % 1.0) - 1.0
        wave_out = np.tanh(wave_out * 3.0)
        wave_out *= np.exp(-t * 1.5)

    elif type_ == "noise":
        wave_out = np.random.uniform(-1, 1, samples).astype(np.float32)
        f_type = preset.get("filter", "lpf")
        if HAS_SCIPY:
            if f_type == "bpf": # ビルドアップ用バンドパス
                b, a = signal.butter(2, [1000/(44100/2), 3000/(44100/2)], btype='band')
                wave_out = signal.lfilter(b, a, wave_out)
            elif f_type == "hpf":
                b, a = signal.butter(2, 2000/(44100/2), btype='high')
                wave_out = signal.lfilter(b, a, wave_out)
            else:
                b, a = signal.butter(2, 2000/(44100/2), btype='low')
                wave_out = signal.lfilter(b, a, wave_out)
        wave_out *= np.exp(-t * (1.0/dur))

    # ==========================================
    # 3. 追加ギミック＆DSP（エフェクト）
    # ==========================================
    # 逆再生（リバース）
    if preset.get("reverse", False):
        wave_out = wave_out[::-1]

    # フェイク・サイドチェイン（Pump）
    sc_rate = preset.get("sidechain_rate")
    if sc_rate:
        pump = 0.5 - 0.5 * np.cos(2 * np.pi * sc_rate * t)
        wave_out *= pump

    # 特殊DSP群
    for efx in preset.get("dsp", []):
        if efx == "sub_boost":
            # ルートの1オクターブ下のサイン波を足して激重にする
            sub_phase = 2 * np.pi * np.cumsum(freq_array * 0.5) / 44100
            wave_out += np.sin(sub_phase) * 1.5
        elif efx == "dist": wave_out = np.tanh(wave_out * 4.0)
        elif efx == "bitcrush":
            steps = 8
            wave_out = np.round(wave_out * steps) / steps
        elif efx == "comp":
            wave_out = np.tanh(wave_out * 1.5)
            # ▼▼▼ ここを追加！！！（モノラルをステレオ2chに変換） ▼▼▼
    wave_out = np.column_stack((wave_out, wave_out))
    # ▲▲▲ ここまで追加 ▲▲▲

    return wave_out

# ==========================================
# 12. Live Synth Generation (XY Pad)
# ==========================================
def generate_live_grain(x, y, samples=1024, wave_type="supersaw"):
    """
    Generates a small audio grain for the Live XY Pad.
    x: Pitch (0.0-1.0) -> C3 to C6
    y: Filter/Modulation (0.0-1.0)
    """
    min_f, max_f = 130.81, 1046.5 # C3 to C6
    freq = min_f * (max_f / min_f) ** x

    raw = np.zeros(samples, dtype=np.float32)
    t = np.arange(samples) / FS

    if wave_type == "karplus": 
        # Inline Karplus-Strong
        N = int(FS / freq)
        N = max(2, N)
        buf = np.random.uniform(-1, 1, N).astype(np.float32)
        if N < samples:
            raw[:N] = buf
            for i in range(N, samples, N):
                block_end = min(i + N, samples)
                prev_block = raw[i-N:i]
                new_block = 0.5 * (prev_block + np.roll(prev_block, 1)) * 0.996
                raw[i:block_end] = new_block[:block_end - i]
        else:
             raw = buf[:samples]
            
    elif wave_type == "supersaw":
        for detune in [0.99, 1.0, 1.01]:
             raw += (2 * (t * freq * detune - np.floor(t * freq * detune + 0.5)))
        raw /= 3.0
        
    elif wave_type == "pop" or wave_type == "adult":
        # Vocal Grain
        raw = np.sign(np.sin(2*np.pi*freq*t))
        center_f = 300 + y * 2000
        width = 200 + y * 300
        if HAS_SCIPY:
             b, a = signal.butter(2, [center_f/(FS/2), (center_f + width)/(FS/2)], btype='band')
             raw = signal.lfilter(b, a, raw)
        else:
             raw *= (1 - y)
             raw = 0.8 * raw + 0.2 * np.roll(raw, 1)
             
    else: 
        # Standard Waves
        if wave_type == "sqr": raw = np.sign(np.sin(2*np.pi*freq*t))
        elif wave_type == "saw": raw = 2 * (t * freq - np.floor(t * freq + 0.5))
        elif wave_type == "noise": raw = np.random.uniform(-1, 1, samples)
        else: raw = np.sin(2*np.pi*freq*t)
        
    raw = raw.astype(np.float32)
    
    # Simple LPF based on Y-axis
    lpf_str = (1.0 - y) * 0.9 
    if lpf_str > 0.1:
        processed = raw.copy()
        iters = int(lpf_str * 5)
        for _ in range(iters):
             processed = (processed + np.roll(processed, 1)) * 0.5
    else:
        processed = raw

    # Windowing
    window = np.hanning(samples).astype(np.float32)
    processed *= window

    # Mono to Stereo
    return np.column_stack((processed, processed)) * 0.5

# ==========================================
# 13. UI Components (Popups)
# ==========================================

# --- Standard Popups ---

class ErrorPopup(ModalView):
    def __init__(self, title, message, **kwargs):
        super().__init__(size_hint=(0.8, 0.4), auto_dismiss=True, **kwargs)
        content = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content.add_widget(Label(text=title, font_size='20sp', color=(1,0.3,0.3,1), size_hint_y=0.3))
        content.add_widget(Label(text=message, halign='center', valign='middle', size_hint_y=0.4))
        btn = Button(text="CLOSE", size_hint_y=0.3, background_color=(0.4, 0.4, 0.4, 1))
        btn.bind(on_press=self.dismiss)
        content.add_widget(btn)
        self.add_widget(content)

class TextInputPopup(ModalView):
    def __init__(self, title, default_text, callback, **kwargs):
        super().__init__(size_hint=(0.8, 0.4), auto_dismiss=True, **kwargs)
        self.callback = callback
        box = BoxLayout(orientation='vertical', padding=20, spacing=20)
        box.add_widget(Label(text=title, size_hint_y=0.3))
        self.txt_input = TextInput(text=default_text, multiline=False, size_hint_y=0.3)
        box.add_widget(self.txt_input)
        
        btn_box = BoxLayout(spacing=10, size_hint_y=0.4)
        btn_cancel = Button(text="CANCEL", background_color=(0.5,0.5,0.5,1))
        btn_cancel.bind(on_press=self.dismiss)
        btn_save = Button(text="CONFIRM", background_color=(0,0.8,0.4,1))
        btn_save.bind(on_press=self.on_confirm)
        btn_box.add_widget(btn_cancel); btn_box.add_widget(btn_save)
        box.add_widget(btn_box); self.add_widget(box)

    def on_confirm(self, instance):
        self.callback(self.txt_input.text); self.dismiss()

class FileLoadPopup(ModalView):
    """
    Generic File Loader Popup.
    Fixed: Accepts 'filters' argument to switch between JSON (Presets) and Audio (Sampler).
    """
    def __init__(self, callback, filters=None, **kwargs):
        super().__init__(size_hint=(0.9, 0.9), auto_dismiss=True, **kwargs)
        self.callback = callback
        
        # Default to audio if not specified
        if filters is None:
            filters = ['*.wav', '*.mp3', '*.ogg', '*.m4a']
            
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with layout.canvas.before:
            Color(0.1, 0.1, 0.1, 1); Rectangle(pos=self.pos, size=self.size)
        
        from kivy.uix.filechooser import FileChooserListView
        
        # Try to open Download folder on Android for convenience, else CWD
        start_path = "/storage/emulated/0/Download"
        if not os.path.exists(start_path):
            start_path = os.getcwd()
            
        self.file_chooser = FileChooserListView(path=start_path, filters=filters, size_hint_y=0.9)
        layout.add_widget(self.file_chooser)
        
        btn_box = BoxLayout(size_hint_y=0.1, spacing=10)
        btn_cancel = Button(text="CANCEL", background_color=(0.5, 0.5, 0.5, 1))
        btn_cancel.bind(on_press=self.dismiss)
        
        btn_load = Button(text="LOAD", background_color=(0, 0.8, 0.4, 1))
        btn_load.bind(on_press=self.on_load)
        
        btn_box.add_widget(btn_cancel); btn_box.add_widget(btn_load)
        layout.add_widget(btn_box)
        self.add_widget(layout)
        
    def on_load(self, instance):
        selection = self.file_chooser.selection
        if selection:
            # Return the single selected filename string
            self.callback(selection[0])
            self.dismiss()

class SamplerConfigPopup(ModalView):
    def __init__(self, current_slots, callback, **kwargs):
        super().__init__(size_hint=(0.95, 0.9), auto_dismiss=True, **kwargs)
        self.slots = [dict(s) for s in current_slots]
        self.callback = callback
        self.app = App.get_running_app()
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with layout.canvas.before:
            Color(0.12, 0.12, 0.15, 1); Rectangle(pos=self.pos, size=self.size)
            
        layout.add_widget(Label(text="SAMPLER SETUP", font_size='20sp', bold=True, size_hint_y=0.1, color=(1, 0.5, 0.8, 1)))
        
        grid = GridLayout(cols=1, spacing=5, size_hint_y=0.75)
        self.ui_rows = []
        
        for i in range(6):
            row_box = BoxLayout(orientation='horizontal', spacing=5)
            slot = self.slots[i]
            
            row_box.add_widget(Label(text=f"Pad {i+1}", size_hint_x=0.1, font_size='12sp'))
            
            info_text = slot.get("name", "Empty")
            if slot.get("type") == "file": 
                info_text = "(File) " + os.path.basename(slot.get("path", ""))
            lbl_info = Label(text=info_text, size_hint_x=0.3, font_size='11sp', color=(0.8, 0.8, 0.8, 1))
            row_box.add_widget(lbl_info)
            
            spin_preset = Spinner(text="Preset", values=list(SE_PRESETS.keys()), size_hint_x=0.2)
            spin_preset.bind(text=lambda instance, val, idx=i: self.on_preset_select(idx, val))
            row_box.add_widget(spin_preset)
            
            btn_rec = ToggleButton(text="REC", size_hint_x=0.15, background_color=(0.8, 0.2, 0.2, 1), font_size='10sp')
            btn_rec.bind(state=lambda instance, val, idx=i: self.on_rec_toggle(instance, val, idx))
            row_box.add_widget(btn_rec)
            
            btn_file = Button(text="LOAD", size_hint_x=0.15, background_color=(0.2, 0.6, 0.8, 1), font_size='10sp')
            btn_file.bind(on_press=lambda x, idx=i: self.open_file_browser(idx))
            row_box.add_widget(btn_file)
            
            loop_box = BoxLayout(orientation='vertical', size_hint_x=0.1)
            chk_loop = CheckBox(active=slot.get("loop", False))
            chk_loop.bind(active=lambda instance, val, idx=i: self.on_loop_change(idx, val))
            loop_box.add_widget(chk_loop)
            loop_box.add_widget(Label(text="Loop", font_size='8sp'))
            row_box.add_widget(loop_box)
            
            grid.add_widget(row_box)
            self.ui_rows.append({"label": lbl_info, "rec": btn_rec})
            
        layout.add_widget(grid)
        
        btn_save = Button(text="SAVE & CLOSE", background_color=(0, 0.8, 0.4, 1), size_hint_y=0.15)
        btn_save.bind(on_press=self.on_save)
        layout.add_widget(btn_save)
        self.add_widget(layout)

    def on_preset_select(self, idx, val):
        self.slots[idx] = {"type": "preset", "name": val, "loop": self.slots[idx].get("loop", False)}
        self.ui_rows[idx]["label"].text = val

    def on_loop_change(self, idx, val):
        self.slots[idx]["loop"] = val

    def on_rec_toggle(self, instance, state, idx):
        if state == 'down':
            instance.text = "STOP"
            for r in self.ui_rows: 
                if r["rec"] != instance: r["rec"].disabled = True
            self.app.start_recording(idx)
        else:
            instance.text = "REC"
            path = self.app.stop_recording()
            if path:
                self.slots[idx] = {"type": "file", "path": path, "name": "Rec", "loop": self.slots[idx].get("loop", False)}
                self.ui_rows[idx]["label"].text = "(Rec) " + os.path.basename(path)
            for r in self.ui_rows: r["rec"].disabled = False

    def open_file_browser(self, idx):
        def on_file_selected(path):
            self.slots[idx] = {"type": "file", "path": path, "name": "File", "loop": self.slots[idx].get("loop", False)}
            self.ui_rows[idx]["label"].text = os.path.basename(path)
        FileLoadPopup(on_file_selected).open()

    def on_save(self, instance):
        self.callback(self.slots)
        self.dismiss()


class EngineConfigPopup(ModalView):
    def __init__(self, app_instance, **kwargs):
        super().__init__(size_hint=(0.85, 0.9), auto_dismiss=True, **kwargs)
        self.app = app_instance
        self.config = self.app.engine_config
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        with layout.canvas.before:
            Color(0.12, 0.12, 0.15, 1); Rectangle(pos=self.pos, size=self.size)
        
        layout.add_widget(Label(text="ENGINE CONFIG", font_size='20sp', bold=True, size_hint_y=0.1, color=(0.4, 0.8, 1, 1)))
        
        scroll = ScrollView(size_hint_y=0.75)
        grid = GridLayout(cols=1, spacing=15, size_hint_y=None, padding=[0, 10])
        grid.bind(minimum_height=grid.setter('height'))
        
        def create_slider_row(label, key, min_val, max_val, step=0.01):
            row = BoxLayout(orientation='vertical', size_hint_y=None, height=70)
        
            top = BoxLayout(size_hint_y=0.4)
            top.add_widget(Label(text=label, size_hint_x=0.7, halign='left'))
            val_lbl = Label(text=f"{self.config.get(key, 0):.2f}", size_hint_x=0.3, color=(1, 1, 0.5, 1))
            top.add_widget(val_lbl)
            row.add_widget(top)
            
            val = self.config.get(key, 0)
            val = max(min_val, min(max_val, val))
            
            slider = Slider(min=min_val, max=max_val, value=val, step=step, size_hint_y=0.6)
            def on_val(instance, value):
                self.config[key] = value
                val_lbl.text = f"{value:.2f}"
            slider.bind(value=on_val)
            
            row.add_widget(slider)
            return row

        grid.add_widget(Label(text="--- General ---", size_hint_y=None, height=30))
        grid.add_widget(create_slider_row("Master Volume", "master_vol", 0.0, 2.0))
        grid.add_widget(create_slider_row("Swing Amount", "swing_amount", 0.0, 0.5))
        grid.add_widget(create_slider_row("Velocity Humanize", "vel_humanize", 0.0, 0.5))
        
        grid.add_widget(Label(text="--- Vocal Settings (Global) ---", size_hint_y=None, height=30))
        grid.add_widget(create_slider_row("Vocal Gender (Formant)", "vocal_formant_shift", 0.5, 2.0))
        # [NEW] Added global sliders for Breath and Consonant
        grid.add_widget(create_slider_row("Breath Volume", "breath_vol", 0.0, 0.5))
        grid.add_widget(create_slider_row("Breath Chance", "breath_chance", 0.0, 1.0))
        grid.add_widget(create_slider_row("Consonant Attack", "cons_attack", 0.5, 2.0))
        
        grid.add_widget(Label(text="--- Sidechain ---", size_hint_y=None, height=30))
        grid.add_widget(create_slider_row("SC Strength (0=Off)", "sc_strength", 0.0, 1.0))
        grid.add_widget(create_slider_row("SC Release (sec)", "sc_release", 0.05, 0.5))
        
        grid.add_widget(Label(text="--- Global Reverb ---", size_hint_y=None, height=30))
        grid.add_widget(create_slider_row("Reverb Mix", "reverb_mix", 0.0, 1.0))
        grid.add_widget(create_slider_row("Reverb Decay", "reverb_decay", 0.1, 0.9))

        scroll.add_widget(grid)
        layout.add_widget(scroll)
        
        btn_close = Button(text="APPLY & CLOSE", size_hint_y=0.15, background_color=(0, 0.8, 0.4, 1))
        btn_close.bind(on_press=self.dismiss)
        layout.add_widget(btn_close)
        self.add_widget(layout)

# ==========================================
# 13.5 Harmony Style System (New Architecture)
# ==========================================
HARMONY_DB = {
    "emopop": {
        "desc": "Kawaii/Floating/Future",
        "roots": ["IV", "III", "vi", "I"],
        "extensions": ["maj7", "add9", "9", "6/9"],
        "transitions": {
            "IV": ["III", "V"], "III": ["vi"], "vi": ["I", "ii", "IV"], 
            "I": ["IV", "ii"], "ii": ["V"], "V": ["I", "vi"]
        },
        "quality_map": {"III": "7", "vi": "m7", "ii": "m7", "V": "7", "I": "maj7", "IV": "maj7"},
        "cadence_targets": ["I", "V"],
        "complexity": 0.85
    },
    "City Pop": {
        "desc": "Jazzy/Nostalgic/Urban",
        "roots": ["IV", "ii"],
        "extensions": ["maj7", "9", "13", "m9"],
        "transitions": {
            "IV": ["iii", "III"], "iii": ["vi"], "III": ["vi"], 
            "vi": ["ii"], "ii": ["V"], "V": ["I", "IV"], "I": ["IV", "ii"]
        },
        "quality_map": {"IV": "maj7", "iii": "m7", "III": "7", "vi": "m7", "ii": "m9", "V": "13", "I": "maj9"},
        "complexity": 0.9
    },
    "Capsule": {
        "desc": "Nakata/Minimal/Pedal",
        "roots": ["IV", "V", "iii", "vi"],
        "extensions": ["", "sus4", "maj7"],
        "pedal_bass": True, 
        "transitions": { "IV": ["V"], "V": ["iii", "I"], "iii": ["vi"], "vi": ["IV", "ii"] },
        "complexity": 0.3
    },
    "Komuro (TK)": {
        "desc": "Dramatic/Minor/Anthemic",
        "roots": ["vi", "IV", "V", "I"],
        "extensions": ["", "add9", "sus4"],
        "transitions": { "vi": ["IV", "ii"], "IV": ["V"], "V": ["I", "vi"], "I": ["V", "vi"] },
        "quality_map": {"V": "", "vi": "m"}, 
        "complexity": 0.4
    },
    "Vkei": {
        "desc": "Dark/Melodic/HarmonicMinor",
        "roots": ["vi", "IV", "V", "vi"], 
        "extensions": ["m", "dim", "aug"],
        "transitions": { "vi": ["ii", "IV"], "ii": ["V"], "V": ["vi", "I"], "IV": ["V", "dim"] },
        "quality_map": {"V": "7", "vi": "m"}, 
        "complexity": 0.6
    }
}

class HarmonyEngine:
    ROMAN_TO_INTERVAL = {
        "I": 0, "ii": 2, "II": 2, "iii": 4, "III": 4,
        "IV": 5, "iv": 5, "V": 7, "vi": 9, "VI": 9, "vii": 11
    }
    CHROMATIC_SCALE = NOTE_ORDER

    def __init__(self, key="C", rng_seed=None):
        self.key = key
        self.key_index = self.CHROMATIC_SCALE.index(key)
        self.rng = random.Random(rng_seed)

    def _get_base_note(self, roman):
        interval = self.ROMAN_TO_INTERVAL.get(roman, 0)
        note_index = (self.key_index + interval) % 12
        return self.CHROMATIC_SCALE[note_index]

    def _build_chord_string(self, base, roman, style):
        quality = ""
        if "force_quality" in style:
            quality = style["force_quality"]
        elif "quality_map" in style and roman in style["quality_map"]:
            quality = style["quality_map"][roman]
        else:
            if roman in ["ii", "iii", "vi"]: quality = "m"
            elif roman in ["vii"]: quality = "dim"

        if self.rng.random() < style.get("complexity", 0.5):
            ext_pool = style.get("extensions", [""])
            ext = self.rng.choice(ext_pool)
            if ext:
                if any(x in ext for x in ["maj", "m", "dim", "aug"]):
                    quality = ext
                elif "7" in ext and quality == "m":
                    quality = "m" + ext if "m" not in ext else ext
                else:
                    quality += ext

        if style.get("pedal_bass", False) and self.rng.random() > 0.4:
            if base != self.key: 
                quality += f"/{self.key}"

        return f"{base}{quality}"

    def generate_section(self, style_name, bars=4, harmonic_rhythm=True):
        style = HARMONY_DB.get(style_name, HARMONY_DB["emopop"])
        progression = []
        
        if "fixed_progression" in style:
            prog = style["fixed_progression"]
            for i in range(bars):
                roman = prog[i % len(prog)]
                base = self._get_base_note(roman)
                chord_str = self._build_chord_string(base, roman, style)
                progression.append(chord_str)
            return progression

        current_roman = self.rng.choice(style.get("roots", ["I"]))
        transitions = style.get("transitions", {})
        
        for bar in range(bars):
            bar_chords = []
            
            # 和声的リズム: '_' を使って1小節内のコードを分割
            split_bar = harmonic_rhythm and bar < (bars - 1) and self.rng.random() < (style.get("complexity", 0.5) * 0.5)
            chords_in_bar = 2 if split_bar else 1
            
            for _ in range(chords_in_bar):
                if bar == bars - 1 and "cadence_targets" in style:
                    if current_roman not in style["cadence_targets"] and self.rng.random() > 0.3:
                         current_roman = self.rng.choice(style["cadence_targets"])

                base = self._get_base_note(current_roman)
                chord_str = self._build_chord_string(base, current_roman, style)
                bar_chords.append(chord_str)
                
                possible_next = transitions.get(current_roman, ["I", "V", "vi", "IV"])
                current_roman = self.rng.choice(possible_next)
                
            # ここで "_" 繋ぎにする！ (例: "Fmaj7_G")
            progression.append("_".join(bar_chords) if len(bar_chords) > 1 else bar_chords[0])

        return progression

    def generate_full_song_map(self, style_name):
        return {
            "I": self.generate_section(style_name, 4),
            "V": self.generate_section(style_name, 8, harmonic_rhythm=False),
            "V2": self.generate_section(style_name, 8, harmonic_rhythm=False),
            "C": self.generate_section(style_name, 8),
            "B": self.generate_section(style_name, 4),
            "O": self.generate_section(style_name, 4)
        }

# ==========================================
# HyperNekoProduct v16.1 (Fixed Version)
# Part 4: Widgets, Screens, App Logic, and Entry Point
# ==========================================
# ==========================================
# Audio Recorder (Robust Implementation)
# ==========================================
class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.is_android = (platform == 'android')
        self.temp_output = "/storage/emulated/0/Download/pydroid_rec.mp4"
        self.recorder = None

    def start(self):
        if self.recording: return
        self.recording = True
        
        if self.is_android:
            try:
                from jnius import autoclass
                # Androidの録音エンジンを呼び出す
                MediaRecorder = autoclass('android.media.MediaRecorder')
                AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
                OutputFormat = autoclass('android.media.MediaRecorder$OutputFormat')
                AudioEncoder = autoclass('android.media.MediaRecorder$AudioEncoder')

                self.recorder = MediaRecorder()
                self.recorder.setAudioSource(AudioSource.MIC)
                self.recorder.setOutputFormat(OutputFormat.MPEG_4)
                self.recorder.setAudioEncoder(AudioEncoder.AAC)
                self.recorder.setAudioSamplingRate(44100)
                self.recorder.setOutputFile(self.temp_output)
                
                self.recorder.prepare()
                self.recorder.start()
                logger.info(f"MediaRecorder started: {self.temp_output}")
            except Exception as e:
                logger.error(f"MediaRecorder Fail: {e}")
                self.recording = False
        else:
            logger.info("PC Recording (Dummy)")

    def stop(self, filename="user_voice.wav"):
        if not self.recording: return None
        self.recording = False

        if self.is_android and self.recorder:
            try:
                self.recorder.stop()
                self.recorder.release()
                self.recorder = None
                logger.info("Recording stopped and saved.")
                # 本来はWAV変換が必要ですが、まずはファイルが存在することを返す
                return self.temp_output
            except Exception as e:
                logger.error(f"MediaRecorder Stop Fail: {e}")
                return None
        return None

            
        # Save WAV
        try:
            raw_data = b''.join(self.frames)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                # 実際に録音で使われたレートを使用する（デフォルトはFS）
                rate = getattr(self, 'actual_fs', self.fs)
                wf.setframerate(rate)
                wf.writeframes(raw_data)
            return filename
                
            logger.info(f"Recording saved: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save wav: {e}")
            return None

    def _pc_record_loop(self):
        """PC Loop using PyAudio"""
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.fs,
                input=True,
                frames_per_buffer=1024
            )
            while self.recording:
                data = self.stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)
        except ImportError:
            logger.error("PyAudio not installed.")
        except Exception as e:
            logger.error(f"PC Rec Error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pa:
                self.pa.terminate()

    def _android_record_loop(self):
        try:
            from jnius import autoclass
            AudioRecord = autoclass('android.media.AudioRecord')
            AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
            AudioFormat = autoclass('android.media.AudioFormat')
            
            # 試行するサンプリングレートの優先順位
            # 44100はAndroidで最も標準的、16000は音声認識などで使われる
            sample_rates = [44100, 48000, 16000, 8000]
            recorder = None
            final_rate = 44100

            for rate in sample_rates:
                try:
                    min_buf = AudioRecord.getMinBufferSize(
                        rate, 
                        AudioFormat.CHANNEL_IN_MONO, 
                        AudioFormat.ENCODING_PCM_16BIT
                    )
                    if min_buf <= 0: continue
                    
                    # インスタンス作成
                    r = AudioRecord(
                        AudioSource.MIC, 
                        rate,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        min_buf * 2
                    )
                    
                    # ★ここで初期化状態をチェック
                    if r.getState() == AudioRecord.STATE_INITIALIZED:
                        recorder = r
                        final_rate = rate
                        logger.info(f"Successfully initialized AudioRecord at {rate}Hz")
                        break
                    else:
                        r.release()
                except Exception:
                    continue

            if not recorder:
                logger.error("All AudioRecord attempts failed.")
                self.recording = False
                return

            self.audio_recorder = recorder
            # 実際に使われたレートを保存（後でWAV保存時に必要）
            self.actual_fs = final_rate 
            
            recorder.startRecording()
            logger.info("Recording actually started!")
            
            buffer_size = 2048
            buffer_byte = bytearray(buffer_size)
            
            while self.recording:
                read_result = recorder.read(buffer_byte, 0, buffer_size)
                if read_result > 0:
                    self.frames.append(bytes(buffer_byte[:read_result]))
                elif read_result < 0:
                    logger.error(f"Read error: {read_result}")
                    break
                    
            recorder.stop()
            recorder.release()
            
        except Exception as e:
            logger.error(f"Critical Android Rec Error: {e}")
        finally:
            self.recording = False

# ==========================================
# 15. Custom Widgets (XYPad, TrackWidget)
# ==========================================

class XYPad(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_touch = None
        self.num_strings = 6
        self.last_string = -1  # 最後に弾いた弦を記憶
        
        with self.canvas:
            Color(0.2, 0.2, 0.25, 1)
            self.bg = Rectangle(pos=self.pos, size=self.size)
            
            # ▼▼▼ 十字線の代わりに「6本の弦」を描画 ▼▼▼
            Color(0.6, 0.6, 0.6, 0.5)
            self.string_lines = [Line(points=[], width=1.5) for _ in range(self.num_strings)]
            
            Color(1, 0.5, 0.2, 1)
            self.cursor = Ellipse(size=(20, 20), pos=(-100, -100))
            
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.app = App.get_running_app()

    def update_canvas(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
        # 画面の高さに合わせて弦の位置を更新
        for i in range(self.num_strings):
            sy = self.y + (i + 0.5) * (self.height / self.num_strings)
            self.string_lines[i].points = [self.x, sy, self.x + self.width, sy]

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.active_touch = touch
            self.last_string = -1 # タッチ開始時にリセット
            self.update_cursor(touch)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch == self.active_touch:
            self.update_cursor(touch)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch == self.active_touch:
            self.active_touch = None
            self.cursor.pos = (-100, -100)
            self.last_string = -1
            return True
        return super().on_touch_up(touch)

    def update_cursor(self, touch):
        x, y = touch.pos
        norm_x = (x - self.x) / self.width
        norm_y = (y - self.y) / self.height
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        cx = self.x + norm_x * self.width
        cy = self.y + norm_y * self.height
        self.cursor.pos = (cx - 10, cy - 10)
        
        # ▼▼▼ ストローク判定ロジック ▼▼▼
        # Y座標から「今どの弦の上にいるか(0〜5)」を計算
        current_string = int(norm_y * self.num_strings)
        current_string = max(0, min(self.num_strings - 1, current_string))
        
        # 前のフレームと違う弦の上にいれば「弦を弾いた（跨いだ）」と判定！
        if current_string != self.last_string:
            self.last_string = current_string
            if self.app and hasattr(self.app, 'play_guitar_strum'):
                # アプリ側に「X座標（コード指定用）」と「弾いた弦番号」を渡す
                self.app.play_guitar_strum(norm_x, current_string)

# ==========================================
# UI: Track Settings Popup (Restored & Enhanced)
# ==========================================
class TrackSettingsPopup(ModalView):
    """
    Detailed settings for a single track.
    Restored features from v4 (Rhythm Editor, LFO, Bitcrusher) 
    and integrated v5 features (AI Fills, Chorus Octave).
    """
    def on_import_midi(self, instance):
        """MIDIファイルを読み込み、歌詞を入力してトラックに適用する"""
        
        def on_file_selected(filename):
            if not filename: return
            
            # ファイル選択後に「歌詞入力ポップアップ」を出す
            def on_lyrics_entered(text):
                # 解析実行
                new_sequence = import_midi_to_sequence(filename, base_lyric_text=text)
                
                if new_sequence:
                    # トラック情報を上書き
                    self.spec["sequence"] = new_sequence
                    self.spec["keep_sequence"] = True # AI自動生成をブロック
                    
                    # 画面を閉じて再生成
                    self.callback()
                    self.dismiss()
                else:
                    # エラー時は閉じるだけ（ログは出る）
                    pass

            # デフォルト歌詞
            default_lyric = "ららら"
            # もし現在のセクション設定に歌詞があればそれを初期値にする
            # (これは簡易的な取得方法です)
            
            TextInputPopup("Enter Lyrics (Hiragana):", default_lyric, on_lyrics_entered).open()

        # MIDIファイルのみ表示するフィルタ
        FileLoadPopup(on_file_selected, filters=['*.mid', '*.midi']).open()
    
    def __init__(self, spec, callback, **kwargs):
        super().__init__(size_hint=(0.95, 0.95), auto_dismiss=True, **kwargs)
        self.spec = spec
        self.callback = callback
        
        # --- Deep copies for editing (v4 Logic) ---
        self.pattern_map_edit = copy.deepcopy(spec.get("pattern_map", {}))
        self.lfo_params_edit = copy.deepcopy(spec.get("lfo_params", {"active": False, "rate": 1.0, "depth": 0.0}))
        self.bitcrush_params_edit = copy.deepcopy(spec.get("bitcrush_params", {"rate": 0.2, "depth": 12}))
        
        default_div = spec.get("div", 16)
        self.div_map_edit = copy.deepcopy(spec.get("div_map", {k: default_div for k in ["I", "V", "V2", "C", "B", "O"]}))
        
        # ▼▼▼ 追加: style_map の編集用コピーを作成 ▼▼▼
        default_style = spec.get("style", "Normal")
        self.style_map_edit = copy.deepcopy(spec.get("style_map", {k: default_style for k in ["I", "V", "V2", "C", "B", "O"]}))
        # ▲▲▲ 追加終わり ▲▲▲
        # ▼▼▼ 新規追加: gate_map の編集用コピーを作成 ▼▼▼
        self.gate_map_edit = copy.deepcopy(spec.get("gate_map", {}))
        # ▲▲▲ 新規追加終わり ▲▲▲

        # Updated Vocal Params
        default_vocal_params = {
            "attack": 0.05, "release": 0.1, 
            "vibrato_depth": 0.05, "vibrato_rate": 5.0, 
            "reverb_send": 0.0,
            "breath_vol": 0.05, "breath_chance": 0.5, "cons_attack": 1.0,
            "whisper_amount": 0.0, "scoop_amount": 0.2
        }
        self.vocal_params_edit = copy.deepcopy(spec.get("vocal_params", default_vocal_params))
        for k, v in default_vocal_params.items():
            if k not in self.vocal_params_edit: self.vocal_params_edit[k] = v
        
        self.current_edit_section = "V"
        self.is_vocal = "vocal" in spec["role"]
        self.is_drum = spec.get("is_drum", False)

        # --- Layout Setup ---
        layout = BoxLayout(orientation='vertical', padding=10, spacing=5)
        with layout.canvas.before:
            Color(0.15, 0.15, 0.18, 1); Rectangle(pos=self.pos, size=self.size)

        # -- Header --
        header = BoxLayout(orientation='horizontal', size_hint_y=0.08)
        header.add_widget(Label(text=f"Settings: {spec['role'].upper()}", font_size='18sp', bold=True, halign='left'))
        
        self.chk_xy = CheckBox(active=spec.get("xy_route", False), size_hint_x=0.2)
        header.add_widget(Label(text="XY Route", size_hint_x=0.2, font_size='11sp', color=(0, 0.8, 1, 1)))
        header.add_widget(self.chk_xy)
        layout.add_widget(header)

        # -- Main Scroll View --
        scroll = ScrollView(size_hint_y=0.8)
        content_box = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=5)
        content_box.bind(minimum_height=content_box.setter('height'))

        # -----------------------------
        # 1. Section & Basic Params (v4 Style)
        # -----------------------------
        sec_sel_box = BoxLayout(size_hint_y=None, height=40)
        sec_sel_box.add_widget(Label(text="EDIT SECTION:", size_hint_x=0.3, color=(1, 0.8, 0.2, 1), bold=True))
        self.spin_sec_edit = Spinner(text="V", values=["I", "V", "V2", "C", "B", "O"], size_hint_x=0.7)
        self.spin_sec_edit.bind(text=self.on_edit_section_change)
        sec_sel_box.add_widget(self.spin_sec_edit)
        content_box.add_widget(sec_sel_box)

        basic_grid = GridLayout(cols=2, spacing=5, size_hint_y=None, height=80)
        
        basic_grid.add_widget(Label(text="Div (Current Sec):", halign='right'))
        init_div = str(self.div_map_edit.get("V", 16))
        self.spin_div = Spinner(text=init_div, values=[str(x) for x in [1, 2, 4, 8, 12, 16, 20, 24, 32]], size_hint_x=0.6)
        self.spin_div.bind(text=self.on_div_change)
        basic_grid.add_widget(self.spin_div)
        
        basic_grid.add_widget(Label(text="Wave Type:", halign='right'))
        self.spin_wave = Spinner(text=spec.get("wave", "sin"), 
            values=("sin", "saw", "sqr", "noise", "supersaw", "karplus", 
                    "pop", "adult", "whisper", "utada",
                    "piano", "legacy_piano", "heavy_guitar", "organ", "musicbox",
                    "fm_bass", "fm_bell", "pluck", 
                    "celtic_flute", "fiddle", "bagpipe", "harp"),   # <--- ここに追加！
            size_hint_x=0.6)
        basic_grid.add_widget(self.spin_wave)
         # ▼▼▼ ここに新規追加: Note Length スライダー ▼▼▼
        basic_grid.add_widget(Label(text="Note Length (Gate):", halign='right', color=(0.6, 1.0, 0.8, 1)))
        
        len_box = BoxLayout(size_hint_x=0.6)
        # 初期値は 1.0 (等倍)。0.1倍(極短) から 4.0倍(超ロング) まで調整可能にする
        current_len = spec.get("note_len", 1.0)
        self.lbl_note_len = Label(text=f"x{current_len:.1f}", size_hint_x=0.3)
        self.slider_note_len = Slider(min=0.1, max=4.0, value=current_len, step=0.1, size_hint_x=0.7)
        self.slider_note_len.bind(value=lambda i,v: setattr(self.lbl_note_len, 'text', f"x{v:.1f}"))
        
        len_box.add_widget(self.slider_note_len)
        len_box.add_widget(self.lbl_note_len)
        basic_grid.add_widget(len_box)
        # ▲▲▲ 新規追加終わり ▲▲▲
        content_box.add_widget(basic_grid)

        # Style & Rhythm
        style_box = BoxLayout(size_hint_y=None, height=35)
        # ▼▼▼ 修正: ラベルを少し変更 ▼▼▼
        style_box.add_widget(Label(text="Arp/Style (Current Sec):", size_hint_x=0.4)) 
        # ▲▲▲ 修正終わり ▲▲▲
        self.spin_style = Spinner(text=spec.get("style", "Normal"), 
            values=("Normal", "Up", "Down", "Rand", "Legato", 
                    "Glides", "PingPong", "Root", "Octave","Pentatonic", 
                    "-1 Oct", "+1 Oct",
                    "Arp","PowerChord","FullChord","PowerChord_Low", "FullChord_Low","Piano_Comping"), size_hint_x=0.6)
                    # ▼▼▼ 追加: スタイル変更時のバインド ▼▼▼
        self.spin_style.bind(text=self.on_style_change)
        # ▲▲▲ 追加終わり ▲▲▲
        style_box.add_widget(self.spin_style)
        content_box.add_widget(style_box)
        
        # ▼▼▼ ここに新規追加: カスタムコード入力枠 ▼▼▼
        custom_chord_box = BoxLayout(size_hint_y=None, height=35)
        custom_chord_box.add_widget(Label(text="Custom Chords:", size_hint_x=0.4, color=(1, 0.6, 0.8, 1)))
        self.txt_custom_chords = TextInput(
            text=spec.get("custom_chords", ""), 
            hint_text="Override Global (e.g. Gm7, C7)", 
            multiline=False, size_hint_x=0.6
        )
        custom_chord_box.add_widget(self.txt_custom_chords)
        content_box.add_widget(custom_chord_box)
        # ▲▲▲ 新規追加終わり ▲▲▲
        # ▼▼▼ 新規追加: S/E ゲートマスク入力枠 ▼▼▼
        gate_box = BoxLayout(size_hint_y=None, height=35)
        gate_box.add_widget(Label(text="S/E Gate (Current Sec):", size_hint_x=0.4, color=(0.6, 1.0, 0.8, 1)))
        
        # 初期値は現在のセクション("V")のものを取得
        init_gate = self.gate_map_edit.get("V", "")
        self.txt_gate_mask = TextInput(
            text=init_gate,
            hint_text="e.g. ----S---E--- (S=Start, E=End)",
            multiline=False, size_hint_x=0.6
        )
        # テキストが変更されたらすぐ辞書に保存するバインド
        self.txt_gate_mask.bind(text=self.on_gate_mask_change)
        gate_box.add_widget(self.txt_gate_mask)
        content_box.add_widget(gate_box)
        # ▲▲▲ 新規追加終わり ▲▲▲    
        
        # Active Sections (Toggle Buttons)
        content_box.add_widget(Label(text="Active Sections (Global):", size_hint_y=None, height=20, color=(0.7, 0.7, 0.7, 1)))
        sec_grid = GridLayout(cols=6, spacing=2, size_hint_y=None, height=30)
        self.sec_toggles = {}
        current_actives = spec.get("active_sections", ["I", "V", "V2", "C", "B", "O"])
        
        for code in ["I", "V", "V2", "C", "B", "O"]:
            btn = ToggleButton(text=code, state='down' if code in current_actives else 'normal', font_size='11sp')
            self.sec_toggles[code] = btn
            sec_grid.add_widget(btn)
        content_box.add_widget(sec_grid)

        # -----------------------------
        # 2. RHYTHM EDITOR (The detailed grid from v4)
        # -----------------------------
        content_box.add_widget(Label(text="--- RHYTHM EDITOR ---", size_hint_y=None, height=30, color=(1, 0.5, 0.2, 1)))
        pat_ctrl = BoxLayout(size_hint_y=None, height=30)
        pat_ctrl.add_widget(Label(text="Load Template:", size_hint_x=0.4))
        self.spin_template = Spinner(text="Select...", values=list(RHYTHM_TEMPLATES.keys()), size_hint_x=0.6)
        self.spin_template.bind(text=self.on_template_load)
        pat_ctrl.add_widget(self.spin_template)
        content_box.add_widget(pat_ctrl)

        self.step_layout = GridLayout(cols=8, spacing=2, size_hint_y=None, height=70)
        self.step_buttons = []
        for i in range(16):
            # ▼▼▼ 変更: ToggleButton から Button に変更し、初期状態をOFFの色にする ▼▼▼
            btn = Button(text=str(i+1), font_size='9sp', background_color=(0.3, 0.3, 0.3, 1))
            # ▲▲▲ 変更終わり ▲▲▲
            btn.index = i
            btn.bind(on_press=self.on_step_toggle)
            self.step_buttons.append(btn)
            self.step_layout.add_widget(btn)
        content_box.add_widget(self.step_layout)
        
        self.refresh_ui_for_section("V")

        # -----------------------------
        # 3. AI & Special Features (v5 Features Integrated)
        # -----------------------------
        content_box.add_widget(Label(text="--- AI & ARRANGE ---", size_hint_y=None, height=30, color=(0.4, 0.8, 1, 1)))
        
        # Chorus Octave Shift
        row_oct = BoxLayout(size_hint_y=None, height=35)
        row_oct.add_widget(Label(text="Chorus +1 Octave:", size_hint_x=0.7))
        self.chk_oct = CheckBox(active=spec.get("chorus_octave", True), size_hint_x=0.3)
        row_oct.add_widget(self.chk_oct)
        content_box.add_widget(row_oct)

        # Drum Fills (Only for Drums)
        if self.is_drum:
            row_fill = BoxLayout(size_hint_y=None, height=35)
            row_fill.add_widget(Label(text="Enable Fills:", size_hint_x=0.7))
            self.chk_fill = CheckBox(active=spec.get("allow_fill", True), size_hint_x=0.3)
            row_fill.add_widget(self.chk_fill)
            content_box.add_widget(row_fill)
            
            # Fill Section Toggles
            content_box.add_widget(Label(text="Allow Fills In:", font_size='11sp', size_hint_y=None, height=20))
            row_secs = BoxLayout(size_hint_y=None, height=35)
            current_fill_secs = spec.get("fill_sections", ["I", "V", "V2", "C", "B", "O"])
            self.fill_sec_toggles = {}
            for sec in ["I", "V", "C", "B", "O"]:
                s_col = BoxLayout(orientation='vertical')
                s_col.add_widget(Label(text=sec, font_size='9sp'))
                chk = CheckBox(active=(sec in current_fill_secs))
                self.fill_sec_toggles[sec] = chk
                s_col.add_widget(chk)
                row_secs.add_widget(s_col)
            content_box.add_widget(row_secs)

        # -----------------------------
        # 4. VOCAL SETTINGS (Detailed v4 Style)
        # -----------------------------
        if self.is_vocal:
            # Create a new box for "Voice Character"
            char_box = BoxLayout(orientation='vertical', size_hint_y=None, height=550, padding=5, spacing=5)
            with char_box.canvas.before:
                 Color(0.25, 0.2, 0.3, 1); Rectangle(pos=char_box.pos, size=char_box.size)
            
            char_box.add_widget(Label(text="--- VOICE THICKNESS & CHARACTER ---", bold=True, color=(1, 0.6, 0.8, 1), size_hint_y=None, height=25))

            # Phase Mode Spinner
            row_phase = BoxLayout(size_hint_y=None, height=30)
            row_phase.add_widget(Label(text="Phase Mode:", size_hint_x=0.4, font_size='11sp'))
            self.spin_phase = Spinner(
                text=self.vocal_params_edit.get("phase_mode", "continuous"),
                values=("continuous", "reset", "random"),
                size_hint_x=0.6
            )
            def _update_phase(inst, val): self.vocal_params_edit["phase_mode"] = val
            self.spin_phase.bind(text=_update_phase)
            row_phase.add_widget(self.spin_phase)
            char_box.add_widget(row_phase)

            # Sliders
            def add_char_slider(lbl, key, min_v, max_v):
                row = BoxLayout(size_hint_y=None, height=30)
                row.add_widget(Label(text=lbl, size_hint_x=0.4, font_size='11sp'))
                val_lbl = Label(text=f"{self.vocal_params_edit.get(key, 0):.2f}", size_hint_x=0.2, font_size='11sp')
                slider = Slider(min=min_v, max=max_v, value=self.vocal_params_edit.get(key, 0), size_hint_x=0.4)
                def _update_v(inst, val):
                    self.vocal_params_edit[key] = val
                    val_lbl.text = f"{val:.2f}"
                slider.bind(value=_update_v)
                row.add_widget(val_lbl); row.add_widget(slider)
                char_box.add_widget(row)

            add_char_slider("Harmonics (Buzz)", "harmonic_mix", 0.0, 0.5)
            add_char_slider("Saturation (Drive)", "comp_drive", 0.5, 1.5)
            add_char_slider("Micro Detune", "detune_amount", 0.0, 1.0)
            
            # --- 既存の3つの下に追加してください ---

            # === 1. 声の表情 (Expression) ===
            # ささやき成分（ウィスパーボイス）
            add_char_slider("Whisper (Airy)", "whisper_amount", 0.0, 1.0)
            
            # ブレスノイズ（息の量）
            add_char_slider("Breath Vol", "breath_vol", 0.0, 0.5)
            
            # ブレスが入る確率
            add_char_slider("Breath Chance", "breath_chance", 0.0, 1.0)

            # === 2. 滑舌と歌い方 (Articulation) ===
            # 子音の強さ（アタック感）
            add_char_slider("Consonant Len", "cons_attack", 0.5, 2.0)
            
            # しゃくり（音程のずり上げ）の量
            add_char_slider("Scoop Amount", "scoop_amount", 0.0, 0.5)

            # === 3. ビブラート設定 (Vibrato) ===
            # ビブラートの深さ
            add_char_slider("Vibrato Depth", "vibrato_depth", 0.0, 0.15)
            
            # ビブラートの速さ (Hz)
            add_char_slider("Vibrato Rate", "vibrato_rate", 2.0, 8.0)

            # === 4. 音量エンベロープ (Envelope) ===
            # 音の立ち上がり（小さいほど鋭い）
            add_char_slider("Vol Attack", "attack", 0.005, 0.2)
            
            # 音の余韻（大きいほど伸びる）
            add_char_slider("Vol Release", "release", 0.01, 0.5)

            
            content_box.add_widget(char_box)

        # -----------------------------
        # 5. LFO & Effects (Restored v4 Style)
        # -----------------------------
        # LFO
        lfo_box = BoxLayout(orientation='vertical', size_hint_y=None, height=110, padding=5)
        with lfo_box.canvas.before:
            Color(0.2, 0.2, 0.25, 1); Rectangle(pos=lfo_box.pos, size=lfo_box.size)
            
        lfo_head = BoxLayout(size_hint_y=None, height=30)
        lfo_head.add_widget(Label(text="LFO (Tremolo)", bold=True, color=(1, 0.8, 0.2, 1)))
        self.chk_lfo = CheckBox(active=self.lfo_params_edit.get("active", False), size_hint_x=0.2)
        lfo_head.add_widget(self.chk_lfo)
        lfo_box.add_widget(lfo_head)

        rate_box = BoxLayout(size_hint_y=None, height=30)
        rate_box.add_widget(Label(text="Rate", size_hint_x=0.3))
        self.lbl_lfo_rate = Label(text=f"{self.lfo_params_edit.get('rate', 1.0):.1f} Hz", size_hint_x=0.2)
        rate_box.add_widget(self.lbl_lfo_rate)
        self.slider_lfo_rate = Slider(min=0.1, max=15.0, value=self.lfo_params_edit.get("rate", 1.0))
        self.slider_lfo_rate.bind(value=lambda i,v: setattr(self.lbl_lfo_rate, 'text', f"{v:.1f} Hz"))
        rate_box.add_widget(self.slider_lfo_rate)
        lfo_box.add_widget(rate_box)

        depth_box = BoxLayout(size_hint_y=None, height=30)
        depth_box.add_widget(Label(text="Depth", size_hint_x=0.3))
        self.lbl_lfo_depth = Label(text=f"{int(self.lfo_params_edit.get('depth', 0.0)*100)} %", size_hint_x=0.2)
        depth_box.add_widget(self.lbl_lfo_depth)
        self.slider_lfo_depth = Slider(min=0.0, max=1.0, value=self.lfo_params_edit.get("depth", 0.0))
        self.slider_lfo_depth.bind(value=lambda i,v: setattr(self.lbl_lfo_depth, 'text', f"{int(v*100)} %"))
        depth_box.add_widget(self.slider_lfo_depth)
        lfo_box.add_widget(depth_box)
        content_box.add_widget(lfo_box)

        # Bitcrusher
        bc_box = BoxLayout(orientation='vertical', size_hint_y=None, height=90, padding=5)
        bc_box.add_widget(Label(text="Bitcrusher Params", bold=True, size_hint_y=None, height=25, color=(1, 0.4, 0.4, 1)))
        
        redux_box = BoxLayout(size_hint_y=None, height=30)
        redux_box.add_widget(Label(text="Crush", size_hint_x=0.3))
        self.slider_bc_rate = Slider(min=0.0, max=0.95, value=self.bitcrush_params_edit.get("rate", 0.2))
        redux_box.add_widget(self.slider_bc_rate)
        bc_box.add_widget(redux_box)

        bits_box = BoxLayout(size_hint_y=None, height=30)
        bits_box.add_widget(Label(text="Bits", size_hint_x=0.3))
        self.lbl_bc_bits = Label(text=f"{int(self.bitcrush_params_edit.get('depth', 12))} bit", size_hint_x=0.2)
        bits_box.add_widget(self.lbl_bc_bits)
        self.slider_bc_depth = Slider(min=2, max=16, step=1, value=self.bitcrush_params_edit.get("depth", 12))
        self.slider_bc_depth.bind(value=lambda i,v: setattr(self.lbl_bc_bits, 'text', f"{int(v)} bit"))
        bits_box.add_widget(self.slider_bc_depth)
        bc_box.add_widget(bits_box)
        content_box.add_widget(bc_box)

        # -----------------------------
        #  [NEW] 3-Band Parametric EQ
        # -----------------------------
        eq_box = BoxLayout(orientation='vertical', size_hint_y=None, height=140, padding=5)
        with eq_box.canvas.before:
            Color(0.18, 0.18, 0.22, 1); Rectangle(pos=eq_box.pos, size=eq_box.size)
            
        eq_box.add_widget(Label(text="3-Band EQ (dB)", bold=True, size_hint_y=None, height=25, color=(0.6, 1, 0.6, 1)))

        # 現在の設定を取得
        current_eq = self.spec.get("eq_params", {"low": 0.0, "mid": 0.0, "high": 0.0})
        self.eq_sliders = {}

        def add_eq_slider(band, label):
            row = BoxLayout(size_hint_y=None, height=30)
            row.add_widget(Label(text=label, size_hint_x=0.3, font_size='11sp'))
            
            val_lbl = Label(text=f"{current_eq[band]:.1f}", size_hint_x=0.2, font_size='11sp')
            
            # Range: -12dB to +12dB
            slider = Slider(min=-12.0, max=12.0, value=current_eq[band], step=0.5)
            slider.bind(value=lambda i,v: setattr(val_lbl, 'text', f"{v:.1f}"))
            
            row.add_widget(val_lbl)
            row.add_widget(slider)
            eq_box.add_widget(row)
            self.eq_sliders[band] = slider

        add_eq_slider("high", "High (5k)")
        add_eq_slider("mid", "Mid (1k)")
        add_eq_slider("low", "Low (150)")
        
        content_box.add_widget(eq_box)

# [NEW] MIDI Import Button
        content_box.add_widget(Label(text="--- EXTERNAL DATA ---", size_hint_y=None, height=30, color=(1, 0.5, 0.8, 1)))
        
        btn_midi_import = Button(text="IMPORT MIDI & LYRICS", background_color=(0.8, 0.4, 0.2, 1), size_hint_y=None, height=40)
        btn_midi_import.bind(on_press=self.on_import_midi) 
        content_box.add_widget(btn_midi_import)
        
        scroll.add_widget(content_box)
        layout.add_widget(scroll)

        # -- Apply Button --
        btn_apply = Button(text="APPLY & REGENERATE", background_color=(0, 0.7, 0.5, 1), size_hint_y=0.12)
        btn_apply.bind(on_press=self.on_apply)
        layout.add_widget(btn_apply)
        self.add_widget(layout)

    def on_edit_section_change(self, instance, value):
        self.current_edit_section = value
        self.refresh_ui_for_section(value)

    def on_div_change(self, instance, value):
        self.div_map_edit[self.current_edit_section] = int(value)
        
        # ▼▼▼ 新規追加: スタイル変更時の処理 ▼▼▼
    def on_style_change(self, instance, value):
        self.style_map_edit[self.current_edit_section] = value
    # ▲▲▲ 追加終わり ▲▲▲
    # ▼▼▼ 新規追加: S/Eゲート変更時の処理 ▼▼▼
    def on_gate_mask_change(self, instance, value):
        self.gate_map_edit[self.current_edit_section] = value
    # ▲▲▲ 新規追加終わり ▲▲▲

    def refresh_ui_for_section(self, section_code):
        val = self.div_map_edit.get(section_code, 16)
        self.spin_div.text = str(val)
        # ▼▼▼ 追加: Styleの更新 ▼▼▼
        style_val = self.style_map_edit.get(section_code, "Normal")
        self.spin_style.text = style_val
        # ▲▲▲ 追加終わり ▲▲▲
        # ▼▼▼ 新規追加: S/Eゲートテキストの更新 ▼▼▼
        gate_val = self.gate_map_edit.get(section_code, "")
        self.txt_gate_mask.text = gate_val
        # ▲▲▲ 新規追加終わり ▲▲▲
        pattern = self.pattern_map_edit.get(section_code)
        if pattern is None:
            pattern = [0]*16
            self.pattern_map_edit[section_code] = pattern
        for i, btn in enumerate(self.step_buttons):
            val = pattern[i] if i < len(pattern) else 0
            # ▼▼▼ 変更: 0, 1, 2 に合わせて色とテキストを更新する ▼▼▼
            if val == 1:
                btn.background_color = (0.8, 0.2, 0.2, 1)  # 赤: 鳴らす(アタック)
                btn.text = f"{i+1}\nON"
            elif val == 2:
                btn.background_color = (0.2, 0.6, 0.8, 1)  # 青: 伸ばす(タイ)
                btn.text = f"{i+1}\n->"
            else:
                btn.background_color = (0.3, 0.3, 0.3, 1)  # 灰: 休み
                btn.text = str(i+1)
            # ▲▲▲ 変更終わり ▲▲▲
            if i % 4 == 0: btn.color = (1, 1, 0.5, 1)
            else: btn.color = (1, 1, 1, 1)

    def on_template_load(self, instance, value):
        template = RHYTHM_TEMPLATES.get(value)
        if template:
            self.pattern_map_edit[self.current_edit_section] = template[:]
            self.refresh_ui_for_section(self.current_edit_section)
        self.spin_template.text = "Select..."

    # ▼▼▼ 丸ごと上書き変更 ▼▼▼
    def on_step_toggle(self, instance):
        idx = instance.index
        if self.current_edit_section not in self.pattern_map_edit:
            self.pattern_map_edit[self.current_edit_section] = [0]*16
            
        # 今のパターン値を取得
        current_val = self.pattern_map_edit[self.current_edit_section][idx]
        
        # 0(休) → 1(鳴) → 2(伸) → 0... の順で切り替え
        next_val = (current_val + 1) % 3
        
        # 配列に保存
        self.pattern_map_edit[self.current_edit_section][idx] = next_val
        
        # 押したボタンの見た目だけを即座に更新
        if next_val == 1:
            instance.background_color = (0.8, 0.2, 0.2, 1)
            instance.text = f"{idx+1}\nON"
        elif next_val == 2:
            instance.background_color = (0.2, 0.6, 0.8, 1)
            instance.text = f"{idx+1}\n->"
        else:
            instance.background_color = (0.3, 0.3, 0.3, 1)
            instance.text = str(idx+1)
    # ▲▲▲ 変更終わり ▲▲▲

    def on_apply(self, instance):
        self.spec["div_map"] = self.div_map_edit
        self.spec["div"] = int(self.spin_div.text) 
        self.spec["wave"] = self.spin_wave.text
        # ▼▼▼ ここに新規追加: note_len を保存 ▼▼▼
        self.spec["note_len"] = self.slider_note_len.value
        # ▲▲▲ 新規追加終わり ▲▲▲
        self.spec["xy_route"] = self.chk_xy.active
        # ▼▼▼ 修正: style_map を保存し、style (デフォルト) は現在の値をセット ▼▼▼
        self.spec["style_map"] = self.style_map_edit
        self.spec["style"] = self.spin_style.text # UI上の最新値をフォールバックとして保存
        # ▲▲▲ 修正終わり ▲▲▲
        # ▼▼▼ 追加: カスタムコードの保存 ▼▼▼
        self.spec["custom_chords"] = self.txt_custom_chords.text
        # ▲▲▲ 追加終わり ▲▲▲
        # ▼▼▼ 新規追加: gate_mapの保存 ▼▼▼
        self.spec["gate_map"] = self.gate_map_edit
        # ▲▲▲ 新規追加終わり ▲▲▲        
        self.spec["eq_params"] = {
            "low": self.eq_sliders["low"].value,
            "mid": self.eq_sliders["mid"].value,
            "high": self.eq_sliders["high"].value
        }
        
        new_actives = [code for code, btn in self.sec_toggles.items() if btn.state == 'down']
        self.spec["active_sections"] = new_actives
        
        self.spec["lfo_params"] = {
            "active": self.chk_lfo.active, "rate": self.slider_lfo_rate.value, "depth": self.slider_lfo_depth.value
        }
        self.spec["bitcrush_params"] = {
            "rate": self.slider_bc_rate.value, "depth": int(self.slider_bc_depth.value)
        }
        
        # v5 AI Params
        self.spec["chorus_octave"] = self.chk_oct.active
        if self.is_drum:
            self.spec["allow_fill"] = self.chk_fill.active
            new_fill_secs = [code for code, chk in self.fill_sec_toggles.items() if chk.active]
            self.spec["fill_sections"] = new_fill_secs
        
        if self.is_vocal: self.spec["vocal_params"] = self.vocal_params_edit
        self.spec["pattern_map"] = self.pattern_map_edit
            
            
        
        self.callback()
        self.dismiss()

class VocalFileImportPopup(ModalView):
    def __init__(self, app_instance, **kwargs):
        super().__init__(size_hint=(0.9, 0.8), auto_dismiss=True, **kwargs)
        self.app = app_instance
        self.target_vowels = ['a', 'i', 'u', 'e', 'o']
        # ステータス管理用ラベル
        self.status_labels = {}
        
        layout = BoxLayout(orientation='vertical', padding=15, spacing=15)
        with layout.canvas.before:
            Color(0.1, 0.12, 0.15, 1); Rectangle(pos=self.pos, size=self.size)

        # ヘッダー
        layout.add_widget(Label(text="IMPORT VOICE SAMPLES", font_size='20sp', bold=True, size_hint_y=0.1, color=(0.4, 1, 0.8, 1)))
        layout.add_widget(Label(text="Load .wav files for each vowel to clone voice.", font_size='12sp', size_hint_y=0.05, color=(0.7, 0.7, 0.7, 1)))

        # 母音ごとの行を作成
        grid = GridLayout(cols=1, spacing=10, size_hint_y=0.7)
        
        for vowel in self.target_vowels:
            row = BoxLayout(orientation='horizontal', spacing=10)
            
            # 母音ラベル (A, I, U...)
            row.add_widget(Label(text=vowel.upper(), font_size='20sp', bold=True, size_hint_x=0.15, color=(1, 0.8, 0.2, 1)))
            
            # ステータス表示 (Empty / Analyzed)
            status_text = "Ready" if vowel in USER_VOICE_DATA else "Empty"
            col = (0.5, 1, 0.5, 1) if vowel in USER_VOICE_DATA else (0.5, 0.5, 0.5, 1)
            lbl_status = Label(text=status_text, size_hint_x=0.35, color=col)
            self.status_labels[vowel] = lbl_status
            row.add_widget(lbl_status)
            
            # Loadボタン
            btn_load = Button(text="LOAD WAV", size_hint_x=0.3, background_color=(0.2, 0.6, 0.8, 1))
            # lambdaで変数を固定して渡す
            btn_load.bind(on_press=lambda x, v=vowel: self.open_file_browser(v))
            row.add_widget(btn_load)
            
            # Resetボタン
            btn_reset = Button(text="X", size_hint_x=0.1, background_color=(0.8, 0.3, 0.3, 1))
            btn_reset.bind(on_press=lambda x, v=vowel: self.clear_data(v))
            row.add_widget(btn_reset)
            
            grid.add_widget(row)

        layout.add_widget(grid)

        # 閉じるボタン
        btn_close = Button(text="CLOSE", size_hint_y=0.15, background_color=(0.4, 0.4, 0.4, 1))
        btn_close.bind(on_press=self.dismiss)
        layout.add_widget(btn_close)
        
        self.add_widget(layout)

    def open_file_browser(self, vowel):
        """ファイルブラウザを開く"""
        def on_file_selected(filename):
            if filename:
                self.analyze_file(vowel, filename)
        
        # WAVファイルのみ許可
        FileLoadPopup(on_file_selected, filters=['*.wav']).open()

    def analyze_file(self, vowel, filepath):
        """選択されたファイルを解析してUSER_VOICE_DATAに格納"""
        lbl = self.status_labels[vowel]
        lbl.text = "Analyzing..."
        lbl.color = (1, 1, 0, 1) # 黄色
        
        # UIをフリーズさせないようスレッドで実行
        threading.Thread(target=self._analyze_thread, args=(vowel, filepath)).start()

    def _analyze_thread(self, vowel, filepath):
        try:
            # 既存のVoiceCloneAnalyzerを使用
            freqs = VoiceCloneAnalyzer.analyze_spectral_envelope(filepath)
            
            # メインスレッドでUI更新
            def _update_ui(dt):
                lbl = self.status_labels[vowel]
                if freqs:
                    USER_VOICE_DATA[vowel] = freqs
                    lbl.text = f"OK! {int(freqs[0])}/{int(freqs[1])}Hz"
                    lbl.color = (0.2, 1, 0.4, 1) # 緑
                else:
                    lbl.text = "Failed (No Peaks)"
                    lbl.color = (1, 0.2, 0.2, 1) # 赤
            
            Clock.schedule_once(_update_ui, 0)
            
        except Exception as e:
            traceback.print_exc()
            def _error_ui(dt):
                lbl = self.status_labels[vowel]
                lbl.text = "Error"
                lbl.color = (1, 0.2, 0.2, 1)
            Clock.schedule_once(_error_ui, 0)

    def clear_data(self, vowel):
        """解析データを消去してデフォルトに戻す"""
        if vowel in USER_VOICE_DATA:
            del USER_VOICE_DATA[vowel]
        
        lbl = self.status_labels[vowel]
        lbl.text = "Empty"
        lbl.color = (0.5, 0.5, 0.5, 1)


#class VocalCapturePopup(ModalView):
  #  def __init__(self, app_instance, **kwargs):
     #   super().__init__(size_hint=(0.8, 0.6), auto_dismiss=True, **kwargs)
    #    self.app = app_instance
   #     self.target_vowels = ['a', 'i', 'u', 'e', 'o']
   #     self.current_idx = 0
        
       # layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
      #  with layout.canvas.before:
          #  Color(0.1, 0.12, 0.15, 1); Rectangle(pos=self.pos, size=self.size)

      #  layout.add_widget(Label(text="SELF-VOCALOID SETUP", font_size='20sp', bold=True, size_hint_y=0.15, color=(0.4, 1, 0.8, 1)))
        
      #  self.lbl_instruction = Label(text=f"Please say: '{self.target_vowels[0].upper()}'", font_size='24sp', size_hint_y=0.3)
        #layout.add_widget(self.lbl_instruction)
        
      #  self.btn_rec = ToggleButton(text="HOLD TO RECORD", size_hint_y=0.2, background_color=(1, 0.3, 0.3, 1))
      #  self.btn_rec.bind(state=self.on_rec_state)
     #   layout.add_widget(self.btn_rec)
        
     #   self.progress_bar = Label(text="Progress: 0/5", size_hint_y=0.1)
      #  layout.add_widget(self.progress_bar)

      #  btn_close = Button(text="FINISH", size_hint_y=0.15)
     #   btn_close.bind(on_press=self.dismiss)
      #  layout.add_widget(btn_close)
        
        #self.add_widget(layout)

    # ----------------------------------------------------
    # In class VocalCapturePopup: Replace on_rec_state
    # ----------------------------------------------------
    def on_rec_state(self, instance, state):
        if state == 'down':
            # 録音開始
            self.app.recorder.start()
            self.lbl_instruction.text = "Recording... Keep holding."
            self.lbl_instruction.color = (1, 0.5, 0.5, 1)
        else:
            # 録音停止 & 解析プロセス
            self.lbl_instruction.text = "Analyzing..."
            self.lbl_instruction.color = (0.5, 1, 0.5, 1)
            
            # スレッドで処理（UIフリーズ防止）
            threading.Thread(target=self._process_recording).start()

    def _process_recording(self):
        # 1. ファイル保存
        vowel = self.target_vowels[self.current_idx]
        filename = f"user_voice_{vowel}.wav"
        path = self.app.recorder.stop(filename)
        
        if not path:
            Clock.schedule_once(lambda dt: setattr(self.lbl_instruction, 'text', "Error: No audio."))
            return

        # 2. 解析 (Cepstral Analysis)
        freqs = VoiceCloneAnalyzer.analyze_spectral_envelope(path)
        
        # 3. 結果の適用 (Main ThreadでUI更新)
        def _update_ui(dt):
            if freqs:
                # グローバル辞書に保存
                USER_VOICE_DATA[vowel] = freqs
                logger.info(f"Analyzed {vowel}: {freqs}")
                
                # UI更新
                self.current_idx = (self.current_idx + 1) % 5
                next_v = self.target_vowels[self.current_idx].upper()
                self.lbl_instruction.text = f"Success! Next: Say '{next_v}'"
                self.lbl_instruction.color = (1, 1, 1, 1)
                self.progress_bar.text = f"Captured: {len(USER_VOICE_DATA)}/5"
            else:
                self.lbl_instruction.text = "Analysis Failed. Speak clearer."
                self.lbl_instruction.color = (1, 0.3, 0.3, 1)
        
        Clock.schedule_once(_update_ui, 0)
# ==========================================
# UI: Track Widget (Fixed: Added on_regen)
# ==========================================
class TrackWidget(BoxLayout):
    """
    Enhanced TrackWidget with DSP Toggles and Piano Roll Edit.
    Fixed: Added 'on_regen' method for compatibility with TrackSettingsPopup.
    """
    def __init__(self, spec, app, track_idx, **kwargs):
        super().__init__(orientation='horizontal', padding=5, spacing=5, **kwargs)
        self.spec = spec
        self.app = app
        self.track_idx = track_idx
        
        self.raw_data = None
        self.status = "Init"
        self.progress = 0.0
        self.is_active = spec.get("initial_active", True)
        
        # Local mirror of DSP settings for UI state
        self.dsp_chain = list(spec.get("dsp", []))
        self.lfo_active = spec.get("lfo_params", {}).get("active", False)
        
        # -- 1. Status & Active Check (Left Column) --
        left_col = BoxLayout(orientation='vertical', size_hint_x=0.1)
        self.chk_active = CheckBox(active=self.is_active, size_hint_y=0.5)
        self.chk_active.bind(active=self.on_active_toggle)
        left_col.add_widget(self.chk_active)
        
        # Level Meter
        self.meter_canvas = Widget(size_hint_y=0.5)
        with self.meter_canvas.canvas:
            Color(0.2, 0.2, 0.2, 1)
            Rectangle(pos=self.meter_canvas.pos, size=self.meter_canvas.size)
            
            # Dynamic Meter Color & Bar
            self.meter_color = Color(0, 1, 0.5, 1)
            self.meter_bar = Rectangle(size=(0, 0))
            
        self.meter_canvas.bind(pos=self.update_meter_pos, size=self.update_meter_pos)
        left_col.add_widget(self.meter_canvas)
        self.add_widget(left_col)
        
        # -- 2. Main Controls (Right Column) --
        main_col = BoxLayout(orientation='vertical', size_hint_x=0.9, spacing=2)
        
        # 2a. Header
        header = BoxLayout(size_hint_y=0.25) # 少し高さを調整
        role_name = spec['role'].replace("_", " ").upper()
        # 色分け：ドラム系はオレンジ、ボーカルは黄色、その他は白
        name_col = (1, 0.7, 0.2, 1) if spec.get("is_drum") else ((1, 1, 0.4, 1) if "vocal" in spec['role'] else (1,1,1,1))
        
        self.lbl_role = Label(text=role_name, font_size='11sp', bold=True, halign='left', color=name_col)
        self.lbl_status = Label(text="Wait...", font_size='10sp', color=(0.7, 0.7, 0.7, 1), size_hint_x=0.4)
        
        # Piano Roll Edit Button
        btn_piano = Button(text="NOTE", font_size='10sp', background_color=(0.9, 0.8, 0.2, 1), size_hint_x=0.2)
        btn_piano.bind(on_press=self.on_piano_roll)
        
        # Settings Popup Button
        btn_settings = Button(text="SET", font_size='10sp', background_color=(0.2, 0.6, 0.8, 1), size_hint_x=0.2)
        btn_settings.bind(on_press=self.on_settings)
        
        header.add_widget(self.lbl_role)
        header.add_widget(self.lbl_status)
        header.add_widget(btn_piano)
        header.add_widget(btn_settings)
        main_col.add_widget(header)
        
        # 2b. Sliders (Vol + Pan in one row)
        sliders_row = BoxLayout(orientation='horizontal', size_hint_y=0.35)
        
        # Vol Slider
        sliders_row.add_widget(Label(text="VOL", font_size='8sp', size_hint_x=0.1))
        self.slider_vol = Slider(min=0.0, max=1.5, value=spec.get("vol", 1.0), cursor_size=(15,15), size_hint_x=0.4)
        self.slider_vol.bind(value=self.on_param_change)
        sliders_row.add_widget(self.slider_vol)
        
        # Pan Slider
        sliders_row.add_widget(Label(text="PAN", font_size='8sp', size_hint_x=0.1))
        self.slider_pan = Slider(min=-1.0, max=1.0, value=spec.get("pan", 0.0), cursor_size=(15,15), size_hint_x=0.4)
        self.slider_pan.bind(value=self.on_param_change)
        sliders_row.add_widget(self.slider_pan)
        
        main_col.add_widget(sliders_row)
        
        # 2c. DSP Toggles (Grid Layout for more buttons)
        # ---------------------------------------------------------
        # 【修正】ボタンを増やしてGridにする
        dsp_grid = GridLayout(cols=5, spacing=2, size_hint_y=0.45) # 2段組み
        
        self.dsp_buttons = {}
        # 表示名と内部IDの対応表
        toggles = [
            ("CMP", "comp"),          # コンプレッサー (New!)
            ("HPF", "high_pass"),     # ハイパス (New!)
            ("LPF", "lpf"),           # ローパス
            ("RV.S", "reverb_short"), # ショートリバーブ (New!)
            ("RV.L", "reverb_long"),  # ロングリバーブ (New!)
            ("DLY", "delay"),         # ディレイ
            ("WID", "wide"),          # ワイド
            ("DST", "dist"),          # 歪み
            ("BIT", "bitcrush"),      # ビットクラッシュ
            ("LFO", "lfo")            # LFO
        ]
        
        for label, key in toggles:
            is_active = False
            if key == "lfo": is_active = self.lfo_active
            else: is_active = (key in self.dsp_chain)
            
            # 色分け: リバーブ系は紫、コンプは赤、LFOは黄色
            bg_col_active = (0.4, 0.8, 0.6, 1) # Default Green
            if "reverb" in key: bg_col_active = (0.6, 0.4, 0.9, 1)
            elif "comp" in key: bg_col_active = (0.9, 0.4, 0.4, 1)
            elif "lfo" in key:  bg_col_active = (0.9, 0.8, 0.2, 1)

            btn = ToggleButton(text=label, font_size='8sp', state='down' if is_active else 'normal',
                               background_color=bg_col_active if is_active else (0.25, 0.25, 0.25, 1))
            
            btn.dsp_key = key
            btn.active_col = bg_col_active # アクティブ時の色を保存
            btn.bind(on_press=self.on_dsp_toggle)
            
            self.dsp_buttons[key] = btn
            dsp_grid.add_widget(btn)
            
        main_col.add_widget(dsp_grid)
        self.add_widget(main_col)

    def reset(self):
        self.raw_data = None
        self.status = "Queued"
        self.progress = 0.0
        self.lbl_status.text = "Queued"
        self.lbl_status.color = (0.7, 0.7, 0.7, 1)

    def on_active_toggle(self, instance, value):
        self.is_active = value
        self.lbl_role.color = self.spec.get("col", (1,1,1)) if value else (0.4, 0.4, 0.4, 1)
        
        if self.app:
            # 【修正】データが無いのにONにされた場合、生成をリクエストする
            if value and (self.raw_data is None or len(self.raw_data) == 0):
                self.status = "Generating..."
                self.app.request_track_regeneration(self)
            else:
                self.app.on_track_toggled()

        
    def set_active(self, active):
        self.chk_active.active = active

    def on_param_change(self, instance, value):
        if instance == self.slider_vol: self.spec["vol"] = value
        if instance == self.slider_pan: self.spec["pan"] = value
        if self.app and self.app.user_started:
            self.app.on_track_toggled() 

    def on_dsp_toggle(self, instance):
        key = getattr(instance, 'dsp_key', None)
        if not key: return
        
        is_active = (instance.state == 'down')
        # 保存しておいた色を使う
        active_col = getattr(instance, 'active_col', (0.4, 0.8, 0.6, 1))
        instance.background_color = active_col if is_active else (0.25, 0.25, 0.25, 1)

        if key == "lfo":
            if "lfo_params" not in self.spec: self.spec["lfo_params"] = {}
            self.spec["lfo_params"]["active"] = is_active
            self.lfo_active = is_active
        else:
            if is_active:
                if key not in self.dsp_chain: self.dsp_chain.append(key)
            else:
                if key in self.dsp_chain: self.dsp_chain.remove(key)
            self.spec["dsp"] = self.dsp_chain
            
        if self.app: self.app.refresh_single_track(self.spec)

    def on_piano_roll(self, instance):
        if self.app: self.app.open_track_editor(self.track_idx)

    def on_settings(self, instance):
        # Pass on_regen as the callback
        TrackSettingsPopup(self.spec, lambda: self.on_regen(None)).open()

    def on_regen(self, instance):
        """Bridge method: Called by TrackSettingsPopup callback to trigger audio update."""
        # 【修正】設定変更後は必ず再作曲（リフィル等）を行うためフラグを折る
        self.spec["keep_sequence"] = False
        if self.app: self.app.refresh_single_track(self.spec)

    def update_meter_pos(self, *args):
        self.meter_bar.pos = (self.meter_canvas.x, self.meter_canvas.y)
        self.meter_bar.size = (self.meter_canvas.width, self.meter_canvas.height * self.progress)
        
        if self.status == "Generating...":
             self.meter_color.rgba = (1, 1, 0, 0.5) 
        else:
             self.meter_color.rgba = (0, 1, 0.5, 1)

    def update_vis(self, playing, elapsed):
        if playing and self.raw_data is not None and self.is_active:
            idx = int(elapsed * FS)
            if hasattr(self.raw_data, 'shape') and 0 <= idx < self.raw_data.shape[0]:
                amp = abs(self.raw_data[idx][0]) + abs(self.raw_data[idx][1])
                h = min(1.0, amp * 2.0)
                self.progress = h 
            else:
                self.progress = 0
        else:
            if self.status != "Generating...": self.progress = 0
        self.update_meter_pos()

# ==========================================
# 16. Screens (Restored & Adapted)
# ==========================================
class StartScreen(Screen):
    """
    Initial configuration screen.
    Includes: BPM, Key, Genre (Sound), and Harmony Style (Composition) controls.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = AnchorLayout(anchor_x='center', anchor_y='center')
        
        # BG
        with layout.canvas.before:
            Color(0.05, 0.05, 0.1, 1)
            Rectangle(size=(2000, 2000))
            Color(0.5, 0, 0.5, 0.2)
            Ellipse(pos=(300, 300), size=(600, 600))

        main_box = BoxLayout(orientation='vertical', size_hint=(0.95, 0.95), spacing=10)
        
        # --- Top Bar ---
        top_bar = BoxLayout(size_hint_y=0.08, spacing=10)
        top_bar.add_widget(Label(text="HyperNeko", font_size='20sp', bold=True, color=(0.4, 1, 0.8, 1), size_hint_x=0.5))
        
        btn_load = Button(text="LOAD PRESET", background_color=(0.8, 0.5, 0.2, 1), size_hint_x=0.25)
        btn_load.bind(on_press=self.on_load_preset)
        top_bar.add_widget(btn_load)
        
        btn_conf = Button(text="ENGINE CFG", background_color=(0.5, 0.2, 0.8, 1), size_hint_x=0.25)
        btn_conf.bind(on_press=self.open_config)
        top_bar.add_widget(btn_conf)
        main_box.add_widget(top_bar)

        # --- Scrollable Settings Area ---
        scroll = ScrollView(size_hint_y=0.77)
        self.settings_box = GridLayout(cols=1, spacing=15, size_hint_y=None, padding=[10, 10])
        self.settings_box.bind(minimum_height=self.settings_box.setter('height'))
        
        # 1. BPM / Key / Seed / Genre (Sound)
        param_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=10)
        
        # BPM
        param_row.add_widget(Label(text="BPM:", size_hint_x=0.1))
        self.txt_bpm = TextInput(text="160", multiline=False, input_filter='int', size_hint_x=0.15)
        param_row.add_widget(self.txt_bpm)
        
        # Key Transpose
        param_row.add_widget(Label(text="Key:", size_hint_x=0.1, color=(1, 1, 0.5, 1)))
        key_values = [str(i) for i in range(-12, 13)]
        self.spin_key = Spinner(text="0", values=key_values, size_hint_x=0.15)
        param_row.add_widget(self.spin_key)

        # Seed
        param_row.add_widget(Label(text="Seed:", size_hint_x=0.1, halign='right'))
        self.txt_seed = TextInput(text="", hint_text="Rnd", multiline=False, input_filter='int', size_hint_x=0.2)
        param_row.add_widget(self.txt_seed)
        
        # Genre Select (Sound Only)
        btn_genre = Button(text="SOUND GENRE", background_color=(0.3, 0.6, 0.9, 1), size_hint_x=0.25)
        btn_genre.bind(on_press=lambda x: App.get_running_app().open_style_menu(x))
        param_row.add_widget(btn_genre)

        self.settings_box.add_widget(param_row)

        # ==========================================
        # [NEW] HARMONY STYLE CONTROLS (Composition)
        # ==========================================
        harmony_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=45, spacing=10, padding=[0,5])
        
        harmony_row.add_widget(Label(text="Harmony Style:", size_hint_x=0.25, color=(1, 0.5, 0.8, 1), bold=True))
        
        # Harmony Spinner (Uses global HARMONY_DB keys)
        style_keys = list(HARMONY_DB.keys()) if 'HARMONY_DB' in globals() else ["emopop", "Capsule", "Komuro (TK)", "Vkei", "Hendrix", "Canon"]
        self.spin_harmony = Spinner(
            text="emopop", 
            values=style_keys, 
            size_hint_x=0.35,
            background_color=(0.2, 0.2, 0.25, 1)
        )
        harmony_row.add_widget(self.spin_harmony)
        
        # Apply Button (Overwrites Chords)
        btn_apply_h = Button(text="APPLY COMPOSITION", size_hint_x=0.4, background_color=(0.8, 0.2, 0.5, 1))
        btn_apply_h.bind(on_press=self.apply_harmony_logic)
        harmony_row.add_widget(btn_apply_h)
        
        self.settings_box.add_widget(harmony_row)
        # ==========================================

        # 2. Structure
        struct_row = BoxLayout(orientation='vertical', size_hint_y=None, height=70)
        struct_row.add_widget(Label(text="Structure (I, V, V2, C, B, O):", size_hint_y=0.4, halign='left', color=(0.7, 0.7, 1, 1)))
        self.txt_struct = TextInput(text="I, V, V2, C, B, C, O", multiline=False, size_hint_y=0.6)
        struct_row.add_widget(self.txt_struct)
        self.settings_box.add_widget(struct_row)

        # Auto-fill Checkbox
        auto_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        auto_row.add_widget(Label(text="Auto-fill Chords / Patterns:", size_hint_x=0.8, halign='left'))
        self.chk_auto = CheckBox(active=True, size_hint_x=0.2)
        auto_row.add_widget(self.chk_auto)
        self.settings_box.add_widget(auto_row)

        # 3. Chords & Lyrics Input Fields
        self.settings_box.add_widget(Label(text="--- CHORDS & LYRICS ---", size_hint_y=None, height=30, color=(1, 0.8, 0.2, 1)))
        
        # Intro
        self.txt_intro, self.txt_lyric_intro = self._add_section_input(
            "Intro (I):", "Abm, Ebm, B, E", "はいぱーねこにゃん"
        )
        # Verse1
        self.txt_verse, self.txt_lyric_verse = self._add_section_input(
            "Verse 1 (V) - Main:", "Fm, Cm, Ab, Db", "あれはやみのそなた"
        )
        # Verse2
        self.txt_verse2, self.txt_lyric_verse2 = self._add_section_input(
            "Verse 2 (V2):", "Bbm, Fm, Db, Gb", "たゆたうのはわたしか"
        )
        # Chorus
        self.txt_chorus, self.txt_lyric_chorus = self._add_section_input(
            "Chorus (C):", "B, Eb, Abm, Ebm", "きみとのやくそくはたしか"
        )
        # Bridge
        self.txt_bridge, self.txt_lyric_bridge = self._add_section_input(
            "Bridge (B):", "Bbm, Fm, Db, Gb", "にゃんにゃーにゃーにゃん"
        )
        # Outro
        self.txt_outro, self.txt_lyric_outro = self._add_section_input(
            "Outro (O):", "Db, Gb, Bbm, Fm", "いつかこんなうたうたうためのあんさ"
        )

        scroll.add_widget(self.settings_box)
        main_box.add_widget(scroll)

        # --- Start Button ---
        self.btn_start = Button(text="INITIALIZE SYSTEM", font_size='22sp', background_color=(0.2, 0.6, 0.8, 1), size_hint_y=0.1)
        self.btn_start.bind(on_press=self.on_start)
        main_box.add_widget(self.btn_start)
        
        self.lbl_status = Label(text="System Ready.", color=(0.5, 0.5, 0.5, 1), size_hint_y=0.05, font_size='12sp')
        main_box.add_widget(self.lbl_status)
        
        layout.add_widget(main_box)
        self.add_widget(layout)
        
        self.app = App.get_running_app()

    def _add_section_input(self, label_text, default_chord, default_lyric):
        box = BoxLayout(orientation='vertical', size_hint_y=None, height=110, padding=[0, 5])
        box.add_widget(Label(text=label_text, size_hint_y=0.2, halign='left', color=(0.8, 0.8, 0.8, 1)))
        
        chord_row = BoxLayout(size_hint_y=0.4)
        chord_row.add_widget(Label(text="Chords:", size_hint_x=0.2, font_size='10sp'))
        txt_chord = TextInput(text=default_chord, multiline=False, size_hint_x=0.8)
        chord_row.add_widget(txt_chord)
        box.add_widget(chord_row)
        
        lyric_row = BoxLayout(size_hint_y=0.4)
        lyric_row.add_widget(Label(text="Lyrics:", size_hint_x=0.2, font_size='10sp', color=(1, 0.6, 0.8, 1)))
        txt_lyric = TextInput(text=default_lyric, hint_text="Kana...", multiline=False, size_hint_x=0.8)
        lyric_row.add_widget(txt_lyric)
        box.add_widget(lyric_row)
        
        self.settings_box.add_widget(box)
        return txt_chord, txt_lyric

    # [NEW] Applies the selected Harmony Style to the chord inputs
    def apply_harmony_logic(self, instance):
        if 'HarmonyEngine' not in globals():
            self.lbl_status.text = "Error: HarmonyEngine not found."
            return

        style_name = self.spin_harmony.text
        self.lbl_status.text = f"Composing with style: {style_name}..."
        
        try:
            # Generate Map using the global HarmonyEngine
            new_map = HarmonyEngine().generate_full_song_map(style_name)
            
            # Apply to Text Inputs (Join lists into strings)
            self.txt_intro.text = ", ".join(new_map["I"])
            self.txt_verse.text = ", ".join(new_map["V"])
            self.txt_verse2.text = ", ".join(new_map["V2"])
            self.txt_chorus.text = ", ".join(new_map["C"])
            self.txt_bridge.text = ", ".join(new_map["B"])
            self.txt_outro.text = ", ".join(new_map["O"])
            
            self.lbl_status.text = f"Composition Applied: {style_name}"
        except Exception as e:
            self.lbl_status.text = f"Composition Error: {str(e)}"
            print(f"Harmony Gen Error: {e}")

    def open_config(self, instance): 
        EngineConfigPopup(self.app).open()

    def on_load_preset(self, instance):
        """
        Open the file browser filtered for JSON files to load presets.
        """
        def on_load(filename):
            if not filename: return
            try:
                data = PresetManager.load_preset(filename)
                if data:
                    self._perform_load(data, filename)
                else:
                    self.lbl_status.text = "Error: Invalid or corrupted preset file."
            except Exception as e:
                self.lbl_status.text = f"Load Error: {e}"
        
        # Use specific filters for JSON
        FileLoadPopup(on_load, filters=['*.json']).open()

    def _perform_load(self, data, filename):
        try:
            g = data["global"]
            self.txt_bpm.text = str(g.get("bpm", 160))
            self.txt_struct.text = g.get("structure_str", "")
            self.txt_seed.text = str(g.get("seed", ""))
            self.chk_auto.active = g.get("auto_fill", True)
            self.spin_key.text = str(g.get("key_offset", 0))
            
            cmap = g.get("chords_map", {})
            self.txt_intro.text = ", ".join(cmap.get("I", []))
            self.txt_verse.text = ", ".join(cmap.get("V", []))
            self.txt_verse2.text = ", ".join(cmap.get("V2", []))
            self.txt_chorus.text = ", ".join(cmap.get("C", []))
            self.txt_bridge.text = ", ".join(cmap.get("B", []))
            self.txt_outro.text = ", ".join(cmap.get("O", []))
            
            lmap = g.get("lyrics_map", {})
            self.txt_lyric_intro.text = lmap.get("I", "")
            self.txt_lyric_verse.text = lmap.get("V", "")
            self.txt_lyric_verse2.text = lmap.get("V2", "")
            self.txt_lyric_chorus.text = lmap.get("C", "")
            self.txt_lyric_bridge.text = lmap.get("B", "")
            self.txt_lyric_outro.text = lmap.get("O", "")
            
            if "engine" in data:
                for k, v in data["engine"].items(): self.app.engine_config[k] = v

            if "sampler" in data:
                self.app.sampler_mapping = data["sampler"]

            # Store track config so on_start can use it
            self.app.loaded_track_data = data.get("tracks", None)
            
            self.lbl_status.text = f"Loaded: {os.path.basename(filename)}"
        except Exception as e:
            self.lbl_status.text = f"Preset Error: {e}"
            traceback.print_exc()

    def on_start(self, instance):
        try:
            bpm_val = int(self.txt_bpm.text)
            seed_val = int(self.txt_seed.text) if self.txt_seed.text.strip() else random.randint(0, 999999)
            key_val = int(self.spin_key.text)
            
            struct_str = self.txt_struct.text.strip() or "V"
            def get_chords(w): return [c.strip() for c in w.text.split(',') if c.strip()]
            
            chords_map = {
                'I': get_chords(self.txt_intro), 'V': get_chords(self.txt_verse),
                'V2': get_chords(self.txt_verse2), 'C': get_chords(self.txt_chorus),
                'B': get_chords(self.txt_bridge), 'O': get_chords(self.txt_outro)
            }
            
            lyrics_map = {
                'I': self.txt_lyric_intro.text, 'V': self.txt_lyric_verse.text,
                'V2': self.txt_lyric_verse2.text, 'C': self.txt_lyric_chorus.text,
                'B': self.txt_lyric_bridge.text, 'O': self.txt_lyric_outro.text
            }

            self.app.seed_offset = seed_val
            self.app.bpm = bpm_val
            self.app.structure_str = struct_str
            self.app.chords_map = chords_map
            self.app.auto_fill = self.chk_auto.active
            self.app.key_offset = key_val 
            
            self.app.engine_config["lyrics_map"] = lyrics_map

            self.btn_start.disabled = True
            self.btn_start.text = "GENERATING..."
            self.app.apply_settings_and_start(bpm_val, struct_str, chords_map, self.chk_auto.active, key_val)
            
        except ValueError: self.lbl_status.text = "Error: Invalid Input (Check BPM/Seed)"

    def update_gen_progress(self, progress):
        if progress >= 1.0:
            self.lbl_status.text = "Complete!"
            self.btn_start.text = "ENTER STUDIO"
            self.btn_start.background_color = (1, 0.2, 0.6, 1)
            self.btn_start.disabled = False
            try: self.btn_start.unbind(on_press=self.on_start)
            except: pass
            self.btn_start.bind(on_press=self.go_to_main)
        else:
            self.lbl_status.text = f"Generating... {int(progress*100)}%"

    def go_to_main(self, instance):
        self.app.transition_to_main()


# ==========================================
# UI: Piano Roll Editor
# ==========================================
class NoteWidget(Button):
    """A single note widget. Drag to move."""
    def __init__(self, note_event, px_per_sec, px_per_pitch, base_pitch, editor, **kwargs):
        super().__init__(**kwargs)
        self.note = note_event
        self.px_per_sec = px_per_sec
        self.px_per_pitch = px_per_pitch
        self.base_pitch = base_pitch
        self.editor = editor # Ref to parent editor for callbacks
        
        self.background_normal = ''
        self.background_color = (0.4, 0.7, 1.0, 0.9)
        self.border = (0,0,0,0)
        
        self.text = note_event.lyric if note_event.lyric else ""
        self.font_size = '11sp'
        self.color = (1,1,1,1)
        
        self.update_geometry()

    def update_geometry(self):
        if self.note.pitch <= 0: midi = 0
        else: midi = 69 + 12 * math.log2(self.note.pitch / 440.0)
        
        self.x = self.note.start_time * self.px_per_sec
        self.y = (midi - self.base_pitch) * self.px_per_pitch
        self.width = max(15, self.note.duration * self.px_per_sec)
        self.height = self.px_per_pitch - 2

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            if touch.is_double_tap:
                self.edit_lyric_popup()
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            parent_x = self.parent.x if self.parent else 0
            new_x = touch.x - parent_x
            
            # Snap X (Time)
            grid_sec = 60 / BPM / 4 # 16th note
            grid_px = grid_sec * self.px_per_sec
            new_x = round(new_x / grid_px) * grid_px
            self.x = max(0, new_x)
            
            # Snap Y (Pitch) - Visual feedback during drag
            parent_y = self.parent.y if self.parent else 0
            new_y = touch.y - parent_y
            grid_y = self.px_per_pitch
            new_y = round(new_y / grid_y) * grid_y
            self.y = new_y
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            
            # Commit changes
            new_time = self.x / self.px_per_sec
            
            rel_y = self.y / self.px_per_pitch
            new_midi = round(self.base_pitch + rel_y)
            new_freq = 440.0 * (2 ** ((new_midi - 69) / 12.0))
            
            self.note.start_time = max(0, new_time)
            self.note.pitch = new_freq
            self.update_geometry() # Snap properly
            return True
        return super().on_touch_up(touch)

    def edit_lyric_popup(self):
        def cb(text):
            self.note.lyric = text
            self.text = text
        TextInputPopup("Edit Lyric", self.note.lyric, cb).open()


class PianoRollEditor(ModalView):
    def __init__(self, track_spec, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.spec = track_spec
        self.app = app_instance
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = False
        
        # --- Config ---
        global BPM
        self.bpm = BPM
        self.bars = len(track_spec.get("chords", ["C"]*4))
        self.total_sec = (60/self.bpm) * 4 * self.bars
        
        self.px_per_sec = 120
        self.px_per_pitch = 24
        self.base_pitch = 36 # C2 start
        self.num_pitches = 60 # 5 Octaves range
        
        content_w = max(Window.width, self.total_sec * self.px_per_sec + 200)
        content_h = self.num_pitches * self.px_per_pitch
        
        # --- Layout ---
        root = BoxLayout(orientation='vertical')
        
        # 1. Toolbar
        top = BoxLayout(size_hint_y=None, height=50, padding=5, spacing=10)
        with top.canvas.before:
            Color(0.15, 0.15, 0.2, 1); Rectangle(pos=top.pos, size=top.size)
            
        top.add_widget(Label(text=f"Edit: {track_spec['role']}", bold=True, size_hint_x=0.3))
        
        btn_add = Button(text="+ Note", background_color=(0.3, 0.7, 1, 1), size_hint_x=0.15)
        btn_add.bind(on_press=self.add_center_note)
        top.add_widget(btn_add)
        
        btn_play = Button(text="Preview Track", background_color=(0.2, 0.8, 0.4, 1), size_hint_x=0.2)
        btn_play.bind(on_press=self.preview_track)
        top.add_widget(btn_play)
        
        btn_save = Button(text="Save & Close", background_color=(0.8, 0.5, 0.2, 1), size_hint_x=0.2)
        btn_save.bind(on_press=self.save_and_close)
        top.add_widget(btn_save)
        
        root.add_widget(top)
        
        # 2. Main Editor Area (ScrollView)
        self.scroll = ScrollView(size_hint=(1, 1), do_scroll_x=True, do_scroll_y=True)
        self.content = RelativeLayout(size_hint=(None, None), size=(content_w, content_h))
        
        # Draw Background Grid & Labels
        with self.content.canvas.before:
            Color(0.1, 0.1, 0.12, 1)
            Rectangle(pos=(0,0), size=(content_w, content_h))
            
            # Beat Grid
            sec_per_beat = 60/self.bpm
            Color(0.25, 0.25, 0.3, 1)
            for i in range(int(self.total_sec / sec_per_beat) + 1):
                x = i * sec_per_beat * self.px_per_sec
                # Bar lines brighter
                if i % 4 == 0: Line(points=[x, 0, x, content_h], width=1.5)
                else: Line(points=[x, 0, x, content_h], width=1)
            
            # Pitch Grid
            Color(0.2, 0.2, 0.25, 1)
            for i in range(self.num_pitches):
                y = i * self.px_per_pitch
                Line(points=[0, y, content_w, y])
                
        # Add Pitch Labels (Manual Labels on the left)
        # Note: In a RelativeLayout, simple Labels might move. 
        # For simplicity, we add them to content, they will scroll away. 
        # Ideally, use a separate fixed widget for keys.
        for i in range(self.num_pitches):
            midi = self.base_pitch + i
            note_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][midi % 12]
            octave = midi // 12 - 1
            if note_name == "C":
                lbl = Label(text=f"{note_name}{octave}", font_size='10sp', 
                            pos=(5, i * self.px_per_pitch), size_hint=(None, None), size=(30, self.px_per_pitch),
                            color=(1,1,1,0.5))
                self.content.add_widget(lbl)

        # Add Notes
        self.sequence = track_spec.get("sequence", [])
        self.widgets = []
        for note in self.sequence:
            w = NoteWidget(note, self.px_per_sec, self.px_per_pitch, self.base_pitch, self)
            self.content.add_widget(w)
            self.widgets.append(w)
            
        self.scroll.add_widget(self.content)
        root.add_widget(self.scroll)
        self.add_widget(root)

    def add_center_note(self, instance):
        # Add a note at the center of the viewport or start
        center_sec = (self.scroll.scroll_x * self.total_sec) + 1.0 # Offset slightly
        mid_pitch = 60 # C4
        freq = 440 * (2**((mid_pitch-69)/12))
        
        new_note = NoteEvent(center_sec, 0.5, freq, 0.8, "la")
        self.sequence.append(new_note)
        
        w = NoteWidget(new_note, self.px_per_sec, self.px_per_pitch, self.base_pitch, self)
        self.content.add_widget(w)
        self.widgets.append(w)

    def preview_track(self, instance):
        # Trigger single track regen and play via App
        # Save sequence first
        self.spec["sequence"] = self.sequence
        self.spec["keep_sequence"] = True
        self.app.refresh_single_track(self.spec)
        # Playback trigger is handled by refresh callback usually, or user presses Play in Main

    def save_and_close(self, instance):
        self.spec["sequence"] = self.sequence
        self.spec["keep_sequence"] = True # Lock sequence from AI overwrites
        self.app.refresh_single_track(self.spec) # Update Audio
        self.dismiss()

# Hz <=> MIDI Note 変換ヘルパー
def hz_to_midi(hz):
    if hz <= 0: return 0
    return 69 + 12 * np.log2(hz / 440.0)

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

class MelodicSculptorGrid(Widget):
    # ★ 引数に section_names を追加
    def __init__(self, spec, section_names=None, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.sequence = spec.get("sequence", [])
        
        # ★ セクション名のリストを保持
        self.section_names = section_names or []
        
        # 編集モード切替 ("NOTE" または "PITCH")
        self.edit_mode = "NOTE" 
        
        # Y軸の表示範囲 (C2〜C7)
        self.min_pitch = 36 
        self.max_pitch = 96 
        
        # X軸の設定
        self.step_duration = 0.125
        self.section_steps = 64
        self.pitch_resolution = 4 
        
        self.size_hint = (None, None)
        self.height = 1200 
        
        self.active_points = []
        self.pitch_curve = self.spec.get("pitch_curve", {})
        self.last_pitch_touch = None 
        
        self._load_sequence()
        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def _load_sequence(self):
        self.active_points = []
        if not self.sequence:
            self.total_steps = self.section_steps
            self.width = self.total_steps * 25
            return
            
        for note in self.sequence:
            start_time = getattr(note, "start_time", 0.0)
            pitch_hz = getattr(note, "pitch", 440.0)
            
            step = int(round(start_time / self.step_duration))
            midi_pitch = hz_to_midi(pitch_hz)
            
            params = getattr(note, "params", {})
            if params is None: params = {}
            pin_state = params.get("pin_state", 0)
            
            self.active_points.append({
                "step": step,
                "midi": midi_pitch,
                "note_ref": note,
                "pin": pin_state
            })
            
        self.active_points.sort(key=lambda x: x["step"])
        max_step = self.active_points[-1]["step"] if self.active_points else 0
        self.total_steps = max(self.section_steps, max_step + 16)
        self.width = self.total_steps * 30

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.y < self.y + 40 or touch.x < self.x + 40:
                return super().on_touch_down(touch)
                
            touch.ud['start_pos'] = touch.pos
            touch.ud['editing'] = True
            touch.ud['is_drag'] = False
            self.last_pitch_touch = None
            self.modify_point(touch)
            return True 
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.ud.get('editing') and self.collide_point(*touch.pos):
            sx, sy = touch.ud['start_pos']
            if touch.ud.get('is_drag') or abs(touch.x - sx) > 10 or abs(touch.y - sy) > 10:
                touch.ud['is_drag'] = True
                self.modify_point(touch)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.ud.get('editing'):
            # NOTEモードかつドラッグしていない時だけピン切り替え
            if not touch.ud.get('is_drag') and self.edit_mode == "NOTE":
                self.cycle_pin(touch)
            touch.ud['editing'] = False
            self.last_pitch_touch = None
            return True
        return super().on_touch_up(touch)

    def modify_point(self, touch):
        if not self.active_points and self.edit_mode == "NOTE": return
        step_w = self.width / max(1, self.total_steps)
        pitch_h = self.height / max(1, self.max_pitch - self.min_pitch)
        
        if self.edit_mode == "NOTE":
            # NOTEモード: 既存の点を半音単位(クオンタイズ)で上下に動かす
            target_step = int((touch.x - self.x) / step_w)
            target_step = max(0, min(self.total_steps - 1, target_step))
            target_midi = self.min_pitch + ((touch.y - self.y) / pitch_h)
            target_midi = max(self.min_pitch, min(self.max_pitch, target_midi))
            target_midi = round(target_midi) # 半音吸着
            
            for pt in self.active_points:
                if pt["step"] == target_step:
                    pt["midi"] = target_midi
                    break

        elif self.edit_mode == "PITCH":
            # PITCHモード: 高解像度＆吸着なしで自由なピッチカーブを記録
            sub_step_w = step_w / self.pitch_resolution
            target_sub_step = int((touch.x - self.x) / sub_step_w)
            
            target_midi = self.min_pitch + ((touch.y - self.y) / pitch_h)
            target_midi = max(self.min_pitch, min(self.max_pitch, target_midi)) # 吸着なし(小数点のまま)
            
            # 指を速くスワイプしたときに隙間を補間する処理
            if self.last_pitch_touch is not None:
                last_sub_step, last_midi = self.last_pitch_touch
                step_diff = target_sub_step - last_sub_step
                if abs(step_diff) > 1:
                    step_dir = 1 if step_diff > 0 else -1
                    for i in range(1, abs(step_diff)):
                        interp_sub_step = last_sub_step + i * step_dir
                        t = i / abs(step_diff)
                        interp_midi = last_midi + (target_midi - last_midi) * t
                        self.pitch_curve[interp_sub_step] = interp_midi
            
            self.pitch_curve[target_sub_step] = target_midi
            self.last_pitch_touch = (target_sub_step, target_midi)

        self.update_canvas()

    def cycle_pin(self, touch):
        if not self.active_points: return
        step_w = self.width / max(1, self.total_steps)
        pitch_h = self.height / max(1, self.max_pitch - self.min_pitch)
        tap_step = int((touch.x - self.x) / step_w)
        tap_midi = self.min_pitch + ((touch.y - self.y) / pitch_h)
        for pt in self.active_points:
            if pt["step"] == tap_step and abs(pt["midi"] - tap_midi) < 2.5:
                pt["pin"] = (pt["pin"] + 1) % 5
                break
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.clear()
        with self.canvas:
            step_w = self.width / max(1, self.total_steps)
            pitch_h = self.height / max(1, self.max_pitch - self.min_pitch)
            
            # 1. セクション背景
            section_colors = [
                (0.1, 0.4, 0.5, 0.3), (0.2, 0.5, 0.2, 0.3),
                (0.4, 0.4, 0.1, 0.3), (0.6, 0.2, 0.2, 0.3),
                (0.4, 0.1, 0.4, 0.3), (0.2, 0.2, 0.2, 0.3)
            ]
            
            for i in range(0, self.total_steps, self.section_steps):
                sec_idx = i // self.section_steps
                color_idx = sec_idx % len(section_colors)
                Color(*section_colors[color_idx])
                
                draw_steps = min(self.section_steps, self.total_steps - i)
                Rectangle(pos=(self.x + i * step_w, self.y), size=(draw_steps * step_w, self.height))
                
                # セクション境界線
                Color(1, 1, 1, 0.6)
                Line(points=[self.x + i * step_w, self.y, self.x + i * step_w, self.top], width=1.5)

                # ★【追加】セクション名の文字描画 (左上に透かしで表示)
                name = self.section_names[sec_idx] if sec_idx < len(self.section_names) else f"Sec {sec_idx+1}"
                lbl = CoreLabel(text=name, font_size=40, bold=True, color=(1, 1, 1, 0.4)) # 半透明の白文字
                lbl.refresh()
                tex = lbl.texture
                if tex:
                    Color(1, 1, 1, 1) # テクスチャを描画する時のベースカラー
                    # 各セクションの左上(少し余白を取る)に配置
                    Rectangle(pos=(self.x + i * step_w + 15, self.top - 60), size=tex.size, texture=tex)
            # 2. 横線
            Color(1, 1, 1, 0.1)
            for i in range(self.max_pitch - self.min_pitch + 1):
                if (self.min_pitch + i) % 12 == 0:
                    Line(points=[self.x, self.y + i*pitch_h, self.right, self.y + i*pitch_h], width=1.5)
                else:
                    Line(points=[self.x, self.y + i*pitch_h, self.right, self.y + i*pitch_h], width=0.5)

            # 3. メロディ波形 (NOTEモードの点と線)
            # 現在のモードに応じて透明度を変更し、どちらを編集しているか分かりやすくする
            note_alpha = 1.0 if self.edit_mode == "NOTE" else 0.4
            Color(0.2, 1.0, 0.8, note_alpha) 
            line_points = []
            for pt in self.active_points:
                px = self.x + (pt["step"] + 0.5) * step_w
                py = self.y + (pt["midi"] - self.min_pitch) * pitch_h
                line_points.extend([px, py])
            if len(line_points) >= 4:
                Line(points=line_points, width=3)

            for pt in self.active_points:
                px = self.x + (pt["step"] + 0.5) * step_w
                py = self.y + (pt["midi"] - self.min_pitch) * pitch_h
                
                if pt["pin"] == 1: Color(1.0, 0.2, 0.2, note_alpha)
                elif pt["pin"] == 2: Color(0.2, 0.4, 1.0, note_alpha)
                elif pt["pin"] == 3: Color(1.0, 1.0, 1.0, note_alpha)
                elif pt["pin"] == 4: Color(0.1, 0.1, 0.1, note_alpha)
                else: Color(0.2, 1.0, 0.8, note_alpha)
                Ellipse(pos=(px-12, py-12), size=(24, 24))

            # 4. ピッチカーブ波形 (PITCHモードのフリーハンド曲線)
            if self.pitch_curve:
                pitch_alpha = 1.0 if self.edit_mode == "PITCH" else 0.5
                Color(1.0, 0.4, 0.8, pitch_alpha) # ピンク色
                
                sub_step_w = step_w / self.pitch_resolution
                sorted_sub_steps = sorted(self.pitch_curve.keys())
                
                pitch_line_points = []
                last_ss = None
                
                for ss in sorted_sub_steps:
                    pmidi = self.pitch_curve[ss]
                    px = self.x + (ss + 0.5) * sub_step_w
                    py = self.y + (pmidi - self.min_pitch) * pitch_h
                    
                    # 繋がっていない描画(隙間が空いている)場合は別々の線として描く
                    if last_ss is not None and ss - last_ss > 1:
                        if len(pitch_line_points) >= 4:
                            Line(points=pitch_line_points, width=2.5)
                        pitch_line_points = []
                        
                    pitch_line_points.extend([px, py])
                    last_ss = ss
                    
                if len(pitch_line_points) >= 4:
                    Line(points=pitch_line_points, width=2.5)

            # 5. スクロールガイド
            Color(0, 0, 0, 0.5)
            Rectangle(pos=(self.x, self.y), size=(self.width, 40))
            Rectangle(pos=(self.x, self.y), size=(40, self.height))

    def apply_changes(self):
        # NOTEデータの保存
        for pt in self.active_points:
            new_hz = midi_to_hz(pt["midi"])
            pt["note_ref"].pitch = new_hz
            if not hasattr(pt["note_ref"], "params") or pt["note_ref"].params is None:
                pt["note_ref"].params = {}
            pt["note_ref"].params["pin_state"] = pt["pin"]
            
        # PITCHカーブデータの保存
        self.spec["pitch_curve"] = self.pitch_curve

class MelodicSculptorEditor(ModalView):
    def __init__(self, spec, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.95, 0.9)
        self.spec = spec
        self.app = app_instance
        
        # ★【追加】アプリ本体から structure_str (例: "I-V-C-B-O") を取得してリスト化
        struct_str = ""
        try:
            # メイン画面のテキスト入力欄から取得を試みる
            if hasattr(self.app, 'txt_struct'):
                struct_str = self.app.txt_struct.text
            elif hasattr(self.app, 'screen_start') and hasattr(self.app.screen_start, 'txt_struct'):
                struct_str = self.app.screen_start.txt_struct.text
        except Exception as e:
            print("Failed to get structure:", e)
            
        if not struct_str:
            struct_str = "I-V-C-O" # 取得できなかった場合のデフォルト

        # "I", "V", "C" などをフルネームにマッピング
        name_map = {
            "I": "Intro", "V": "Verse", "V2": "Verse 2", 
            "C": "Chorus", "B": "Bridge", "O": "Outro"
        }
        
        # ハイフンで分割し、マッピング辞書で変換
        section_names = []
        for s in struct_str.replace(" ", "").split("-"):
            if s:
                section_names.append(name_map.get(s, s)) # 辞書にない場合はそのまま表示
        
        layout = BoxLayout(orientation='vertical')
        
        # ヘッダー領域
        header = BoxLayout(size_hint_y=0.1)
        role = spec.get("role", "Track")
        header.add_widget(Label(text=f"Sculptor: {role}", bold=True, size_hint_x=0.2))
        
        self.btn_mode = ToggleButton(text="Mode: NOTE (基本)", size_hint_x=0.3, background_color=(0.2, 0.8, 0.8, 1))
        self.btn_mode.bind(on_release=self.toggle_mode)
        header.add_widget(self.btn_mode)
        
        btn_apply = Button(text="Apply & Render", size_hint_x=0.3, background_color=(0.2, 0.8, 0.2, 1))
        btn_apply.bind(on_release=self.on_apply)
        header.add_widget(btn_apply)
        
        btn_close = Button(text="Cancel", size_hint_x=0.2)
        btn_close.bind(on_release=self.dismiss)
        header.add_widget(btn_close)
        
        layout.add_widget(header)
        
        # 波形編集グリッド
        self.scroll = ScrollView(size_hint=(1, 0.85), do_scroll_x=True, do_scroll_y=True)
        # ★引数に section_names を渡す
        self.grid = MelodicSculptorGrid(self.spec, section_names=section_names) 
        self.scroll.add_widget(self.grid)
        layout.add_widget(self.scroll)
        
        self.lbl_hint = Label(text="【NOTEモード】余白スワイプで移動 / 点ドラッグで音程移動 / 点タップでピン", size_hint_y=0.05, color=(0.8, 0.8, 0.8, 1))
        layout.add_widget(self.lbl_hint)
            
        self.add_widget(layout)

    def toggle_mode(self, instance):
        if instance.state == 'down':
            instance.text = "Mode: PITCH (曲線)"
            instance.background_color = (1.0, 0.4, 0.8, 1) # PITCHモードはピンク
            self.grid.edit_mode = "PITCH"
            self.lbl_hint.text = "【PITCHモード】画面をなぞって細かいピッチベンド・ビブラートを自由に描画"
        else:
            instance.text = "Mode: NOTE (基本)"
            instance.background_color = (0.2, 0.8, 0.8, 1) # NOTEモードは水色
            self.grid.edit_mode = "NOTE"
            self.lbl_hint.text = "【NOTEモード】余白スワイプで移動 / 点ドラッグで音程移動 / 点タップでピン"
        self.grid.update_canvas()

    def on_apply(self, *args):
        if hasattr(self, 'grid'):
            self.grid.apply_changes()
            self.spec["keep_sequence"] = True
            self.dismiss()
            self.app.refresh_single_track(self.spec)

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 画面全体のルートレイアウト（縦並び）
        root = BoxLayout(orientation='vertical')
        
        # -- 1. Top Bar (上部メニュー) --
        top = BoxLayout(size_hint_y=0.08, padding=5)
        with top.canvas.before:
            Color(0.1, 0.1, 0.12, 1); Rectangle(pos=top.pos, size=top.size)
            
        self.title_lbl = Label(text="HyperNeko Studio", font_size='18sp', bold=True, size_hint_x=0.35, halign='left')
        top.add_widget(self.title_lbl)
        
        self.lbl_time = Label(text="0:00", size_hint_x=0.15, font_size='16sp', color=(0, 1, 1, 1))
        top.add_widget(self.lbl_time)
        
        self.lbl_section = Label(text="Sec: -", size_hint_x=0.15, font_size='11sp')
        top.add_widget(self.lbl_section)

        # [NEW] スタイル変更ボタン
        self.btn_style = Button(text="STYLE", size_hint_x=0.15, background_color=(0.3, 0.6, 0.9, 1))
        top.add_widget(self.btn_style)
        
        self.btn_export = Button(text="MENU", size_hint_x=0.2, background_color=(0.8, 0.4, 0.1, 1))
        top.add_widget(self.btn_export)
        
        # トップバーをルートに追加（1回だけ！）
        root.add_widget(top)
        
        # -- 2. Transport (再生・停止・シークバー) --
        transport = BoxLayout(size_hint_y=0.1, padding=5, spacing=10)
        self.btn_stop = Button(text="STOP", background_color=(0.8, 0.2, 0.2, 1), size_hint_x=0.2)
        self.btn_play = Button(text="PLAY", background_color=(0, 0.8, 0.4, 1), size_hint_x=0.2)
        transport.add_widget(self.btn_stop)
        transport.add_widget(self.btn_play)
        
        self.slider_seek = Slider(min=0, max=100, value=0, size_hint_x=0.6)
        transport.add_widget(self.slider_seek)
        
        # トランスポートをルートに追加（1回だけ！）
        root.add_widget(transport)
        
        # -- 3. Main Content (トラックリスト + 右パネル) --
        center = BoxLayout(orientation='horizontal', size_hint_y=0.65)
        
        # 左側: トラックリスト
        self.track_scroll = ScrollView(size_hint_x=0.6)
        self.grid = GridLayout(cols=1, spacing=2, size_hint_y=None)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.track_scroll.add_widget(self.grid)
        center.add_widget(self.track_scroll)
        
        # 右側: XYパッド & サンプラー
        right = BoxLayout(orientation='vertical', size_hint_x=0.4, padding=5, spacing=5)
        
        # XY Pad
        right.add_widget(Label(text="LIVE SYNTH (XY)", size_hint_y=0.05, font_size='11sp'))
        self.xy_pad = XYPad(size_hint_y=0.45)
        right.add_widget(self.xy_pad)
        
        # Sampler Pads
        right.add_widget(Label(text="SAMPLER / SE", size_hint_y=0.05, font_size='11sp'))
        sampler_grid = GridLayout(cols=3, spacing=5, size_hint_y=0.35)
        self.sampler_btns = []
        for i in range(6):
            btn = ToggleButton(text=f"Pad {i+1}", font_size='11sp')
            self.sampler_btns.append(btn)
            sampler_grid.add_widget(btn)
        right.add_widget(sampler_grid)
        
        btn_sampler_cfg = Button(text="Sampler Config", size_hint_y=0.1)
        self.btn_sampler_cfg = btn_sampler_cfg
        right.add_widget(btn_sampler_cfg)
        
        # センターエリアをルートに追加（1回だけ！）
        center.add_widget(right)
        root.add_widget(center)
        
        # -- 4. Macros (下部スイッチ) --
        macros = BoxLayout(size_hint_y=0.08, padding=5, spacing=10)
        self.tgl_off_vocal = ToggleButton(text="OFF VOCAL", size_hint_x=0.25)
        self.tgl_loop = ToggleButton(text="LOOP PLAY", size_hint_x=0.25)
        macros.add_widget(self.tgl_off_vocal)
        macros.add_widget(self.tgl_loop)
        
        # マクロをルートに追加（1回だけ！）
        root.add_widget(macros)
        
        # 最後にルートレイアウトを画面に追加
        self.add_widget(root)

    def update_sampler_labels(self, mapping):
        for i, btn in enumerate(self.sampler_btns):
            if i < len(mapping):
                name = mapping[i].get("name", "Empty")
                if mapping[i].get("type") == "file": name = "File"
                btn.text = f"{name[:6]}"

# ==========================================
# 17. RagnarokApp (Main Application Logic)
# ==========================================

class RagnarokApp(App):
    def build(self):
        self.title = "HyperNeko (Fixed Edition)"
        Window.clearcolor = (0.05, 0.05, 0.08, 1)
        
        # Core State
        self.engine_config = copy.deepcopy(DEFAULT_ENGINE_CONFIG)
        self.sampler_mapping = []
        for name in DEFAULT_SAMPLER_MAPPING:
            self.sampler_mapping.append({"type": "preset", "name": name, "loop": False})

        self.se_cache = {}
        self.active_channels = [None] * 6 
        self.recorder = AudioRecorder()

        # UI Setup
        self.sm = ScreenManager(transition=FadeTransition())
        self.screen_start = StartScreen(name='start')
        self.screen_main = MainScreen(name='main')
        self.sm.add_widget(self.screen_start)
        self.sm.add_widget(self.screen_main)
        
        # Wire up Main Screen events
        self.screen_main.update_sampler_labels(self.sampler_mapping)
        self.screen_main.btn_stop.bind(on_press=self.stop_manual)
        self.screen_main.btn_play.bind(on_press=self.toggle_play_pause)
        self.screen_main.btn_export.bind(on_press=self.open_export_menu)
        # [NEW] Bind Style Button
        self.screen_main.btn_style.bind(on_press=lambda x: self.open_style_menu(x))
        self.screen_main.slider_seek.bind(on_touch_down=self.on_seek_touch_down)
        self.screen_main.slider_seek.bind(on_touch_up=self.on_seek_touch_up)
        
        self.screen_main.tgl_off_vocal.bind(state=lambda i, s: self.set_off_vocal_macro(s=='down'))
        self.screen_main.tgl_loop.bind(state=lambda i, s: self.set_loop_mode(s=='down'))
        self.screen_main.btn_sampler_cfg.bind(on_press=lambda x: self.open_sampler_menu())
        
        for i, btn in enumerate(self.screen_main.sampler_btns):
            btn.bind(on_press=lambda x, idx=i: self.toggle_sample(idx, x))
        
        # Logic State
        self.tracks = []
        self.playing = False
        self.paused = False
        self.user_started = False
        self.loop_enabled = False
        self.is_seeking = False
        self.seed_offset = int(time.time())
        self.master_mix_cache = None
        self.start_ticks = 0
        self.paused_pos_ms = 0   
        self.master_sound = None
        
        self.loaded_track_data = None 
        self.bpm = BPM
        self.structure_str = ""
        self.chords_map = {}
        self.auto_fill = True
        self.key_offset = 0 
        
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.sm

    # --- ここから追加 ---
    def on_start(self):
        # 権限リクエストはOS側で許可済みなので、エラーを避けるために呼ばない
        logger.info("Android permissions: Assumed granted by OS settings.")

    # --- ここまで追加 ---

    # --- Setup & Gen ---
    
    def apply_settings_and_start(self, bpm, struct_str, chords_map, auto_fill, key_offset=0):
        self.master_mix_cache = None
        self.master_sound = None
        if self.playing: self.stop_playback()
        
        # Call global rebuilder
        rebuild_specs(bpm, struct_str, chords_map, auto_fill, self.loaded_track_data, key_offset=key_offset)
        self.start_generation_process()
        
    def start_generation_process(self):
        try:
            self.screen_main.title_lbl.text = f"HyperNeko [{BPM} BPM]"
            self.screen_main.slider_seek.max = max(1.0, TOTAL_DURATION)
            self.screen_main.grid.clear_widgets()
            self.tracks = []
            
            # Create Track Widgets (Increased height for buttons)
            for i, spec in enumerate(TRACK_SPECS):
                t_widget = TrackWidget(spec, App.get_running_app(), i)
                t_widget.size_hint_y = None
                t_widget.height = 170  # Increased height to fit DSP buttons
                self.screen_main.grid.add_widget(t_widget)
                self.tracks.append(t_widget)
            
            logger.info(f"Generation Started. Seed: {self.seed_offset}")
            
            # Reset widgets state
            for t in self.tracks: 
                t.reset()
                t.status = "Queued"

            # Start the Main Worker Thread
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()
                
        except Exception as e:
            logger.critical("Init Error", exc_info=True)
            self.show_global_error("Init Error", str(e))

    def open_track_editor(self, track_idx):
        if track_idx < 0 or track_idx >= len(self.tracks): return
        spec = self.tracks[track_idx].spec
        
        # 未使用のピアノロールを廃止し、メロディックスカルプター(波形認証UI)を開く
        MelodicSculptorEditor(spec, self).open()

    def refresh_single_track(self, spec):
        """
        Re-renders a single track (after manual edit) and mixes it back.
        Fix: Updates widget status and triggers remix.
        """
        # 1. 該当するTrackWidgetを探す（これがないと画面更新できない）
        target_widget = None
        for t in self.tracks:
            if t.spec == spec:
                target_widget = t
                break
        
        if not target_widget: return

        # 2. ステータスを「生成中」にする
        target_widget.status = "Generating..."
        self.show_global_error("Rendering", f"Updating {spec['role']}...")
        
        # Run in thread to not freeze UI
        def _rerender():
            try:
                # 【重要】設定変更（フィル追加など）を反映させるため、意図的にsequenceを消して再作曲させる
                # ただし、PianoRollで保存した場合(keep_sequence=True)は楽譜を維持する
                if not spec.get("keep_sequence", False):
                    spec.pop("sequence", None) 

                # Generate Audio (ここで新しい設定に基づいて音声を作る)
                audio = generate_poly_stem(spec, self.seed_offset, self.engine_config, None)
                
                # Update Widget Data (Main Thread)
                def _update_ui(dt):
                    # 3. 生成された音声をWidgetにセットする（これで音が変わる）
                    target_widget.raw_data = audio
                    target_widget.status = "Ready"
                    target_widget.progress = 1.0
                    
                    # 4. 全体をミックスし直して再生を更新
                    self.master_mix_cache = None # キャッシュクリア
                    self.on_track_toggled() 
                    
                    self.show_global_error("Done", "Track updated!")
                
                Clock.schedule_once(_update_ui, 0)

            except Exception as e:
                traceback.print_exc()
                err_msg = str(e) # <--- Capture string here immediately
                def _err(dt):
                    target_widget.status = "Error"
                    self.show_global_error("Error", err_msg) # <--- Use the string, not 'e'
                Clock.schedule_once(_err, 0)

            
        threading.Thread(target=_rerender).start()

    def _worker(self):
        """
        Main Worker Thread:
        Generates all tracks sequentially, mixes them, applies mastering, and exports.
        Calls global generate_poly_stem directly.
        """
        try:
            self.is_generating = True
            total_tracks = len(self.tracks)
            
            # 1. Generate All Tracks
            self.master_mix_cache = None
            
            for i, t_widget in enumerate(self.tracks):
                if not t_widget.is_active: 
                    t_widget.status = "Skip"
                    continue
                
                # Update UI Status
                msg = f"Gen Track {i+1}/{total_tracks}: {t_widget.spec['role']}"
                Clock.schedule_once(lambda dt, m=msg: setattr(self.screen_start.lbl_status, 'text', m), 0)
                
                # Generate Audio directly (Global function call)
                audio = generate_poly_stem(
                    t_widget.spec, 
                    self.seed_offset + (i * 55), 
                    self.engine_config, 
                    None # callback
                )
                audio = np.asarray(audio)

                if audio.ndim == 1:
                   audio = np.column_stack((audio, audio))
                
                # Assign to widget for visualization
                t_widget.raw_data = audio
                t_widget.status = "Ready"
                t_widget.progress = 1.0
                
                # Mix into Master
                if audio is not None and len(audio) > 0:
                    if self.master_mix_cache is None:
                        self.master_mix_cache = audio.copy()
                    else:
                        # Resize buffer if needed
                        curr_len = len(self.master_mix_cache)
                        new_len = len(audio)
                        if new_len > curr_len:
                            tmp = np.zeros((new_len, 2), dtype=np.float32)
                            tmp[:curr_len] = self.master_mix_cache
                            self.master_mix_cache = tmp
                        elif new_len < curr_len:
                            audio = np.pad(audio, ((0, curr_len - new_len), (0, 0)), 'constant')
                            
                        self.master_mix_cache += audio

            # 2. Mastering
            Clock.schedule_once(lambda dt: setattr(self.screen_start.lbl_status, 'text', "Mastering..."), 0)
            
            if self.master_mix_cache is not None:
                # Apply Multiband Limiter (Global function)
                self.master_mix_cache = mastering_limiter(self.master_mix_cache, FS)

            # 3. Export Temp WAV for Playback
            Clock.schedule_once(lambda dt: setattr(self.screen_start.lbl_status, 'text', "Finalizing..."), 0)
            wav_filename = f"neko_temp_{self.seed_offset}.wav"
            
            if self.master_mix_cache is not None:
                # Clip safely
                clipped = np.clip(self.master_mix_cache, -1.0, 1.0)
                wave_data = (clipped * 32767).astype(np.int16)
                
                # Pygame buffer approach
                if self.master_sound: self.master_sound.stop()
                self.master_sound = pygame.sndarray.make_sound(np.ascontiguousarray(wave_data))

            # 4. Finish
            Clock.schedule_once(lambda dt: self.screen_start.update_gen_progress(1.0), 0)

        except Exception as e:
            traceback.print_exc()
            Clock.schedule_once(lambda dt, msg=str(e): setattr(self.screen_start.lbl_status, 'text', f"Error: {msg}"), 0)
        finally:
            self.is_generating = False

    def request_track_regeneration(self, track_widget):
        # 【修正】古い start_gen メソッドは廃止されたので、
        # 新しい生成メソッド (refresh_single_track) に処理を委譲する
            self.refresh_single_track(track_widget.spec)


    def transition_to_main(self):
        self.user_started = True
        self.sm.current = 'main'
        self.start_playback()

    # --- Playback Logic ---

    def start_playback(self, start_offset_ms=0, start_paused=False):
        # In this fixed version, we rely on the self.master_sound generated in _worker
        # If tracks were regenerated individually, we might need to remix.
        
        try:
            # Check if remix needed
            if self.master_mix_cache is None:
                # Remix logic... (Simplified for brevity, assumes all ready)
                valid_tracks = [t for t in self.tracks if t.status == "Ready"]
                if valid_tracks:
                     base_len = max(len(t.raw_data) for t in valid_tracks if t.raw_data is not None)
                     self.master_mix_cache = np.zeros((base_len, 2), dtype=np.float32)
                     
                     for t in valid_tracks:
                         if not t.is_active or t.raw_data is None: continue
                         l = len(t.raw_data)
                         vol = t.slider_vol.value
                         pan = t.slider_pan.value
                         # Simple Pan
                         left = 1.0 - max(0, pan); right = 1.0 + min(0, pan)
                         
                         mix_add = t.raw_data * vol
                         mix_add[:, 0] *= left
                         mix_add[:, 1] *= right
                         
                         self.master_mix_cache[:l] += mix_add
                     
                     # Master Limiter
                     self.master_mix_cache = mastering_limiter(self.master_mix_cache, FS)
                     
                     # Re-create Sound object
                     clipped = np.clip(self.master_mix_cache, -1.0, 1.0)
                     wave_data = (clipped * 32767).astype(np.int16)
                     self.master_sound = pygame.sndarray.make_sound(np.ascontiguousarray(wave_data))

            if not self.master_sound: return

            # Seek logic for Pygame Sound is limited. We might play from start if seek not supported.
            # Pygame mixer doesn't support seeking on Sound objects easily. 
            # We must slice the array and create a new sound if we want to seek.
            
            # Since self.master_mix_cache holds the full float data:
            start_sample = int((start_offset_ms / 1000.0) * FS)
            if start_sample >= len(self.master_mix_cache):
                if self.loop_enabled: start_sample = 0; start_offset_ms = 0
                else: self.stop_playback(manual=False); return

            sliced_audio = self.master_mix_cache[start_sample:]
            audio_int16 = (np.clip(sliced_audio, -1, 1) * 32767).astype(np.int16)
            
            if self.master_sound: self.master_sound.stop()
            self.master_sound = pygame.sndarray.make_sound(np.ascontiguousarray(audio_int16))
            self.master_sound.play()

            if start_paused:
                pygame.mixer.pause()
                self.playing = False; self.paused = True
                self.paused_pos_ms = start_offset_ms
                self.screen_main.btn_play.text = "RESUME"
                self.screen_main.btn_play.background_color = (0, 0.5, 1, 1)
            else:
                self.playing = True; self.paused = False
                self.screen_main.btn_play.text = "PAUSE"
                self.screen_main.btn_play.background_color = (0, 0.8, 0.4, 1)
            
            self.start_ticks = pygame.time.get_ticks() - start_offset_ms

        except Exception as e:
            logger.error("Playback failed", exc_info=True)
            self.show_global_error("Playback Error", str(e))

    def stop_playback(self, manual=False):
        self.playing = False
        self.paused = False
        if self.master_sound: self.master_sound.stop()
        
        self.screen_main.btn_play.text = "PLAY"
        self.screen_main.btn_play.background_color = (0, 0.8, 0.4, 1)
        self.screen_main.slider_seek.value = 0
        self.screen_main.lbl_time.text = "0:00 / 0:00"
        self.screen_main.lbl_section.text = "Section: -"
        self.paused_pos_ms = 0
        
        # Loop handled in update

    def toggle_play_pause(self, instance):
        if not self.user_started: return
        
        if self.playing:
            pygame.mixer.pause()
            self.playing = False
            self.paused = True
            self.paused_pos_ms = pygame.time.get_ticks() - self.start_ticks
            self.screen_main.btn_play.text = "RESUME"
            self.screen_main.btn_play.background_color = (0, 0.5, 1, 1)
        elif self.paused:
            pygame.mixer.unpause()
            self.playing = True
            self.paused = False
            self.start_ticks = pygame.time.get_ticks() - self.paused_pos_ms
            self.screen_main.btn_play.text = "PAUSE"
            self.screen_main.btn_play.background_color = (0, 0.8, 0.4, 1)
        else:
            self.start_playback(0)

    def stop_manual(self, instance): 
        self.stop_playback(manual=True)

    def on_track_toggled(self):
        # If tracks change state, we need to remix.
        self.master_mix_cache = None
        if self.playing or self.paused:
            current_ms = (pygame.time.get_ticks() - self.start_ticks) if self.playing else self.paused_pos_ms
            self.start_playback(start_offset_ms=current_ms, start_paused=self.paused)
    
    def set_off_vocal_macro(self, is_off_vocal):
        changed = False
        for t in self.tracks:
            if "vocal" in t.spec["role"]:
                if t.is_active == is_off_vocal: # If currently active and we want off-vocal (False), flip it
                    t.set_active(not is_off_vocal)
                    changed = True
        if changed: self.on_track_toggled()
        
    def set_loop_mode(self, enabled): self.loop_enabled = enabled

    def on_seek_touch_down(self, instance, touch):
        if instance.collide_point(*touch.pos): self.is_seeking = True

    def on_seek_touch_up(self, instance, touch):
        if instance.collide_point(*touch.pos) or self.is_seeking:
            self.is_seeking = False
            seek_ms = instance.value * 1000
            self.start_playback(start_offset_ms=seek_ms, start_paused=self.paused)

    # --- Sampler & Live Synth Delegates ---
    def start_recording(self, slot_idx): self.recorder.start()
    def stop_recording(self):
        filename = f"rec_neko_{int(time.time())}.wav"
        return self.recorder.stop(filename)

    def toggle_sample(self, index, btn_instance):
        if index < 0 or index >= len(self.sampler_mapping): return
        slot = self.sampler_mapping[index]
        is_loop = slot.get("loop", False)
        
        if self.active_channels[index] is not None and self.active_channels[index].get_busy():
            self.active_channels[index].stop()
            self.active_channels[index] = None
            btn_instance.state = 'normal'
            return
            
        sound = self._load_sound_for_slot(slot)
        if not sound: 
            btn_instance.state = 'normal'; return
            
        loops = -1 if is_loop else 0
        chan = pygame.mixer.find_channel(True)
        if chan:
            chan.set_volume(0.8)
            chan.play(sound, loops=loops)
            self.active_channels[index] = chan
            if not is_loop: 
                btn_instance.state = 'normal'
                self.active_channels[index] = None 
        else:
            btn_instance.state = 'normal'

    def _load_sound_for_slot(self, slot):
        key = f"{slot['type']}_{slot.get('name')}_{slot.get('path')}"
        if key in self.se_cache: return self.se_cache[key]
        sound = None
        try:
            if slot["type"] == "file":
                path = slot.get("path")
                if path and os.path.exists(path): sound = pygame.mixer.Sound(path)
            else: 
                name = slot.get("name")
                preset = SE_PRESETS.get(name)
                if preset:
                    wave_data = generate_se_wave(preset)
                    audio_int16 = (soft_limit(wave_data) * 32767).astype(np.int16)
                    sound = pygame.sndarray.make_sound(np.ascontiguousarray(audio_int16))
        except Exception as e: logger.error(f"Sound Load Error: {e}")
        
        if sound: self.se_cache[key] = sound
        return sound

    def open_sampler_menu(self): SamplerConfigPopup(self.sampler_mapping, self._update_sampler_map).open()
    def _update_sampler_map(self, new_map): 
        self.sampler_mapping = new_map
        self.se_cache = {} 
        self.screen_main.update_sampler_labels(self.sampler_mapping)

    def update_live_synth(self, x, y):
        target_wave = "supersaw" 
        # Find a suitable track to steal settings from or default
        for t in self.tracks:
            if t.spec.get("xy_route", False) and t.is_active: 
                target_wave = t.spec.get("wave", "sin")
                break 
        
        grain = generate_live_grain(x, y, samples=2048, wave_type=target_wave)
        audio_int16 = (grain * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(np.ascontiguousarray(audio_int16))
        chan = pygame.mixer.find_channel(True)
        if chan: 
            chan.set_volume(0.5)
            chan.play(sound)

    def stop_live_synth(self): pass

    # --- Update Loop ---
    def update(self, dt):
        if not self.user_started and self.tracks:
            # Fake progress calc during generation
            progress_list = [t.progress for t in self.tracks]
            avg = sum(progress_list) / len(progress_list) if progress_list else 0
            if all(t.status in ["Ready", "Error", "Skip"] for t in self.tracks): avg = 1.0
            self.screen_start.update_gen_progress(avg)

        if self.user_started:
            current_ms = (pygame.time.get_ticks() - self.start_ticks) if self.playing else self.paused_pos_ms
            elapsed_sec = current_ms / 1000.0
            
            for t in self.tracks: 
                t.update_vis(self.playing, elapsed_sec)
            
            if self.playing and elapsed_sec >= TOTAL_DURATION:
                if self.loop_enabled:
                    self.start_playback(0)
                else:
                    self.stop_playback(manual=False)
                    return
            
            if not self.is_seeking: 
                self.screen_main.slider_seek.value = elapsed_sec
            
            cur_min = int(elapsed_sec // 60)
            cur_sec = int(elapsed_sec % 60)
            tot_min = int(TOTAL_DURATION // 60)
            tot_sec = int(TOTAL_DURATION % 60)
            self.screen_main.lbl_time.text = f"{cur_min}:{cur_sec:02d} / {tot_min}:{tot_sec:02d}"

            if BPM > 0:
                sec_per_bar = 240.0 / BPM
                current_bar = elapsed_sec / sec_per_bar
                current_code = "-"
                if 'SECTION_TIMINGS' in globals() and SECTION_TIMINGS:
                    for start_bar, code in reversed(SECTION_TIMINGS):
                        if current_bar >= start_bar: current_code = code; break
                self.screen_main.lbl_section.text = f"Playing: {current_code}"


    def play_guitar_strum(self, norm_x, string_idx):
        """
        XYPadで弦を弾いた瞬間に呼ばれる
        norm_x: 0.0〜1.0 (X軸の位置。これでコードを決める)
        string_idx: 0〜5 (どの弦を弾いたか。0が低音弦、5が高音弦)
        """
        # 1. X軸の位置からルート音（コード）を決める (例: C, D, E, F, G, A, B)
        root_notes = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94] # C3〜B3
        chord_idx = int(norm_x * len(root_notes))
        chord_idx = max(0, min(len(root_notes)-1, chord_idx))
        base_freq = root_notes[chord_idx]
        
        # 2. 弾いた弦(0〜5)に合わせて、コードの構成音(度数)を割り当てる
        # ギターのオープンコード風（Root, 5th, Octave, Major 3rd, 5th, Octave）
        intervals = [0, 7, 12, 16, 19, 24] 
        note_freq = base_freq * (2 ** (intervals[string_idx] / 12.0))
        
        # 3. ギターの音色（Karplus-Strong）を生成 (長さは1.5秒程度で減衰させる)
        # ※すでに v21.x の render_track_sequence 内で使われている generate_karplus_custom を流用するか、
        # もしくは generate_live_grain(wave_type="karplus") を単発用に少し改変して使います。
        
        samples = int(44100 * 1.5) 
        wave_data = generate_live_grain(0, 0, samples=samples, wave_type="karplus") 
        # (※実際の generate_live_grain は x,y を引数に取るので、note_freqを直接渡せるように少し改造するとベターです)
        
        # ピッチを直接指定して生成する簡易Karplus:
        N = int(44100 / note_freq)
        raw = np.zeros(samples, dtype=np.float32)
        buf = np.random.uniform(-1, 1, max(2, N)).astype(np.float32)
        raw[:N] = buf
        for i in range(N, samples, N):
            block_end = min(i + N, samples)
            prev = raw[i-N:i]
            # ローパスフィルタをかけつつ減衰(0.995)させてリアルな弦の響きに
            new_block = 0.5 * (prev + np.roll(prev, 1)) * 0.995
            raw[i:block_end] = new_block[:block_end - i]
            
        # ステレオ化してPygameで再生
        stereo = np.column_stack((raw, raw))
        sound = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
        
        # 空いているチャンネルを探して鳴らす
        channel = pygame.mixer.find_channel()
        if channel:
            channel.play(sound)

         # ==========================================
    # Menu & Export (Updated with JSON Save)
    # ==========================================
    def show_global_error(self, title, message): 
        Clock.schedule_once(lambda dt: ErrorPopup(title, message).open())

    def open_export_menu(self, instance):
        # Improved popup for export options & navigation
        content = BoxLayout(orientation='vertical', padding=20, spacing=10)
        content.add_widget(Label(text="Main Menu", font_size='20sp', bold=True, size_hint_y=0.15))
        
        # [NEW] Save Preset Button
        btn_json = Button(text="Save Project (JSON)", background_color=(0.4, 0.8, 0.4, 1))
        btn_json.bind(on_press=lambda x: self._export_preset_dialog())
        content.add_widget(btn_json)

        # Existing Export Buttons
        btn_wav = Button(text="Save WAV (Mix)", background_color=(0.2, 0.6, 1, 1))
        btn_wav.bind(on_press=lambda x: self._export_wav_dialog())
        content.add_widget(btn_wav)
        
        btn_midi = Button(text="Save MIDI", background_color=(0.8, 0.6, 0.2, 1))
        btn_midi.bind(on_press=lambda x: self._export_midi_dialog())
        content.add_widget(btn_midi)
        
        # ▼▼▼ 追加場所はここ！！ (Backボタンの前に入れるのがおすすめ) ▼▼▼
        # Self-Vocaloid (録音画面) を開くボタン
        btn_rec = Button(text="SELF-VOCALOID REC", background_color=(1, 0.3, 0.3, 1)) # 赤っぽい色
        # 引数エラーを防ぐため lambda を使用
        btn_rec.bind(on_press=lambda x: self._open_vocal_capture())
        content.add_widget(btn_rec)
        # ▲▲▲ 追加ここまで ▲▲▲
        
        # Return Button
        btn_back = Button(text="Back to Title", background_color=(0.5, 0.5, 0.5, 1))
        btn_back.bind(on_press=self.return_to_title)
        content.add_widget(btn_back)
        
        self.menu_popup = ModalView(size_hint=(0.6, 0.7)) # ボタンが増えたので少し縦長に
        self.menu_popup.add_widget(content)
        self.menu_popup.open()
    
    def return_to_title(self, instance):
        """完全リセットを行ってタイトルへ戻る（修正版）"""
        # ポップアップがあれば閉じる
        if hasattr(self, 'menu_popup') and self.menu_popup:
            self.menu_popup.dismiss()
        
        # 1. 再生停止とオーディオエンジンのリセット
        self.stop_playback()
        pygame.mixer.stop() # 全チャンネル強制停止
        
        # 2. フラグのリセット
        self.playing = False
        self.paused = False
        self.user_started = False
        self.master_mix_cache = None
        self.master_sound = None
        
        # 3. 前回の曲データとトラック情報の消去
        self.loaded_track_data = None 
        self.tracks = []  # トラックリストを空にする
        if hasattr(self, 'screen_main') and hasattr(self.screen_main, 'grid'):
             self.screen_main.grid.clear_widgets() # メイン画面のトラック表示も消しておく
        
        # 4. スタート画面のボタン状態を強制的に「初期状態」に戻す
        btn = self.screen_start.btn_start
        btn.text = "INITIALIZE SYSTEM"
        btn.background_color = (0.2, 0.6, 0.8, 1) # 青色に戻す
        btn.disabled = False
        
        # 5. バインドの付け替え
        try: btn.unbind(on_press=self.screen_start.go_to_main)
        except: pass
        try: btn.unbind(on_press=self.screen_start.on_start)
        except: pass
        
        # 生成用メソッドを再バインド
        btn.bind(on_press=self.screen_start.on_start)
        
        # ステータス表示のリセット
        self.screen_start.lbl_status.text = "System Ready."
        
        # 6. 画面遷移
        self.sm.transition.direction = 'right'
        self.sm.current = 'start'
        
        logging.info("Returned to Title Screen (Full Reset Complete).")

    # --- ヘルパーメソッド群 ---

    def _export_preset_dialog(self):
        """Dialog to save the current state as a JSON preset file."""
        if hasattr(self, 'menu_popup'): self.menu_popup.dismiss()
        
        def cb(text):
            filename = text if text.endswith(".json") else text + ".json"
            try:
                data = PresetManager.create_preset_data(self)
                success, msg = PresetManager.save_preset(filename, data)
                self.show_global_error("Save Result", msg)
            except Exception as e:
                self.show_global_error("Save Error", str(e))
                traceback.print_exc()
        
        default_name = f"neko_project_{self.seed_offset}.json"
        TextInputPopup("Save Project As:", default_name, cb).open()

    def _export_wav_dialog(self):
        if hasattr(self, 'menu_popup'): self.menu_popup.dismiss()
        def cb(text):
            filename = text if text.endswith(".wav") else text + ".wav"
            try:
                if self.master_mix_cache is not None:
                    clipped = np.clip(self.master_mix_cache, -1.0, 1.0)
                    wave_data = (clipped * 32767).astype(np.int16)
                    with wave.open(filename, 'w') as wf:
                        wf.setnchannels(2)
                        wf.setsampwidth(2)
                        wf.setframerate(FS)
                        wf.writeframes(wave_data.tobytes())
                    self.show_global_error("Success", f"Saved to {filename}")
                else:
                    self.show_global_error("Error", "No audio generated yet.")
            except Exception as e:
                self.show_global_error("Export Error", str(e))
                
        TextInputPopup("Save WAV As:", f"neko_{self.seed_offset}.wav", cb).open()

    def _export_midi_dialog(self):
        if hasattr(self, 'menu_popup'): self.menu_popup.dismiss()
        def cb(text):
            filename = text if text.endswith(".mid") else text + ".mid"
            if export_midi_file(filename, self.seed_offset, self.engine_config):
                self.show_global_error("Success", f"Saved to {filename}")
            else:
                self.show_global_error("Error", "MIDI export failed.")
        TextInputPopup("Save MIDI As:", f"neko_{self.seed_offset}.mid", cb).open()

    def _open_vocal_capture(self):
        if hasattr(self, 'menu_popup'): self.menu_popup.dismiss()
        # 新しいインポート画面を開く
        VocalFileImportPopup(self).open()

    def open_style_menu(self, instance):
        content = BoxLayout(orientation='vertical', padding=20, spacing=10)
        with content.canvas.before: 
            Color(0.1, 0.1, 0.15, 1); Rectangle(pos=content.pos, size=content.size)
            
        content.add_widget(Label(text="SELECT GENRE", font_size='20sp', bold=True, size_hint_y=0.15))
        
        scroll = ScrollView(size_hint_y=0.85)
        grid = GridLayout(cols=2, spacing=10, size_hint_y=None, padding=10)
        grid.bind(minimum_height=grid.setter('height'))
        
        for genre in GENRE_STYLES.keys():
            btn = Button(text=genre, background_color=(0.3, 0.6, 0.9, 1), size_hint_y=None, height=50)
            btn.bind(on_press=lambda x, g=genre: self.apply_genre_style(g))
            grid.add_widget(btn)
        
        scroll.add_widget(grid)
        content.add_widget(scroll)
        
        self.genre_popup = ModalView(size_hint=(0.8, 0.8))
        self.genre_popup.add_widget(content)
        self.genre_popup.open()

    def apply_genre_style(self, genre):
        """
        Applies a genre style. 
        If on Start Screen: Updates UI text and prepares internal data for initial generation.
        If on Main Screen: Updates BPM, modifies active tracks live, and triggers regeneration.
        """
        if hasattr(self, 'genre_popup') and self.genre_popup:
            self.genre_popup.dismiss()
            
        data = GENRE_STYLES.get(genre)
        if not data: return
        
        # 1. Update Global Engine Config (Parameters like swing, bitcrush, etc.)
        eng = data.get("engine", {})
        new_bpm = eng.get("bpm", 160)
        
        for k, v in eng.items():
            if k != "bpm": 
                self.engine_config[k] = v

        # 2. Logic Split based on Current Screen
        if self.sm.current == 'start':
            # --- START SCREEN LOGIC (Pre-generation) ---
            # Update UI text
            self.screen_start.txt_bpm.text = str(new_bpm)
            
            # Reset loaded data
            self.loaded_track_data = []
            common = data.get("common", {})
            
            # Construct track configurations from the style data
            for role, settings in data.items():
                if role in ["engine", "common"]: continue
                
                track_conf = settings.copy()
                track_conf["role"] = role
                
                # Merge common DSP effects
                if "dsp" in common:
                    current_dsp = track_conf.get("dsp", [])
                    # Append common DSPs (e.g. compression on everything)
                    track_conf["dsp"] = list(set(current_dsp + common["dsp"]))
                
                self.loaded_track_data.append(track_conf)
                
            self.screen_start.lbl_status.text = f"Selected: {genre}"

        elif self.sm.current == 'main':
            # --- MAIN SCREEN LOGIC (Live Update) ---
            # Stop audio to prevent glitches during update
            self.stop_playback()
            
            # Update BPM Globally
            self.bpm = new_bpm
            
            # 【修正】BPM変更に伴い、曲の総再生時間(秒)も再計算する
            global BPM, TOTAL_DURATION, TOTAL_BARS
            BPM = new_bpm 
            
            # TOTAL_BARS（総小節数）は変わっていないので、新しいBPMで時間を計算し直す
            if 'TOTAL_BARS' in globals() and TOTAL_BARS > 0:
                TOTAL_DURATION = TOTAL_BARS * 4 * (60 / BPM)
                # シークバーの表示もバグらないように最大値を更新
                self.screen_main.slider_seek.max = max(1.0, TOTAL_DURATION)

            self.screen_main.title_lbl.text = f"HyperNeko [{BPM} BPM]"
            self.master_mix_cache = None
            
            # Restart Worker Thread to regenerate audio
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()

    def _apply_style_live(self, style_data):
        """
        Helper method to update existing TrackWidgets on the Main Screen
        without destroying/recreating them.
        """
        common = style_data.get("common", {})
        
        for t_widget in self.tracks:
            role = t_widget.spec["role"]
            
            # Find matching settings in the style data
            # The style data keys are role names (e.g., "kick", "vocal_lead")
            # We match exactly here.
            new_settings = style_data.get(role)
            
            if new_settings:
                # 1. Update Volume
                if "vol" in new_settings:
                    vol = new_settings["vol"]
                    t_widget.slider_vol.value = vol
                    t_widget.spec["vol"] = vol
                
                # 2. Update Rhythm Pattern
                if "pattern" in new_settings:
                    pat_key = new_settings["pattern"]
                    # If the style specifies a pattern template, apply it to all sections
                    if pat_key in RHYTHM_TEMPLATES:
                        new_pat = RHYTHM_TEMPLATES[pat_key]
                        for sec in t_widget.spec["pattern_map"]:
                            t_widget.spec["pattern_map"][sec] = new_pat[:]

                # 3. Update DSP Chain
                target_dsp = new_settings.get("dsp", [])
                if "dsp" in common: 
                    target_dsp = target_dsp + common["dsp"]
                
                # Update spec and UI state
                t_widget.spec["dsp"] = list(set(target_dsp)) # Remove duplicates
                t_widget.dsp_chain = t_widget.spec["dsp"]
                
                # Refresh the toggle buttons on the track widget
                for key, btn in t_widget.dsp_buttons.items():
                    is_active = (key in t_widget.dsp_chain)
                    btn.state = 'down' if is_active else 'normal'
                    btn.background_color = (0.4, 0.8, 0.6, 1) if is_active else (0.25, 0.25, 0.25, 1)

                # 4. Other Parameters
                if "wave" in new_settings: t_widget.spec["wave"] = new_settings["wave"]
                if "style" in new_settings: t_widget.spec["style"] = new_settings["style"]
                
                # 5. Active State (Solo/Mute logic from style)
                if "active" in new_settings:
                    should_active = new_settings["active"]
                    t_widget.set_active(should_active)
                    t_widget.spec["initial_active"] = should_active

    def load_preset_data(self, data):
        """
        Loads a saved JSON preset into the engine variables.
        """
        meta = data.get("global", {})
        
        # Update Start Screen UI
        if hasattr(self, 'screen_start'):
            self.screen_start.txt_bpm.text = str(meta.get("bpm", 160))
            self.screen_start.txt_struct.text = meta.get("structure_str", "")
            self.screen_start.txt_seed.text = str(meta.get("seed", ""))
            self.screen_start.chk_auto.active = meta.get("auto_fill", True)
            self.screen_start.spin_key.text = str(meta.get("key_offset", 0))

            # Restore Lyrics / Chords maps
            cmap = meta.get("chords_map", {})
            self.screen_start.txt_intro.text = ", ".join(cmap.get("I", []))
            self.screen_start.txt_verse.text = ", ".join(cmap.get("V", []))
            self.screen_start.txt_verse2.text = ", ".join(cmap.get("V2", []))
            self.screen_start.txt_chorus.text = ", ".join(cmap.get("C", []))
            self.screen_start.txt_bridge.text = ", ".join(cmap.get("B", []))
            self.screen_start.txt_outro.text = ", ".join(cmap.get("O", []))
            
            lmap = meta.get("lyrics_map", {})
            self.screen_start.txt_lyric_intro.text = lmap.get("I", "")
            self.screen_start.txt_lyric_verse.text = lmap.get("V", "")
            self.screen_start.txt_lyric_verse2.text = lmap.get("V2", "")
            self.screen_start.txt_lyric_chorus.text = lmap.get("C", "")
            self.screen_start.txt_lyric_bridge.text = lmap.get("B", "")
            self.screen_start.txt_lyric_outro.text = lmap.get("O", "")

        # Load Engine & Sampler Data
        self.engine_config = data.get("engine", DEFAULT_ENGINE_CONFIG)
        self.sampler_mapping = data.get("sampler", self.sampler_mapping)
        
        # Important: Store track overrides so they are used when 'INITIALIZE SYSTEM' is clicked
        self.loaded_track_data = data.get("tracks", [])
        self.key_offset = meta.get("key_offset", 0)


# ==========================================
# 18. Execution
# ==========================================
if __name__ == '__main__':
    try:
        RagnarokApp().run()
    except Exception as e:
        print("CRITICAL CRASH:")
        traceback.print_exc()