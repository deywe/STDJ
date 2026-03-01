"""
STDJ WARP MONITOR — Qt/PyQtGraph Edition v2 (fix PyQt6 QImage.bits())
Dependências: pip install pyqtgraph PyQt6 pandas pyarrow numpy
"""

import sys, os, hashlib, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError:
    print("❌  Instale: pip install pyqtgraph PyQt6"); sys.exit(1)

pg.setConfigOptions(antialias=True, useOpenGL=True, enableExperimental=True)

# ══════════════════════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════════════════════
BG      = '#020812'; PANEL   = '#040d1a'; BORDER  = '#0a2a4a'
CYAN    = '#00f5ff'; CYAN_D  = '#0077aa'; GREEN   = '#00ff88'
RED     = '#ff2244'; ORANGE  = '#ff8800'; PURPLE  = '#aa44ff'
TEXT_LO = '#3a6080'; TEXT_MID= '#6ab0d0'; TEXT_HI = '#c8eeff'

def qc(h, a=255):
    c = QtGui.QColor(h); c.setAlpha(a); return c

# ══════════════════════════════════════════════════════════════════════════════
# DADOS
# ══════════════════════════════════════════════════════════════════════════════
TELEMETRY_FILE = "stdj_secure_telemetry.parquet"
if not os.path.exists(TELEMETRY_FILE):
    print("❌  Dataset não encontrado."); sys.exit(1)

df          = pd.read_parquet(TELEMETRY_FILE)
N           = len(df)
veracity_v  = df['veracity'].values.astype(np.float64)
warp_v      = df['warp_velocity_c'].values / 1e10
status_arr  = df['status'].values
is_stable   = (status_arr == 'STABLE')
frames_idx  = np.arange(N, dtype=np.float64)
print(f"✅  {N:,} frames carregados.")

# ── heatmap como array numpy direto (evita QImage.bits()) ────────────────────
side  = int(np.sqrt(N)) + 1
pad   = side * side
ver_p = np.pad(veracity_v, (0, pad - N), constant_values=np.nan)
matrix = ver_p.reshape((side, side))

mn, mx    = np.nanmin(matrix), np.nanmax(matrix)
norm_mat  = (matrix - mn) / max(mx - mn, 1e-12)
norm_mat  = np.nan_to_num(norm_mat, nan=0.0)

# colormap azul-escuro → ciano → branco, shape (H, W, 3) uint8
r_ch = np.clip(norm_mat * 1.5 - 0.5, 0, 1)
g_ch = np.clip(norm_mat * 2.0 - 0.8, 0, 1)
b_ch = np.clip(norm_mat * 1.2,       0, 1)
# pyqtgraph ImageItem espera (W, H, 3) ou (W, H) — transpomos
HMAP_ARR = (np.stack([r_ch, g_ch, b_ch], axis=-1) * 255).astype(np.uint8)
HMAP_ARR = HMAP_ARR.transpose(1, 0, 2)   # (W, H, 3)
HMAP_W, HMAP_H = HMAP_ARR.shape[:2]

# ── scatter veracity: downsample para max 8000 pontos visíveis ────────────────
MAX_SC = 8000
if N > MAX_SC:
    sc_idx = np.linspace(0, N - 1, MAX_SC, dtype=int)
else:
    sc_idx = np.arange(N)
sc_x      = frames_idx[sc_idx]
sc_y      = veracity_v[sc_idx]
sc_stable = is_stable[sc_idx]
sc_colors = [pg.mkBrush(GREEN if s else RED) for s in sc_stable]

# ══════════════════════════════════════════════════════════════════════════════
# JANELA
# ══════════════════════════════════════════════════════════════════════════════
class STDJMonitor(QtWidgets.QMainWindow):
    _sha_ready = QtCore.pyqtSignal(str, str, bool)   # sig, sha_str, valid

    def __init__(self):
        super().__init__()
        self.setWindowTitle('STDJ WARP MONITOR — HOLOGRAPHIC DISPLAY')
        self.resize(1600, 900)
        self.setStyleSheet(f'background:{BG}; color:{TEXT_HI};')
        self.frame_idx  = 0
        self.playing    = False
        self._sha_pool  = ThreadPoolExecutor(max_workers=2)
        self._sha_busy  = False
        self._fps_acc   = 0
        self._fps_t     = time.perf_counter()

        self._sha_ready.connect(self._apply_sha)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        self._build_header(root)
        self._build_body(root)
        self._build_sensor(root)
        self._build_controls(root)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)   # 60 Hz target

        self._go_to(0)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lbl(self, txt, color=TEXT_MID, size=8, bold=False):
        l = QtWidgets.QLabel(txt)
        l.setStyleSheet(
            f'color:{color};font-family:monospace;font-size:{size}pt;'
            f'font-weight:{"bold" if bold else "normal"};background:transparent;'
        )
        return l

    def _panel(self, title, tc=CYAN):
        f = QtWidgets.QFrame()
        f.setStyleSheet(
            f'QFrame{{background:{PANEL};border:1px solid {BORDER};border-radius:4px;}}'
        )
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(8, 6, 8, 6); lay.setSpacing(3)
        h = self._lbl(f'[ {title} ]', color=tc, size=7, bold=True)
        h.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(h)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BORDER};'); lay.addWidget(sep)
        return f, lay

    def _btn(self, label, color, cb):
        b = QtWidgets.QPushButton(label)
        b.setFixedHeight(32)
        b.setStyleSheet(f"""
            QPushButton{{background:{PANEL};color:{color};
              border:1px solid {BORDER};border-radius:4px;
              font-family:monospace;font-size:8pt;font-weight:bold;padding:0 12px;}}
            QPushButton:hover{{background:{BORDER};}}
            QPushButton:pressed{{background:#0a1f40;}}
        """)
        b.clicked.connect(cb); return b

    # ── header ────────────────────────────────────────────────────────────────
    def _build_header(self, root):
        col = QtWidgets.QVBoxLayout()
        t = self._lbl('◈  S T D J   W A R P   M O N I T O R  ◈',
                      color=CYAN, size=14, bold=True)
        t.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        s = self._lbl(
            f'TELEMETRY STREAM · {N:,} FRAMES · SUB-PLANCK · 1200 QUBITS',
            color=TEXT_LO, size=7)
        s.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        col.addWidget(t); col.addWidget(s)
        root.addLayout(col)

    # ── body ──────────────────────────────────────────────────────────────────
    def _build_body(self, root):
        body = QtWidgets.QHBoxLayout(); body.setSpacing(6)

        # LEFT
        left, ll = self._panel('FRAME TELEMETRY', CYAN)
        left.setFixedWidth(245); self._build_tele(ll)
        body.addWidget(left)

        # CENTER
        cc = QtWidgets.QVBoxLayout(); cc.setSpacing(4)
        hf, hl = self._panel('WARP FIELD FIDELITY MATRIX', CYAN)
        self._build_heatmap(hl); cc.addWidget(hf, stretch=5)
        cr = QtWidgets.QHBoxLayout(); cr.setSpacing(4)
        self._build_warp_chart(cr)
        self._build_ver_chart(cr)
        cc.addLayout(cr, stretch=2)
        body.addLayout(cc, stretch=1)

        # RIGHT
        right, rl = self._panel('FIELD ANALYSIS', PURPLE)
        right.setFixedWidth(225); self._build_stats(rl)
        body.addWidget(right)

        root.addLayout(body, stretch=1)

    # ── telemetry panel ───────────────────────────────────────────────────────
    def _build_tele(self, lay):
        def row(k, vc=TEXT_HI):
            h = QtWidgets.QHBoxLayout()
            h.addWidget(self._lbl(k, TEXT_LO, 7))
            v = self._lbl('—', vc, 8, True)
            v.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            h.addWidget(v); lay.addLayout(h); return v

        self.t_frame  = row('FRAME',    TEXT_HI)
        self.t_real   = row('T_REAL',   TEXT_MID)
        self.t_verac  = row('VERACITY', GREEN)
        self.t_warp   = row('WARP',     CYAN)
        self.t_stdj   = row('T_STDJ',   PURPLE)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BORDER};'); lay.addWidget(sep)

        self.t_status = self._lbl('—', GREEN, 12, True)
        self.t_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.t_status)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep2.setStyleSheet(f'color:{BORDER};'); lay.addWidget(sep2)

        lay.addWidget(self._lbl('SHA-256', TEXT_LO, 6))
        self.t_sha   = self._lbl('—', TEXT_MID, 6); self.t_sha.setWordWrap(True)
        self.t_valid = self._lbl('—', GREEN, 8, True)
        self.t_valid.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.t_sha); lay.addWidget(self.t_valid)
        lay.addStretch()

    # ── heatmap ───────────────────────────────────────────────────────────────
    def _build_heatmap(self, lay):
        gv = pg.GraphicsLayoutWidget(); gv.setBackground(PANEL)
        vb = gv.addViewBox()
        vb.setAspectLocked(True); vb.invertY(True)

        img = pg.ImageItem()
        img.setImage(HMAP_ARR)          # (W, H, 3) uint8 — sem QImage
        vb.addItem(img)

        self._hmap_marker = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(ORANGE), pen=pg.mkPen(RED, width=1.5))
        self._hmap_vline  = pg.InfiniteLine(angle=90, pen=pg.mkPen(ORANGE, width=0.8, alpha=120))
        self._hmap_hline  = pg.InfiniteLine(angle=0,  pen=pg.mkPen(ORANGE, width=0.8, alpha=120))
        vb.addItem(self._hmap_marker)
        vb.addItem(self._hmap_vline)
        vb.addItem(self._hmap_hline)
        lay.addWidget(gv)

    # ── warp chart ────────────────────────────────────────────────────────────
    def _build_warp_chart(self, row_lay):
        pw = pg.PlotWidget(background=PANEL)
        pw.setTitle('<span style="color:#aa44ff;font-size:7pt">WARP VELOCITY</span>')
        pw.hideAxis('bottom'); pw.hideAxis('left'); pw.setMinimumHeight(120)
        pw.plot(warp_v, pen=pg.mkPen(PURPLE, width=1.0))
        self._warp_marker = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(ORANGE))
        self._warp_vline  = pg.InfiniteLine(angle=90, pen=pg.mkPen(ORANGE, width=0.8, alpha=150))
        pw.addItem(self._warp_marker); pw.addItem(self._warp_vline)
        row_lay.addWidget(pw)

    # ── veracity chart ────────────────────────────────────────────────────────
    def _build_ver_chart(self, row_lay):
        pw = pg.PlotWidget(background=PANEL)
        pw.setTitle('<span style="color:#00f5ff;font-size:7pt">VERACITY TIMELINE</span>')
        pw.hideAxis('bottom'); pw.hideAxis('left'); pw.setMinimumHeight(120)
        sp = pg.ScatterPlotItem(
            x=sc_x, y=sc_y, size=2, pxMode=True,
            brush=sc_colors, pen=None)
        pw.addItem(sp)
        pw.addLine(y=0.99, pen=pg.mkPen(CYAN_D, width=0.8,
                   style=QtCore.Qt.PenStyle.DashLine))
        self._ver_marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(ORANGE))
        self._ver_vline  = pg.InfiniteLine(angle=90, pen=pg.mkPen(ORANGE, width=1.0, alpha=180))
        pw.addItem(self._ver_marker); pw.addItem(self._ver_vline)
        row_lay.addWidget(pw)

    # ── stats panel ───────────────────────────────────────────────────────────
    def _build_stats(self, lay):
        n_s = is_stable.sum(); n_t = N - n_s

        def stat_row(k, v, c):
            h = QtWidgets.QHBoxLayout()
            h.addWidget(self._lbl(k, TEXT_LO, 6))
            vl = self._lbl(v, c, 7, True)
            vl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            h.addWidget(vl); lay.addLayout(h)

        lay.addWidget(self._lbl('FIELD STABILITY', TEXT_LO, 6))
        self._gauge = QtWidgets.QProgressBar()
        self._gauge.setRange(0, 1000)
        self._gauge.setValue(int(n_s / N * 1000))
        self._gauge.setFormat(f'{n_s/N:.1%}')
        self._gauge.setStyleSheet(f"""
            QProgressBar{{background:{BORDER};border-radius:3px;
              color:{BG};font-size:7pt;font-weight:bold;height:14px;}}
            QProgressBar::chunk{{background:{GREEN};border-radius:3px;}}
        """)
        lay.addWidget(self._gauge)
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BORDER};'); lay.addWidget(sep)

        stat_row('STABLE',       f'{n_s:,}',                GREEN)
        stat_row('TURBULENT',    f'{n_t:,}',                RED)
        stat_row('AVG VERACITY', f'{veracity_v.mean():.8f}',CYAN)
        stat_row('MIN VERACITY', f'{veracity_v.min():.8f}', ORANGE)
        stat_row('MAX VERACITY', f'{veracity_v.max():.8f}', GREEN)
        stat_row('AVG WARP',     f'{warp_v.mean():.3f}e10c',PURPLE)
        stat_row('MAX WARP',     f'{warp_v.max():.3f}e10c', CYAN)
        lay.addStretch()

    # ── sensor bar ────────────────────────────────────────────────────────────
    def _build_sensor(self, root):
        f = QtWidgets.QFrame(); f.setFixedHeight(40)
        f.setStyleSheet(
            f'QFrame{{background:{PANEL};border:1px solid {BORDER};border-radius:4px;}}')
        h = QtWidgets.QHBoxLayout(f)
        h.setContentsMargins(10, 4, 10, 4)
        h.addWidget(self._lbl('FIELD FIDELITY SENSOR', TEXT_LO, 6))
        self._sensor = QtWidgets.QProgressBar()
        self._sensor.setRange(0, 10000); self._sensor.setTextVisible(True)
        self._sensor.setStyleSheet(f"""
            QProgressBar{{background:{BORDER};border-radius:3px;
              color:{BG};font-size:8pt;font-weight:bold;}}
            QProgressBar::chunk{{background:{GREEN};border-radius:3px;}}
        """)
        h.addWidget(self._sensor, stretch=1); root.addWidget(f)

    # ── controls ──────────────────────────────────────────────────────────────
    def _build_controls(self, root):
        row = QtWidgets.QHBoxLayout(); row.setSpacing(6)
        row.addWidget(self._btn('↺  RESET',  RED,    lambda: self._set_play(False) or self._go_to(0)))
        row.addWidget(self._btn('◀  PREV',   CYAN,   lambda: self._set_play(False) or self._go_to(self.frame_idx-1)))
        row.addWidget(self._btn('▶  PLAY',   GREEN,  lambda: self._set_play(True)))
        row.addWidget(self._btn('⏸  PAUSE',  ORANGE, lambda: self._set_play(False)))
        row.addWidget(self._btn('NEXT  ▶',   CYAN,   lambda: self._set_play(False) or self._go_to(self.frame_idx+1)))
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, N-1)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal{{background:{BORDER};height:4px;border-radius:2px;}}
            QSlider::handle:horizontal{{background:{CYAN};width:14px;height:14px;
              margin:-5px 0;border-radius:7px;}}
            QSlider::sub-page:horizontal{{background:{CYAN_D};border-radius:2px;}}
        """)
        self._slider.valueChanged.connect(lambda v: self._go_to(v))
        row.addWidget(self._slider, stretch=1)
        self._fps_lbl = self._lbl('— fps', ORANGE, 7)
        row.addWidget(self._fps_lbl)
        root.addLayout(row)

    # ══════════════════════════════════════════════════════════════════════════
    # LÓGICA
    # ══════════════════════════════════════════════════════════════════════════
    def _set_play(self, v): self.playing = v

    def _tick(self):
        if self.playing:
            self._go_to((self.frame_idx + 1) % N, from_timer=True)
        self._fps_acc += 1
        if self._fps_acc >= 30:
            dt = time.perf_counter() - self._fps_t
            self._fps_lbl.setText(f'{self._fps_acc/dt:.0f} fps')
            self._fps_acc = 0; self._fps_t = time.perf_counter()

    def _go_to(self, idx, from_timer=False):
        idx = int(np.clip(idx, 0, N-1))
        self.frame_idx = idx
        row = df.iloc[idx]
        v   = veracity_v[idx]; w = warp_v[idx]; stb = is_stable[idx]

        # info
        self.t_frame.setText(f'{idx+1:>6} / {N}')
        self.t_real.setText(f'{row["real_time_s"]:.4f} s')
        self.t_verac.setText(f'{v:.8f}')
        self.t_warp.setText(f'{w:.4f} ×10¹⁰c')
        self.t_stdj.setText(f'{row["stdj_time_u"]:.3e} u')
        self.t_status.setText(f'◉  {row["status"]}')
        self.t_status.setStyleSheet(
            f'color:{"#00ff88" if stb else "#ff2244"};'
            f'font-family:monospace;font-size:12pt;font-weight:bold;background:transparent;')

        # SHA async
        if not self._sha_busy:
            self._sha_busy = True
            row_copy = row.copy()
            sig_emit = self._sha_ready
            def _sha_work():
                d = row_copy.to_dict()
                sig = d.pop('signature', '')
                rc  = hashlib.sha256(str(d).encode()).hexdigest()
                sig_emit.emit(sig, rc, sig == rc)
            self._sha_pool.submit(_sha_work)

        # heatmap crosshair
        hr = idx // HMAP_W; hc = idx % HMAP_W
        self._hmap_marker.setData([hc+.5], [hr+.5])
        self._hmap_vline.setValue(hc+.5)
        self._hmap_hline.setValue(hr+.5)

        # charts
        self._warp_marker.setData([idx], [w])
        self._warp_vline.setValue(idx)
        self._ver_marker.setData([idx], [v])
        self._ver_vline.setValue(idx)

        # sensor
        norm = int(np.clip((v-0.97)/0.03, 0, 1)*10000)
        self._sensor.setValue(norm)
        self._sensor.setFormat(f'{v:.8f}')
        chunk = GREEN if stb else RED
        self._sensor.setStyleSheet(f"""
            QProgressBar{{background:{BORDER};border-radius:3px;
              color:{BG};font-size:8pt;font-weight:bold;}}
            QProgressBar::chunk{{background:{chunk};border-radius:3px;}}
        """)

        # slider sync
        if not from_timer:
            self._slider.blockSignals(True)
            self._slider.setValue(idx)
            self._slider.blockSignals(False)

    @QtCore.pyqtSlot(str, str, bool)
    def _apply_sha(self, sig, rc, valid):
        self._sha_busy = False
        self.t_sha.setText(sig[:32]+'\n'+sig[32:] if sig else rc[:32]+'\n'+rc[32:])
        self.t_valid.setText('✓ SIGNATURE VALID' if valid else '✗ CORRUPT')
        self.t_valid.setStyleSheet(
            f'color:{"#00ff88" if valid else "#ff2244"};'
            f'font-family:monospace;font-size:8pt;font-weight:bold;background:transparent;')

    def keyPressEvent(self, e):
        k = e.key()
        if   k == QtCore.Qt.Key.Key_Space: self.playing = not self.playing
        elif k == QtCore.Qt.Key.Key_Right: self._set_play(False); self._go_to(self.frame_idx+1)
        elif k == QtCore.Qt.Key.Key_Left:  self._set_play(False); self._go_to(self.frame_idx-1)
        elif k == QtCore.Qt.Key.Key_R:     self._set_play(False); self._go_to(0)
        elif k == QtCore.Qt.Key.Key_Escape: self.close()

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    win = STDJMonitor()
    win.show()
    sys.exit(app.exec())
