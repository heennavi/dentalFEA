"""
fatigue_diagrams.py
Matplotlib-based visualization widget for S-N (Wöhler) and Haigh (Mean Stress) diagrams.
Embedded seamlessly inside PySide6 with dark theme styling.

Uses offscreen FigureCanvasAgg and QPixmap rendering to completely prevent
OpenGL context collisions with VTK (QVTKRenderWindowInteractor) and eliminate
resizeEvent re-layout recursion loops.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QSizePolicy, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from fatigue.models import FatigueMaterial, FatigueLoadCase, MeanStressCorrectionMethod


class FatigueDiagramWidget(QWidget):
    """
    Embedded 2D diagram widget displaying S-N curve and Haigh diagram.
    Uses pure CPU Agg rasterization converted to QPixmap for rock-solid stability.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c4048;
                background-color: #21252b;
            }
            QTabBar::tab {
                background: #1e2227;
                color: #abb2bf;
                padding: 4px 8px;
                font-size: 11px;
                border: 1px solid #3c4048;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #282c34;
                color: #61afef;
                font-weight: bold;
            }
        """)

        # 1. S-N 곡선 이미지 라벨
        self.lbl_sn = QLabel(self)
        self.lbl_sn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sn.setStyleSheet("background-color: #21252b;")
        self.lbl_sn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tab_widget.addTab(self.lbl_sn, "S-N 선도 (Wöhler)")

        # 2. Haigh 선도 이미지 라벨
        self.lbl_haigh = QLabel(self)
        self.lbl_haigh.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_haigh.setStyleSheet("background-color: #21252b;")
        self.lbl_haigh.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tab_widget.addTab(self.lbl_haigh, "Haigh 선도 (평균응력)")

        self.fig_sn: Optional[Figure] = None
        self.ax_sn: Optional[Any] = None
        self.fig_haigh: Optional[Figure] = None
        self.ax_haigh: Optional[Any] = None

        layout.addWidget(self.tab_widget)

        # 초기 안내 렌더링
        self.render_empty()

    def _fig_to_pixmap(self, fig: Figure) -> QPixmap:
        """Convert a Matplotlib Figure to QPixmap via memory buffer."""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = canvas.buffer_rgba()
        w, h = fig.get_size_inches() * fig.dpi
        w, h = int(w), int(h)
        qimg = QImage(rgba, w, h, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def _style_axes(self, ax) -> None:
        """Apply consistent dark theme styling to a matplotlib axis."""
        ax.set_facecolor("#21252b")
        ax.tick_params(colors="#abb2bf", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#4b5263")
        ax.grid(True, linestyle="--", alpha=0.3, color="#5c6370")

    def render_empty(self) -> None:
        """Render initial placeholder text before analysis is run."""
        # S-N empty
        fig_sn = Figure(figsize=(3.8, 2.6), facecolor="#21252b", dpi=100)
        ax_sn = fig_sn.add_subplot(111)
        self._style_axes(ax_sn)
        ax_sn.text(
            0.5, 0.5, "Run fatigue analysis to display S-N curve",
            color="#5c6370", fontsize=9.5, ha="center", va="center", transform=ax_sn.transAxes
        )
        self.fig_sn = fig_sn
        self.ax_sn = ax_sn
        self.lbl_sn.setPixmap(self._fig_to_pixmap(fig_sn))

        # Haigh empty
        fig_h = Figure(figsize=(3.8, 2.6), facecolor="#21252b", dpi=100)
        ax_h = fig_h.add_subplot(111)
        self._style_axes(ax_h)
        ax_h.text(
            0.5, 0.5, "Run fatigue analysis to display Haigh diagram",
            color="#5c6370", fontsize=9.5, ha="center", va="center", transform=ax_h.transAxes
        )
        self.fig_haigh = fig_h
        self.ax_haigh = ax_h
        self.lbl_haigh.setPixmap(self._fig_to_pixmap(fig_h))

    def update_diagrams(
        self,
        material: FatigueMaterial,
        load_case: FatigueLoadCase,
        sigma_a_hotspot: float = 0.0,
        sigma_m_hotspot: float = 0.0,
        life_hotspot: float = float("inf"),
    ) -> None:
        """
        Plot both S-N curve and Haigh diagram with current operating point.
        """
        self._update_sn_curve(material, load_case, sigma_a_hotspot, sigma_m_hotspot, life_hotspot)
        self._update_haigh_diagram(material, load_case, sigma_a_hotspot, sigma_m_hotspot)

    def _update_sn_curve(
        self,
        material: FatigueMaterial,
        load_case: FatigueLoadCase,
        sigma_a: float,
        sigma_m: float,
        life: float,
    ) -> None:
        """Render S-N curve with Basquin model and operating point."""
        fig = Figure(figsize=(3.8, 2.6), facecolor="#21252b", dpi=100)
        ax = fig.add_subplot(111)
        self._style_axes(ax)

        se = float(material.fatigue_limit_Se)
        ne = float(material.endurance_cycles_Ne)
        sut = float(material.tensile_strength_Sut)

        # Cycles 범위: 10^3 ~ 10^8
        n_arr = np.logspace(3.0, 8.0, 200)

        # Basquin 파라미터 확인
        if material.basquin_sigma_f_prime and material.basquin_b:
            s_f_p = material.basquin_sigma_f_prime
            b_exp = material.basquin_b
            s_curve = s_f_p * ((2.0 * n_arr) ** b_exp)
            s_curve = np.where(n_arr >= ne, se, np.maximum(s_curve, se))
        else:
            s1 = 0.9 * sut
            from fatigue.sn_curve import fit_basquin_from_two_points
            s_f_p, b_exp = fit_basquin_from_two_points(1e3, s1, ne, se)
            s_curve = s_f_p * ((2.0 * n_arr) ** b_exp)
            s_curve = np.where(n_arr >= ne, se, np.maximum(s_curve, se))

        ax.semilogx(n_arr, s_curve, color="#61afef", linewidth=1.8, label="Basquin S-N")
        ax.axhline(se, color="#98c379", linestyle="--", linewidth=1.1, label=f"Se ({se:.0f} MPa)")

        # 등가 응력 진폭 sigma_ar 계산
        from fatigue.mean_stress import correct_mean_stress_goodman
        sigma_ar = correct_mean_stress_goodman(sigma_a, sigma_m, Sut=sut) if sut > 0 else sigma_a

        # 동작점 플롯
        disp_life = min(life, 5e7) if not np.isinf(life) else ne * 2
        ax.scatter(
            [disp_life], [sigma_ar],
            color="#e06c75", s=80, marker="*", zorder=5,
            label=f"Hotspot ({sigma_ar:.0f} MPa)"
        )

        ax.set_xlabel("Cycles (N)", color="#abb2bf", fontsize=8)
        ax.set_ylabel("Amplitude (MPa)", color="#abb2bf", fontsize=8)
        ax.set_title(f"S-N Curve: {material.name}", color="#dcdfe4", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", facecolor="#282c34", edgecolor="#3c4048", fontsize=7, labelcolor="#abb2bf")
        fig.tight_layout(pad=1.0)

        self.fig_sn = fig
        self.ax_sn = ax
        self.lbl_sn.setPixmap(self._fig_to_pixmap(fig))

    def _update_haigh_diagram(
        self,
        material: FatigueMaterial,
        load_case: FatigueLoadCase,
        sigma_a: float,
        sigma_m: float,
    ) -> None:
        """Render Haigh diagram with Goodman, Gerber, Soderberg curves and operating point."""
        fig = Figure(figsize=(3.8, 2.6), facecolor="#21252b", dpi=100)
        ax = fig.add_subplot(111)
        self._style_axes(ax)

        se = float(material.fatigue_limit_Se)
        sut = float(material.tensile_strength_Sut)
        sy = float(material.yield_strength_Sy)

        sm_max = max(sut * 1.05, 100.0)
        sm_arr = np.linspace(0.0, sut, 200)

        # 1. Goodman line: sa = Se * (1 - sm / Sut)
        sa_goodman = np.maximum(se * (1.0 - sm_arr / sut), 0.0)
        ax.plot(sm_arr, sa_goodman, color="#e5c07b", linewidth=1.6, label="Goodman")

        # 2. Gerber parabola: sa = Se * (1 - (sm / Sut)^2)
        sa_gerber = np.maximum(se * (1.0 - (sm_arr / sut) ** 2), 0.0)
        ax.plot(sm_arr, sa_gerber, color="#61afef", linewidth=1.3, linestyle=":", label="Gerber")

        # 3. Soderberg line: sa = Se * (1 - sm / Sy)
        sm_sod = np.linspace(0.0, sy, 200)
        sa_soderberg = np.maximum(se * (1.0 - sm_sod / sy), 0.0)
        ax.plot(sm_sod, sa_soderberg, color="#c678dd", linewidth=1.1, linestyle="--", label="Soderberg")

        # 4. Yield line (Langer criterion): sa + sm = Sy
        sm_yield = np.linspace(0.0, sy, 200)
        sa_yield = np.maximum(sy - sm_yield, 0.0)
        ax.plot(sm_yield, sa_yield, color="#abb2bf", linewidth=0.9, linestyle="-.", label="Yield")

        # 안전 영역 음영 채우기 (Goodman 기준)
        ax.fill_between(sm_arr, 0.0, sa_goodman, color="#98c379", alpha=0.12, label="Safe")

        # 핫스팟 동작점
        ax.scatter(
            [sigma_m], [sigma_a],
            color="#e06c75", s=70, marker="D", zorder=5,
            label=f"Hotspot (σm={sigma_m:.0f}, σa={sigma_a:.0f})"
        )

        ax.set_xlim(-20.0, sm_max)
        ax.set_ylim(0.0, max(se * 1.3, sy * 1.1))
        ax.set_xlabel("Mean Stress σ_m (MPa)", color="#abb2bf", fontsize=8)
        ax.set_ylabel("Amplitude σ_a (MPa)", color="#abb2bf", fontsize=8)
        ax.set_title(f"Haigh: {material.name}", color="#dcdfe4", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", facecolor="#282c34", edgecolor="#3c4048", fontsize=7, labelcolor="#abb2bf")
        fig.tight_layout(pad=1.0)

        self.fig_haigh = fig
        self.ax_haigh = ax
        self.lbl_haigh.setPixmap(self._fig_to_pixmap(fig))
