from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QLinearGradient, QPainterPath
from PySide6.QtWidgets import QPushButton, QStyle, QStyleOptionButton


class WaveformButton(QPushButton):

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"

    def __init__(self, parent=None):
        super().__init__(parent)

        self._state = self.IDLE
        self._phase = 0.0
        self._idle_phase = 0.0

        self._waveform = np.zeros(512)
        self._peak_level = 0.0

        self._num_particles = 60
        self._particles: list[dict] = []

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state

        if state in (self.IDLE, self.RECORDING):
            self._waveform = np.zeros(512)
            self._peak_level = 0.0
        elif state == self.TRANSCRIBING:
            self._init_particles()

        self.update()

    def get_state(self) -> str:
        return self._state

    def update_waveform(self, samples: np.ndarray) -> None:
        if self._state != self.RECORDING or len(samples) == 0:
            return

        mono = samples.astype(np.float32) / 32768.0

        # Auto-gain: boosts quiet audio while keeping loud/quiet visually distinct
        # Increase the 0.02 noise floor to suppress more background; decrease to show more subtle sound
        # Increase the 3.0 max gain to boost quiet sounds more; decrease to keep them smaller
        peak = np.max(np.abs(mono))
        if peak > 0.015:
            gain = min(3.0, 0.5 / peak)
            mono = mono * gain
        else:
            mono *= 0.1

        if len(mono) >= len(self._waveform):
            indices = np.linspace(0, len(mono) - 1, len(self._waveform), dtype=int)
            resampled = mono[indices]
            # Smoothing kernel: increase the 5 for a smoother waveform, decrease for more jagged detail
            kernel = np.ones(5) / 5
            smoothed = np.convolve(resampled, kernel, mode="same")
            # Blend with previous frame: closer to 0 = snappier response, closer to 1 = slower/smoother transitions
            blend = 0.45
            self._waveform = self._waveform * blend + smoothed * (1 - blend)
        else:
            # Smoothing kernel: increase the 5 for a smoother waveform, decrease for more jagged detail
            kernel = np.ones(7) / 7
            smoothed = np.convolve(mono, kernel, mode="same")
            # Blend with previous frame: closer to 0 = snappier response, closer to 1 = slower/smoother transitions
            blend = 0.25
            length = len(smoothed)
            self._waveform[:length] = (
                self._waveform[:length] * blend + smoothed * (1 - blend)
            )
            self._waveform[length:] = 0.0

        # Level bar sensitivity: increase the 3.5 multiplier to make the bar fill up faster at lower volumes
        self._peak_level = min(1.0, peak * 5.5)
        self.update()

    def _tick(self) -> None:
        self._phase += 0.04
        self._idle_phase += 0.03

        if self._state == self.TRANSCRIBING:
            self._advance_particles()

        self.update()

    def _init_particles(self) -> None:
        self._particles = []
        for _ in range(self._num_particles):
            self._particles.append(self._make_particle(randomize_progress=True))

    def _make_particle(self, randomize_progress: bool = False) -> dict:
        return {
            "progress": np.random.uniform(0.0, 1.0) if randomize_progress else 0.0,
            "speed": np.random.uniform(0.004, 0.012),
            "line": np.random.randint(0, 5),
            "wave_offset": np.random.uniform(0, 2 * np.pi),
            "drift_y": np.random.uniform(-3, 3),
            "size": np.random.uniform(1.5, 3.0),
        }

    def _advance_particles(self) -> None:
        for p in self._particles:
            p["progress"] += p["speed"]
            if p["progress"] > 1.0:
                p.update(self._make_particle(randomize_progress=False))

    def paintEvent(self, event) -> None:
        opt = QStyleOptionButton()
        self.initStyleOption(opt)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.style().drawControl(QStyle.ControlElement.CE_PushButtonBevel, opt, painter, self)

        painter.save()
        painter.setClipRect(self.rect())

        w = self.width()
        h = self.height()

        if self._state == self.RECORDING:
            self._draw_recording(painter, w, h)
        elif self._state == self.TRANSCRIBING:
            self._draw_transcribing(painter, w, h)
        else:
            self._draw_idle(painter, w, h)

        painter.restore()
        painter.end()

    def _draw_idle(self, painter: QPainter, w: int, h: int) -> None:
        margin_x = 10
        draw_w = w - margin_x * 2
        mid_y = h / 2.0

        num_points = 256
        path = QPainterPath()
        for i in range(num_points):
            t = i / (num_points - 1)
            x = margin_x + t * draw_w
            y_val = (
                np.sin(t * 8.0 + self._idle_phase) * 0.06
                + np.sin(t * 12.0 - self._idle_phase * 0.7) * 0.03
                + np.sin(t * 20.0 + self._idle_phase * 1.3) * 0.015
            )
            y = mid_y - y_val * h * 4.0
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # Slowly cycle hue over time; adjust the 0.15 divisor to change cycle speed
        hue = (self._idle_phase * 0.15) % 1.0
        # Offset hues for a gradient that shifts across the wave
        hue_left = hue % 1.0
        hue_mid = (hue + 0.25) % 1.0
        hue_right = (hue + 0.5) % 1.0

        glow_grad = QLinearGradient(margin_x, 0, w - margin_x, 0)
        glow_grad.setColorAt(0.0, QColor.fromHsvF(hue_left, 0.5, 0.55, 0.10))
        glow_grad.setColorAt(0.5, QColor.fromHsvF(hue_mid, 0.5, 0.55, 0.10))
        glow_grad.setColorAt(1.0, QColor.fromHsvF(hue_right, 0.5, 0.55, 0.10))
        painter.setPen(QPen(glow_grad, 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)

        line_grad = QLinearGradient(margin_x, 0, w - margin_x, 0)
        line_grad.setColorAt(0.0, QColor.fromHsvF(hue_left, 0.6, 0.7, 0.22))
        line_grad.setColorAt(0.5, QColor.fromHsvF(hue_mid, 0.6, 0.7, 0.22))
        line_grad.setColorAt(1.0, QColor.fromHsvF(hue_right, 0.6, 0.7, 0.22))
        painter.setPen(QPen(line_grad, 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)

        self._draw_label(painter, w)

    def _draw_recording(self, painter: QPainter, w: int, h: int) -> None:
        margin_x = 10
        draw_w = w - margin_x * 2
        mid_y = h / 2.0
        half_h = h * 0.4

        num_points = len(self._waveform)

        wave_path = QPainterPath()
        for i in range(num_points):
            x = margin_x + (i / (num_points - 1)) * draw_w
            y = mid_y - self._waveform[i] * half_h * 7.55
            if i == 0:
                wave_path.moveTo(x, y)
            else:
                prev = wave_path.currentPosition()
                cx = (prev.x() + x) / 2
                cy = (prev.y() + y) / 2
                wave_path.quadTo(prev.x(), prev.y(), cx, cy)
                wave_path.quadTo(cx, cy, x, y)

        painter.setPen(QPen(QColor(0, 160, 255, 18), 6.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(wave_path)

        # change this up or down to adjust the minimum brightness or volume-driven range
        intensity = 0.5 + self._peak_level * 0.2
        gradient = QLinearGradient(margin_x, 0, w - margin_x, 0)
        gradient.setColorAt(0.0, QColor(0, int(180 * intensity), int(255 * intensity)))
        gradient.setColorAt(0.5, QColor(int(120 * intensity), int(230 * intensity), int(255 * intensity)))
        gradient.setColorAt(1.0, QColor(0, int(160 * intensity), int(255 * intensity)))

        thickness = 1.5 + self._peak_level * 1.0
        painter.setPen(QPen(gradient, thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(wave_path)

        self._draw_label(painter, w)

    def _draw_transcribing(self, painter: QPainter, w: int, h: int) -> None:
        margin_x = 12
        draw_w = w - margin_x * 2
        draw_cy = h / 2.0

        wave_x_start = margin_x
        wave_x_end = margin_x + draw_w * 0.30
        wave_w = wave_x_end - wave_x_start

        text_x_start = margin_x + draw_w * 0.55
        text_x_end = w - margin_x
        text_w = text_x_end - text_x_start

        num_wave_pts = 80
        wave_path = QPainterPath()
        dissolve_pulse = 0.5 + 0.5 * np.sin(self._phase * 1.2)

        for i in range(num_wave_pts):
            t = i / (num_wave_pts - 1)
            x = wave_x_start + t * wave_w
            amp = (
                np.sin(t * 10.0 + self._phase * 3.0) * 0.4
                + np.sin(t * 16.0 - self._phase * 2.0) * 0.2
                + np.sin(t * 6.0 + self._phase * 1.5) * 0.3
            )
            edge_fade = 1.0 - t * 0.6
            y = draw_cy - amp * (h * 0.25) * edge_fade
            if i == 0:
                wave_path.moveTo(x, y)
            else:
                prev = wave_path.currentPosition()
                mx = (prev.x() + x) / 2
                my = (prev.y() + y) / 2
                wave_path.quadTo(prev.x(), prev.y(), mx, my)

        wave_hue = (self._phase * 0.05) % 1.0

        painter.setPen(QPen(QColor.fromHsvF(wave_hue, 0.5, 0.6, 0.06), 5.0,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(wave_path)

        wave_grad = QLinearGradient(wave_x_start, 0, wave_x_end, 0)
        wave_alpha_left = 0.4 + 0.15 * dissolve_pulse
        wave_alpha_right = 0.06 + 0.06 * dissolve_pulse
        wave_grad.setColorAt(0.0, QColor.fromHsvF(wave_hue, 0.6, 0.85, wave_alpha_left))
        wave_grad.setColorAt(0.7, QColor.fromHsvF(wave_hue, 0.5, 0.7, wave_alpha_left * 0.5))
        wave_grad.setColorAt(1.0, QColor.fromHsvF(wave_hue, 0.4, 0.6, wave_alpha_right))
        painter.setPen(QPen(wave_grad, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(wave_path)

        num_lines = 5
        available_h = h * 0.6
        line_spacing = min(available_h / (num_lines + 1), 12)
        lines_top = draw_cy - (num_lines - 1) * line_spacing / 2.0
        width_factors = [0.95, 0.70, 0.85, 0.60, 0.40]

        text_hue = (self._phase * 0.05 + 0.35) % 1.0

        for i in range(num_lines):
            ly = lines_top + i * line_spacing

            fill = (np.sin(self._phase * 1.2 - i * 0.6) + 1) / 2.0
            line_len = text_w * width_factors[i] * (0.3 + 0.7 * fill)
            line_alpha = 0.10 + 0.18 * fill

            glow_grad = QLinearGradient(text_x_start, ly, text_x_start + line_len, ly)
            glow_grad.setColorAt(0.0, QColor.fromHsvF(text_hue, 0.3, 0.7, line_alpha * 0.3))
            glow_grad.setColorAt(1.0, QColor.fromHsvF(text_hue, 0.3, 0.5, 0.0))
            painter.setPen(QPen(glow_grad, 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(text_x_start, ly), QPointF(text_x_start + line_len, ly))

            line_grad = QLinearGradient(text_x_start, ly, text_x_start + line_len, ly)
            line_grad.setColorAt(0.0, QColor.fromHsvF(text_hue, 0.4, 0.8, line_alpha))
            line_grad.setColorAt(0.8, QColor.fromHsvF(text_hue, 0.4, 0.8, line_alpha * 0.7))
            line_grad.setColorAt(1.0, QColor.fromHsvF(text_hue, 0.3, 0.6, 0.0))
            painter.setPen(QPen(line_grad, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(text_x_start, ly), QPointF(text_x_start + line_len, ly))

            if fill > 0.6:
                cursor_blink = (np.sin(self._phase * 5.0 + i * 2.0) + 1) / 2.0
                cursor_x = text_x_start + line_len
                cursor_alpha = line_alpha * cursor_blink
                painter.setPen(QPen(QColor.fromHsvF(text_hue, 0.3, 0.95, cursor_alpha), 1.2))
                painter.drawLine(QPointF(cursor_x, ly - 4), QPointF(cursor_x, ly + 4))

        for p in self._particles:
            prog = p["progress"]

            src_x = wave_x_end - 5
            wave_t = p["wave_offset"]
            src_y = draw_cy + np.sin(wave_t + self._phase * 3.0) * h * 0.18

            line_idx = p["line"] % num_lines
            dst_y = lines_top + line_idx * line_spacing
            dst_fill = (np.sin(self._phase * 1.2 - line_idx * 0.6) + 1) / 2.0
            dst_x = text_x_start + text_w * width_factors[line_idx] * (0.3 + 0.7 * dst_fill) * 0.5

            t = prog
            ease = t * t * (3 - 2 * t)

            px = src_x + (dst_x - src_x) * ease
            arc_height = (h * 0.1) * np.sin(np.pi * t) * np.sign(p["drift_y"])
            py = src_y + (dst_y - src_y) * ease + arc_height + p["drift_y"] * (1 - ease)

            if prog < 0.15:
                alpha = prog / 0.15
            elif prog > 0.85:
                alpha = (1.0 - prog) / 0.15
            else:
                alpha = 1.0
            alpha *= 0.4

            p_hue = wave_hue + (text_hue - wave_hue) * ease
            p_hue = p_hue % 1.0

            p_color = QColor.fromHsvF(p_hue, 0.5, 0.85, alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(p_color)
            painter.drawEllipse(QPointF(px, py), p["size"], p["size"])

        painter.setBrush(Qt.BrushStyle.NoBrush)

        self._draw_label(painter, w)

    def _draw_label(self, painter: QPainter, w: int) -> None:
        label = self.text()
        if label:
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor(80, 170, 220, 200))
            painter.drawText(QRectF(0, 4, w, 28), Qt.AlignmentFlag.AlignCenter, label)

    def stop(self) -> None:
            self._timer.stop()