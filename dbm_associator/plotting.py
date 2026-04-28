import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QPushButton, QTextEdit, QVBoxLayout

from .utils import epoch_to_mpl_scalar


class MplPanel(FigureCanvas):
    def __init__(self, title, ylabel, logy=False):
        fig = Figure(figsize=(8, 2.5), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        if logy:
            self.ax.set_yscale('log')
        self.ax.grid(True, alpha=0.25)
        self.boundary_artists = []
        self.setMinimumHeight(220)

    def enterEvent(self, event):
        try:
            return super().enterEvent(event)
        except AttributeError:
            event.accept()

    def clear_plot(self, logy=False):
        self.ax.cla()
        self.ax.grid(True, alpha=0.25)
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        if logy:
            self.ax.set_yscale('log')
        self.boundary_artists = []

    def draw_verticals(self, times_and_colors):
        for art in self.boundary_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.boundary_artists = []
        for t, c in times_and_colors:
            if np.isfinite(t):
                v = self.ax.axvline(epoch_to_mpl_scalar(t), color=c, lw=1.5, alpha=0.9)
                self.boundary_artists.append(v)
        self.figure.canvas.draw_idle()


class ResultsDialog(QDialog):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        layout.addWidget(te)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class SimpleMplCanvas(FigureCanvas):
    def __init__(self, xlabel, ylabel):
        fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.25)
