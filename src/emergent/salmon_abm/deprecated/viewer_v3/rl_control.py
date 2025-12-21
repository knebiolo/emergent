"""Reinforcement-learning control form widget.

Exposes simple controls (Start, Stop, Reset) and a few hyperparameters.
Signals can be connected by the viewer shim to start/stop training loops.
"""
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal


class RLControlWidget(QtWidgets.QGroupBox):
    start_training = pyqtSignal()
    stop_training = pyqtSignal()
    reset_training = pyqtSignal()
    params_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__('RL Controls', parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Start')
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.reset_btn = QtWidgets.QPushButton('Reset')
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        # Parameters: episode length, lr, epsilon
        self.ep_len_spin = QtWidgets.QSpinBox()
        self.ep_len_spin.setRange(1, 100000)
        self.ep_len_spin.setValue(600)
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 10.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-3)
        self.lr_spin.setValue(0.001)
        self.eps_spin = QtWidgets.QDoubleSpinBox()
        self.eps_spin.setRange(0.0, 1.0)
        self.eps_spin.setDecimals(3)
        self.eps_spin.setSingleStep(0.01)
        self.eps_spin.setValue(0.1)

        form = QtWidgets.QFormLayout()
        form.addRow('Episode Len', self.ep_len_spin)
        form.addRow('LR', self.lr_spin)
        form.addRow('Epsilon', self.eps_spin)
        layout.addLayout(form)

        self.setLayout(layout)

        # Connect signals
        self.start_btn.clicked.connect(self.start_training.emit)
        self.stop_btn.clicked.connect(self.stop_training.emit)
        self.reset_btn.clicked.connect(self.reset_training.emit)
        self.ep_len_spin.valueChanged.connect(self._on_params_changed)
        self.lr_spin.valueChanged.connect(self._on_params_changed)
        self.eps_spin.valueChanged.connect(self._on_params_changed)

    def _on_params_changed(self, *_):
        self.params_changed.emit(self.get_params())

    def get_params(self):
        return {
            'episode_length': int(self.ep_len_spin.value()),
            'learning_rate': float(self.lr_spin.value()),
            'epsilon': float(self.eps_spin.value())
        }
