"""Левая панель: папка с данными, параметры VO, кнопки управления."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from vo.api import VOParams, find_sequences


class ControlPanel(QtWidgets.QWidget):
    start_requested = QtCore.Signal(object, object)   # Path, VOParams
    pause_toggled = QtCore.Signal(bool)
    stop_requested = QtCore.Signal()
    export_requested = QtCore.Signal()
    data_root_changed = QtCore.Signal(object)         # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_root = Path.cwd()
        self._build_inputs()
        self._build_buttons()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._data_group())
        layout.addWidget(self._params_group())
        layout.addWidget(self._buttons_group())
        self.set_running(False)
        self.set_export_enabled(False)

    # ---------- сборка ----------
    def _build_inputs(self) -> None:
        self.root_label = QtWidgets.QLabel()
        self.root_label.setWordWrap(True)
        self.btn_browse = QtWidgets.QPushButton("Выбрать папку…")
        self.btn_browse.clicked.connect(self._browse)

        self.sequence = QtWidgets.QComboBox()
        self.gt_scale = QtWidgets.QCheckBox("Масштаб шага из GT")

        self.min_flow = QtWidgets.QDoubleSpinBox()
        self.min_flow.setRange(0.0, 20.0); self.min_flow.setSingleStep(0.5)
        self.max_corners = QtWidgets.QSpinBox()
        self.max_corners.setRange(50, 1000); self.max_corners.setSingleStep(50)
        self.px_tol = QtWidgets.QDoubleSpinBox()
        self.px_tol.setRange(0.1, 10.0); self.px_tol.setSingleStep(0.1)

    def _build_buttons(self) -> None:
        self.btn_start = QtWidgets.QPushButton("Запустить")
        self.btn_start.setObjectName("primary")
        self.btn_pause = QtWidgets.QPushButton("Пауза")
        self.btn_pause.setCheckable(True)
        self.btn_stop = QtWidgets.QPushButton("Остановить")
        self.btn_export = QtWidgets.QPushButton("Сохранить результаты…")

        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        self.btn_export.clicked.connect(self.export_requested.emit)

    def _data_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Данные")
        col = QtWidgets.QVBoxLayout(box)
        col.addWidget(self.root_label)
        col.addWidget(self.btn_browse)
        col.addWidget(self.sequence)
        col.addWidget(self.gt_scale)
        return box

    def _params_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Параметры")
        form = QtWidgets.QFormLayout(box)
        form.addRow("Мин. поток, px", self.min_flow)
        form.addRow("Макс. углов", self.max_corners)
        form.addRow("Допуск RANSAC, px", self.px_tol)
        return box

    def _buttons_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Управление")
        col = QtWidgets.QVBoxLayout(box)
        for b in (self.btn_start, self.btn_pause, self.btn_stop, self.btn_export):
            col.addWidget(b)
        return box

    # ---------- публичный интерфейс ----------
    def set_data_root(self, path: Path) -> None:
        """Запомнить папку и обновить список последовательностей."""
        self._data_root = Path(path)
        self.root_label.setText(str(self._data_root))
        self.sequence.clear()
        for seq in find_sequences(self._data_root):
            self.sequence.addItem(seq.name, seq)
        self.btn_start.setEnabled(self.sequence.count() > 0)

    def params(self) -> VOParams:
        base = VOParams()
        return VOParams(
            max_corners=self.max_corners.value(),
            px_tol=self.px_tol.value(),
            min_flow_px=self.min_flow.value(),
            min_tracked=base.min_tracked,
            min_inliers_pose=base.min_inliers_pose,
            max_depth=base.max_depth,
            min_scale_pts=base.min_scale_pts,
            min_fwd=base.min_fwd,
            gt_scale=self.gt_scale.isChecked(),
            window_m=base.window_m,
        )

    def set_params(self, p: VOParams) -> None:
        self.max_corners.setValue(p.max_corners)
        self.px_tol.setValue(p.px_tol)
        self.min_flow.setValue(p.min_flow_px)
        self.gt_scale.setChecked(p.gt_scale)

    def set_running(self, running: bool) -> None:
        for w in (self.btn_browse, self.sequence, self.gt_scale, self.min_flow,
                  self.max_corners, self.px_tol):
            w.setEnabled(not running)
        self.btn_start.setEnabled(not running and self.sequence.count() > 0)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
        if not running:
            self.btn_pause.blockSignals(True)
            self.btn_pause.setChecked(False)
            self.btn_pause.setText("Пауза")
            self.btn_pause.blockSignals(False)

    def set_export_enabled(self, enabled: bool) -> None:
        self.btn_export.setEnabled(enabled)

    # ---------- внутреннее ----------
    def _browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Папка с последовательностями", str(self._data_root))
        if folder:
            self.set_data_root(Path(folder))
            self.data_root_changed.emit(Path(folder))

    def _on_start_clicked(self) -> None:
        seq_dir = self.sequence.currentData()
        if seq_dir is not None:
            self.start_requested.emit(seq_dir, self.params())

    def _on_pause_toggled(self, paused: bool) -> None:
        self.btn_pause.setText("Продолжить" if paused else "Пауза")
        self.pause_toggled.emit(paused)