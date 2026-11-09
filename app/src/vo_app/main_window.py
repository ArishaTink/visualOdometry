"""Главное окно: собирает виджеты и связывает их с потоком вычислений."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from vo.api import EvaluationReport, FrameResult, VOParams, VOSession, save_report
from vo_app.settings import AppSettings
from vo_app.widgets import (ControlPanel, MetricsPanel, StatsPlots,
                            TrajectoryPlot, VideoView)
from vo_app.worker import VOWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings=None, session_factory=VOSession):
        super().__init__()
        self.setWindowTitle("Monocular Visual Odometry")
        self.resize(1450, 880)

        self._settings = settings or AppSettings()
        self._session_factory = session_factory
        self._worker: VOWorker | None = None
        self._report: EvaluationReport | None = None
        self._sequence_name = ""

        self.controls = ControlPanel()
        self.metrics = MetricsPanel()
        self.video = VideoView()
        self.trajectory = TrajectoryPlot()
        self.stats = StatsPlots()

        self.setCentralWidget(self._build_layout())
        self._connect_signals()
        self._restore_settings()
        self.statusBar().showMessage("Выберите папку с данными и нажмите «Запустить»")

    # ---------- компоновка ----------
    def _build_layout(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(300)
        col = QtWidgets.QVBoxLayout(sidebar)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.controls)
        col.addWidget(self.metrics)
        col.addStretch()

        root = QtWidgets.QWidget()
        root.setObjectName("root")
        row = QtWidgets.QHBoxLayout(root)
        row.addWidget(sidebar)
        row.addWidget(self._build_center(), 1)
        return root

    def _build_center(self) -> QtWidgets.QSplitter:
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(self.video)
        top.addWidget(self.trajectory)

        center = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        center.addWidget(top)
        center.addWidget(self.stats)
        center.setSizes([560, 240])
        return center

    def _connect_signals(self) -> None:
        self.controls.start_requested.connect(self._start)
        self.controls.pause_toggled.connect(self._pause)
        self.controls.stop_requested.connect(self._stop)
        self.controls.export_requested.connect(self._export)
        self.controls.data_root_changed.connect(self._settings.set_data_root)

    def _restore_settings(self) -> None:
        self.controls.set_params(self._settings.params())
        self.controls.set_data_root(self._settings.data_root())
        geometry = self._settings.geometry()
        if geometry:
            self.restoreGeometry(geometry)

    # ---------- действия пользователя ----------
    def _start(self, sequence_dir: Path, params: VOParams) -> None:
        self._stop_worker()
        self._reset_views()
        self._sequence_name = Path(sequence_dir).name
        self._settings.set_params(params)

        self._worker = VOWorker(self._session_factory, sequence_dir, params)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.report_ready.connect(self._on_report)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_thread_finished)

        self.controls.set_running(True)
        self.statusBar().showMessage(f"Обработка {self._sequence_name}…")
        self._worker.start()

    def _pause(self, paused: bool) -> None:
        if self._worker:
            self._worker.pause(paused)
        self.statusBar().showMessage("Пауза" if paused else "Обработка…")

    def _stop(self) -> None:
        self._stop_worker()
        self.statusBar().showMessage("Остановлено")

    def _export(self) -> None:
        if self._report is None:
            return
        start = str(Path.cwd() / "results" / self._sequence_name)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Куда сохранить результаты", start)
        if folder:
            paths = save_report(self._report, folder)
            self.statusBar().showMessage(f"Сохранено файлов: {len(paths)} → {folder}")

    # ---------- события потока ----------
    @QtCore.Slot(object)
    def _on_frame(self, r: FrameResult) -> None:
        self.video.show_frame(r.image, r.tracks)
        self.trajectory.append(r.position, r.gt_position)
        self.stats.append(r.n_inliers, r.step)
        self.metrics.update_frame(r)

    @QtCore.Slot(object)
    def _on_report(self, report: EvaluationReport) -> None:
        self._report = report
        self.trajectory.show_aligned(report.est_aligned)
        self.metrics.show_report(report)
        self.controls.set_export_enabled(True)
        self.statusBar().showMessage("Готово")

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self.statusBar().showMessage("Ошибка")
        QtWidgets.QMessageBox.critical(self, "Ошибка обработки", message)

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self.controls.set_running(False)

    # ---------- служебное ----------
    def _stop_worker(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()

    def _reset_views(self) -> None:
        self._report = None
        self.controls.set_export_enabled(False)
        self.video.clear_view()
        self.trajectory.reset()
        self.stats.reset()
        self.metrics.reset()

    def closeEvent(self, event) -> None:
        self._stop_worker()
        self._settings.set_params(self.controls.params())
        self._settings.set_geometry(self.saveGeometry())
        super().closeEvent(event)