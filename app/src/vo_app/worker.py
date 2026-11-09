"""Фоновый поток: гоняет сессию кадр за кадром и шлёт результаты в окно."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from PySide6 import QtCore

from vo.api import VOParams


class VOWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(object)      # FrameResult
    report_ready = QtCore.Signal(object)     # EvaluationReport
    failed = QtCore.Signal(str)

    def __init__(self, session_factory: Callable, sequence_dir: Path, params: VOParams):
        super().__init__()
        self._factory = session_factory
        self._sequence_dir = sequence_dir
        self._params = params
        self._stop_flag = False
        self._resume = threading.Event()
        self._resume.set()

    # --- управление из GUI-потока ---
    def pause(self, paused: bool) -> None:
        self._resume.clear() if paused else self._resume.set()

    def stop(self) -> None:
        self._stop_flag = True
        self._resume.set()                   # разбудить, если стоим на паузе

    # --- тело потока ---
    def run(self) -> None:
        try:
            session = self._factory(self._sequence_dir, self._params)
            while not session.finished:
                self._resume.wait()
                if self._stop_flag:
                    return
                self.frame_ready.emit(session.step())
            self.report_ready.emit(session.report())
        except Exception as exc:             # показываем ошибку в окне, а не роняем приложение
            self.failed.emit(f"{type(exc).__name__}: {exc}")