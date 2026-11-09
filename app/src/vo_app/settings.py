"""Сохранение настроек между запусками (QSettings: реестр Windows / ~/.config)."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from PySide6 import QtCore

from vo.api import VOParams


class AppSettings:
    def __init__(self):
        self._s = QtCore.QSettings("visualOdometry", "vo_app")

    # --- папка с данными ---
    def data_root(self) -> Path:
        default = Path.cwd() / "data"
        return Path(self._s.value("data_root", str(default), type=str))

    def set_data_root(self, path: Path) -> None:
        self._s.setValue("data_root", str(path))

    # --- параметры VO ---
    def params(self) -> VOParams:
        defaults = VOParams()
        values = {}
        for f in fields(VOParams):
            default = getattr(defaults, f.name)
            values[f.name] = self._s.value(f"params/{f.name}", default, type=type(default))
        return VOParams(**values)

    def set_params(self, params: VOParams) -> None:
        for f in fields(VOParams):
            self._s.setValue(f"params/{f.name}", getattr(params, f.name))

    # --- геометрия окна ---
    def geometry(self):
        return self._s.value("window/geometry")

    def set_geometry(self, data) -> None:
        self._s.setValue("window/geometry", data)