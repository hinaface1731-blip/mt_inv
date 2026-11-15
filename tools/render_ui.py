"""Render the real Qt interface for visual review, without external data/services."""
import argparse
import base64
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6 import QtWidgets
from mtinv.gui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ui-previews")
    parser.add_argument("--log-preview", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.resize(1500, 980)
    w.show()
    app.processEvents()
    w.grab().save(str(out / "00-welcome.png"))
    w.shell.load_demo()
    for index in range(6):
        w.tabs.setCurrentIndex(index)
        app.processEvents()
        for widget in w.tabs.currentWidget().findChildren(QtWidgets.QWidget):
            if hasattr(widget, "canvas"):
                widget.canvas.draw()
        app.processEvents()
        w.grab().save(str(out / f"{index+1:02}-page.png"))
    w.tabs.setCurrentIndex(0)
    w.resize(1100, 760)
    app.processEvents()
    w.curves.plot.canvas.draw()
    w.grab().save(str(out / "07-compact.png"))
    if args.log_preview:
        for filename in ("00-welcome.png", "01-page.png", "05-page.png", "07-compact.png"):
            print(f"UI_IMAGE:{filename}:" + base64.b64encode((out / filename).read_bytes()).decode())
    w.close()
    app.processEvents()


if __name__ == "__main__":
    main()
