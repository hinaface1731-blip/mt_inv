"""Render the real Qt interface for visual review, without external data/services."""
import argparse
import base64
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6 import QtCore, QtWidgets
from mtinv.gui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ui-previews")
    parser.add_argument("--log-preview", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    old_hook = sys.excepthook
    sys.excepthook = lambda *exc: errors.append("".join(traceback.format_exception(*exc)))
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
        for filename in sorted(p.name for p in out.glob("*.png")):
            from PySide6 import QtGui
            image = QtGui.QImage(str(out / filename))
            if image.width() > 1200:
                image = image.scaledToWidth(1200, QtCore.Qt.SmoothTransformation)
            buf = QtCore.QBuffer()
            buf.open(QtCore.QIODevice.WriteOnly)
            image.save(buf, "JPG", 85)
            encoded = base64.b64encode(bytes(buf.data())).decode()
            for part, start in enumerate(range(0, len(encoded), 4000)):
                print(f"UI_CHUNK:{filename}:{part}:" + encoded[start:start+4000])
    w.close()
    app.processEvents()
    sys.excepthook = old_hook
    if errors:
        raise RuntimeError("Qt callback failed during visual review:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
