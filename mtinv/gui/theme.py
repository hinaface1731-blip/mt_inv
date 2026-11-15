"""Shared desktop design tokens and scalable line icons."""
from PySide6 import QtCore, QtGui, QtWidgets, QtSvg

ACCENT = "#087f8c"
INK = "#172b3a"
MUTED = "#596b7c"

_PATHS = {
    "wave": '<path d="M2 13h4l2-7 4 13 3-10 2 4h5"/>',
    "map": '<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3zM9 3v15M15 6v15"/>',
    "section": '<path d="M3 4v16h18M6 8h4l3 5 4-3 4 5M6 14h3l4 4 4-3 4 3"/>',
    "layers": '<path d="m3 7 9-4 9 4-9 4zM3 12l9 4 9-4M3 17l9 4 9-4"/>',
    "grid": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>',
    "link": '<path d="m10 8 3-3a4 4 0 0 1 6 6l-3 3M14 16l-3 3a4 4 0 0 1-6-6l3-3M8 16l8-8"/>',
    "folder": '<path d="M3 19V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11z"/>',
    "save": '<path d="M4 3h13l4 4v14H3V3zM7 3v6h10V3M7 21v-8h10v8"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    "sidebar": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 4 2c-1.5 1-1.5 1-1.5 3M12 17h.01"/>',
    "play": '<path d="m9 5 11 7-11 7z"/>',
    "chevron": '<path d="m9 5 7 7-7 7"/>',
}


def icon(name, color=MUTED, size=20):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
           f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.7" '
           f'stroke-linecap="round" stroke-linejoin="round">{_PATHS[name]}</svg>')
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    pix = QtGui.QPixmap(size * 2, size * 2)
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    renderer.render(painter)
    painter.end()
    pix.setDevicePixelRatio(2)
    return QtGui.QIcon(pix)


STYLE = """
QMainWindow, QDialog { background: #f3f6f9; color: #172b3a; }
QWidget { font-family: 'Segoe UI', 'DejaVu Sans', sans-serif; font-size: 13px; color: #172b3a; }
QLabel { background: transparent; }
QLabel[role='muted'] { color: #596b7c; }
QLabel[role='section'] { color: #172b3a; font-weight: 600; padding-top: 10px; padding-bottom: 4px; }
QLabel#pageTitle { font-size: 24px; font-weight: 600; }
QLabel#pageDescription { color: #596b7c; }
QLabel#eyebrow { color: #087f8c; font-size: 11px; font-weight: 600; }
QFrame#rail { background: #142838; border: 0; }
QFrame#rail QLabel { color: #c5d1db; }
QFrame#rail QLabel#brand { color: #f6fbff; font-size: 23px; font-weight: 600; }
QFrame#rail QLabel#brandSub { color: #91a7ba; font-size: 11px; }
QListWidget#navigation { background: transparent; border: 0; outline: 0; }
QListWidget#navigation::item { color: #c5d1db; border-radius: 7px; padding: 8px; margin: 2px 0; }
QListWidget#navigation::item:hover { background: #213d50; }
QListWidget#navigation::item:selected { background: #275368; color: white; }
QListWidget#navigation::item:focus { border: 1px solid #63d6d0; }
QFrame#rail QPushButton { color: #c5d1db; background: transparent; border: 1px solid #365063; }
QFrame#stationPanel { background: white; border: 0; border-right: 1px solid #dfe6ed; }
QLabel#stationHeading { font-size: 15px; font-weight: 600; }
QLabel#stationCount, QLabel#dataBadge { color: #42596b; background: #eaf0f5; border-radius: 5px; padding: 5px 8px; font-size: 11px; }
QLabel#stationInfo { background: #f5f8fa; border: 1px solid #e4eaf0; border-radius: 8px; padding: 12px; }
QFrame#workspaceHeader { background: #f3f6f9; }
QFrame#workspacePanel { background: white; border: 1px solid #dfe6ed; border-radius: 10px; }
QFrame#emptyState { background: white; border: 1px solid #dfe6ed; border-radius: 12px; }
QLabel#emptyTitle { font-size: 28px; font-weight: 600; }
QLabel#emptyDescription { color: #596b7c; font-size: 14px; }
QLabel#emptySymbol { color: #087f8c; background: #e1f3f2; border-radius: 14px; padding: 20px; font-size: 30px; font-weight: 600; }
QPushButton { background: white; border: 1px solid #cbd6df; border-radius: 6px; padding: 6px 10px; min-height: 19px; }
QPushButton:hover { background: #edf5f7; border-color: #91b7c0; }
QPushButton:pressed { background: #deedf0; }
QPushButton:focus, QToolButton:focus { border: 2px solid #087f8c; }
QPushButton[role='primary'] { color: white; background: #087f8c; border: 1px solid #087f8c; font-weight: 600; }
QPushButton[role='primary']:hover { background: #066773; }
QPushButton[role='danger'] { color: #ac3e3e; background: #fff5f5; border-color: #e9c5c5; }
QPushButton:disabled { color: #83929e; background: #f1f4f7; border-color: #dfe6ed; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 5px; }
QToolButton:hover, QToolButton:checked { background: #e5f1f3; }
QToolButton#disclosure { text-align: left; font-weight: 600; padding: 8px; border: 1px solid #dfe6ed; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white; border: 1px solid #cbd6df; border-radius: 5px; padding: 5px 8px; min-height: 20px; selection-background-color: #d5eeed; selection-color: #172b3a; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #087f8c; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background: #f1f4f7; color: #83929e; }
QComboBox { padding-right: 24px; }
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox QAbstractItemView { background: white; color: #172b3a; border: 1px solid #cbd6df; selection-background-color: #d5eeed; selection-color: #172b3a; }
QCheckBox, QRadioButton { spacing: 6px; padding: 3px 0; }
QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; }
QGroupBox { background: white; border: 1px solid #dfe6ed; border-radius: 8px; margin-top: 14px; padding: 14px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #42596b; }
QListWidget { background: white; border: 1px solid #dfe6ed; border-radius: 6px; outline: 0; padding: 3px; }
QListWidget::item { padding: 7px 8px; border-radius: 4px; }
QListWidget::item:selected { color: #075d68; background: #dcf0ef; }
QListWidget::item:hover { background: #edf4f7; }
QListWidget#stationList { border: 0; padding: 0; }
QTabWidget::pane { background: white; border: 1px solid #dfe6ed; border-radius: 7px; }
QTabWidget#workspaceTabs::pane { border: 0; }
QTabBar::tab { background: #f0f4f7; color: #596b7c; border: 1px solid #dfe6ed; padding: 8px 12px; margin-right: 3px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
QTabBar::tab:selected { background: white; color: #087f8c; border-bottom: 2px solid #087f8c; }
QScrollArea { border: 0; background: transparent; }
QScrollBar:vertical { background: #f1f4f7; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #bccbd5; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #f1f4f7; height: 9px; }
QScrollBar::handle:horizontal { background: #bccbd5; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QPlainTextEdit, QTextEdit { background: #f6f8fa; color: #42596b; border: 1px solid #dfe6ed; border-radius: 6px; padding: 8px; }
QPlainTextEdit { font-family: Consolas, monospace; font-size: 12px; }
QSplitter::handle { background: #e4eaf0; width: 3px; }
QStatusBar { background: white; color: #596b7c; border-top: 1px solid #dfe6ed; padding: 3px; }
QStatusBar::item { border: 0; }
QMenuBar { background: #f3f6f9; padding: 2px 8px; }
QMenuBar::item:selected, QMenu::item:selected { background: #dceef0; color: #172b3a; }
QMenu { background: white; border: 1px solid #cbd6df; padding: 5px; }
QMenu::item { padding: 7px 24px; }
QToolTip { background: #142838; color: white; border: 0; padding: 6px; }
QToolBar { background: white; border: 0; border-bottom: 1px solid #e4eaf0; spacing: 2px; }
QProgressBar { background: #dfedef; border: 0; border-radius: 2px; height: 3px; }
QProgressBar::chunk { background: #087f8c; }
QLabel#activity { background: #e1f3ed; color: #276048; border-radius: 5px; padding: 5px 8px; font-size: 11px; }
QLabel#activity[busy='true'] { background: #e0eff7; color: #21627d; }
"""


def apply_theme(window):
    app = QtWidgets.QApplication.instance()
    if not app.property("mtinv_style"):
        app.setStyle("Fusion")
        app.setProperty("mtinv_style", True)
    palette = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.Window, "#f3f6f9"), (QtGui.QPalette.Base, "#ffffff"),
                        (QtGui.QPalette.WindowText, INK), (QtGui.QPalette.Text, INK),
                        (QtGui.QPalette.Button, "#ffffff"), (QtGui.QPalette.ButtonText, INK),
                        (QtGui.QPalette.Highlight, ACCENT), (QtGui.QPalette.HighlightedText, "#ffffff")):
        palette.setColor(role, QtGui.QColor(color))
    window.setPalette(palette)
    window.setStyleSheet(STYLE)
