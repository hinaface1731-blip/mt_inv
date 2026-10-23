"""Запуск MT-Inv: python run.py [папка_с_EDI | файлы.edi | проект.json]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtinv.gui.main_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
