import sys

from PyQt5.QtWidgets import QApplication

from .gui import CMEGUI


def run():
    app = QApplication(sys.argv)
    w = CMEGUI()
    w.resize(1200, 950)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
