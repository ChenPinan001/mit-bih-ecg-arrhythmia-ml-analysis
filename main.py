import os
os.environ["QT_LOGGING_RULES"] = "qt.image.png=false"
import sys
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import main

if __name__ == '__main__':
    main()
