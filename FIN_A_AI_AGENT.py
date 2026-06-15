import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import build_app

if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True)
