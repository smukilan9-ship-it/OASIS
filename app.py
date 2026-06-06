"""
app.py — IHC Analyzer Desktop App
pywebview + HTML/CSS/JS frontend
"""
import sys
import webview
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from webui.api import API

def main():
    api = API()
    html_path = str(Path(__file__).parent / "webui" / "index.html")
    window = webview.create_window(
        title="IHC Analyzer",
        url=f"file://{html_path}",
        js_api=api,
        width=1280,
        height=820,
        min_size=(1100, 700),
        background_color="#FFFFFF",
    )
    api.set_window(window)
    webview.start(debug=False)

if __name__ == "__main__":
    main()