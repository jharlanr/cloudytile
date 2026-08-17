"""
Browser-based binary labeling GUI for tile JPG frames.

Replaces the Labelbox workflow (export_labelbox.py): serves one JPG at a
time on localhost and records labels straight into labels.csv, the format
consumed by CloudyTileDataset.

Usage:
    python engine/labeling/label_gui.py --image_dir /path/to/jpgs
    python engine/labeling/label_gui.py --image_dir jpgs --labels_csv mine.csv --port 5050

    Then label in the browser:
        left arrow  = 0 (not useful: cloudy / no data)
        right arrow = 1 (useful: clear)
        u / backspace = undo (go back to the previous tile)
        h = help overlay with example tiles of each class
"""
import argparse
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cloudytile.labels import LabelStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets"

# Module-level config — populated by main()
IMAGE_DIR = None
STORE = None
FILENAMES = []  # sorted JPG basenames in IMAGE_DIR

# Flask's dev server handles requests on multiple threads; serialize
# label writes so two saves never interleave on the same temp file.
_store_lock = threading.Lock()

app = Flask(__name__, static_folder=None)


def progress():
    n0, n1 = STORE.counts(FILENAMES)
    return {"total": len(FILENAMES), "labeled": n0 + n1, "n0": n0, "n1": n1}


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/api/state")
def api_state():
    """Everything the frontend needs: file list, known labels, progress."""
    labels = {f: STORE.get(f) for f in FILENAMES if f in STORE}
    return jsonify({"files": FILENAMES, "labels": labels, **progress()})


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.json
    filename = data["filename"]
    if filename not in FILENAMES:
        return jsonify({"error": f"unknown filename: {filename}"}), 400
    with _store_lock:
        STORE.set(filename, int(data["label"]))
        STORE.save()
    return jsonify({"status": "ok", **progress()})


@app.route("/image/<path:filename>")
def image(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/assets/<path:filename>")
def asset(filename):
    """Example tiles for the help overlay."""
    return send_from_directory(ASSETS_DIR, filename)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Browser-based binary labeling GUI for tile JPG frames."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory of JPG frames (as produced by extract_jpgs.py)",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="labels/labels_v2.csv",
        help="Labels CSV to resume from and write to "
             "(default: labels/labels_v2.csv). Deliberately not "
             "labels/labels.csv — that holds the older Labelbox campaign over "
             "a different frame set, and the two must not be merged.",
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global IMAGE_DIR, STORE, FILENAMES

    args = parse_args(argv)

    # resolve() because send_from_directory treats relative paths as
    # relative to the Flask app dir, not the CWD
    IMAGE_DIR = Path(args.image_dir).resolve()
    if not IMAGE_DIR.is_dir():
        print(f"ERROR: image directory does not exist: {IMAGE_DIR}")
        sys.exit(1)

    FILENAMES = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))
    if not FILENAMES:
        print(f"ERROR: no .jpg files found in {IMAGE_DIR}")
        sys.exit(1)

    STORE = LabelStore(args.labels_csv)

    url = f"http://localhost:{args.port}"
    p = progress()
    print("\nLabeling server")
    print(f"  Image dir:  {IMAGE_DIR}")
    print(f"  Labels CSV: {Path(args.labels_csv).resolve()}")
    print(f"  Tiles:      {p['labeled']}/{p['total']} already labeled "
          f"(0: {p['n0']}, 1: {p['n1']})")
    print(f"\n  Opening {url} ...\n")

    # Silence Flask request logging
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    if not args.no_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
