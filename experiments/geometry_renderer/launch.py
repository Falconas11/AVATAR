"""Local-only launcher with telemetry disabled and bounded in-memory file input."""
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    import cv2
    cv2.setNumThreads(1)
    from streamlit.web import cli
    sys.argv = ['streamlit', 'run', str(Path(__file__).with_name('app.py')),
                '--server.address=127.0.0.1', f'--server.port={args.port}',
                '--server.headless=true', '--browser.gatherUsageStats=false',
                '--server.maxUploadSize=10', '--server.fileWatcherType=none',
                '--client.toolbarMode=minimal']
    raise SystemExit(cli.main())


if __name__ == '__main__':
    main()
