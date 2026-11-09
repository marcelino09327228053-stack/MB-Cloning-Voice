"""Reuse this studio when running; otherwise start it on a free local port."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import json
from pathlib import Path
import runpy
import socket
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PORTS = range(7865, 7886)


def existing_studio():
    for port in PORTS:
        with socket.socket() as probe:
            probe.settimeout(0.05)
            if probe.connect_ex(('127.0.0.1', port)) != 0:
                continue
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/config', timeout=0.3) as response:
                config = json.load(response)
            ids = {c.get('props', {}).get('elem_id') for c in config.get('components', [])}
            if config.get('title') == 'MB Cloning Voice' and {'sidebar', 'generate-btn', 'result-panel'} <= ids:
                return f'http://127.0.0.1:{port}'
        except (OSError, ValueError):
            continue
    return None

def available_port():
    for port in PORTS:
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('All studio ports (7865-7885) are occupied. Close an unused studio window and retry.')


def main():
    url = existing_studio()
    if url:
        print(f'MB Cloning Voice is already running: {url}', flush=True)
        webbrowser.open(url)
        return
    runpy.run_path(str(ROOT / 'mb_cloning_voice.py'), run_name='__main__')


if __name__ == '__main__':
    main()


