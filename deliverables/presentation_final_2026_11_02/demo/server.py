"""Loopback-only static demo server. Standard library; no inference imports."""
import argparse
import hashlib
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
IDENTITY = 'CV5570_OFFLINE_DEMO_20261102_V1'

def verify_assets():
    manifest = json.loads((HERE / 'asset_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        p = (HERE / name).resolve()
        if HERE not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Missing or modified demo asset: {name}. Rebuild demo assets.')
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5570)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    assets = verify_assets()

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            super().end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/health':
                payload = json.dumps({'identity': IDENTITY, 'token': args.token,
                    'pid': os.getpid(), 'root': str(HERE), 'assets': len(assets), 'ready': True}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            name = 'index.html' if path == '/' else path.lstrip('/')
            if name not in assets:
                self.send_error(404)
                return
            self.path = '/' + name
            super().do_GET()

        def do_HEAD(self):
            name = 'index.html' if urlsplit(self.path).path == '/' else urlsplit(self.path).path.lstrip('/')
            if name not in assets:
                self.send_error(404)
                return
            super().do_HEAD()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(HERE)))
    print(f'{IDENTITY} ready http://127.0.0.1:{args.port}/ pid={os.getpid()}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
