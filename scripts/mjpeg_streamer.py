#!/usr/bin/env python3
"""
mjpeg_streamer.py — Ultra-low latency MJPEG HTTP Live Streamer for headless robots.

Requires zero third-party dependencies (uses Python standard library http.server).
View live camera feeds in any web browser (Chrome, Edge, Safari, Firefox, mobile).
"""

import cv2
import time
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


def get_local_ip() -> str:
    """Detect LAN IP address for easy browser access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = 'localhost'
    finally:
        s.close()
    return ip


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _MJPEGHandler(BaseHTTPRequestHandler):
    streamer = None  # Class-level reference to MJPEGStreamer instance

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SO-ARM101 Live Camera Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            margin: 0;
            background: #121214;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        .header {{
            margin-bottom: 12px;
            text-align: center;
        }}
        h1 {{
            margin: 0;
            font-size: 1.4rem;
            color: #00e676;
            letter-spacing: 0.5px;
        }}
        .badge {{
            display: inline-block;
            margin-top: 4px;
            padding: 3px 8px;
            background: #1f1f23;
            border: 1px solid #333;
            border-radius: 4px;
            font-size: 0.8rem;
            color: #888;
        }}
        .container {{
            position: relative;
            background: #000;
            border: 2px solid #28282e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.7);
            max-width: 95vw;
        }}
        img {{
            display: block;
            width: auto;
            max-width: 95vw;
            max-height: 80vh;
            object-fit: contain;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 SO-ARM101 Live Camera Feed</h1>
        <span class="badge">RealSense D405 • Low Latency Live Stream</span>
    </div>
    <div class="container">
        <img src="/stream.mjpg" alt="Live Stream" />
    </div>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            while self.streamer and self.streamer.running:
                jpeg = self.streamer.get_jpeg()
                if jpeg is not None:
                    try:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                        break
                time.sleep(0.02)  # max ~50 fps
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress spamming terminal with continuous HTTP GET log messages
        pass


class MJPEGStreamer:
    """
    Lightweight background HTTP server that broadcasts OpenCV BGR frames
    as an MJPEG stream over standard HTTP.
    """
    def __init__(self, port=8080, quality=75):
        self.port = port
        self.quality = quality
        self.current_jpeg = None
        self.lock = threading.Lock()
        self.running = True
        self.server = None
        self.thread = None
        self.local_ip = get_local_ip()

        # Bind handler
        handler_class = _MJPEGHandler
        handler_class.streamer = self

        # Try binding specified port; fall back if already in use
        for p in [port, port + 1, port + 2, 5000, 8888]:
            try:
                self.server = _ThreadedHTTPServer(('0.0.0.0', p), handler_class)
                self.port = p
                break
            except OSError:
                continue

        if self.server is None:
            print("⚠️  Could not bind any HTTP port for MJPEG streaming.")
            return

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        print("\n" + "═"*65)
        print(" 🌐 LIVE VIDEO STREAM ACTIVE")
        print(f" 👉 View on your phone / PC: http://{self.local_ip}:{self.port}")
        print(f" 👉 View via localhost:      http://localhost:{self.port}")
        print("═"*65 + "\n")

    def update_frame(self, frame_bgr):
        """Encode new OpenCV BGR frame as JPEG and store for streaming clients."""
        if frame_bgr is None:
            return
        try:
            success, encoded = cv2.imencode(
                '.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if success:
                with self.lock:
                    self.current_jpeg = encoded.tobytes()
        except Exception:
            pass

    def get_jpeg(self):
        """Retrieve most recently encoded JPEG byte chunk."""
        with self.lock:
            return self.current_jpeg

    def stop(self):
        """Shut down the HTTP streamer."""
        self.running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass


# Global singleton instance
_GLOBAL_STREAMER = None

def get_streamer(port=8080) -> MJPEGStreamer:
    global _GLOBAL_STREAMER
    if _GLOBAL_STREAMER is None:
        _GLOBAL_STREAMER = MJPEGStreamer(port=port)
    return _GLOBAL_STREAMER
