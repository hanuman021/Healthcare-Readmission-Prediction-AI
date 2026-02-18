"""
Simple HTTP Server to serve the Healthcare App
Run this instead of opening the HTML file directly
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    PORT = 8000
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  Healthcare AI App Server                                 ║
    ╚══════════════════════════════════════════════════════════╝
    
    ✅ Server running at: http://localhost:{PORT}
    
    📝 Instructions:
    1. Make sure Flask backend is running on port 5000
    2. Open your browser and go to:
       👉 http://localhost:{PORT}/healthcare-app-enhanced.html
    
    3. Default login:
       Username: admin
       Password: admin123
    
    Press Ctrl+C to stop the server
    """)
    
    httpd = HTTPServer(('localhost', PORT), CORSRequestHandler)
    httpd.serve_forever()