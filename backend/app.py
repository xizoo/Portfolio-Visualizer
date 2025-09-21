"""
app.py

Main entry point for the backend service.
Creates the Flask app, registers routes, and runs the server.
"""

from flask import Flask
from api.routes import api_blueprint

def create_app():
    """
    Factory function to create and configure the Flask app.
    """
    app = Flask(__name__)

    # Register API routes under /api prefix
    app.register_blueprint(api_blueprint, url_prefix="/api")

    return app

if __name__ == "__main__":
    app = create_app()
    # Runs on http://localhost:5000 by default
    app.run(debug=True)
