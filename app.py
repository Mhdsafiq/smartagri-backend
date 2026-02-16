from flask import Flask
from flask_cors import CORS
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'message': 'SmartAgroAI Backend is running'}

    # Import routes
    from routes.weather_routes import weather_bp
    from routes.yield_routes import yield_bp
    from routes.fertilizer_routes import fertilizer_bp

    # Register blueprints
    app.register_blueprint(weather_bp, url_prefix='/api')
    app.register_blueprint(yield_bp, url_prefix='/api')
    app.register_blueprint(fertilizer_bp, url_prefix='/api')

    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
