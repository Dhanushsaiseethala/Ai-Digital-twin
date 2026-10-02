"""
Flask API for LightGBM server health predictions.
Run this separately to provide model predictions to the frontend.

Usage:
    python server_health_api.py
    
Then the Load Distribution frontend will call http://localhost:5000/predict_health
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server_health_predictor import get_server_health_predictions, predict_server_health_from_features

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication


@app.route('/predict_health', methods=['POST'])
def predict_health():
    """
    Endpoint to predict health for multiple servers.
    
    Request body:
    {
        "servers": [
            {"id": "Rack A01-S01", "rackId": "Rack A01", "position": [0, 1, 0]},
            ...
        ]
    }
    
    Response:
    {
        "predictions": [
            {"id": "Rack A01-S01", "health": "Healthy", "score": 85.5},
            ...
        ]
    }
    """
    try:
        data = request.json
        servers_data = data.get('servers', [])
        
        if not servers_data:
            return jsonify({'error': 'No servers provided'}), 400
        
        # Get predictions from the model
        predictions = get_server_health_predictions(servers_data)
        
        # Format response
        result = []
        for server, (health_cat, health_score) in zip(servers_data, predictions):
            result.append({
                'id': server.get('id'),
                'health': health_cat,
                'score': round(health_score, 1)
            })
        
        return jsonify({
            'success': True,
            'predictions': result
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict_health/model', methods=['POST'])
def predict_health_model():
    """Return model outputs only when the trained model and all inputs exist."""
    data = request.get_json(silent=True) or {}
    servers_data = data.get('servers', [])
    if not servers_data:
        return jsonify({'success': False, 'error': 'No servers provided'}), 400

    predictions = []
    for server in servers_data:
        prediction, error = predict_server_health_from_features(server.get('features', {}))
        predictions.append({
            'id': server.get('id'),
            'prediction': prediction,
            'status': 'available' if error is None else 'unavailable',
            'message': error
        })

    available = all(item['status'] == 'available' for item in predictions)
    return jsonify({
        'success': True,
        'status': 'available' if available else 'unavailable',
        'message': None if available else 'LightGBM unavailable for this experiment',
        'predictions': predictions
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    print("Starting LightGBM Model API Server on http://localhost:5000")
    print("CORS enabled for frontend communication")
    print("\nEndpoints:")
    print("  POST /predict_health - Get health predictions for servers")
    print("  GET /health - Health check")
    app.run(debug=False, host='localhost', port=5000)
