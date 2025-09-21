"""
api/routes.py

Defines API endpoints (routes) for portfolio optimization.
Connects frontend requests → optimizer (business logic) → serializer (JSON response).
"""

from flask import Blueprint, request, jsonify
from portfolio.optimizer import optimize_portfolio
from utils.serializer import serialize_results

# Create a blueprint for API routes
api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/optimize", methods=["POST"])
def optimize():
    """
    POST /api/optimize
    Request JSON format:
    {
        "stocks": ["AAPL", "TSLA", "MSFT"]
    }

    Response JSON format:
    {
        "results": {
            "expected_return": 0.14,
            "volatility": 0.15,
            "weights": {
                "AAPL": 0.33,
                "TSLA": 0.33,
                "MSFT": 0.33
            }
        },
        "plots": {
            "pie_chart": "<base64 string>",
            "efficient_frontier": "<base64 string>"
        }
    }
    """
    try:
        data = request.get_json()
        stocks = data.get("stocks", [])

        if not stocks:
            return jsonify({"error": "Stock list is required"}), 400

        # Run optimizer
        results, plots = optimize_portfolio(stocks)

        # Serialize results
        response = serialize_results(results, plots)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
