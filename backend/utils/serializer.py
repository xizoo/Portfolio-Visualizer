"""
utils/serializer.py

Helpers to serialize Python objects (like matplotlib figures)
into JSON-safe formats for API responses.
"""

import base64
import io

def fig_to_base64(fig):
    """
    Convert a matplotlib figure into a base64-encoded PNG string.

    Args:
        fig (matplotlib.figure.Figure)

    Returns:
        str: base64 string of the image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded

def serialize_results(results: dict, plots: dict):
    """
    Combine results and plots into a JSON-safe dictionary.

    Args:
        results (dict): numerical results (return, volatility, weights)
        plots (dict): dict of matplotlib figures

    Returns:
        dict: JSON-serializable object with results and base64 plots
    """
    encoded_plots = {name: fig_to_base64(fig) for name, fig in plots.items()}

    return {
        "results": results,
        "plots": encoded_plots
    }
