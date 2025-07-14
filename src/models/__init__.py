"""
Models Package
Machine learning models for Transfermarkt data.
"""

# Use simple model instead of XGBoost due to compatibility issues
try:
    from .simple_prediction import SimpleMarketValuePredictor as MarketValuePredictor
    from .simple_prediction import SimpleTransferValueAnalyzer as TransferValueAnalyzer
    __all__ = [
        'MarketValuePredictor',
        'TransferValueAnalyzer'
    ]
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    # Provide minimal fallback classes
    class MarketValuePredictor:
        def __init__(self):
            self.available = False
            
        def train(self, *args, **kwargs):
            return {"error": "Models not available"}
            
        def predict(self, *args, **kwargs):
            return []
    
    class TransferValueAnalyzer:
        def __init__(self):
            self.available = False
            
        def analyze(self, *args, **kwargs):
            return {"error": "Models not available"}
    
    __all__ = [
        'MarketValuePredictor',
        'TransferValueAnalyzer'
    ]
