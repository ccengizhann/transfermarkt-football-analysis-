"""
Data Collection Package
Manages data collection operations from Transfermarkt API.
"""

from .api_client import TransfermarktAPIClient, APIConfig
from .data_collector import DataCollector, get_popular_players, get_top_leagues

__all__ = [
    'TransfermarktAPIClient',
    'APIConfig', 
    'DataCollector',
    'get_popular_players',
    'get_top_leagues'
]
