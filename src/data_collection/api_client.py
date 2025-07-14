"""
Transfermarkt API Client
This module contains the base client class for interacting with the Transfermarkt API.
"""

import requests
import time
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import json

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration settings"""
    base_url: str = "https://transfermarkt-api.fly.dev"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


class TransfermarktAPIClient:
    """
    Transfermarkt API client class
    
    This class manages all HTTP requests to the Transfermarkt API
    and provides basic error handling, retry logic, and rate limiting.
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize the API client
        
        Args:
            config: API configuration, uses default if None
        """
        self.config = config or APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TransfermarktDataAnalysis/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Makes HTTP request to the API
        
        Args:
            endpoint: API endpoint (e.g., '/players/search/messi')
            params: Query parameters
            
        Returns:
            JSON response dictionary
            
        Raises:
            requests.RequestException: In case of API error
        """
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Making API request: {url}")
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                # Short wait for rate limiting
                time.sleep(0.5)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(self.config.retry_delay * (attempt + 1))
    
    # PLAYER ENDPOINTS
    def search_players(self, query: str, page: int = 1) -> Dict:
        """
        Search for players
        
        Args:
            query: Search term (player name)
            page: Page number
            
        Returns:
            Search results dictionary
        """
        return self._make_request(f"/players/search/{query}", {"page_number": page})
    
    def get_player_profile(self, player_id: str) -> Dict:
        """
        Get player profile information
        
        Args:
            player_id: Player ID
            
        Returns:
            Player profile information
        """
        return self._make_request(f"/players/{player_id}/profile")
    
    def get_player_market_value(self, player_id: str) -> Dict:
        """
        Get player market value history
        
        Args:
            player_id: Player ID
            
        Returns:
            Market value history
        """
        return self._make_request(f"/players/{player_id}/market_value")
    
    def get_player_transfers(self, player_id: str) -> Dict:
        """
        Get player transfer history
        
        Args:
            player_id: Player ID
            
        Returns:
            Transfer history
        """
        return self._make_request(f"/players/{player_id}/transfers")
    
    def get_player_stats(self, player_id: str) -> Dict:
        """
        Get player statistics
        
        Args:
            player_id: Player ID
            
        Returns:
            Player statistics
        """
        return self._make_request(f"/players/{player_id}/stats")
    
    def get_player_injuries(self, player_id: str, page: int = 1) -> Dict:
        """
        Get player injury history
        
        Args:
            player_id: Player ID
            page: Page number
            
        Returns:
            Injury history
        """
        return self._make_request(f"/players/{player_id}/injuries", {"page_number": page})
    
    # CLUB ENDPOINTS
    def search_clubs(self, query: str, page: int = 1) -> Dict:
        """
        Search for clubs
        
        Args:
            query: Search term (club name)
            page: Page number
            
        Returns:
            Search results
        """
        return self._make_request(f"/clubs/search/{query}", {"page_number": page})
    
    def get_club_profile(self, club_id: str) -> Dict:
        """
        Get club profile information
        
        Args:
            club_id: Club ID
            
        Returns:
            Club profile information
        """
        return self._make_request(f"/clubs/{club_id}/profile")
    
    def get_club_players(self, club_id: str, season_id: Optional[str] = None) -> Dict:
        """
        Get club player squad
        
        Args:
            club_id: Club ID
            season_id: Season ID (optional)
            
        Returns:
            Player squad
        """
        params = {"season_id": season_id} if season_id else None
        return self._make_request(f"/clubs/{club_id}/players", params)
    
    # COMPETITION ENDPOINTS
    def search_competitions(self, query: str, page: int = 1) -> Dict:
        """
        Search for leagues/competitions
        
        Args:
            query: Search term
            page: Page number
            
        Returns:
            Search results
        """
        return self._make_request(f"/competitions/search/{query}", {"page_number": page})
    
    def get_competition_clubs(self, competition_id: str, season_id: Optional[str] = None) -> Dict:
        """
        Get clubs in a league/competition
        
        Args:
            competition_id: League ID
            season_id: Season ID (optional)
            
        Returns:
            League clubs
        """
        params = {"season_id": season_id} if season_id else None
        return self._make_request(f"/competitions/{competition_id}/clubs", params)


# Utility functions
def save_data_to_json(data: Union[Dict, List], filename: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filename: File name
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Data saved to {filename}")


def load_data_from_json(filename: str) -> Union[Dict, List]:
    """
    Load data from JSON file
    
    Args:
        filename: File name
        
    Returns:
        Loaded data
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Data loaded from {filename}")
    return data


if __name__ == "__main__":
    # Test code
    client = TransfermarktAPIClient()
    
    # Messi search test
    try:
        result = client.search_players("messi")
        print("Messi search result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
