"""
Data Collection Module
This module coordinates large-scale data collection operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set
import logging
import time
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .api_client import TransfermarktAPIClient, save_data_to_json

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Class that manages large-scale data collection operations
    """
    
    def __init__(self, client: Optional[TransfermarktAPIClient] = None, 
                 output_dir: str = "data"):
        """
        Initialize DataCollector
        
        Args:
            client: API client, creates new one if None
            output_dir: Output directory
        """
        self.client = client or TransfermarktAPIClient()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Collected data tracking
        self.collected_player_ids: Set[str] = set()
        self.collected_club_ids: Set[str] = set()
    
    def collect_top_players_data(self, player_names: List[str], 
                                max_results_per_search: int = 5) -> Dict:
        """
        Search for specified player names and collect their detailed data
        
        Args:
            player_names: Player names to search for
            max_results_per_search: Maximum number of results per search
            
        Returns:
            Toplanan veri dictionary
        """
        logger.info(f"Starting data collection for {len(player_names)} players")
        
        all_players_data = {}
        
        for name in player_names:
            try:
                logger.info(f"'{name}' aranıyor...")
                search_results = self.client.search_players(name)
                
                if 'results' in search_results:
                    players = search_results['results'][:max_results_per_search]
                    
                    for player in players:
                        player_id = player.get('id')
                        if player_id and player_id not in self.collected_player_ids:
                            player_data = self._collect_single_player_data(player_id)
                            if player_data:
                                all_players_data[player_id] = player_data
                                self.collected_player_ids.add(player_id)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting data for '{name}': {e}")
        
        # Veriyi kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"players_data_{timestamp}.json")
        save_data_to_json(all_players_data, filename)
        
        logger.info(f"Total {len(all_players_data)} player data collected")
        return all_players_data
    
    def collect_league_clubs_data(self, league_name: str, 
                                 season_id: Optional[str] = None) -> Dict:
        """
        Collect data for all clubs in a league
        
        Args:
            league_name: League name
            season_id: Season ID
            
        Returns:
            Club data dictionary
        """
        logger.info(f"Collecting club data for '{league_name}' league")
        
        try:
            # Liga ara
            competition_search = self.client.search_competitions(league_name)
            
            if 'results' not in competition_search or not competition_search['results']:
                logger.warning(f"'{league_name}' ligası bulunamadı")
                return {}
            
            # İlk sonucu al
            competition = competition_search['results'][0]
            competition_id = competition.get('id')
            
            if not competition_id:
                logger.error("League ID not found")
                return {}
            
            # Get clubs in the league
            clubs_data = self.client.get_competition_clubs(competition_id, season_id)
            
            all_clubs_data = {}
            
            if 'clubs' in clubs_data:
                for club in clubs_data['clubs']:
                    club_id = club.get('id')
                    if club_id and club_id not in self.collected_club_ids:
                        club_data = self._collect_single_club_data(club_id, season_id)
                        if club_data:
                            all_clubs_data[club_id] = club_data
                            self.collected_club_ids.add(club_id)
                        
                        time.sleep(0.5)  # Rate limiting
            
            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"league_{league_name}_{timestamp}.json")
            save_data_to_json(all_clubs_data, filename)
            
            logger.info(f"Collected {len(all_clubs_data)} club data for '{league_name}' league")
            return all_clubs_data
            
        except Exception as e:
            logger.error(f"Error collecting league data: {e}")
            return {}
    
    def _collect_single_player_data(self, player_id: str) -> Optional[Dict]:
        """
        Collect all available data for a single player
        
        Args:
            player_id: Player ID
            
        Returns:
            Player data dictionary
        """
        player_data = {'id': player_id}
        
        try:
            # Profile information
            logger.info(f"Collecting profile information for player {player_id}")
            profile = self.client.get_player_profile(player_id)
            player_data['profile'] = profile
            
            # Market value
            try:
                market_value = self.client.get_player_market_value(player_id)
                player_data['market_value'] = market_value
            except Exception as e:
                logger.warning(f"Could not get market value for player {player_id}: {e}")
            
            # Transfers
            try:
                transfers = self.client.get_player_transfers(player_id)
                player_data['transfers'] = transfers
            except Exception as e:
                logger.warning(f"Could not get transfer data for player {player_id}: {e}")
            
            # Statistics
            try:
                stats = self.client.get_player_stats(player_id)
                player_data['stats'] = stats
            except Exception as e:
                logger.warning(f"Could not get statistics for player {player_id}: {e}")
            
            # Injuries
            try:
                injuries = self.client.get_player_injuries(player_id)
                player_data['injuries'] = injuries
            except Exception as e:
                logger.warning(f"Could not get injury data for player {player_id}: {e}")
            
            time.sleep(0.5)  # Rate limiting
            return player_data
            
        except Exception as e:
            logger.error(f"Error collecting data for player {player_id}: {e}")
            return None
    
    def _collect_single_club_data(self, club_id: str, 
                                 season_id: Optional[str] = None) -> Optional[Dict]:
        """
        Collect all available data for a single club
        
        Args:
            club_id: Club ID
            season_id: Season ID
            
        Returns:
            Club data dictionary
        """
        club_data = {'id': club_id}
        
        try:
            # Profile information
            logger.info(f"Collecting profile information for club {club_id}")
            profile = self.client.get_club_profile(club_id)
            club_data['profile'] = profile
            
            # Players
            try:
                players = self.client.get_club_players(club_id, season_id)
                club_data['players'] = players
            except Exception as e:
                logger.warning(f"Could not get player data for club {club_id}: {e}")
            
            return club_data
            
        except Exception as e:
            logger.error(f"Error collecting data for club {club_id}: {e}")
            return None
    
    def collect_top_leagues_data(self, leagues: List[str], 
                                season_id: Optional[str] = None) -> Dict:
        """
        Birden fazla liga için veri toplar
        
        Args:
            leagues: Liga isimleri listesi
            season_id: Sezon ID'si
            
        Returns:
            Tüm liga verileri
        """
        all_leagues_data = {}
        
        for league in leagues:
            logger.info(f"'{league}' ligası işleniyor...")
            league_data = self.collect_league_clubs_data(league, season_id)
            if league_data:
                all_leagues_data[league] = league_data
            
            # Ligalar arası bekleme
            time.sleep(2)
        
        # Tüm veriyi kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"all_leagues_data_{timestamp}.json")
        save_data_to_json(all_leagues_data, filename)
        
        return all_leagues_data


def get_popular_players() -> List[str]:
    """
    Returns a list of popular player names
    
    Returns:
        List of player names
    """
    return [
        "Messi", "Ronaldo", "Neymar", "Mbappe", "Haaland",
        "Benzema", "Lewandowski", "Salah", "De Bruyne", "Modric",
        "Vinicius", "Pedri", "Bellingham", "Osimhen", "Kane",
        "Gundogan", "Kimmich", "Muller", "Griezmann", "Dembele"
    ]


def get_top_leagues() -> List[str]:
    """
    Büyük Avrupa ligalarının listesini döndürür
    
    Returns:
        Liga isimleri listesi
    """
    return [
        "Premier League",
        "La Liga", 
        "Bundesliga",
        "Serie A",
        "Ligue 1"
    ]


if __name__ == "__main__":
    # Test kodları
    collector = DataCollector()
    
    # Data collection test for popular players
    popular_players = get_popular_players()[:5]  # First 5 players
    logger.info("Starting data collection for popular players...")
    
    result = collector.collect_top_players_data(popular_players)
    print(f"Number of players collected: {len(result)}")
