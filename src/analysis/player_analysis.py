"""
Data Analysis Module
Functions for analyzing player and club data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, date
import json
import re

logger = logging.getLogger(__name__)


class PlayerAnalyzer:
    """
    Class for analyzing player data
    """
    
    def __init__(self, players_data: Dict):
        """
        Initialize PlayerAnalyzer
        
        Args:
            players_data: Player data dictionary
        """
        self.players_data = players_data
        self.players_df = self._create_players_dataframe()
    
    def _create_players_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame from player data
        
        Returns:
            Player DataFrame
        """
        players_list = []
        
        for player_id, player_data in self.players_data.items():
            try:
                profile = player_data.get('profile', {})
                market_value_data = player_data.get('market_value', {})
                
                player_info = {
                    'player_id': player_id,
                    'name': profile.get('name', ''),
                    'dateOfBirth': profile.get('dateOfBirth', ''),
                    'age': profile.get('age', 0),
                    'height': profile.get('height', ''),
                    'foot': profile.get('foot', ''),
                    'position': profile.get('position', {}).get('main', ''),
                    'nationality': self._extract_nationality(profile.get('citizenship', [])),
                    'current_club': profile.get('club', {}).get('name', ''),
                    'current_market_value': self._extract_market_value(market_value_data.get('marketValue', '')),
                    'market_value_history': market_value_data.get('marketValueHistory', [])
                }
                
                players_list.append(player_info)
                
            except Exception as e:
                logger.warning(f"Error processing player {player_id}: {e}")
        
        df = pd.DataFrame(players_list)
        
        # Veri tiplerini düzenle
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if 'current_market_value' in df.columns:
            df['current_market_value'] = pd.to_numeric(df['current_market_value'], errors='coerce')
        if 'dateOfBirth' in df.columns:
            df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], errors='coerce')
        
        return df
    
    def _extract_nationality(self, citizenship: List[Dict]) -> str:
        """
        Vatandaşlık bilgisinden ülke ismini çıkarır
        
        Args:
            citizenship: Vatandaşlık bilgisi listesi
            
        Returns:
            Ülke ismi
        """
        if citizenship and isinstance(citizenship, list) and len(citizenship) > 0:
            return citizenship[0].get('name', '')
        return ''
    
    def _extract_market_value(self, market_value_str: str) -> Optional[float]:
        """
        Market değeri string'inden sayısal değer çıkarır
        
        Args:
            market_value_str: Market değeri string'i (örn: "€75.00m")
            
        Returns:
            Market değeri (milyon Euro cinsinden)
        """
        if not market_value_str or market_value_str == '-':
            return None
        
        # Regex ile sayıları ve birimleri çıkar
        pattern = r'€?([\d,.]+)([kmKM])?'
        match = re.search(pattern, str(market_value_str))
        
        if match:
            value_str = match.group(1).replace(',', '.')
            unit = match.group(2)
            
            try:
                value = float(value_str)
                
                if unit and unit.lower() == 'k':
                    value = value / 1000  # k'yı milyon'a çevir
                elif unit and unit.lower() == 'm':
                    value = value  # Zaten milyon cinsinden
                
                return value
            except ValueError:
                pass
        
        return None
    
    def get_top_players_by_value(self, n: int = 10) -> pd.DataFrame:
        """
        Get the most valuable players by market value
        
        Args:
            n: Number of players to return
            
        Returns:
            DataFrame of most valuable players
        """
        return (self.players_df
                .dropna(subset=['current_market_value'])
                .nlargest(n, 'current_market_value')
                .reset_index(drop=True))
    
    def analyze_age_distribution(self) -> Dict[str, Any]:
        """
        Yaş dağılımını analiz eder
        
        Returns:
            Yaş analizi sonuçları
        """
        age_stats = self.players_df['age'].describe()
        
        age_ranges = {
            'Young (Under 23)': len(self.players_df[self.players_df['age'] < 23]),
            'Prime (23-29)': len(self.players_df[(self.players_df['age'] >= 23) & 
                                               (self.players_df['age'] <= 29)]),
            'Experienced (30+)': len(self.players_df[self.players_df['age'] > 29])
        }
        
        return {
            'statistics': age_stats.to_dict(),
            'age_ranges': age_ranges
        }
    
    def analyze_position_distribution(self) -> pd.Series:
        """
        Pozisyon dağılımını analiz eder
        
        Returns:
            Pozisyon dağılımı
        """
        return self.players_df['position'].value_counts()
    
    def analyze_nationality_distribution(self) -> pd.Series:
        """
        Uyruk dağılımını analiz eder
        
        Returns:
            Uyruk dağılımı
        """
        return self.players_df['nationality'].value_counts()
    
    def analyze_market_value_trends(self, player_id: str) -> Optional[pd.DataFrame]:
        """
        Belirli bir oyuncunun market değeri trendini analiz eder
        
        Args:
            player_id: Oyuncu ID'si
            
        Returns:
            Market değeri trend DataFrame'i
        """
        if player_id not in self.players_data:
            return None
        
        market_history = (self.players_data[player_id]
                         .get('market_value', {})
                         .get('marketValueHistory', []))
        
        if not market_history:
            return None
        
        trend_data = []
        for entry in market_history:
            trend_data.append({
                'date': entry.get('date', ''),
                'age': entry.get('age', 0),
                'club': entry.get('clubName', ''),
                'market_value': self._extract_market_value(entry.get('marketValue', ''))
            })
        
        df = pd.DataFrame(trend_data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df.sort_values('date', ascending=True) if not df.empty else None
    
    def compare_players(self, player_ids: List[str]) -> pd.DataFrame:
        """
        Birden fazla oyuncuyu karşılaştırır
        
        Args:
            player_ids: Karşılaştırılacak oyuncu ID'leri
            
        Returns:
            Karşılaştırma DataFrame'i
        """
        comparison_data = []
        
        for player_id in player_ids:
            if player_id in self.players_data:
                player_row = self.players_df[self.players_df['player_id'] == player_id]
                if not player_row.empty:
                    comparison_data.append(player_row.iloc[0])
        
        return pd.DataFrame(comparison_data).reset_index(drop=True)


class ClubAnalyzer:
    """
    Kulüp verilerini analiz eden sınıf
    """
    
    def __init__(self, clubs_data: Dict):
        """
        ClubAnalyzer'ı başlatır
        
        Args:
            clubs_data: Kulüp verileri dictionary
        """
        self.clubs_data = clubs_data
        self.clubs_df = self._create_clubs_dataframe()
    
    def _create_clubs_dataframe(self) -> pd.DataFrame:
        """
        Kulüp verilerinden DataFrame oluşturur
        
        Returns:
            Kulüp DataFrame'i
        """
        clubs_list = []
        
        for club_id, club_data in self.clubs_data.items():
            try:
                profile = club_data.get('profile', {})
                players_data = club_data.get('players', {})
                
                club_info = {
                    'club_id': club_id,
                    'name': profile.get('name', ''),
                    'official_name': profile.get('officialName', ''),
                    'founded': profile.get('foundedOn', ''),
                    'members': profile.get('members', 0),
                    'stadium_name': profile.get('stadium', {}).get('name', ''),
                    'stadium_capacity': profile.get('stadium', {}).get('capacity', 0),
                    'total_market_value': profile.get('squadSize', {}).get('totalMarketValue', 0),
                    'average_age': profile.get('squadSize', {}).get('averageAge', 0),
                    'foreign_players': profile.get('squadSize', {}).get('foreigners', 0),
                    'squad_size': profile.get('squadSize', {}).get('size', 0),
                    'player_count': len(players_data.get('players', []))
                }
                
                clubs_list.append(club_info)
                
            except Exception as e:
                logger.warning(f"Kulüp {club_id} işlenirken hata: {e}")
        
        df = pd.DataFrame(clubs_list)
        
        # Veri tiplerini düzenle
        numeric_columns = ['members', 'stadium_capacity', 'total_market_value', 
                          'average_age', 'foreign_players', 'squad_size', 'player_count']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'founded' in df.columns:
            df['founded'] = pd.to_datetime(df['founded'], errors='coerce')
        
        return df
    
    def get_top_clubs_by_value(self, n: int = 10) -> pd.DataFrame:
        """
        Toplam market değerine göre en değerli kulüpleri getirir
        
        Args:
            n: Getirilecek kulüp sayısı
            
        Returns:
            En değerli kulüpler DataFrame'i
        """
        return (self.clubs_df
                .dropna(subset=['total_market_value'])
                .nlargest(n, 'total_market_value')
                .reset_index(drop=True))
    
    def analyze_squad_composition(self, club_id: str) -> Optional[Dict]:
        """
        Kulüp kadro kompozisyonunu analiz eder
        
        Args:
            club_id: Kulüp ID'si
            
        Returns:
            Kadro analizi sonuçları
        """
        if club_id not in self.clubs_data:
            return None
        
        players_data = (self.clubs_data[club_id]
                       .get('players', {})
                       .get('players', []))
        
        if not players_data:
            return None
        
        # DataFrame oluştur
        players_df = pd.DataFrame(players_data)
        
        analysis = {
            'total_players': len(players_df),
            'positions': players_df['position'].value_counts().to_dict(),
            'nationalities': players_df['nationality'].value_counts().to_dict(),
            'age_distribution': {
                'mean': players_df['age'].mean() if 'age' in players_df.columns else 0,
                'median': players_df['age'].median() if 'age' in players_df.columns else 0,
                'std': players_df['age'].std() if 'age' in players_df.columns else 0
            }
        }
        
        return analysis


def load_and_analyze_data(data_file: str) -> Tuple[PlayerAnalyzer, Optional[ClubAnalyzer]]:
    """
    Veri dosyasını yükler ve analiz sınıflarını oluşturur
    
    Args:
        data_file: JSON veri dosyası yolu
        
    Returns:
        PlayerAnalyzer ve ClubAnalyzer tuple'ı
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Eğer veri oyuncu verisi ise
    if any('profile' in item for item in data.values() if isinstance(item, dict)):
        player_analyzer = PlayerAnalyzer(data)
        return player_analyzer, None
    
    # Eğer veri kulüp verisi ise
    # Bu durumu geliştir...
    
    return PlayerAnalyzer(data), None


if __name__ == "__main__":
    # Test kodları buraya eklenebilir
    pass
