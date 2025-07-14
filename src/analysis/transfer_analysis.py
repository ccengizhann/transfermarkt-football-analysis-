"""
Transfer Analizi Modülü
Transfer verilerinin detaylı analizi için fonksiyonlar.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, date
import json
import re

logger = logging.getLogger(__name__)


class TransferAnalyzer:
    """
    Transfer verilerini analiz eden sınıf
    """
    
    def __init__(self, players_data: Dict):
        """
        TransferAnalyzer'ı başlatır
        
        Args:
            players_data: Oyuncu verileri (transfer bilgileri dahil)
        """
        self.players_data = players_data
        self.transfers_df = self._create_transfers_dataframe()
    
    def _create_transfers_dataframe(self) -> pd.DataFrame:
        """
        Transfer verilerinden DataFrame oluşturur
        
        Returns:
            Transfer DataFrame'i
        """
        transfers_list = []
        
        for player_id, player_data in self.players_data.items():
            try:
                transfers_data = player_data.get('transfers', {}).get('transfers', [])
                player_profile = player_data.get('profile', {})
                player_name = player_profile.get('name', '')
                
                for transfer in transfers_data:
                    transfer_info = {
                        'player_id': player_id,
                        'player_name': player_name,
                        'transfer_id': transfer.get('id', ''),
                        'date': transfer.get('date', ''),
                        'season': transfer.get('season', ''),
                        'from_club_id': transfer.get('clubFrom', {}).get('id', ''),
                        'from_club_name': transfer.get('clubFrom', {}).get('name', ''),
                        'to_club_id': transfer.get('clubTo', {}).get('id', ''),
                        'to_club_name': transfer.get('clubTo', {}).get('name', ''),
                        'market_value': self._extract_transfer_value(transfer.get('marketValue')),
                        'fee': self._extract_transfer_value(transfer.get('fee')),
                        'upcoming': transfer.get('upcoming', False)
                    }
                    
                    transfers_list.append(transfer_info)
                    
            except Exception as e:
                logger.warning(f"Oyuncu {player_id} transfer verileri işlenirken hata: {e}")
        
        df = pd.DataFrame(transfers_list)
        
        # Veri tiplerini düzenle
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_columns = ['market_value', 'fee']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _extract_transfer_value(self, value: Any) -> Optional[float]:
        """
        Transfer değerini sayısal formata çevirir
        
        Args:
            value: Transfer değeri (string veya sayı)
            
        Returns:
            Milyon Euro cinsinden değer
        """
        if value is None or value == '' or value == '-':
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # String ise parse et
        value_str = str(value)
        
        # Free transfer durumları
        if 'free' in value_str.lower() or 'ablösefrei' in value_str.lower():
            return 0.0
        
        # Loan durumları
        if 'loan' in value_str.lower() or 'leihe' in value_str.lower():
            return None
        
        # Regex ile sayıları ve birimleri çıkar
        pattern = r'€?([\d,.]+)([kmKMB])?'
        match = re.search(pattern, value_str)
        
        if match:
            value_num_str = match.group(1).replace(',', '.')
            unit = match.group(2)
            
            try:
                value_num = float(value_num_str)
                
                if unit:
                    unit = unit.lower()
                    if unit == 'k':
                        value_num = value_num / 1000  # k'yı milyon'a çevir
                    elif unit == 'm':
                        value_num = value_num  # Zaten milyon
                    elif unit == 'b':
                        value_num = value_num * 1000  # Milyar'ı milyon'a çevir
                
                return value_num
            except ValueError:
                pass
        
        return None
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """
        Genel transfer istatistiklerini hesaplar
        
        Returns:
            Transfer istatistikleri
        """
        # Sadece geçmiş transferleri (upcoming=False) analiz et
        past_transfers = self.transfers_df[self.transfers_df['upcoming'] == False]
        
        # Fee olan transferler
        paid_transfers = past_transfers.dropna(subset=['fee'])
        paid_transfers = paid_transfers[paid_transfers['fee'] > 0]
        
        stats = {
            'total_transfers': len(past_transfers),
            'paid_transfers': len(paid_transfers),
            'free_transfers': len(past_transfers[past_transfers['fee'] == 0]),
            'total_transfer_volume': paid_transfers['fee'].sum() if not paid_transfers.empty else 0,
            'average_transfer_fee': paid_transfers['fee'].mean() if not paid_transfers.empty else 0,
            'median_transfer_fee': paid_transfers['fee'].median() if not paid_transfers.empty else 0,
            'highest_transfer_fee': paid_transfers['fee'].max() if not paid_transfers.empty else 0,
            'most_expensive_transfer': None
        }
        
        # En pahalı transfer detayı
        if not paid_transfers.empty:
            most_expensive = paid_transfers.loc[paid_transfers['fee'].idxmax()]
            stats['most_expensive_transfer'] = {
                'player': most_expensive['player_name'],
                'from_club': most_expensive['from_club_name'],
                'to_club': most_expensive['to_club_name'],
                'fee': most_expensive['fee'],
                'date': most_expensive['date']
            }
        
        return stats
    
    def analyze_transfer_trends_by_year(self) -> pd.DataFrame:
        """
        Yıllara göre transfer trendlerini analiz eder
        
        Returns:
            Yıllık transfer trend DataFrame'i
        """
        # Tarih olan transferleri al
        dated_transfers = self.transfers_df.dropna(subset=['date'])
        dated_transfers = dated_transfers[dated_transfers['upcoming'] == False]
        
        # Yıl sütunu ekle
        dated_transfers['year'] = dated_transfers['date'].dt.year
        
        # Fee olan transferleri al
        paid_transfers = dated_transfers.dropna(subset=['fee'])
        paid_transfers = paid_transfers[paid_transfers['fee'] > 0]
        
        # Yıllık istatistikler
        yearly_stats = paid_transfers.groupby('year').agg({
            'fee': ['count', 'sum', 'mean', 'median', 'max'],
            'transfer_id': 'count'
        }).round(2)
        
        yearly_stats.columns = ['transfer_count', 'total_volume', 'avg_fee', 'median_fee', 'max_fee', 'total_transfers']
        
        return yearly_stats.reset_index()
    
    def analyze_most_active_clubs(self, direction: str = 'both') -> pd.DataFrame:
        """
        En aktif kulüpleri analiz eder (alım/satım bazında)
        
        Args:
            direction: 'buying', 'selling' veya 'both'
            
        Returns:
            Aktif kulüpler DataFrame'i
        """
        past_transfers = self.transfers_df[self.transfers_df['upcoming'] == False]
        
        if direction == 'buying':
            club_activity = past_transfers.groupby(['to_club_id', 'to_club_name']).agg({
                'fee': ['count', 'sum', 'mean'],
                'transfer_id': 'count'
            }).round(2)
            club_activity.columns = ['paid_transfers', 'total_spent', 'avg_spent', 'total_transfers']
            
        elif direction == 'selling':
            club_activity = past_transfers.groupby(['from_club_id', 'from_club_name']).agg({
                'fee': ['count', 'sum', 'mean'],
                'transfer_id': 'count'
            }).round(2)
            club_activity.columns = ['paid_transfers', 'total_received', 'avg_received', 'total_transfers']
            
        else:  # both
            buying = past_transfers.groupby(['to_club_id', 'to_club_name']).agg({
                'fee': ['count', 'sum'],
                'transfer_id': 'count'
            })
            buying.columns = ['paid_buys', 'total_spent', 'total_buys']
            
            selling = past_transfers.groupby(['from_club_id', 'from_club_name']).agg({
                'fee': ['count', 'sum'],
                'transfer_id': 'count'
            })
            selling.columns = ['paid_sales', 'total_received', 'total_sales']
            
            # İki tabloyu birleştir
            club_activity = buying.join(selling, how='outer').fillna(0)
            club_activity['net_spend'] = club_activity['total_spent'] - club_activity['total_received']
            club_activity['total_activity'] = club_activity['total_buys'] + club_activity['total_sales']
        
        return club_activity.reset_index().sort_values('total_activity' if direction == 'both' else 'total_transfers', 
                                                      ascending=False)
    
    def analyze_player_transfer_patterns(self, player_id: str) -> Optional[Dict]:
        """
        Belirli bir oyuncunun transfer paternlerini analiz eder
        
        Args:
            player_id: Oyuncu ID'si
            
        Returns:
            Transfer patern analizi
        """
        player_transfers = self.transfers_df[self.transfers_df['player_id'] == player_id]
        
        if player_transfers.empty:
            return None
        
        # Geçmiş transferler
        past_transfers = player_transfers[player_transfers['upcoming'] == False]
        
        # Fee olan transferler
        paid_transfers = past_transfers.dropna(subset=['fee'])
        paid_transfers = paid_transfers[paid_transfers['fee'] > 0]
        
        analysis = {
            'total_transfers': len(past_transfers),
            'paid_transfers': len(paid_transfers),
            'free_transfers': len(past_transfers[past_transfers['fee'] == 0]),
            'total_transfer_value': paid_transfers['fee'].sum() if not paid_transfers.empty else 0,
            'average_transfer_fee': paid_transfers['fee'].mean() if not paid_transfers.empty else 0,
            'highest_transfer_fee': paid_transfers['fee'].max() if not paid_transfers.empty else 0,
            'transfer_frequency': len(past_transfers) / max(1, len(past_transfers['season'].unique())),
            'clubs_played': past_transfers['to_club_name'].unique().tolist(),
            'transfer_timeline': past_transfers[['date', 'from_club_name', 'to_club_name', 'fee']].to_dict('records')
        }
        
        return analysis
    
    def find_transfer_bargains(self, market_value_threshold: float = 0.5) -> pd.DataFrame:
        """
        Transfer pazarlıklarını bulur (fee < market_value * threshold)
        
        Args:
            market_value_threshold: Market değerinin hangi oranı altındaki transferler
            
        Returns:
            Pazarlık transferleri DataFrame'i
        """
        # Fee ve market value olan transferler
        valid_transfers = self.transfers_df.dropna(subset=['fee', 'market_value'])
        valid_transfers = valid_transfers[
            (valid_transfers['fee'] > 0) & 
            (valid_transfers['market_value'] > 0) &
            (valid_transfers['upcoming'] == False)
        ]
        
        # Pazarlık kriterine uyan transferler
        bargains = valid_transfers[
            valid_transfers['fee'] < (valid_transfers['market_value'] * market_value_threshold)
        ]
        
        # Pazarlık oranını hesapla
        bargains = bargains.copy()
        bargains['bargain_ratio'] = bargains['fee'] / bargains['market_value']
        bargains['savings'] = bargains['market_value'] - bargains['fee']
        
        return bargains.sort_values('bargain_ratio').reset_index(drop=True)
    
    def analyze_transfer_inflation(self) -> pd.DataFrame:
        """
        Transfer enflasyonunu analiz eder
        
        Returns:
            Transfer enflasyon analizi
        """
        yearly_trends = self.analyze_transfer_trends_by_year()
        
        if len(yearly_trends) < 2:
            return yearly_trends
        
        # Yıllık büyüme oranlarını hesapla
        yearly_trends = yearly_trends.sort_values('year')
        yearly_trends['avg_fee_growth'] = yearly_trends['avg_fee'].pct_change() * 100
        yearly_trends['volume_growth'] = yearly_trends['total_volume'].pct_change() * 100
        yearly_trends['transfer_count_growth'] = yearly_trends['transfer_count'].pct_change() * 100
        
        return yearly_trends


if __name__ == "__main__":
    # Test kodları buraya eklenebilir
    pass
