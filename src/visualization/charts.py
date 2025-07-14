"""
Visualization Module
Tools for visualizing Transfermarkt data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# Matplotlib font settings
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Seaborn style settings
sns.set_style("whitegrid")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class PlayerVisualizer:
    """
    Class for visualizing player data
    """
    
    def __init__(self, players_df: pd.DataFrame):
        """
        PlayerVisualizer'ı başlatır
        
        Args:
            players_df: Oyuncu DataFrame'i
        """
        self.players_df = players_df
    
    def plot_market_value_distribution(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Market değeri dağılımını görselleştirir
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Oyuncu Market Değeri Dağılımı', fontsize=16, fontweight='bold')
        
        # Market değeri olan oyuncuları al
        mv_data = self.players_df.dropna(subset=['current_market_value'])
        
        if mv_data.empty:
            logger.warning("Market değeri verisi bulunamadı")
            return fig
        
        # 1. Histogram
        axes[0, 0].hist(mv_data['current_market_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Market Değeri Histogramı')
        axes[0, 0].set_xlabel('Market Değeri (€M)')
        axes[0, 0].set_ylabel('Oyuncu Sayısı')
        
        # 2. Box plot
        axes[0, 1].boxplot(mv_data['current_market_value'], patch_artist=True, 
                          boxprops=dict(facecolor='lightcoral'))
        axes[0, 1].set_title('Market Değeri Box Plot')
        axes[0, 1].set_ylabel('Market Değeri (€M)')
        
        # 3. Top 10 oyuncu
        top_players = mv_data.nlargest(10, 'current_market_value')
        bars = axes[1, 0].barh(range(len(top_players)), top_players['current_market_value'], 
                              color='lightgreen')
        axes[1, 0].set_yticks(range(len(top_players)))
        axes[1, 0].set_yticklabels(top_players['name'], fontsize=8)
        axes[1, 0].set_title('En Değerli 10 Oyuncu')
        axes[1, 0].set_xlabel('Market Değeri (€M)')
        
        # Değerleri barların üzerine ekle
        for i, (idx, row) in enumerate(top_players.iterrows()):
            axes[1, 0].text(row['current_market_value'] + 1, i, f'€{row["current_market_value"]:.1f}M', 
                           va='center', fontsize=7)
        
        # 4. Pozisyona göre ortalama market değeri
        if 'position' in mv_data.columns:
            pos_avg = mv_data.groupby('position')['current_market_value'].mean().sort_values(ascending=False)
            axes[1, 1].bar(range(len(pos_avg)), pos_avg.values, color='orange', alpha=0.7)
            axes[1, 1].set_xticks(range(len(pos_avg)))
            axes[1, 1].set_xticklabels(pos_avg.index, rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_title('Pozisyona Göre Ort. Market Değeri')
            axes[1, 1].set_ylabel('Ortalama Market Değeri (€M)')
        
        plt.tight_layout()
        return fig
    
    def plot_age_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Yaş dağılımını görselleştirir
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Oyuncu Yaş Dağılımı', fontsize=16, fontweight='bold')
        
        age_data = self.players_df.dropna(subset=['age'])
        
        if age_data.empty:
            logger.warning("Yaş verisi bulunamadı")
            return fig
        
        # 1. Histogram
        axes[0].hist(age_data['age'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0].set_title('Yaş Histogramı')
        axes[0].set_xlabel('Yaş')
        axes[0].set_ylabel('Oyuncu Sayısı')
        axes[0].axvline(age_data['age'].mean(), color='red', linestyle='--', 
                       label=f'Ortalama: {age_data["age"].mean():.1f}')
        axes[0].legend()
        
        # 2. Yaş grupları pasta grafiği
        age_groups = pd.cut(age_data['age'], bins=[0, 23, 29, 35, 50], 
                           labels=['Genç (≤23)', 'Asal (24-29)', 'Tecrübeli (30-35)', 'Veteran (35+)'])
        age_group_counts = age_groups.value_counts()
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        wedges, texts, autotexts = axes[1].pie(age_group_counts.values, labels=age_group_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Yaş Grupları Dağılımı')
        
        plt.tight_layout()
        return fig
    
    def plot_position_analysis(self, figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Pozisyon analizini görselleştirir
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Pozisyon Analizi', fontsize=16, fontweight='bold')
        
        pos_data = self.players_df.dropna(subset=['position'])
        
        if pos_data.empty:
            logger.warning("Pozisyon verisi bulunamadı")
            return fig
        
        # 1. Pozisyon dağılımı
        position_counts = pos_data['position'].value_counts()
        axes[0, 0].bar(range(len(position_counts)), position_counts.values, color='lightcoral')
        axes[0, 0].set_xticks(range(len(position_counts)))
        axes[0, 0].set_xticklabels(position_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Pozisyon Dağılımı')
        axes[0, 0].set_ylabel('Oyuncu Sayısı')
        
        # 2. Pozisyona göre ortalama yaş
        if 'age' in pos_data.columns:
            pos_age = pos_data.groupby('position')['age'].mean().sort_values(ascending=False)
            axes[0, 1].barh(range(len(pos_age)), pos_age.values, color='lightgreen')
            axes[0, 1].set_yticks(range(len(pos_age)))
            axes[0, 1].set_yticklabels(pos_age.index)
            axes[0, 1].set_title('Pozisyona Göre Ortalama Yaş')
            axes[0, 1].set_xlabel('Ortalama Yaş')
        
        # 3. Pozisyona göre market değeri (eğer varsa)
        if 'current_market_value' in pos_data.columns:
            mv_pos_data = pos_data.dropna(subset=['current_market_value'])
            if not mv_pos_data.empty:
                pos_mv = mv_pos_data.groupby('position')['current_market_value'].mean().sort_values(ascending=False)
                axes[1, 0].bar(range(len(pos_mv)), pos_mv.values, color='gold')
                axes[1, 0].set_xticks(range(len(pos_mv)))
                axes[1, 0].set_xticklabels(pos_mv.index, rotation=45, ha='right')
                axes[1, 0].set_title('Pozisyona Göre Ort. Market Değeri')
                axes[1, 0].set_ylabel('Market Değeri (€M)')
        
        # 4. Uyruk dağılımı (top 10)
        if 'nationality' in pos_data.columns:
            nationality_counts = pos_data['nationality'].value_counts().head(10)
            axes[1, 1].barh(range(len(nationality_counts)), nationality_counts.values, color='orange')
            axes[1, 1].set_yticks(range(len(nationality_counts)))
            axes[1, 1].set_yticklabels(nationality_counts.index, fontsize=8)
            axes[1, 1].set_title('En Yaygın 10 Uyruk')
            axes[1, 1].set_xlabel('Oyuncu Sayısı')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_market_value_plot(self) -> go.Figure:
        """
        İnteraktif market değeri grafiği oluşturur
        
        Returns:
            Plotly Figure
        """
        mv_data = self.players_df.dropna(subset=['current_market_value'])
        
        if mv_data.empty:
            logger.warning("Market değeri verisi bulunamadı")
            return go.Figure()
        
        # Yaş ve market değeri scatter plot
        fig = px.scatter(
            mv_data, 
            x='age', 
            y='current_market_value',
            hover_data=['name', 'position', 'nationality', 'current_club'],
            color='position' if 'position' in mv_data.columns else None,
            size='current_market_value',
            title='Oyuncu Yaş vs Market Değeri',
            labels={
                'age': 'Yaş',
                'current_market_value': 'Market Değeri (€M)',
                'position': 'Pozisyon'
            }
        )
        
        fig.update_layout(
            title_font_size=16,
            showlegend=True,
            height=600
        )
        
        return fig
    
    def create_player_comparison_radar(self, player_ids: List[str], 
                                     metrics: List[str] = None) -> go.Figure:
        """
        Oyuncu karşılaştırma radar grafiği oluşturur
        
        Args:
            player_ids: Karşılaştırılacak oyuncu ID'leri
            metrics: Karşılaştırılacak metrikler
            
        Returns:
            Plotly Figure
        """
        if metrics is None:
            metrics = ['current_market_value', 'age']
        
        # Seçilen oyuncuları al
        selected_players = self.players_df[self.players_df['player_id'].isin(player_ids)]
        
        if selected_players.empty:
            logger.warning("Seçilen oyuncular bulunamadı")
            return go.Figure()
        
        fig = go.Figure()
        
        for _, player in selected_players.iterrows():
            values = []
            for metric in metrics:
                if metric in player and pd.notna(player[metric]):
                    values.append(player[metric])
                else:
                    values.append(0)
            
            # Değerleri normalize et (0-100 arası)
            if values:
                max_vals = [selected_players[m].max() for m in metrics if m in selected_players.columns]
                normalized_values = [v/max_v*100 if max_v > 0 else 0 for v, max_v in zip(values, max_vals)]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values + [normalized_values[0]],  # Kapatmak için ilk değeri tekrarla
                    theta=metrics + [metrics[0]],  # Kapatmak için ilk metriği tekrarla
                    fill='toself',
                    name=player['name']
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Oyuncu Karşılaştırması"
        )
        
        return fig


class TransferVisualizer:
    """
    Transfer verilerini görselleştiren sınıf
    """
    
    def __init__(self, transfers_df: pd.DataFrame):
        """
        TransferVisualizer'ı başlatır
        
        Args:
            transfers_df: Transfer DataFrame'i
        """
        self.transfers_df = transfers_df
    
    def plot_transfer_trends(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Transfer trendlerini görselleştirir
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Transfer Trendleri Analizi', fontsize=16, fontweight='bold')
        
        # Tarih olan transferleri al
        dated_transfers = self.transfers_df.dropna(subset=['date'])
        dated_transfers = dated_transfers[dated_transfers['upcoming'] == False]
        
        if dated_transfers.empty:
            logger.warning("Tarihli transfer verisi bulunamadı")
            return fig
        
        # Yıl sütunu ekle
        dated_transfers = dated_transfers.copy()
        dated_transfers['year'] = dated_transfers['date'].dt.year
        
        # 1. Yıllık transfer sayısı
        yearly_counts = dated_transfers.groupby('year').size()
        axes[0, 0].plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Yıllık Transfer Sayısı')
        axes[0, 0].set_xlabel('Yıl')
        axes[0, 0].set_ylabel('Transfer Sayısı')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ücretli transferlerin yıllık hacmi
        paid_transfers = dated_transfers.dropna(subset=['fee'])
        paid_transfers = paid_transfers[paid_transfers['fee'] > 0]
        
        if not paid_transfers.empty:
            yearly_volume = paid_transfers.groupby('year')['fee'].sum()
            axes[0, 1].bar(yearly_volume.index, yearly_volume.values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Yıllık Transfer Hacmi (€M)')
            axes[0, 1].set_xlabel('Yıl')
            axes[0, 1].set_ylabel('Toplam Transfer Ücreti (€M)')
            
            # Değerleri barların üzerine ekle
            for x, y in zip(yearly_volume.index, yearly_volume.values):
                axes[0, 1].text(x, y + y*0.02, f'{y:.0f}M', ha='center', va='bottom', fontsize=8)
        
        # 3. Ortalama transfer ücretleri
        if not paid_transfers.empty:
            yearly_avg = paid_transfers.groupby('year')['fee'].mean()
            axes[1, 0].plot(yearly_avg.index, yearly_avg.values, marker='s', color='green', 
                           linewidth=2, markersize=6)
            axes[1, 0].set_title('Ortalama Transfer Ücreti')
            axes[1, 0].set_xlabel('Yıl')
            axes[1, 0].set_ylabel('Ortalama Ücret (€M)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. En pahalı transferler (top 10)
        if not paid_transfers.empty:
            top_transfers = paid_transfers.nlargest(10, 'fee')
            bars = axes[1, 1].barh(range(len(top_transfers)), top_transfers['fee'], color='gold')
            axes[1, 1].set_yticks(range(len(top_transfers)))
            axes[1, 1].set_yticklabels([f"{row['player_name']}\n({row['year']})" 
                                       for _, row in top_transfers.iterrows()], fontsize=7)
            axes[1, 1].set_title('En Pahalı 10 Transfer')
            axes[1, 1].set_xlabel('Transfer Ücreti (€M)')
            
            # Değerleri barların yanına ekle
            for i, (_, row) in enumerate(top_transfers.iterrows()):
                axes[1, 1].text(row['fee'] + row['fee']*0.02, i, f'€{row["fee"]:.1f}M', 
                               va='center', fontsize=7)
        
        plt.tight_layout()
        return fig
    
    def plot_club_activity(self, figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Kulüp aktivitelerini görselleştirir
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Kulüp Transfer Aktiviteleri', fontsize=16, fontweight='bold')
        
        past_transfers = self.transfers_df[self.transfers_df['upcoming'] == False]
        paid_transfers = past_transfers.dropna(subset=['fee'])
        paid_transfers = paid_transfers[paid_transfers['fee'] > 0]
        
        if paid_transfers.empty:
            logger.warning("Ücretli transfer verisi bulunamadı")
            return fig
        
        # 1. En çok harcayan kulüpler
        spending = paid_transfers.groupby('to_club_name')['fee'].sum().nlargest(10)
        axes[0].barh(range(len(spending)), spending.values, color='lightblue')
        axes[0].set_yticks(range(len(spending)))
        axes[0].set_yticklabels(spending.index, fontsize=8)
        axes[0].set_title('En Çok Harcayan 10 Kulüp')
        axes[0].set_xlabel('Toplam Harcama (€M)')
        
        # Değerleri barların yanına ekle
        for i, value in enumerate(spending.values):
            axes[0].text(value + value*0.02, i, f'€{value:.0f}M', va='center', fontsize=7)
        
        # 2. En çok kazanan kulüpler
        earnings = paid_transfers.groupby('from_club_name')['fee'].sum().nlargest(10)
        axes[1].barh(range(len(earnings)), earnings.values, color='lightgreen')
        axes[1].set_yticks(range(len(earnings)))
        axes[1].set_yticklabels(earnings.index, fontsize=8)
        axes[1].set_title('En Çok Kazanan 10 Kulüp')
        axes[1].set_xlabel('Toplam Kazanç (€M)')
        
        # Değerleri barların yanına ekle
        for i, value in enumerate(earnings.values):
            axes[1].text(value + value*0.02, i, f'€{value:.0f}M', va='center', fontsize=7)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_transfer_timeline(self, player_id: str = None) -> go.Figure:
        """
        İnteraktif transfer zaman çizelgesi oluşturur
        
        Args:
            player_id: Belirli bir oyuncu için filtrele (opsiyonel)
            
        Returns:
            Plotly Figure
        """
        data = self.transfers_df.copy()
        
        if player_id:
            data = data[data['player_id'] == player_id]
            title = f"Oyuncu Transfer Geçmişi"
        else:
            title = "Transfer Zaman Çizelgesi"
        
        # Tarih olan ve ücretli transferleri al
        data = data.dropna(subset=['date', 'fee'])
        data = data[data['fee'] > 0]
        data = data[data['upcoming'] == False]
        
        if data.empty:
            logger.warning("Görselleştirmek için yeterli veri yok")
            return go.Figure()
        
        fig = px.scatter(
            data,
            x='date',
            y='fee',
            hover_data=['player_name', 'from_club_name', 'to_club_name'],
            color='player_name' if not player_id else 'to_club_name',
            size='fee',
            title=title,
            labels={
                'date': 'Transfer Tarihi',
                'fee': 'Transfer Ücreti (€M)'
            }
        )
        
        fig.update_layout(
            title_font_size=16,
            showlegend=True,
            height=600
        )
        
        return fig


def save_plot(fig, filename: str, dpi: int = 300) -> None:
    """
    Matplotlib grafiğini dosyaya kaydeder
    
    Args:
        fig: Matplotlib Figure
        filename: Dosya adı
        dpi: Çözünürlük
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    logger.info(f"Grafik {filename} olarak kaydedildi")


def save_plotly_plot(fig, filename: str) -> None:
    """
    Plotly grafiğini HTML dosyasına kaydeder
    
    Args:
        fig: Plotly Figure
        filename: Dosya adı
    """
    fig.write_html(filename)
    logger.info(f"İnteraktif grafik {filename} olarak kaydedildi")


if __name__ == "__main__":
    # Test kodları buraya eklenebilir
    pass
