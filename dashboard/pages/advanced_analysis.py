"""
📊 Advanced Player Analysis Page
===============================

Detailed player analysis, comparisons and statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def render_advanced_player_analysis(players_df: pd.DataFrame):
    """Advanced player analysis"""
    
    if players_df.empty:
        st.warning("⚠️ Player data not found for analysis")
        return
    
    st.title("📊 Advanced Player Analysis")
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_players = len(players_df)
        st.metric("👥 Total Players", total_players)
    
    with col2:
        avg_age = players_df['age'].mean() if players_df['age'].notna().any() else 0
        st.metric("🎂 Average Age", f"{avg_age:.1f}")
    
    with col3:
        avg_value = players_df['current_market_value'].mean() if players_df['current_market_value'].notna().any() else 0
        st.metric("💰 Average Value", f"€{avg_value:.1f}M")
    
    with col4:
        max_value = players_df['current_market_value'].max() if players_df['current_market_value'].notna().any() else 0
        st.metric("🏆 Maximum Value", f"€{max_value:.1f}M")
    
    with col5:
        unique_clubs = players_df['current_club'].nunique() if players_df['current_club'].notna().any() else 0
        st.metric("🏟️ Total Clubs", unique_clubs)
    
    st.markdown("---")
    
    # Analysis options
    analysis_type = st.selectbox(
        "📈 Select Analysis Type:",
        [
            "Market Value Detail Analysis",
            "Age vs Performance Analysis", 
            "Position-based Comparison",
            "Club Analysis",
            "Nationality Distribution",
            "Statistical Correlations"
        ]
    )
    
    if analysis_type == "Market Value Detail Analysis":
        render_market_value_analysis(players_df)
    
    elif analysis_type == "Age vs Performance Analysis":
        render_age_performance_analysis(players_df)
    
    elif analysis_type == "Position-based Comparison":
        render_position_comparison(players_df)
    
    elif analysis_type == "Club Analysis":
        render_club_analysis(players_df)
    
    elif analysis_type == "Nationality Distribution":
        render_nationality_analysis(players_df)
    
    elif analysis_type == "Statistical Correlations":
        render_correlation_analysis(players_df)

def render_market_value_analysis(players_df: pd.DataFrame):
    """Market value detail analysis"""
    st.markdown("## 💰 Market Value Detail Analysis")
    
    if not players_df['current_market_value'].notna().any():
        st.warning("⚠️ Market value data not found")
        return
    
    mv_data = players_df['current_market_value'].dropna()
    
    # Statistical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Statistical Summary")
        
        stats_data = {
            'Metrik': ['Ortalama', 'Medyan', 'Standart Sapma', 'Minimum', 'Maksimum', 'Q1 (25%)', 'Q3 (75%)'],
            'Değer (€M)': [
                f"{mv_data.mean():.2f}",
                f"{mv_data.median():.2f}",
                f"{mv_data.std():.2f}",
                f"{mv_data.min():.2f}",
                f"{mv_data.max():.2f}",
                f"{mv_data.quantile(0.25):.2f}",
                f"{mv_data.quantile(0.75):.2f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Dağılım Grafiği")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=mv_data,
            nbinsx=20,
            name="Market Değeri",
            marker_color='skyblue',
            opacity=0.7
        ))
        
        fig.add_vline(x=mv_data.mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"Ortalama: €{mv_data.mean():.1f}M")
        fig.add_vline(x=mv_data.median(), line_dash="dash", line_color="green",
                     annotation_text=f"Medyan: €{mv_data.median():.1f}M")
        
        fig.update_layout(
            title="Market Değeri Dağılımı",
            xaxis_title="Market Değeri (€M)",
            yaxis_title="Oyuncu Sayısı",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Market değeri kategorileri
    st.markdown("### 🏷️ Market Değeri Kategorileri")
    
    def categorize_market_value(value):
        if pd.isna(value):
            return "Veri Yok"
        elif value < 1:
            return "Düşük (< €1M)"
        elif value < 10:
            return "Orta (€1-10M)"
        elif value < 50:
            return "Yüksek (€10-50M)"
        else:
            return "Çok Yüksek (> €50M)"
    
    players_df['market_value_category'] = players_df['current_market_value'].apply(categorize_market_value)
    category_counts = players_df['market_value_category'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Market Değeri Kategorileri",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Kategori Bazlı Oyuncu Sayısı",
            labels={'x': 'Kategori', 'y': 'Oyuncu Sayısı'},
            color=category_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # En değerli oyuncular
    st.markdown("### 🌟 En Değerli Oyuncular")
    
    top_10 = players_df.nlargest(10, 'current_market_value')
    
    fig = px.bar(
        top_10,
        x='current_market_value',
        y='name',
        orientation='h',
        title="En Değerli 10 Oyuncu",
        labels={'current_market_value': 'Market Değeri (€M)', 'name': 'Oyuncu'},
        color='current_market_value',
        color_continuous_scale='plasma',
        hover_data=['age', 'position', 'current_club']
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def render_age_performance_analysis(players_df: pd.DataFrame):
    """Yaş vs performans analizi"""
    st.markdown("## 🎂 Yaş vs Performans Analizi")
    
    if not players_df['age'].notna().any():
        st.warning("⚠️ Yaş verisi bulunamadı")
        return
    
    # Yaş grupları oluştur
    def age_group(age):
        if pd.isna(age):
            return "Veri Yok"
        elif age < 20:
            return "Genç (< 20)"
        elif age < 25:
            return "Gelişen (20-24)"
        elif age < 30:
            return "Olgun (25-29)"
        elif age < 35:
            return "Deneyimli (30-34)"
        else:
            return "Veteran (35+)"
    
    players_df['age_group'] = players_df['age'].apply(age_group)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Yaş grubu dağılımı
        age_counts = players_df['age_group'].value_counts()
        
        fig = px.pie(
            values=age_counts.values,
            names=age_counts.index,
            title="Yaş Grubu Dağılımı",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Yaş vs Market Değeri
        if players_df['current_market_value'].notna().any():
            fig = px.scatter(
                players_df,
                x='age',
                y='current_market_value',
                hover_name='name',
                hover_data=['position', 'current_club'],
                title="Yaş vs Market Değeri İlişkisi",
                labels={'age': 'Yaş', 'current_market_value': 'Market Değeri (€M)'},
                trendline="ols",
                color='position' if players_df['position'].notna().any() else None
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Yaş gruplarına göre istatistikler
    if players_df['current_market_value'].notna().any():
        st.markdown("### 📊 Yaş Gruplarına Göre Market Değeri Analizi")
        
        age_stats = players_df.groupby('age_group')['current_market_value'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        age_stats.columns = ['Oyuncu Sayısı', 'Ortalama (€M)', 'Medyan (€M)', 'Std Sapma', 'Min (€M)', 'Max (€M)']
        st.dataframe(age_stats, use_container_width=True)
        
        # Box plot
        fig = px.box(
            players_df,
            x='age_group',
            y='current_market_value',
            title="Yaş Gruplarına Göre Market Değeri Dağılımı",
            labels={'age_group': 'Yaş Grubu', 'current_market_value': 'Market Değeri (€M)'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_position_comparison(players_df: pd.DataFrame):
    """Pozisyon bazlı karşılaştırma"""
    st.markdown("## ⚽ Pozisyon Bazlı Karşılaştırma")
    
    if not players_df['position'].notna().any():
        st.warning("⚠️ Pozisyon verisi bulunamadı")
        return
    
    position_stats = players_df.groupby('position').agg({
        'current_market_value': ['count', 'mean', 'median', 'std'],
        'age': ['mean', 'min', 'max']
    }).round(2)
    
    position_stats.columns = [
        'Oyuncu Sayısı', 'Ort. Market Değeri', 'Medyan Market Değeri', 'Std Sapma',
        'Ort. Yaş', 'Min Yaş', 'Max Yaş'
    ]
    
    st.markdown("### 📊 Pozisyon İstatistikleri")
    st.dataframe(position_stats, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pozisyona göre oyuncu sayısı
        position_counts = players_df['position'].value_counts()
        
        fig = px.bar(
            x=position_counts.index,
            y=position_counts.values,
            title="Pozisyona Göre Oyuncu Sayısı",
            labels={'x': 'Pozisyon', 'y': 'Oyuncu Sayısı'},
            color=position_counts.values,
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pozisyona göre ortalama market değeri
        if players_df['current_market_value'].notna().any():
            pos_avg_value = players_df.groupby('position')['current_market_value'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=pos_avg_value.index,
                y=pos_avg_value.values,
                title="Pozisyona Göre Ortalama Market Değeri",
                labels={'x': 'Pozisyon', 'y': 'Ortalama Market Değeri (€M)'},
                color=pos_avg_value.values,
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_club_analysis(players_df: pd.DataFrame):
    """Kulüp analizi"""
    st.markdown("## 🏟️ Kulüp Analizi")
    
    if not players_df['current_club'].notna().any():
        st.warning("⚠️ Kulüp verisi bulunamadı")
        return
    
    # En çok oyuncuya sahip kulüpler
    club_counts = players_df['current_club'].value_counts().head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 En Çok Oyuncuya Sahip Kulüpler")
        
        fig = px.bar(
            x=club_counts.values,
            y=club_counts.index,
            orientation='h',
            title="Kulüp Bazlı Oyuncu Sayısı (Top 15)",
            labels={'x': 'Oyuncu Sayısı', 'y': 'Kulüp'},
            color=club_counts.values,
            color_continuous_scale='greens'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if players_df['current_market_value'].notna().any():
            st.markdown("### 💰 En Değerli Kadrolara Sahip Kulüpler")
            
            club_value = players_df.groupby('current_club')['current_market_value'].sum().sort_values(ascending=False).head(15)
            
            fig = px.bar(
                x=club_value.values,
                y=club_value.index,
                orientation='h',
                title="Kulüp Toplam Kadro Değeri (Top 15)",
                labels={'x': 'Toplam Kadro Değeri (€M)', 'y': 'Kulüp'},
                color=club_value.values,
                color_continuous_scale='oranges'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def render_nationality_analysis(players_df: pd.DataFrame):
    """Uyruk dağılımı analizi"""
    st.markdown("## 🌍 Uyruk Dağılımı Analizi")
    
    if not players_df['nationality'].notna().any():
        st.warning("⚠️ Uyruk verisi bulunamadı")
        return
    
    nationality_counts = players_df['nationality'].value_counts().head(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏁 En Yaygın Uyruklar")
        
        fig = px.bar(
            x=nationality_counts.index,
            y=nationality_counts.values,
            title="En Yaygın 20 Uyruk",
            labels={'x': 'Uyruk', 'y': 'Oyuncu Sayısı'},
            color=nationality_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Treemap görünümü
        fig = px.treemap(
            names=nationality_counts.index,
            values=nationality_counts.values,
            title="Uyruk Dağılımı (Treemap)"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_correlation_analysis(players_df: pd.DataFrame):
    """İstatistiksel korelasyon analizi"""
    st.markdown("## 📈 İstatistiksel Korelasyonlar")
    
    # Numerik kolonları seç
    numeric_cols = players_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Korelasyon analizi için yeterli numerik veri yok")
        return
    
    # Korelasyon matrisi
    corr_matrix = players_df[numeric_cols].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔥 Korelasyon Haritası")
        
        fig = px.imshow(
            corr_matrix,
            title="Değişkenler Arası Korelasyon",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Güçlü Korelasyonlar")
        
        # Güçlü korelasyonları bul
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # 0.3'ten büyük korelasyonlar
                    strong_corrs.append({
                        'Değişken 1': corr_matrix.columns[i],
                        'Değişken 2': corr_matrix.columns[j],
                        'Korelasyon': f"{corr_val:.3f}",
                        'Güç': 'Güçlü' if abs(corr_val) > 0.7 else 'Orta'
                    })
        
        if strong_corrs:
            corr_df = pd.DataFrame(strong_corrs)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
        else:
            st.info("Güçlü korelasyon bulunamadı (|r| > 0.3)")
    
    # Scatter plot matrisi
    if len(numeric_cols) <= 5:  # Sadece 5 veya daha az değişken varsa
        st.markdown("### 📊 Scatter Plot Matrisi")
        
        fig = px.scatter_matrix(
            players_df[numeric_cols].dropna(),
            title="Değişkenler Arası İlişkiler"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
