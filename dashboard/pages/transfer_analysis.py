"""
🔄 Transfer Analysis Page
========================

Detailed transfer analysis and trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def render_transfer_analysis(transfers_df: pd.DataFrame):
    """Transfer analysis main function"""
    
    st.title("🔄 Transfer Analysis Dashboard")
    
    if transfers_df.empty:
        st.warning("⚠️ Transfer data not found")
        st.info("""
        📋 For transfer data, check the following:
        - Has data been collected from API?
        - Transfer verileri mevcut mu?
        - Veri formatı doğru mu?
        """)
        return
    
    # Transfer verisi temizleme
    transfers_clean = transfers_df.copy()
    
    # Tarihleri datetime'a çevir
    if 'date' in transfers_clean.columns:
        transfers_clean['date'] = pd.to_datetime(transfers_clean['date'], errors='coerce')
        transfers_clean['year'] = transfers_clean['date'].dt.year
        transfers_clean['month'] = transfers_clean['date'].dt.month
    
    # Ücret verilerini temizle
    if 'fee' in transfers_clean.columns:
        transfers_clean['fee'] = pd.to_numeric(transfers_clean['fee'], errors='coerce').fillna(0)
    
    # Ana metrikler
    render_transfer_metrics(transfers_clean)
    
    st.markdown("---")
    
    # Analiz seçenekleri
    analysis_type = st.selectbox(
        "📈 Transfer Analizi Türü:",
        [
            "Genel Transfer Trendleri",
            "Transfer Ücretleri Analizi",
            "Kulüp Transfer Aktiviteleri",
            "Sezonluk Transfer Dağılımı",
            "Pozisyon Bazlı Transferler",
            "Transfer Ağları"
        ]
    )
    
    if analysis_type == "Genel Transfer Trendleri":
        render_transfer_trends(transfers_clean)
    
    elif analysis_type == "Transfer Ücretleri Analizi":
        render_fee_analysis(transfers_clean)
    
    elif analysis_type == "Kulüp Transfer Aktiviteleri":
        render_club_activity(transfers_clean)
    
    elif analysis_type == "Sezonluk Transfer Dağılımı":
        render_seasonal_analysis(transfers_clean)
    
    elif analysis_type == "Pozisyon Bazlı Transferler":
        render_position_transfers(transfers_clean)
    
    elif analysis_type == "Transfer Ağları":
        render_transfer_networks(transfers_clean)

def render_transfer_metrics(transfers_df: pd.DataFrame):
    """Transfer ana metrikleri"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_transfers = len(transfers_df)
        st.metric("🔄 Toplam Transfer", total_transfers)
    
    with col2:
        paid_transfers = len(transfers_df[transfers_df['fee'] > 0])
        st.metric("💰 Ücretli Transfer", paid_transfers)
    
    with col3:
        free_transfers = len(transfers_df[transfers_df['fee'] == 0])
        st.metric("🆓 Ücretsiz Transfer", free_transfers)
    
    with col4:
        total_fee = transfers_df['fee'].sum()
        st.metric("💸 Toplam Hacim", f"€{total_fee:.1f}M")
    
    with col5:
        avg_fee = transfers_df[transfers_df['fee'] > 0]['fee'].mean() if len(transfers_df[transfers_df['fee'] > 0]) > 0 else 0
        st.metric("📊 Ortalama Ücret", f"€{avg_fee:.1f}M")

def render_transfer_trends(transfers_df: pd.DataFrame):
    """Genel transfer trendleri"""
    st.markdown("## 📈 Genel Transfer Trendleri")
    
    if 'year' not in transfers_df.columns:
        st.warning("⚠️ Tarih verisi bulunamadı")
        return
    
    # Yıllık transfer sayıları
    yearly_transfers = transfers_df.groupby('year').agg({
        'fee': ['count', 'sum', 'mean']
    }).round(2)
    
    yearly_transfers.columns = ['Transfer Sayısı', 'Toplam Hacim (€M)', 'Ortalama Ücret (€M)']
    yearly_transfers = yearly_transfers.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Yıllık transfer sayısı trendi
        fig = px.line(
            yearly_transfers,
            x='year',
            y='Transfer Sayısı',
            title="Yıllık Transfer Sayısı Trendi",
            markers=True,
            line_shape='linear'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Yıllık transfer hacmi trendi
        fig = px.bar(
            yearly_transfers,
            x='year',
            y='Toplam Hacim (€M)',
            title="Yıllık Transfer Hacmi (€M)",
            color='Toplam Hacim (€M)',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Yıllık istatistikler tablosu
    st.markdown("### 📊 Yıllık Transfer İstatistikleri")
    st.dataframe(yearly_transfers, use_container_width=True, hide_index=True)
    
    # Aylık dağılım (eğer mevcut yıl verisi varsa)
    if 'month' in transfers_df.columns:
        st.markdown("### 📅 Aylık Transfer Dağılımı")
        
        current_year = datetime.now().year
        recent_transfers = transfers_df[transfers_df['year'] >= current_year - 2]  # Son 2-3 yıl
        
        if not recent_transfers.empty:
            monthly_transfers = recent_transfers.groupby(['year', 'month']).size().reset_index(name='transfer_count')
            monthly_transfers['year_month'] = monthly_transfers['year'].astype(str) + '-' + monthly_transfers['month'].astype(str).str.zfill(2)
            
            fig = px.line(
                monthly_transfers,
                x='year_month',
                y='transfer_count',
                color='year',
                title="Aylık Transfer Sayısı Trendi (Son Yıllar)",
                markers=True
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

def render_fee_analysis(transfers_df: pd.DataFrame):
    """Transfer ücretleri analizi"""
    st.markdown("## 💰 Transfer Ücretleri Analizi")
    
    paid_transfers = transfers_df[transfers_df['fee'] > 0]
    
    if paid_transfers.empty:
        st.warning("⚠️ Ücretli transfer verisi bulunamadı")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Ücret Dağılımı")
        
        # Transfer ücret kategorileri
        def categorize_fee(fee):
            if fee == 0:
                return "Ücretsiz"
            elif fee < 1:
                return "Düşük (< €1M)"
            elif fee < 10:
                return "Orta (€1-10M)"
            elif fee < 50:
                return "Yüksek (€10-50M)"
            else:
                return "Çok Yüksek (> €50M)"
        
        transfers_df['fee_category'] = transfers_df['fee'].apply(categorize_fee)
        fee_counts = transfers_df['fee_category'].value_counts()
        
        fig = px.pie(
            values=fee_counts.values,
            names=fee_counts.index,
            title="Transfer Ücret Kategorileri",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Ücret Histogramı")
        
        fig = px.histogram(
            paid_transfers,
            x='fee',
            nbins=20,
            title="Transfer Ücretleri Dağılımı",
            labels={'fee': 'Transfer Ücreti (€M)', 'count': 'Transfer Sayısı'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # En pahalı transferler
    st.markdown("### 💎 En Pahalı Transferler")
    
    top_transfers = paid_transfers.nlargest(10, 'fee')
    
    if not top_transfers.empty:
        # Gerekli kolonları kontrol et
        display_cols = ['player_name', 'fee', 'from_club_name', 'to_club_name', 'date']
        available_cols = [col for col in display_cols if col in top_transfers.columns]
        
        if 'player' in top_transfers.columns and 'player_name' not in available_cols:
            available_cols = ['player'] + [col for col in available_cols if col != 'player_name']
        
        if available_cols:
            st.dataframe(
                top_transfers[available_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(top_transfers.head(10), use_container_width=True)
    
    # Ücret istatistikleri
    st.markdown("### 📊 Transfer Ücret İstatistikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ortalama Ücret", f"€{paid_transfers['fee'].mean():.2f}M")
    
    with col2:
        st.metric("Medyan Ücret", f"€{paid_transfers['fee'].median():.2f}M")
    
    with col3:
        st.metric("En Yüksek Ücret", f"€{paid_transfers['fee'].max():.2f}M")
    
    with col4:
        st.metric("Toplam Hacim", f"€{paid_transfers['fee'].sum():.1f}M")

def render_club_activity(transfers_df: pd.DataFrame):
    """Kulüp transfer aktiviteleri"""
    st.markdown("## 🏟️ Kulüp Transfer Aktiviteleri")
    
    # Alıcı kulüpler (en aktif alıcılar)
    if 'to_club_name' in transfers_df.columns:
        st.markdown("### 📥 En Aktif Alıcı Kulüpler")
        
        buyer_activity = transfers_df.groupby('to_club_name').agg({
            'fee': ['count', 'sum', 'mean']
        }).round(2)
        
        buyer_activity.columns = ['Transfer Sayısı', 'Toplam Harcama (€M)', 'Ortalama Ücret (€M)']
        buyer_activity = buyer_activity.sort_values('Toplam Harcama (€M)', ascending=False).head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=buyer_activity.index,
                y=buyer_activity['Transfer Sayısı'],
                title="En Aktif Alıcı Kulüpler (Transfer Sayısı)",
                labels={'x': 'Kulüp', 'y': 'Transfer Sayısı'},
                color=buyer_activity['Transfer Sayısı'],
                color_continuous_scale='blues'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=buyer_activity.index,
                y=buyer_activity['Toplam Harcama (€M)'],
                title="En Çok Harcama Yapan Kulüpler",
                labels={'x': 'Kulüp', 'y': 'Toplam Harcama (€M)'},
                color=buyer_activity['Toplam Harcama (€M)'],
                color_continuous_scale='reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(buyer_activity, use_container_width=True)
    
    # Satıcı kulüpler
    if 'from_club_name' in transfers_df.columns:
        st.markdown("### 📤 En Aktif Satıcı Kulüpler")
        
        seller_activity = transfers_df.groupby('from_club_name').agg({
            'fee': ['count', 'sum', 'mean']
        }).round(2)
        
        seller_activity.columns = ['Transfer Sayısı', 'Toplam Gelir (€M)', 'Ortalama Ücret (€M)']
        seller_activity = seller_activity.sort_values('Toplam Gelir (€M)', ascending=False).head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=seller_activity.index,
                y=seller_activity['Transfer Sayısı'],
                title="En Aktif Satıcı Kulüpler (Transfer Sayısı)",
                labels={'x': 'Kulüp', 'y': 'Transfer Sayısı'},
                color=seller_activity['Transfer Sayısı'],
                color_continuous_scale='greens'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=seller_activity.index,
                y=seller_activity['Toplam Gelir (€M)'],
                title="En Çok Gelir Elde Eden Kulüpler",
                labels={'x': 'Kulüp', 'y': 'Toplam Gelir (€M)'},
                color=seller_activity['Toplam Gelir (€M)'],
                color_continuous_scale='oranges'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

def render_seasonal_analysis(transfers_df: pd.DataFrame):
    """Sezonluk transfer dağılımı"""
    st.markdown("## 📅 Sezonluk Transfer Dağılımı")
    
    if 'month' not in transfers_df.columns:
        st.warning("⚠️ Tarih verisi bulunamadı")
        return
    
    # Aylık transfer dağılımı
    monthly_dist = transfers_df['month'].value_counts().sort_index()
    
    month_names = {
        1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan',
        5: 'Mayıs', 6: 'Haziran', 7: 'Temmuz', 8: 'Ağustos',
        9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=[month_names[m] for m in monthly_dist.index],
            y=monthly_dist.values,
            title="Aylık Transfer Dağılımı",
            labels={'x': 'Ay', 'y': 'Transfer Sayısı'},
            color=monthly_dist.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Transfer dönemleri
        def get_transfer_window(month):
            if month in [1, 2]:
                return "Kış Transfer Dönemi"
            elif month in [6, 7, 8, 9]:
                return "Yaz Transfer Dönemi"
            else:
                return "Sezon İçi"
        
        transfers_df['transfer_window'] = transfers_df['month'].apply(get_transfer_window)
        window_dist = transfers_df['transfer_window'].value_counts()
        
        fig = px.pie(
            values=window_dist.values,
            names=window_dist.index,
            title="Transfer Dönemlerine Göre Dağılım",
            color_discrete_sequence=['#FF9999', '#66B2FF', '#99FF99']
        )
        st.plotly_chart(fig, use_container_width=True)

def render_position_transfers(transfers_df: pd.DataFrame):
    """Pozisyon bazlı transferler"""
    st.markdown("## ⚽ Pozisyon Bazlı Transfer Analizi")
    
    if 'position' not in transfers_df.columns:
        st.warning("⚠️ Pozisyon verisi bulunamadı")
        return
    
    position_stats = transfers_df.groupby('position').agg({
        'fee': ['count', 'sum', 'mean']
    }).round(2)
    
    position_stats.columns = ['Transfer Sayısı', 'Toplam Hacim (€M)', 'Ortalama Ücret (€M)']
    position_stats = position_stats.sort_values('Toplam Hacim (€M)', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=position_stats.index,
            y=position_stats['Transfer Sayısı'],
            title="Pozisyona Göre Transfer Sayısı",
            labels={'x': 'Pozisyon', 'y': 'Transfer Sayısı'},
            color=position_stats['Transfer Sayısı'],
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=position_stats.index,
            y=position_stats['Ortalama Ücret (€M)'],
            title="Pozisyona Göre Ortalama Transfer Ücreti",
            labels={'x': 'Pozisyon', 'y': 'Ortalama Ücret (€M)'},
            color=position_stats['Ortalama Ücret (€M)'],
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(position_stats, use_container_width=True)

def render_transfer_networks(transfers_df: pd.DataFrame):
    """Transfer ağları (kulüpler arası)"""
    st.markdown("## 🌐 Transfer Ağları")
    
    if 'from_club_name' not in transfers_df.columns or 'to_club_name' not in transfers_df.columns:
        st.warning("⚠️ Kulüp transfer verisi bulunamadı")
        return
    
    st.info("🔧 Transfer ağı görselleştirmesi geliştirme aşamasında...")
    
    # Basit transfer akışı tablosu
    transfer_flows = transfers_df.groupby(['from_club_name', 'to_club_name']).agg({
        'fee': ['count', 'sum']
    }).round(2)
    
    transfer_flows.columns = ['Transfer Sayısı', 'Toplam Ücret (€M)']
    transfer_flows = transfer_flows.sort_values('Toplam Ücret (€M)', ascending=False).head(20)
    
    st.markdown("### 🔄 En Büyük Transfer Akışları")
    st.dataframe(transfer_flows, use_container_width=True)
