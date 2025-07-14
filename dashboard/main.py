"""
🏆 Transfermarkt Football Data Analysis Dashboard
===============================================

Modern and interactive web interface for football data analysis.
Professional visualizations and machine learning models.

Author: Data Science Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
import os

# Add src folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Global variables for module availability
CLIENT_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
ANALYSIS_AVAILABLE = False
VISUALIZATION_AVAILABLE = False
MODELS_AVAILABLE = False

try:
    from data_collection import TransfermarktAPIClient, DataCollector, get_popular_players, get_top_leagues
    CLIENT_AVAILABLE = True
    st.success("✅ Data collection module loaded")
except ImportError as e:
    st.warning(f"⚠️ Data collection import error: {e}")

try:
    from analysis import PlayerAnalyzer, TransferAnalyzer, ClubAnalyzer
    ANALYSIS_AVAILABLE = True
    st.success("✅ Analysis module loaded")
except ImportError as e:
    st.warning(f"⚠️ Analysis import error: {e}")

try:
    from visualization import PlayerVisualizer, TransferVisualizer
    VISUALIZATION_AVAILABLE = True
    st.success("✅ Visualization module loaded")
except ImportError as e:
    st.warning(f"⚠️ Visualization import error: {e}")

try:
    from models import MarketValuePredictor, TransferValueAnalyzer
    MODELS_AVAILABLE = True
    st.success("✅ Models module loaded")
except ImportError as e:
    st.warning(f"⚠️ Models import error: {e}")
    MODELS_AVAILABLE = False
except Exception as e:
    st.error(f"❌ Models loading error: {e}")
    if "XGBoost" in str(e):
        st.info("💡 XGBoost issue detected. Will run without XGBoost.")
    MODELS_AVAILABLE = False

# General status
DATA_COLLECTOR_AVAILABLE = CLIENT_AVAILABLE and ANALYSIS_AVAILABLE

# Streamlit page configuration
st.set_page_config(
    page_title="⚽ Transfermarkt Analysis Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/transfermarkt-api',
        'Report a bug': "mailto:support@example.com",
        'About': "# Transfermarkt Futbol Veri Analizi\n\nProfesyonel futbol veri analizi platformu."
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar-header {
        color: #1e88e5;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache fonksiyonları
@st.cache_data
def load_player_data() -> Dict:
    """Load player data"""
    # Access data folder from project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, "data")
    
    if not os.path.exists(data_dir):
        st.warning(f"⚠️ Data folder not found: {data_dir}")
        return {}
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if json_files:
        # Use demo file as priority
        demo_file = "players_data_demo.json"
        if demo_file in json_files:
            file_path = os.path.join(data_dir, demo_file)
        else:
            # Get the latest file
            latest_file = sorted(json_files)[-1]
            file_path = os.path.join(data_dir, latest_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            st.info(f"📂 Data loaded: {os.path.basename(file_path)} ({len(data)} players)")
            return data
        except Exception as e:
            st.error(f"❌ Data loading error: {e}")
            return {}
    else:
        st.warning("⚠️ Player data not found. Please perform data collection first.")
        return {}

@st.cache_data
def create_player_dataframe(players_data: Dict) -> pd.DataFrame:
    """Create player DataFrame"""
    if not players_data:
        return pd.DataFrame()
    
    if not ANALYSIS_AVAILABLE:
        st.info("⚠️ Analysis modules not found. Creating simple DataFrame...")
        # Create simple DataFrame
        data_list = []
        for player_id, player_info in players_data.items():
            if 'profile' in player_info:
                profile = player_info['profile']
                data_list.append({
                    'player_id': player_id,
                    'name': profile.get('name', 'N/A'),
                    'age': profile.get('age'),
                    'position': profile.get('position', {}).get('main') if isinstance(profile.get('position'), dict) else profile.get('position'),
                    'current_market_value': profile.get('marketValue', {}).get('value') if isinstance(profile.get('marketValue'), dict) else None,
                    'current_club': profile.get('club', {}).get('name') if isinstance(profile.get('club'), dict) else profile.get('club'),
                    'nationality': profile.get('nationality', {}).get('name') if isinstance(profile.get('nationality'), dict) else profile.get('nationality')
                })
        return pd.DataFrame(data_list)
    
    try:
        analyzer = PlayerAnalyzer(players_data)
        return analyzer.players_df
    except Exception as e:
        st.error(f"PlayerAnalyzer error: {e}")
        return pd.DataFrame()

@st.cache_data
def create_transfer_dataframe(players_data: Dict) -> pd.DataFrame:
    """Create transfer DataFrame"""
    if not players_data:
        return pd.DataFrame()
    
    if not ANALYSIS_AVAILABLE:
        return pd.DataFrame()  # Transfer analysis temporarily disabled
    
    try:
        analyzer = TransferAnalyzer(players_data)
        return analyzer.transfers_df
    except Exception as e:
        st.error(f"TransferAnalyzer error: {e}")
        return pd.DataFrame()

# Main dashboard functions
def render_header():
    """Main header section"""
    st.markdown('<h1 class="main-header">⚽ Transfermarkt Football Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Data Analysis</h3>
            <p>Player & Transfer Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Visualization</h3>
            <p>Interactive Charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Machine Learning</h3>
            <p>Market Value Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time</h3>
            <p>Live API Data</p>
        </div>
        """, unsafe_allow_html=True)

def render_sidebar_controls():
    """Sidebar controls (main dashboard only)"""
    
    # Data collection section
    st.sidebar.markdown("### 📥 Data Collection")
    
    if st.sidebar.button("🔄 Collect New Data", help="Collect new player data from API"):
        if not CLIENT_AVAILABLE:
            st.sidebar.error("❌ API client module not found!")
            st.sidebar.info("💡 Check modules in src/ folder")
            st.sidebar.code(f"Src folder path: {src_dir}")
            return None, None, None
            
        with st.spinner("Collecting data..."):
            try:
                client = TransfermarktAPIClient()
                data_collector = DataCollector(client, output_dir="../data")
                
                popular_players = get_popular_players()[:5]  # First 5 players
                players_data = data_collector.collect_top_players_data(
                    player_names=popular_players,
                    max_results_per_search=2
                )
                
                # Save data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_file = f"../data/players_data_{timestamp}.json"
                
                os.makedirs("../data", exist_ok=True)
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2, default=str)
                
                st.sidebar.success(f"✅ {len(players_data)} player data collected!")
                st.rerun()  # Refresh page
                
            except Exception as e:
                st.sidebar.error(f"❌ Data collection error: {e}")
                st.sidebar.info("💡 Check API connection and internet connectivity")
    
    # Module status indicator
    st.sidebar.markdown("### 🔧 System Status")
    
    if CLIENT_AVAILABLE:
        st.sidebar.success("✅ API Client")
    else:
        st.sidebar.error("❌ API Client")
    
    if ANALYSIS_AVAILABLE:
        st.sidebar.success("✅ Analysis")
    else:
        st.sidebar.error("❌ Analysis")
    
    if VISUALIZATION_AVAILABLE:
        st.sidebar.success("✅ Visualization")
    else:
        st.sidebar.error("❌ Visualization")
    
    if MODELS_AVAILABLE:
        st.sidebar.success("✅ ML Models")
    else:
        st.sidebar.error("❌ ML Models")
    
    # Analysis options
    st.sidebar.markdown("### 📊 Analysis Options")
    
    analysis_options = st.sidebar.multiselect(
        "Select analysis types:",
        ["Market Value Analysis", "Age Analysis", "Position Analysis", "Transfer Analysis", "Club Analysis"],
        default=["Market Value Analysis", "Age Analysis"]
    )
    
    # Visualization options
    st.sidebar.markdown("### 🎨 Visualization")
    
    chart_type = st.sidebar.selectbox(
        "Chart type:",
        ["Interactive (Plotly)", "Static (Matplotlib)", "Both"]
    )
    
    color_theme = st.sidebar.selectbox(
        "Color theme:",
        ["Default", "Viridis", "Plasma", "Husl", "Set1"]
    )
    
    return analysis_options, chart_type, color_theme

def render_ml_studio(players_df: pd.DataFrame):
    """Advanced ML model studio"""
    st.title("🤖 Machine Learning Model Studio")
    
    if players_df.empty:
        st.warning("⚠️ Player data not found for ML analysis")
        return
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Model Selection and Training")
        
        model_type = st.selectbox(
            "Select model type:",
            ["Market Value Prediction", "Position Classification", "Transfer Value Analysis", "Player Similarity Analysis"]
        )
        
        if model_type == "Market Value Prediction":
            render_ml_prediction()
        else:
            st.info(f"🔧 {model_type} model is under development...")
    
    with col2:
        st.markdown("### ⚙️ Model Parameters")
        
        # Hyperparameter settings
        test_size = st.slider("Test data ratio", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, min_value=0, max_value=999)
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        
        st.markdown("### 📋 Feature Selection")
        
        feature_options = st.multiselect(
            "Features to use:",
            ["Age", "Position", "Market Value", "Club", "Nationality"],
            default=["Age", "Position"]
        )

def render_comparison_analysis(players_df: pd.DataFrame, transfers_df: pd.DataFrame):
    """Comparative analysis page"""
    st.title("📈 Comparative Analysis")
    
    if players_df.empty:
        st.warning("⚠️ Player data not found for comparison")
        return
    
    comparison_type = st.selectbox(
        "Comparison type:",
        ["Player vs Player", "Club vs Club", "Position vs Position", "League vs League", "Age Group vs Age Group"]
    )
    
    if comparison_type == "Player vs Player":
        st.markdown("### 👤 Player Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox("First player:", players_df['name'].unique())
        
        with col2:
            player2 = st.selectbox("Second player:", players_df['name'].unique())
        
        if player1 and player2 and player1 != player2:
            p1_data = players_df[players_df['name'] == player1].iloc[0]
            p2_data = players_df[players_df['name'] == player2].iloc[0]
            
            # Comparison table
            comparison_data = {
                'Feature': ['Market Value (€M)', 'Age', 'Position', 'Club'],
                player1: [
                    f"€{p1_data.get('current_market_value', 'N/A')}M",
                    p1_data.get('age', 'N/A'),
                    p1_data.get('position', 'N/A'),
                    p1_data.get('current_club', 'N/A')
                ],
                player2: [
                    f"€{p2_data.get('current_market_value', 'N/A')}M",
                    p2_data.get('age', 'N/A'),
                    p2_data.get('position', 'N/A'),
                    p2_data.get('current_club', 'N/A')
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    else:
        st.info(f"🔧 {comparison_type} comparison is under development...")

def render_data_management():
    """Veri yönetimi sayfası"""
    st.title("⚙️ Veri Yönetimi")
    
    tab1, tab2, tab3 = st.tabs(["📁 Veri Dosyaları", "🔄 Veri Güncelleme", "🧹 Veri Temizleme"])
    
    with tab1:
        st.markdown("### 📁 Mevcut Veri Dosyaları")
        
        data_dir = "../data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            
            if files:
                files_info = []
                for file in files:
                    file_path = os.path.join(data_dir, file)
                    file_stats = os.stat(file_path)
                    files_info.append({
                        'Dosya': file,
                        'Boyut (KB)': f"{file_stats.st_size / 1024:.1f}",
                        'Değiştirilme Tarihi': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                files_df = pd.DataFrame(files_info)
                st.dataframe(files_df, use_container_width=True, hide_index=True)
            else:
                st.info("📂 Henüz veri dosyası bulunmuyor")
        else:
            st.warning("⚠️ Data klasörü bulunamadı")
    
    with tab2:
        st.markdown("### 🔄 Veri Güncelleme")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌟 Popüler Oyuncuları Güncelle", type="primary"):
                st.info("Popüler oyuncular güncelleniyor...")
        
        with col2:
            if st.button("🏆 Top Ligleri Güncelle"):
                st.info("Top ligler güncelleniyor...")
        
        st.markdown("### ⚙️ Güncelleme Ayarları")
        
        auto_update = st.checkbox("Otomatik güncelleme", value=False)
        update_interval = st.selectbox("Güncelleme sıklığı:", ["Günlük", "Haftalık", "Aylık"])
        
        if auto_update:
            st.success(f"✅ Otomatik güncelleme aktif: {update_interval}")
    
    with tab3:
        st.markdown("### 🧹 Veri Temizleme")
        
        st.warning("⚠️ Bu işlemler geri alınamaz!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Eski Dosyaları Temizle", type="secondary"):
                st.info("Eski dosyalar temizleniyor...")
        
        with col2:
            if st.button("💫 Tüm Cache'i Temizle"):
                st.cache_data.clear()
                st.success("✅ Cache temizlendi!")

def render_sidebar():
    """Eski sidebar fonksiyonu - geriye dönük uyumluluk için"""
    return render_sidebar_controls()

def render_data_overview(players_df: pd.DataFrame, transfers_df: pd.DataFrame):
    """Data overview"""
    st.markdown("## 📋 Data Overview")
    
    if players_df.empty:
        st.warning("⚠️ Player data not found. Please perform data collection first.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total Players",
            value=len(players_df),
            delta=f"+{len(players_df)} new"
        )
    
    with col2:
        market_value_count = players_df['current_market_value'].notna().sum()
        st.metric(
            label="💰 Market Value Data",
            value=market_value_count,
            delta=f"{(market_value_count/len(players_df)*100):.1f}% complete"
        )
    
    with col3:
        if not transfers_df.empty:
            st.metric(
                label="🔄 Transfer Records",
                value=len(transfers_df),
                delta=f"{len(transfers_df[transfers_df['fee'] > 0])} paid"
            )
        else:
            st.metric(label="🔄 Transfer Records", value=0)
    
    with col4:
        avg_age = players_df['age'].mean() if players_df['age'].notna().any() else 0
        st.metric(
            label="🎂 Average Age",
            value=f"{avg_age:.1f}",
            delta=f"±{players_df['age'].std():.1f} std"
        )
    
    # Data quality table
    st.markdown("### 📊 Data Quality Report")
    
    quality_data = {
        'Data Field': ['Name', 'Age', 'Position', 'Market Value', 'Club', 'Nationality'],
        'Total Records': [len(players_df)] * 6,
        'Complete Records': [
            players_df['name'].notna().sum(),
            players_df['age'].notna().sum(),
            players_df['position'].notna().sum(),
            players_df['current_market_value'].notna().sum(),
            players_df['current_club'].notna().sum(),
            players_df['nationality'].notna().sum()
        ]
    }
    
    quality_df = pd.DataFrame(quality_data)
    quality_df['Completeness (%)'] = (quality_df['Complete Records'] / quality_df['Total Records'] * 100).round(1)
    quality_df['Status'] = quality_df['Completeness (%)'].apply(
        lambda x: "✅ Perfect" if x >= 90 else "🟡 Good" if x >= 70 else "🔴 Missing"
    )
    
    st.dataframe(quality_df, use_container_width=True)

def render_player_analysis(players_df: pd.DataFrame, analysis_options: List[str], chart_type: str):
    """Player analysis section"""
    if players_df.empty:
        return
    
    st.markdown("## 👥 Player Analysis")
    
    if "Market Value Analysis" in analysis_options:
        st.markdown("### 💰 Market Value Analysis")
        
        if players_df['current_market_value'].notna().any():
            col1, col2 = st.columns(2)
            
            with col1:
                # Most valuable players
                top_players = players_df.nlargest(10, 'current_market_value')
                
                fig = px.bar(
                    top_players,
                    x='current_market_value',
                    y='name',
                    orientation='h',
                    title="Top 10 Most Valuable Players",
                    labels={'current_market_value': 'Market Value (€M)', 'name': 'Player'},
                    color='current_market_value',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Market value distribution
                fig = px.histogram(
                    players_df,
                    x='current_market_value',
                    nbins=20,
                    title="Market Value Distribution",
                    labels={'current_market_value': 'Market Value (€M)', 'count': 'Number of Players'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("#### 📈 Market Value Statistics")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            mv_data = players_df['current_market_value'].dropna()
            
            with stats_col1:
                st.metric("Average", f"€{mv_data.mean():.1f}M")
            with stats_col2:
                st.metric("Median", f"€{mv_data.median():.1f}M")
            with stats_col3:
                st.metric("Maximum", f"€{mv_data.max():.1f}M")
            with stats_col4:
                st.metric("Standard Deviation", f"€{mv_data.std():.1f}M")
    
    if "Age Analysis" in analysis_options:
        st.markdown("### 🎂 Age Analysis")
        
        if players_df['age'].notna().any():
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig = px.histogram(
                    players_df,
                    x='age',
                    nbins=15,
                    title="Age Distribution",
                    labels={'age': 'Age', 'count': 'Number of Players'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age vs Market Value
                if players_df['current_market_value'].notna().any():
                    fig = px.scatter(
                        players_df,
                        x='age',
                        y='current_market_value',
                        hover_name='name',
                        title="Age vs Market Value",
                        labels={'age': 'Age', 'current_market_value': 'Market Value (€M)'},
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    if "Position Analysis" in analysis_options:
        st.markdown("### ⚽ Position Analysis")
        
        if players_df['position'].notna().any():
            position_counts = players_df['position'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Position distribution (pie chart)
                fig = px.pie(
                    values=position_counts.values,
                    names=position_counts.index,
                    title="Position Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average market value by position
                if players_df['current_market_value'].notna().any():
                    pos_mv = players_df.groupby('position')['current_market_value'].mean().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=pos_mv.index,
                        y=pos_mv.values,
                        title="Average Market Value by Position",
                        labels={'x': 'Position', 'y': 'Average Market Value (€M)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

def render_ml_prediction():
    """Machine learning prediction section"""
    st.markdown("## 🤖 Market Value Prediction Model")
    
    if not MODELS_AVAILABLE:
        st.warning("⚠️ ML modules not found")
        st.info("""
        📋 Required modules for ML analysis:
        - src/models/market_value_prediction.py
        - Scikit-learn libraries
        """)
        return
    
    # Model training section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Model Training")
        
        players_data = load_player_data()
        
        if players_data:
            players_df = create_player_dataframe(players_data)
            
            if not players_df.empty and players_df['current_market_value'].notna().sum() >= 10:
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes."):
                        try:
                            predictor = MarketValuePredictor()
                            results = predictor.train_model(players_df, test_size=0.2, random_state=42)
                            
                            if results:
                                st.success("✅ Model trained successfully!")
                                
                                # Show model results
                                st.markdown("#### 📊 Model Performance")
                                
                                results_data = []
                                for model_name, metrics in results['all_results'].items():
                                    results_data.append({
                                        'Model': model_name,
                                        'Test R²': f"{metrics['test_r2']:.4f}",
                                        'Test RMSE': f"{metrics['test_rmse']:.2f}",
                                        'Test MAE': f"{metrics['test_mae']:.2f}"
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Best model highlight
                                st.info(f"🏆 Best Model: **{results['best_model']}** (R² = {results['best_score']:.4f})")
                                
                                # Feature importance
                                importance_df = predictor.get_feature_importance()
                                if importance_df is not None and not importance_df.empty:
                                    st.markdown("#### 🎯 Feature Importance")
                                    
                                    fig = px.bar(
                                        importance_df.head(10),
                                        x='importance',
                                        y='feature',
                                        orientation='h',
                                        title="Top 10 Most Important Features"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Model comparison
                                st.markdown("#### 📈 Model Comparison")
                                
                                model_names = list(results['all_results'].keys())
                                r2_scores = [results['all_results'][name]['test_r2'] for name in model_names]
                                
                                fig = px.bar(
                                    x=model_names,
                                    y=r2_scores,
                                    title="Model R² Scores Comparison",
                                    labels={'x': 'Model', 'y': 'R² Score'}
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.error("❌ Model training failed")
                                
                        except Exception as e:
                            st.error(f"❌ Model training error: {e}")
                            
            else:
                st.warning("⚠️ Insufficient data for model training (minimum 10 players required)")
        else:
            st.warning("⚠️ Player data not found for model training")
    
    with col2:
        st.markdown("### ⚙️ Model Settings")
        
        test_size = st.slider("Test data ratio", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, min_value=0, max_value=999)
        
        st.markdown("### 📋 Model Bilgileri")
        st.info("""
        **Kullanılan Modeller:**
        - Random Forest
        - XGBoost
        - Gradient Boosting
        
        **Değerlendirme Metrikleri:**
        - R² Score
        - RMSE
        - MAE
        """)

def main():
    """Main dashboard function"""
    
    # Page selector - in sidebar
    st.sidebar.title("🏆 Transfermarkt Analysis")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "📄 Select Page:",
        [
            "🏠 Main Dashboard",
            "📊 Advanced Player Analysis", 
            "🔄 Transfer Analysis",
            "🤖 ML Model Studio",
            "📈 Comparative Analysis",
            "⚙️ Data Management"
        ]
    )
    
    # Data loading (common for all pages)
    with st.spinner("Loading data..."):
        players_data = load_player_data()
        players_df = create_player_dataframe(players_data)
        transfers_df = create_transfer_dataframe(players_data)
    
    # Page routing
    if page == "🏠 Main Dashboard":
        render_main_dashboard(players_df, transfers_df)
    
    elif page == "📊 Advanced Player Analysis":
        # Import advanced analysis page
        try:
            from pages.advanced_analysis import render_advanced_player_analysis
            render_advanced_player_analysis(players_df)
        except ImportError:
            st.error("❌ Advanced analysis module not found")
            st.info("📁 Make sure pages/advanced_analysis.py file exists")
    
    elif page == "🔄 Transfer Analysis":
        # Import transfer analysis page
        try:
            from pages.transfer_analysis import render_transfer_analysis
            render_transfer_analysis(transfers_df)
        except ImportError:
            st.error("❌ Transfer analysis module not found")
            st.info("📁 Make sure pages/transfer_analysis.py file exists")
    
    elif page == "🤖 ML Model Studio":
        render_ml_studio(players_df)
    
    elif page == "📈 Comparative Analysis":
        render_comparison_analysis(players_df, transfers_df)
    
    elif page == "⚙️ Data Management":
        render_data_management()

def render_main_dashboard(players_df: pd.DataFrame, transfers_df: pd.DataFrame):
    """Main dashboard content"""
    # Header
    render_header()
    
    # Sidebar controls
    st.sidebar.markdown("### 🔧 Control Panel")
    analysis_options, chart_type, color_theme = render_sidebar_controls()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Player Analysis", "🔄 Transfer Analysis", "🤖 ML Model"])
    
    with tab1:
        render_data_overview(players_df, transfers_df)
        
        if not players_df.empty:
            st.markdown("### 📋 Oyuncu Verileri")
            
            # Filtreleme seçenekleri
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if players_df['position'].notna().any():
                    positions = ['Tümü'] + list(players_df['position'].dropna().unique())
                    selected_position = st.selectbox("Pozisyon filtresi", positions)
                else:
                    selected_position = 'Tümü'
            
            with col2:
                if players_df['current_club'].notna().any():
                    clubs = ['Tümü'] + list(players_df['current_club'].dropna().unique())
                    selected_club = st.selectbox("Kulüp filtresi", clubs)
                else:
                    selected_club = 'Tümü'
            
            with col3:
                if players_df['age'].notna().any():
                    min_age, max_age = int(players_df['age'].min()), int(players_df['age'].max())
                    age_range = st.slider("Yaş aralığı", min_age, max_age, (min_age, max_age))
                else:
                    age_range = (18, 40)
            
            # Filtreleme uygula
            filtered_df = players_df.copy()
            
            if selected_position != 'Tümü':
                filtered_df = filtered_df[filtered_df['position'] == selected_position]
            
            if selected_club != 'Tümü':
                filtered_df = filtered_df[filtered_df['current_club'] == selected_club]
            
            if players_df['age'].notna().any():
                filtered_df = filtered_df[
                    (filtered_df['age'] >= age_range[0]) & 
                    (filtered_df['age'] <= age_range[1])
                ]
            
            # Filtrelenmiş tabloyu göster
            display_columns = ['name', 'age', 'position', 'current_market_value', 'current_club', 'nationality']
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_columns].head(20),
                use_container_width=True
            )
            
            st.info(f"📊 Toplam {len(filtered_df)} oyuncu (filtrelenmiş: {len(filtered_df)} / {len(players_df)})")
    
    with tab2:
        render_player_analysis(players_df, analysis_options, chart_type)
    
    with tab3:
        st.markdown("## 🔄 Transfer Analizi")
        if not transfers_df.empty:
            # Basit transfer özeti
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Toplam Transfer", len(transfers_df))
            
            with col2:
                paid_count = len(transfers_df[transfers_df['fee'] > 0]) if 'fee' in transfers_df.columns else 0
                st.metric("💰 Ücretli Transfer", paid_count)
            
            with col3:
                total_fee = transfers_df['fee'].sum() if 'fee' in transfers_df.columns else 0
                st.metric("💸 Toplam Hacim", f"€{total_fee:.1f}M")
            
            st.markdown("### 📋 Transfer Verileri Önizleme")
            st.dataframe(transfers_df.head(10), use_container_width=True)
            
            st.info("💡 Detaylı transfer analizleri için '🔄 Transfer Analizi' sayfasını ziyaret edin.")
        else:
            st.warning("⚠️ Transfer verisi bulunamadı")
    
    with tab4:
        # Basitleştirilmiş ML bölümü
        st.markdown("## 🤖 Makine Öğrenmesi Özeti")
        
        if not players_df.empty and players_df['current_market_value'].notna().sum() >= 10:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Model Bilgileri")
                st.info("""
                **Mevcut Modeller:**
                - Random Forest
                - XGBoost  
                - Gradient Boosting
                
                **Tahmin Hedefi:**
                - Oyuncu market değeri
                """)
                
                if st.button("🚀 Hızlı Model Eğitimi"):
                    with st.spinner("Model eğitiliyor..."):
                        try:
                            predictor = MarketValuePredictor()
                            results = predictor.train_model(players_df, test_size=0.2)
                            
                            if results:
                                st.success(f"✅ Model eğitildi! En iyi R² skoru: {results['best_score']:.4f}")
                                st.info(f"🏆 En iyi model: {results['best_model']}")
                            else:
                                st.error("❌ Model eğitimi başarısız")
                        except Exception as e:
                            st.error(f"❌ Hata: {e}")
            
            with col2:
                st.markdown("### 📊 Veri Durumu")
                
                mv_count = players_df['current_market_value'].notna().sum()
                age_count = players_df['age'].notna().sum()
                pos_count = players_df['position'].notna().sum()
                
                st.metric("💰 Market değeri verisi", f"{mv_count}/{len(players_df)}")
                st.metric("🎂 Yaş verisi", f"{age_count}/{len(players_df)}")
                st.metric("⚽ Pozisyon verisi", f"{pos_count}/{len(players_df)}")
                
                if mv_count >= 10:
                    st.success("✅ ML eğitimi için yeterli veri")
                else:
                    st.warning("⚠️ Daha fazla veri gerekli")
            
            st.info("💡 Gelişmiş ML analizleri için '🤖 ML Model Studio' sayfasını ziyaret edin.")
        else:
            st.warning("⚠️ ML analizi için yeterli veri yok")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>⚽ <strong>Transfermarkt Futbol Veri Analizi Dashboard</strong> | 
        📊 Powered by Streamlit & Plotly | 
        🤖 Machine Learning Ready</p>
        <p>📅 Son güncelleme: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
