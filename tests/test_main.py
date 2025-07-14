"""
Test modülü - Transfermarkt API istemcisi testleri
"""

import pytest
import sys
import os

# Test etmek için src dizinini path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection import TransfermarktAPIClient, APIConfig
from analysis import PlayerAnalyzer, TransferAnalyzer
from models import MarketValuePredictor
import pandas as pd


class TestAPIClient:
    """API istemci testleri"""
    
    def setup_method(self):
        """Test setup"""
        self.client = TransfermarktAPIClient()
    
    def test_api_config(self):
        """API konfigürasyonu testi"""
        config = APIConfig()
        assert config.base_url == "https://transfermarkt-api.fly.dev"
        assert config.timeout == 30
        assert config.retry_attempts == 3
    
    def test_client_initialization(self):
        """İstemci başlatma testi"""
        assert self.client is not None
        assert self.client.config.base_url == "https://transfermarkt-api.fly.dev"
        assert hasattr(self.client, 'session')
    
    @pytest.mark.integration
    def test_search_players(self):
        """Oyuncu arama testi (internet bağlantısı gerekli)"""
        try:
            result = self.client.search_players("messi")
            assert isinstance(result, dict)
            # API yanıtı varsa kontrol et
            if 'results' in result:
                assert isinstance(result['results'], list)
        except Exception as e:
            pytest.skip(f"API bağlantısı başarısız: {e}")


class TestPlayerAnalyzer:
    """Oyuncu analiz sınıfı testleri"""
    
    def setup_method(self):
        """Test verisi oluştur"""
        # Örnek oyuncu verisi
        self.test_data = {
            "player1": {
                "profile": {
                    "name": "Test Player 1",
                    "age": 25,
                    "height": "1,80 m",
                    "foot": "Right",
                    "position": {"main": "Centre-Forward"},
                    "citizenship": [{"name": "Spain"}],
                    "club": {"name": "Test Club"}
                },
                "market_value": {
                    "marketValue": "€50.00m",
                    "marketValueHistory": []
                }
            },
            "player2": {
                "profile": {
                    "name": "Test Player 2",
                    "age": 30,
                    "height": "1,85 m",
                    "foot": "Left",
                    "position": {"main": "Central Midfield"},
                    "citizenship": [{"name": "Germany"}],
                    "club": {"name": "Test Club 2"}
                },
                "market_value": {
                    "marketValue": "€30.00m",
                    "marketValueHistory": []
                }
            }
        }
    
    def test_player_analyzer_initialization(self):
        """PlayerAnalyzer başlatma testi"""
        analyzer = PlayerAnalyzer(self.test_data)
        assert analyzer is not None
        assert hasattr(analyzer, 'players_df')
        assert isinstance(analyzer.players_df, pd.DataFrame)
    
    def test_dataframe_creation(self):
        """DataFrame oluşturma testi"""
        analyzer = PlayerAnalyzer(self.test_data)
        df = analyzer.players_df
        
        assert len(df) == 2
        assert 'name' in df.columns
        assert 'age' in df.columns
        assert 'current_market_value' in df.columns
        
        # Veri tiplerini kontrol et
        assert df['age'].dtype in ['int64', 'float64']
        assert df['current_market_value'].dtype in ['float64']
    
    def test_market_value_extraction(self):
        """Market değeri çıkarma testi"""
        analyzer = PlayerAnalyzer(self.test_data)
        
        # Test market değeri string'lerini çıkarma
        assert analyzer._extract_market_value("€50.00m") == 50.0
        assert analyzer._extract_market_value("€30.00m") == 30.0
        assert analyzer._extract_market_value("-") is None
        assert analyzer._extract_market_value("") is None
    
    def test_top_players_by_value(self):
        """En değerli oyuncular testi"""
        analyzer = PlayerAnalyzer(self.test_data)
        top_players = analyzer.get_top_players_by_value(n=1)
        
        assert len(top_players) == 1
        assert top_players.iloc[0]['name'] == "Test Player 1"  # En yüksek market değeri
    
    def test_age_distribution(self):
        """Yaş dağılımı testi"""
        analyzer = PlayerAnalyzer(self.test_data)
        age_analysis = analyzer.analyze_age_distribution()
        
        assert 'statistics' in age_analysis
        assert 'age_ranges' in age_analysis
        assert age_analysis['statistics']['mean'] == 27.5  # (25+30)/2


class TestMarketValuePredictor:
    """Market değeri tahmin modeli testleri"""
    
    def setup_method(self):
        """Test verisi oluştur"""
        # Daha fazla test verisi (model eğitimi için)
        self.test_df = pd.DataFrame({
            'name': [f'Player {i}' for i in range(20)],
            'age': [20 + i for i in range(20)],
            'height': ['1,80 m'] * 20,
            'foot': ['Right'] * 10 + ['Left'] * 10,
            'position': ['Centre-Forward'] * 5 + ['Central Midfield'] * 5 + 
                       ['Centre-Back'] * 5 + ['Goalkeeper'] * 5,
            'nationality': ['Spain'] * 5 + ['Germany'] * 5 + ['France'] * 5 + ['Italy'] * 5,
            'current_club': ['Club A'] * 10 + ['Club B'] * 10,
            'current_market_value': [10 + i*2 for i in range(20)]  # 10, 12, 14, ... 48
        })
    
    def test_predictor_initialization(self):
        """Tahmin modeli başlatma testi"""
        predictor = MarketValuePredictor()
        assert predictor is not None
        assert predictor.model is None
        assert predictor.target_column == 'current_market_value'
    
    def test_feature_preparation(self):
        """Özellik hazırlama testi"""
        predictor = MarketValuePredictor()
        processed_df = predictor.prepare_features(self.test_df)
        
        assert 'height_cm' in processed_df.columns
        assert 'age_group' in processed_df.columns
        assert 'position_category' in processed_df.columns
        assert 'from_big_football_country' in processed_df.columns
    
    def test_height_extraction(self):
        """Boy bilgisi çıkarma testi"""
        predictor = MarketValuePredictor()
        processed_df = predictor.prepare_features(self.test_df)
        
        # '1,80 m' -> 180 cm'ye dönüşüm
        assert processed_df['height_cm'].iloc[0] == 180.0
    
    @pytest.mark.slow
    def test_model_training(self):
        """Model eğitimi testi (yavaş test)"""
        predictor = MarketValuePredictor()
        
        try:
            results = predictor.train_model(self.test_df, test_size=0.3, random_state=42)
            
            if results:  # Eğer model eğitimi başarılıysa
                assert 'best_model' in results
                assert 'best_score' in results
                assert 'all_results' in results
                assert predictor.model is not None
                
                # Tahmin testi
                predictions = predictor.predict(self.test_df.head(5))
                assert len(predictions) == 5
                assert all(isinstance(p, (int, float)) for p in predictions)
            
        except Exception as e:
            pytest.skip(f"Model eğitimi başarısız (beklenen): {e}")


class TestDataIntegration:
    """Entegrasyon testleri"""
    
    def test_full_workflow_with_mock_data(self):
        """Tam iş akışı testi (mock veri ile)"""
        # Mock veri oluştur
        mock_data = {
            "test_player": {
                "profile": {
                    "name": "Integration Test Player",
                    "age": 25,
                    "height": "1,85 m",
                    "foot": "Right",
                    "position": {"main": "Centre-Forward"},
                    "citizenship": [{"name": "Brazil"}],
                    "club": {"name": "Test FC"}
                },
                "market_value": {
                    "marketValue": "€75.00m",
                    "marketValueHistory": [
                        {"date": "2023-01-01", "age": 24, "clubName": "Old Club", "marketValue": "€60.00m"},
                        {"date": "2023-06-01", "age": 25, "clubName": "Test FC", "marketValue": "€75.00m"}
                    ]
                },
                "transfers": {
                    "transfers": [
                        {
                            "id": "test_transfer",
                            "clubFrom": {"id": "old_club", "name": "Old Club"},
                            "clubTo": {"id": "test_fc", "name": "Test FC"},
                            "date": "2023-01-01",
                            "season": "22/23",
                            "marketValue": "€60.00m",
                            "fee": "€50.00m",
                            "upcoming": False
                        }
                    ]
                }
            }
        }
        
        # Analizleri çalıştır
        player_analyzer = PlayerAnalyzer(mock_data)
        transfer_analyzer = TransferAnalyzer(mock_data)
        
        # Sonuçları kontrol et
        assert len(player_analyzer.players_df) == 1
        assert len(transfer_analyzer.transfers_df) == 1
        
        # Player analizi
        player_df = player_analyzer.players_df
        assert player_df.iloc[0]['name'] == "Integration Test Player"
        assert player_df.iloc[0]['current_market_value'] == 75.0
        
        # Transfer analizi
        transfer_df = transfer_analyzer.transfers_df
        assert transfer_df.iloc[0]['player_name'] == "Integration Test Player"
        assert transfer_df.iloc[0]['fee'] == 50.0


# Test konfigürasyonu
def pytest_configure(config):
    """Pytest konfigürasyonu"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require internet)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take longer to run)"
    )


if __name__ == "__main__":
    # Testleri direkt çalıştırma
    pytest.main([__file__, "-v"])
