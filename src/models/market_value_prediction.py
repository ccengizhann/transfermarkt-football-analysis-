"""
Market Value Prediction Model
Player market value prediction using machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# XGBoost'u güvenli şekilde import et
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"XGBoost import error: {e}")
    print("Will run without XGBoost...")
except Exception as e:
    print(f"XGBoost loading error: {e}")
    print("Will run without XGBoost...")

import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketValuePredictor:
    """
    Player market value prediction model
    """
    
    def __init__(self):
        """
        Initialize MarketValuePredictor
        """
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = 'current_market_value'
        self.model_metrics = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering for the model
        
        Args:
            df: Player DataFrame
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        # Eksik değerleri doldur
        df_processed['age'] = df_processed['age'].fillna(df_processed['age'].median())
        df_processed['position'] = df_processed['position'].fillna('Unknown')
        df_processed['nationality'] = df_processed['nationality'].fillna('Unknown')
        df_processed['current_club'] = df_processed['current_club'].fillna('Unknown')
        
        # Yaş grupları oluştur
        df_processed['age_group'] = pd.cut(
            df_processed['age'], 
            bins=[0, 21, 25, 28, 32, 40], 
            labels=['Very Young', 'Young', 'Prime', 'Mature', 'Veteran']
        )
        
        # Boy bilgisini işle (cm'ye çevir)
        def extract_height(height_str):
            if pd.isna(height_str) or height_str == '':
                return None
            try:
                # "1,85 m" formatından sayıyı çıkar
                if 'm' in str(height_str):
                    height_value = float(str(height_str).replace('m', '').replace(',', '.').strip())
                    return height_value * 100  # cm'ye çevir
                elif 'cm' in str(height_str):
                    return float(str(height_str).replace('cm', '').strip())
                else:
                    return float(str(height_str).replace(',', '.'))
            except:
                return None
        
        df_processed['height_cm'] = df_processed['height'].apply(extract_height)
        df_processed['height_cm'] = df_processed['height_cm'].fillna(df_processed['height_cm'].median())
        
        # Dominant ayak kategorisi
        df_processed['foot'] = df_processed['foot'].fillna('Unknown')
        
        # Pozisyon kategorilerini basitleştir
        position_mapping = {
            'Goalkeeper': 'GK',
            'Centre-Back': 'DEF',
            'Left-Back': 'DEF',
            'Right-Back': 'DEF',
            'Defensive Midfield': 'MID',
            'Central Midfield': 'MID',
            'Attacking Midfield': 'MID',
            'Left Midfield': 'MID',
            'Right Midfield': 'MID',
            'Left Winger': 'ATT',
            'Right Winger': 'ATT',
            'Centre-Forward': 'ATT',
            'Second Striker': 'ATT'
        }
        
        df_processed['position_category'] = df_processed['position'].map(position_mapping).fillna('Unknown')
        
        # Büyük liglerden mi (özellik mühendisliği)
        big_leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
        # Bu bilgi şu an mevcut değil, placeholder olarak bırakıyoruz
        df_processed['is_big_league'] = 0
        
        # Büyük ülkelerden mi (futbol açısından)
        big_football_countries = [
            'Germany', 'France', 'Spain', 'Italy', 'England', 'Brazil', 
            'Argentina', 'Netherlands', 'Portugal', 'Belgium', 'Croatia'
        ]
        df_processed['from_big_football_country'] = df_processed['nationality'].isin(big_football_countries).astype(int)
        
        return df_processed
    
    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Veri ön işleme pipeline'ı oluşturur
        
        Args:
            df: DataFrame
            
        Returns:
            ColumnTransformer
        """
        # Sayısal özellikler
        numeric_features = ['age', 'height_cm']
        
        # Kategorik özellikler
        categorical_features = ['position_category', 'foot', 'age_group', 'nationality']
        
        # Boolean özellikler
        boolean_features = ['from_big_football_country', 'is_big_league']
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features),
                ('bool', 'passthrough', boolean_features)
            ]
        )
        
        return preprocessor
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Modeli eğitir ve değerlendirir
        
        Args:
            df: Eğitim DataFrame'i
            test_size: Test seti oranı
            random_state: Random seed
            
        Returns:
            Eğitim sonuçları
        """
        # Veri hazırlığı
        df_processed = self.prepare_features(df)
        
        # Target değişkeni olan kayıtları al
        df_clean = df_processed.dropna(subset=[self.target_column])
        
        if len(df_clean) < 10:
            logger.error("Eğitim için yeterli veri yok")
            return {}
        
        # Feature ve target ayrımı
        feature_columns = ['age', 'height_cm', 'position_category', 'foot', 
                          'age_group', 'nationality', 'from_big_football_country', 'is_big_league']
        
        X = df_clean[feature_columns]
        y = df_clean[self.target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Preprocessor oluştur ve fit et
        self.preprocessor = self.create_preprocessor(X_train)
        
        # Model pipeline'ları tanımla
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=random_state))
            ]),
            'Gradient Boosting': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=random_state))
            ]),
            'Ridge Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', Ridge(alpha=1.0))
            ])
        }
        
        # XGBoost'u sadece mevcut ise ekle
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=random_state))
            ])
            logger.info("XGBoost modeli eklendi")
        else:
            logger.warning("XGBoost bulunamadı, sadece scikit-learn modelleri kullanılacak")
        
        # Model karşılaştırması
        results = {}
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models.items():
            logger.info(f"{name} modeli eğitiliyor...")
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            
            # Model eğit
            model.fit(X_train, y_train)
            
            # Tahminler
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrikler
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'cv_score_mean': -cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'model': model
            }
            
            # En iyi modeli seç (test R2'ye göre)
            if test_r2 > best_score:
                best_score = test_r2
                best_model_name = name
                self.model = model
        
        # Sonuçları kaydet
        self.feature_columns = feature_columns
        self.model_metrics = results
        
        logger.info(f"En iyi model: {best_model_name} (Test R2: {best_score:.4f})")
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'all_results': results,
            'feature_columns': feature_columns
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Yeni veriler için tahmin yapar
        
        Args:
            df: Tahmin yapılacak DataFrame
            
        Returns:
            Tahmin sonuçları
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş")
        
        # Veri hazırlığı
        df_processed = self.prepare_features(df)
        X = df_processed[self.feature_columns]
        
        # Tahmin
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Özellik önemlerini getirir
        
        Returns:
            Özellik önemleri DataFrame'i
        """
        if self.model is None:
            return None
        
        try:
            # Model tipine göre feature importance al
            regressor = self.model.named_steps['regressor']
            
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
            elif hasattr(regressor, 'coef_'):
                importances = np.abs(regressor.coef_)
            else:
                return None
            
            # Feature names al
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            
            # DataFrame oluştur
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.warning(f"Feature importance alınamadı: {e}")
            return None
    
    def save_model(self, filepath: str) -> None:
        """
        Modeli dosyaya kaydeder
        
        Args:
            filepath: Dosya yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model {filepath} olarak kaydedildi")
    
    def load_model(self, filepath: str) -> None:
        """
        Modeli dosyadan yükler
        
        Args:
            filepath: Dosya yolu
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data['model_metrics']
        
        logger.info(f"Model {filepath} dosyasından yüklendi")


class TransferValueAnalyzer:
    """
    Transfer değeri analizleri
    """
    
    def __init__(self):
        """
        TransferValueAnalyzer'ı başlatır
        """
        pass
    
    def analyze_value_vs_fee_correlation(self, transfers_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Market değeri ile transfer ücreti arasındaki korelasyonu analiz eder
        
        Args:
            transfers_df: Transfer DataFrame'i
            
        Returns:
            Korelasyon analizi sonuçları
        """
        # Market değeri ve fee olan transferler
        valid_data = transfers_df.dropna(subset=['market_value', 'fee'])
        valid_data = valid_data[
            (valid_data['fee'] > 0) & 
            (valid_data['market_value'] > 0) &
            (valid_data['upcoming'] == False)
        ]
        
        if len(valid_data) < 10:
            return {'error': 'Yeterli veri yok'}
        
        # Korelasyon hesapla
        correlation = valid_data['market_value'].corr(valid_data['fee'])
        
        # Linear regresyon fit et
        from sklearn.linear_model import LinearRegression
        X = valid_data[['market_value']]
        y = valid_data['fee']
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        # R-squared hesapla
        r2 = lr.score(X, y)
        
        # Katsayılar
        slope = lr.coef_[0]
        intercept = lr.intercept_
        
        return {
            'correlation': correlation,
            'r_squared': r2,
            'slope': slope,
            'intercept': intercept,
            'equation': f'Fee = {slope:.2f} * Market_Value + {intercept:.2f}',
            'sample_size': len(valid_data)
        }
    
    def find_over_undervalued_transfers(self, transfers_df: pd.DataFrame, 
                                      threshold: float = 0.3) -> Dict[str, pd.DataFrame]:
        """
        Aşırı/eksik değerlendirilmiş transferleri bulur
        
        Args:
            transfers_df: Transfer DataFrame'i
            threshold: Değer farkı eşiği
            
        Returns:
            Aşırı/eksik değerlendirilmiş transferler
        """
        # Market değeri ve fee olan transferler
        valid_data = transfers_df.dropna(subset=['market_value', 'fee'])
        valid_data = valid_data[
            (valid_data['fee'] > 0) & 
            (valid_data['market_value'] > 0) &
            (valid_data['upcoming'] == False)
        ].copy()
        
        if valid_data.empty:
            return {'overvalued': pd.DataFrame(), 'undervalued': pd.DataFrame()}
        
        # Fee/Market Value oranı hesapla
        valid_data['fee_to_value_ratio'] = valid_data['fee'] / valid_data['market_value']
        
        # Aşırı değerlendirilmiş (fee >> market_value)
        overvalued = valid_data[valid_data['fee_to_value_ratio'] > (1 + threshold)]
        
        # Eksik değerlendirilmiş (fee << market_value)
        undervalued = valid_data[valid_data['fee_to_value_ratio'] < (1 - threshold)]
        
        return {
            'overvalued': overvalued.sort_values('fee_to_value_ratio', ascending=False),
            'undervalued': undervalued.sort_values('fee_to_value_ratio')
        }


if __name__ == "__main__":
    # Test kodları buraya eklenebilir
    pass
