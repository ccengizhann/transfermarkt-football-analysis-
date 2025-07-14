"""
Simple Market Value Prediction
==============================
Simple prediction model working without XGBoost.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleMarketValuePredictor:
    """
    Simple market value prediction model.
    Uses only scikit-learn library.
    """
    
    def __init__(self):
        """Model initialization."""
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Özellikleri hazırla ve ön işlemlerini yap.
        
        Args:
            df: Player verileri
            
        Returns:
            Hazırlanmış özellik matrisi
        """
        if df.empty:
            return df
            
        features = df.copy()
        
        # Eksik değerleri doldur
        if 'age' in features.columns:
            features['age'] = features['age'].fillna(features['age'].median())
        if 'market_value' in features.columns:
            features['market_value'] = features['market_value'].fillna(0)
        if 'position' in features.columns:
            features['position'] = features['position'].fillna('Unknown')
        if 'nationality' in features.columns:
            features['nationality'] = features['nationality'].fillna('Unknown')
        if 'club' in features.columns:
            features['club'] = features['club'].fillna('Unknown')
            
        # Kategorik değişkenleri encode et
        categorical_cols = ['position', 'nationality', 'club']
        for col in categorical_cols:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit ediyorsa
                    if not self.is_trained:
                        features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features[col].astype(str))
                    else:
                        # Transform ediyorsa, unknown değerleri için -1 kullan
                        known_labels = set(self.label_encoders[col].classes_)
                        features[f'{col}_encoded'] = features[col].astype(str).apply(
                            lambda x: self.label_encoders[col].transform([x])[0] if x in known_labels else -1
                        )
                else:
                    # Daha önce fit edilmişse transform et
                    known_labels = set(self.label_encoders[col].classes_)
                    features[f'{col}_encoded'] = features[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] if x in known_labels else -1
                    )
        
        # Yaş grubu oluştur
        if 'age' in features.columns:
            features['age_group'] = pd.cut(features['age'], 
                                         bins=[0, 20, 25, 30, 35, 50], 
                                         labels=['U20', 'Young', 'Prime', 'Veteran', 'Experienced'])
            features['age_group_encoded'] = features['age_group'].astype(str)
            if 'age_group' not in self.label_encoders:
                self.label_encoders['age_group'] = LabelEncoder()
                if not self.is_trained:
                    features['age_group_encoded'] = self.label_encoders['age_group'].fit_transform(features['age_group_encoded'])
                else:
                    known_labels = set(self.label_encoders['age_group'].classes_)
                    features['age_group_encoded'] = features['age_group_encoded'].apply(
                        lambda x: self.label_encoders['age_group'].transform([x])[0] if x in known_labels else -1
                    )
            else:
                known_labels = set(self.label_encoders['age_group'].classes_)
                features['age_group_encoded'] = features['age_group_encoded'].apply(
                    lambda x: self.label_encoders['age_group'].transform([x])[0] if x in known_labels else -1
                )
        
        # Numerik özellikler
        numeric_features = ['age']
        if 'age' in features.columns:
            numeric_features.append('age')
        
        # Encoded kategorik özellikler
        encoded_features = [col for col in features.columns if col.endswith('_encoded')]
        
        # Final feature set
        final_features = numeric_features + encoded_features
        final_features = [col for col in final_features if col in features.columns]
        
        self.feature_names = final_features
        
        return features[final_features]
    
    def train(self, data: pd.DataFrame, target_column: str = 'market_value') -> Dict[str, Any]:
        """
        Modeli eğit.
        
        Args:
            data: Eğitim verisi
            target_column: Hedef sütun adı
            
        Returns:
            Eğitim sonuçları
        """
        try:
            if data.empty or target_column not in data.columns:
                return {"error": "Veri boş veya hedef sütun bulunamadı"}
            
            # Özellikleri hazırla
            X = self.prepare_features(data)
            y = data[target_column].fillna(0)
            
            if X.empty:
                return {"error": "Özellik matrisi boş"}
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Modelleri eğit ve değerlendir
            results = {}
            best_score = -np.inf
            
            for name, model in self.models.items():
                try:
                    # Model eğitimi
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    
                    pipeline.fit(X_train, y_train)
                    
                    # Tahminler
                    y_pred = pipeline.predict(X_test)
                    
                    # Metrikler
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                    
                    # En iyi modeli seç
                    if r2 > best_score:
                        best_score = r2
                        self.best_model = pipeline
                        
                except Exception as e:
                    results[name] = {"error": str(e)}
            
            self.is_trained = True
            
            return {
                "success": True,
                "results": results,
                "best_model": type(self.best_model.named_steps['model']).__name__ if self.best_model else None,
                "best_r2": best_score,
                "feature_count": len(self.feature_names)
            }
            
        except Exception as e:
            return {"error": f"Eğitim hatası: {str(e)}"}
    
    def predict(self, data: pd.DataFrame) -> List[float]:
        """
        Tahmin yap.
        
        Args:
            data: Tahmin verisi
            
        Returns:
            Tahmin listesi
        """
        try:
            if not self.is_trained or self.best_model is None:
                return [0.0] * len(data)
            
            X = self.prepare_features(data)
            
            if X.empty:
                return [0.0] * len(data)
            
            predictions = self.best_model.predict(X)
            return predictions.tolist()
            
        except Exception as e:
            print(f"Tahmin hatası: {e}")
            return [0.0] * len(data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Özellik önemini getir.
        
        Returns:
            Özellik önem sözlüğü
        """
        if not self.is_trained or self.best_model is None:
            return {}
        
        try:
            model = self.best_model.named_steps['model']
            
            if hasattr(model, 'feature_importances_'):
                # Random Forest gibi tree-based modeller
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear modeller
                importance = abs(model.coef_)
            else:
                return {}
            
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Normalize et
            total = sum(feature_importance.values())
            if total > 0:
                feature_importance = {k: v/total for k, v in feature_importance.items()}
            
            # Sırala
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Feature importance hatası: {e}")
            return {}


class SimpleTransferValueAnalyzer:
    """
    Basit transfer value analizi.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.available = True
        
    def analyze_transfer_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Transfer trendlerini analiz et.
        
        Args:
            data: Transfer verileri
            
        Returns:
            Analiz sonuçları
        """
        try:
            if data.empty:
                return {"error": "Veri boş"}
            
            results = {
                "total_transfers": len(data),
                "unique_players": data['player_name'].nunique() if 'player_name' in data.columns else 0,
                "avg_market_value": data['market_value'].mean() if 'market_value' in data.columns else 0,
                "median_market_value": data['market_value'].median() if 'market_value' in data.columns else 0,
                "total_value": data['market_value'].sum() if 'market_value' in data.columns else 0
            }
            
            # Pozisyona göre dağılım
            if 'position' in data.columns:
                position_stats = data.groupby('position')['market_value'].agg(['count', 'mean', 'sum']) if 'market_value' in data.columns else data['position'].value_counts()
                results['position_analysis'] = position_stats.to_dict()
            
            # Yaş grubu analizi
            if 'age' in data.columns:
                age_stats = data.groupby(pd.cut(data['age'], bins=[0, 20, 25, 30, 35, 50]))['market_value'].agg(['count', 'mean']) if 'market_value' in data.columns else None
                if age_stats is not None:
                    results['age_analysis'] = age_stats.to_dict()
            
            return results
            
        except Exception as e:
            return {"error": f"Analiz hatası: {str(e)}"}
    
    def predict_transfer_success(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Transfer başarısını tahmin et.
        
        Args:
            data: Player verileri
            
        Returns:
            Başarı tahminleri
        """
        try:
            if data.empty:
                return {"error": "Veri boş"}
            
            # Basit başarı skorları
            success_scores = []
            
            for _, player in data.iterrows():
                score = 0.5  # Base score
                
                # Yaş faktörü
                if 'age' in player and pd.notna(player['age']):
                    age = player['age']
                    if 20 <= age <= 28:
                        score += 0.2
                    elif 28 < age <= 32:
                        score += 0.1
                    else:
                        score -= 0.1
                
                # Market value faktörü
                if 'market_value' in player and pd.notna(player['market_value']):
                    mv = player['market_value']
                    if mv > 50000000:  # 50M+
                        score += 0.3
                    elif mv > 20000000:  # 20M+
                        score += 0.2
                    elif mv > 5000000:  # 5M+
                        score += 0.1
                
                # Pozisyon faktörü
                if 'position' in player and pd.notna(player['position']):
                    pos = player['position']
                    if pos in ['CF', 'LW', 'RW', 'AM']:  # Offensive positions
                        score += 0.1
                
                success_scores.append(min(1.0, max(0.0, score)))
            
            return {
                "success_scores": success_scores,
                "avg_success_rate": np.mean(success_scores),
                "high_success_count": sum(1 for s in success_scores if s > 0.7),
                "total_players": len(success_scores)
            }
            
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}
