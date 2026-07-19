import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class EnsemblePredictor:
    """
    محرك ذكاء اصطناعي مزدوج (Dual-Core): يدعم فحص حركة الأسعار والسيولة بالتكامل مع مشاعر الأخبار
    """

    def __init__(self):
        # 1. نماذج المضاربة السريعة (توقع الجلسة القادمة)
        self.xgb_short = XGBRegressor(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.05, 
            random_state=42, 
            n_jobs=-1,
            verbosity=0
        )
        self.rf_short = RandomForestRegressor(
            n_estimators=80, 
            max_depth=5, 
            random_state=42, 
            n_jobs=-1
        )

        # 2. نماذج الاستثمار الموجي (توقع الاتجاه بعد شهرين / 45 جلسة تداول)
        self.xgb_long = XGBRegressor(
            n_estimators=150, 
            max_depth=5, 
            learning_rate=0.03, 
            random_state=42, 
            n_jobs=-1,
            verbosity=0
        )
        self.rf_long = RandomForestRegressor(
            n_estimators=100, 
            max_depth=6, 
            random_state=42, 
            n_jobs=-1
        )

        self.scaler_short = StandardScaler()
        self.scaler_long = StandardScaler()
        self.is_trained = False
        
        # تخزين معلومات التدريب
        self.training_info = {
            'short': {'mean': 0, 'std': 1, 'samples': 0},
            'long': {'mean': 0, 'std': 1, 'samples': 0}
        }

        # قائمة الميزات الأساسية
        self.feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD', 'VWAP', 'CMF', 'Momentum',
            'Daily_Return', 'EMA_50', 'EMA_200', 
            'Dist_From_EMA200', 'Volume_Shock', 'OBV_Slope',
            'BB_Upper', 'BB_Lower', 'ATR'
        ]
        
        # إضافة News_Sentiment إذا كانت موجودة
        self.news_col = 'News_Sentiment'

    def prepare_features(self, df, mode="short"):
        """تجهيز البيانات والميزات الفنية مع عزل الأهداف"""
        df_clean = df.copy()

        # التأكد من وجود كافة الأعمدة
        for col in self.feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        # إضافة News_Sentiment إذا كانت موجودة
        features = self.feature_cols.copy()
        if self.news_col in df_clean.columns:
            features.append(self.news_col)

        X = df_clean[features]

        # تحديد الهدف
        if mode == "short":
            y = df_clean['Close'].shift(-1)
        else:
            y = df_clean['Close'].shift(-45)

        valid_idx = y.notna()
        return X[valid_idx], y[valid_idx], X

    def train_all(self, df):
        """تدريب المحركين القصير والبعيد المدى"""
        if len(df) < 60:
            print(f"⚠️ بيانات غير كافية للتدريب. المطلوب: 60 يوم، المتوفر: {len(df)} يوم")
            return

        try:
            # أ. تدريب المحرك القصير
            X_s, y_s, _ = self.prepare_features(df, mode="short")
            
            if len(X_s) > 10:
                # حفظ معلومات التدريب للحد من التنبؤات غير المنطقية
                self.training_info['short']['mean'] = y_s.mean()
                self.training_info['short']['std'] = y_s.std()
                self.training_info['short']['samples'] = len(X_s)
                
                X_s_scaled = self.scaler_short.fit_transform(X_s)
                self.xgb_short.fit(X_s_scaled, y_s)
                self.rf_short.fit(X_s_scaled, y_s)
                print(f"✅ تم تدريب النموذج القصير على {len(X_s)} عينة")
            else:
                print(f"⚠️ بيانات غير كافية للتدريب القصير: {len(X_s)} عينة")

            # ب. تدريب المحرك الطويل
            X_l, y_l, _ = self.prepare_features(df, mode="long")
            
            if len(X_l) > 10:
                self.training_info['long']['mean'] = y_l.mean()
                self.training_info['long']['std'] = y_l.std()
                self.training_info['long']['samples'] = len(X_l)
                
                X_l_scaled = self.scaler_long.fit_transform(X_l)
                self.xgb_long.fit(X_l_scaled, y_l)
                self.rf_long.fit(X_l_scaled, y_l)
                print(f"✅ تم تدريب النموذج الطويل على {len(X_l)} عينة")
            else:
                print(f"⚠️ بيانات غير كافية للتدريب الطويل: {len(X_l)} عينة")

            self.is_trained = True
            print("✅ اكتمل تدريب محرك الذكاء الاصطناعي!")

        except Exception as e:
            print(f"❌ خطأ أثناء التدريب: {e}")
            self.is_trained = False

    def _validate_prediction(self, predicted_price, current_price):
        """
        التحقق من صحة التنبؤ ومنع القيم غير المنطقية
        
        Parameters:
        -----------
        predicted_price : float
            السعر المتوقع
        current_price : float
            السعر الحالي
            
        Returns:
        --------
        float: السعر المتوقع المعدل
        """
        # الحد الأقصى للتغير المسموح به (10% للقصير، 50% للطويل)
        max_change_pct = 0.10  # 10% للتداول القصير
        min_change_pct = -0.10
        
        # حساب التغير المتوقع
        change_pct = (predicted_price - current_price) / current_price
        
        # إذا كان التغير خارج الحدود، قم بتعديله
        if change_pct > max_change_pct:
            predicted_price = current_price * (1 + max_change_pct)
        elif change_pct < min_change_pct:
            predicted_price = current_price * (1 + min_change_pct)
        
        # التأكد من أن السعر لا يقل عن 0
        if predicted_price < 0:
            predicted_price = current_price * 0.5
        
        # التأكد من أن السعر ليس أكبر من 10 أضعاف السعر الحالي
        if predicted_price > current_price * 3:
            predicted_price = current_price * 1.1
        
        return predicted_price

    def predict_next_price(self, df, strategy_mode="مضاربة سريعة"):
        """
        حساب التنبؤ الفوري وصياغة القرار الميكانيكي
        """
        # تدريب النموذج إذا لم يكن مدرباً
        if not self.is_trained:
            self.train_all(df)
            
            if not self.is_trained:
                current_price = df['Close'].iloc[-1]
                return "⚠️ النموذج غير مدرب", current_price, current_price, current_price, 0

        # عمل نسخة محلية آمنة
        s_df = df.copy()
        for col in self.feature_cols:
            if col not in s_df.columns:
                s_df[col] = 0.0

        # استخراج القيم الحالية
        current_price = s_df['Close'].iloc[-1]
        atr = s_df['ATR'].iloc[-1] if 'ATR' in s_df.columns else (current_price * 0.02)
        cmf = s_df['CMF'].iloc[-1] if 'CMF' in s_df.columns else 0.0
        rsi = s_df['RSI'].iloc[-1] if 'RSI' in s_df.columns else 50.0
        news_sentiment = s_df['News_Sentiment'].iloc[-1] if 'News_Sentiment' in s_df.columns else 0.0

        # تجهيز آخر جلسة للتنبؤ
        features = self.feature_cols.copy()
        if self.news_col in s_df.columns:
            features.append(self.news_col)
        last_session = s_df[features].iloc[[-1]]

        # القيم الافتراضية
        decision_score = 50
        predicted_target = current_price
        best_entry = current_price
        best_exit = current_price
        direction = "مراقبة / انتظار إشارة السيولة ⏳"

        try:
            if "مضاربة" in strategy_mode:
                # ------ وضع المضاربة السريعة ------
                if len(self.scaler_short.mean_) > 0:
                    last_session_scaled = self.scaler_short.transform(last_session)
                    pred_xgb = self.xgb_short.predict(last_session_scaled)[0]
                    pred_rf = self.rf_short.predict(last_session_scaled)[0]
                    
                    # وزن النماذج
                    predicted_target = (pred_xgb * 0.6) + (pred_rf * 0.4)
                    
                    # ✅ التحقق من صحة التنبؤ ومنع القيم غير المنطقية
                    predicted_target = self._validate_prediction(predicted_target, current_price)
                else:
                    predicted_target = current_price

                # حساب نسبة الربح المتوقعة
                expected_gain_pct = ((predicted_target - current_price) / current_price) * 100

                # شروط الشراء
                buy_conditions = (
                    predicted_target > current_price * 1.002 and
                    cmf > -0.05 and
                    rsi < 72 and
                    news_sentiment >= -0.1 and
                    expected_gain_pct > 0.5  # ربح متوقع على الأقل 0.5%
                )

                # شروط البيع
                sell_conditions = (
                    predicted_target < current_price * 0.998 or
                    rsi > 78 or
                    news_sentiment <= -0.4 or
                    expected_gain_pct < -0.5
                )

                if buy_conditions:
                    direction = f"شراء مضاربي لقطة 🟢 (+{expected_gain_pct:.2f}%)"
                    best_entry = min(current_price, current_price - (0.2 * atr))
                    best_exit = predicted_target + (0.3 * atr)
                    decision_score = 75 + min(20, abs(expected_gain_pct) * 2)

                elif sell_conditions:
                    direction = "خروج / تجنب السهم 🔴"
                    best_entry = current_price
                    best_exit = current_price - (1.5 * atr)
                    decision_score = 20

                else:
                    direction = "مراقبة / انتظار إشارة السيولة ⏳"
                    best_entry = current_price
                    best_exit = current_price
                    decision_score = 50

            else:
                # ------ وضع قناص الموجات الاستثمارية ------
                if len(self.scaler_long.mean_) > 0:
                    last_session_scaled = self.scaler_long.transform(last_session)
                    pred_xgb = self.xgb_long.predict(last_session_scaled)[0]
                    pred_rf = self.rf_long.predict(last_session_scaled)[0]
                    predicted_target = (pred_xgb * 0.5) + (pred_rf * 0.5)
                    
                    # ✅ التحقق من صحة التنبؤ (حد أقصى 50% للطويل)
                    max_long_change = 0.50
                    change_pct = (predicted_target - current_price) / current_price
                    if change_pct > max_long_change:
                        predicted_target = current_price * (1 + max_long_change)
                    elif change_pct < -max_long_change:
                        predicted_target = current_price * (1 - max_long_change)
                else:
                    predicted_target = current_price

                expected_gain_pct = ((predicted_target - current_price) / current_price) * 100

                if expected_gain_pct >= 25.0 and cmf > -0.02 and news_sentiment >= 0.0:
                    direction = f"🚀 انفجار موجي استثماري (+{expected_gain_pct:.1f}%)"
                    best_entry = current_price
                    best_exit = predicted_target
                    decision_score = 90
                elif expected_gain_pct >= 10.0 and news_sentiment >= -0.2:
                    direction = f"📈 موجة صعود متوسطة (+{expected_gain_pct:.1f}%)"
                    best_entry = current_price
                    best_exit = predicted_target
                    decision_score = 70
                else:
                    direction = "❌ لا يصلح للاستثمار طويل المدى حالياً"
                    best_entry = current_price
                    best_exit = current_price
                    decision_score = 30

        except Exception as e:
            print(f"⚠️ خطأ في التنبؤ: {e}")
            direction = f"⚠️ خطأ في التنبؤ"
            predicted_target = current_price
            best_entry = current_price
            best_exit = current_price
            decision_score = 0

        # التأكد من أن best_exit منطقي
        if "شراء" in direction and best_exit <= best_entry:
            best_exit = best_entry + (atr * 0.5)

        # التأكد من أن السعر المتوقع معروض بشكل منطقي
        if abs(predicted_target - current_price) / current_price > 0.50:
            predicted_target = current_price * 1.05  # افتراض ربح 5% كحد أقصى

        return direction, predicted_target, best_entry, best_exit, decision_score

    def get_model_info(self):
        """الحصول على معلومات عن النماذج المدربة"""
        return {
            'is_trained': self.is_trained,
            'training_info': self.training_info,
            'feature_count': len(self.feature_cols)
        }
