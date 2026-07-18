import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)


class EnsemblePredictor:
    """محرك ذكاء اصطناعي مزدوج (Dual-Core): يدعم فحص حركة الأسعار والسيولة بالتكامل مع مشاعر الأخبار"""

    def __init__(self):
        # 1. نماذج المضاربة السريعة (توقع الجلسة القادمة)
        self.xgb_short = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
        self.rf_short = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)

        # 2. نماذج الاستثمار الموجي (توقع الاتجاه بعد شهرين / 45 جلسة تداول)
        self.xgb_long = XGBRegressor(n_estimators=250, max_depth=7, learning_rate=0.03, random_state=42, n_jobs=-1)
        self.rf_long = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)

        self.scaler_short = StandardScaler()
        self.scaler_long = StandardScaler()
        self.is_trained = False

        # قائمة الميزات الشاملة والمحمية لمنع الـ KeyError تماماً
        self.feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'VWAP', 'CMF', 'Momentum',
            'Daily_Return', 'EMA_50', 'EMA_200', 'Dist_From_EMA200', 'Volume_Shock', 'OBV_Slope',
            'BB_Upper', 'BB_Lower',  # <--- إضافة حارات بولينجر رسمياً في الفهرس
            'News_Sentiment'
        ]

    def prepare_features(self, df, mode="short"):
        """تجهيز البيانات والميزات الفنية مع عزل الأهداف بناءً على وضع التداول"""
        df_clean = df.copy()

        # 🛡️ حصن الأمان الفوقي: التأكد من وجود كافة الأعمدة لتفادي الـ KeyError نهائياً
        for col in self.feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        X = df_clean[self.feature_cols]

        if mode == "short":
            y = df_clean['Close'].shift(-1)
        else:
            y = df_clean['Close'].shift(-45)

        valid_idx = y.notna()
        return X[valid_idx], y[valid_idx], X

    def train_all(self, df):
        """تدريب المحركين القصير والبعيد المدى بالتوازي من نفس البيانات الحية"""
        if len(df) < 60:
            return

        # أ. تدريب المحرك القصير (مضاربة)
        X_s, y_s, _ = self.prepare_features(df, mode="short")
        X_s_scaled = self.scaler_short.fit_transform(X_s)
        self.xgb_short.fit(X_s_scaled, y_s)
        self.rf_short.fit(X_s_scaled, y_s)

        # ب. تدريب المحرك الطويل (استثمار وموجات كبرى)
        X_l, y_l, _ = self.prepare_features(df, mode="long")
        if len(X_l) > 10:
            X_l_scaled = self.scaler_long.fit_transform(X_l)
            self.xgb_long.fit(X_l_scaled, y_l)
            self.rf_long.fit(X_l_scaled, y_l)

        self.is_trained = True

    def predict_next_price(self, df, strategy_mode="مضاربة سريعة"):
        """حساب التنبؤ الفوري وصياغة القرار الميكانيكي بالتوازي مع قياس نبرة الشارع المالي"""
        if not self.is_trained:
            self.train_all(df)

        # عمل نسخة محلية آمنة ومطابقة للفحص لحظر الأخطاء الصامتة
        s_df = df.copy()
        for col in self.feature_cols:
            if col not in s_df.columns:
                s_df[col] = 0.0

        current_price = s_df['Close'].iloc[-1]
        atr = s_df['ATR'].iloc[-1] if 'ATR' in s_df.columns else (current_price * 0.02)
        cmf = s_df['CMF'].iloc[-1] if 'CMF' in s_df.columns else 0.0
        rsi = s_df['RSI'].iloc[-1] if 'RSI' in s_df.columns else 50.0
        news_sentiment = s_df['News_Sentiment'].iloc[-1] if 'News_Sentiment' in s_df.columns else 0.0

        last_session = s_df[self.feature_cols].iloc[[-1]]
        decision_score = 50  # القيمة الافتراضية لقوة الإشارة لمنع انهيار حساب المخاطر

        if "مضاربة" in strategy_mode:
            last_session_scaled = self.scaler_short.transform(last_session)
            pred_xgb = self.xgb_short.predict(last_session_scaled)[0]
            pred_rf = self.rf_short.predict(last_session_scaled)[0]
            predicted_target = (pred_xgb * 0.6) + (pred_rf * 0.4)

            if predicted_target > current_price * 1.005 and cmf > -0.05 and rsi < 72 and news_sentiment >= -0.1:
                direction = "شراء مضاربي لقطة 🟢"
                best_entry = min(current_price, current_price - (0.2 * atr))
                best_exit = predicted_target + (0.5 * atr)
                decision_score = 85
            elif predicted_target < current_price * 0.995 or rsi > 78 or news_sentiment <= -0.4:
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
            # وضع قناص الموجات الاستثمارية
            last_session_scaled = self.scaler_long.transform(last_session)
            pred_xgb = self.xgb_long.predict(last_session_scaled)[0]
            pred_rf = self.rf_long.predict(last_session_scaled)[0]
            predicted_target = (pred_xgb * 0.5) + (pred_rf * 0.5)

            expected_gain_pct = ((predicted_target - current_price) / current_price) * 100

            if expected_gain_pct >= 35.0 and cmf > -0.02 and news_sentiment >= 0.0:
                direction = f"🚀 انفجار موجي استثماري لقطة (+{expected_gain_pct:.1f}%)"
                best_entry = current_price
                best_exit = predicted_target
                decision_score = 90
            elif expected_gain_pct >= 15.0 and news_sentiment >= -0.2:
                direction = f"📈 موجة صعود متوسطة المدى قيد التجميع (+{expected_gain_pct:.1f}%)"
                best_entry = current_price
                best_exit = predicted_target
                decision_score = 70
            else:
                direction = "❌ السهم لا يصلح للاستثمار طويل المدى حالياً"
                best_entry = current_price
                best_exit = current_price
                decision_score = 30

        # [إرجاع 5 قيم لتوافق سطر 242 بالتمام والكمال]
        return direction, predicted_target, best_entry, best_exit, decision_score