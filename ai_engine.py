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
    
    المميزات:
    - نموذجان للتنبؤ: قصير المدى (مضاربة) وطويل المدى (استثمار)
    - دمج بين XGBoost و Random Forest للتنبؤ الأكثر دقة
    - دعم كامل لمشاعر الأخبار في اتخاذ القرار
    - نظام تقييم ديناميكي لقوة الإشارة (Decision Score)
    """
    
    def __init__(self):
        # 1. نماذج المضاربة السريعة (توقع الجلسة القادمة)
        self.xgb_short = XGBRegressor(
            n_estimators=150, 
            max_depth=5, 
            learning_rate=0.05, 
            random_state=42, 
            n_jobs=-1,
            verbosity=0  # إخفاء رسائل التدريب
        )
        self.rf_short = RandomForestRegressor(
            n_estimators=100, 
            max_depth=6, 
            random_state=42, 
            n_jobs=-1
        )

        # 2. نماذج الاستثمار الموجي (توقع الاتجاه بعد شهرين / 45 جلسة تداول)
        self.xgb_long = XGBRegressor(
            n_estimators=250, 
            max_depth=7, 
            learning_rate=0.03, 
            random_state=42, 
            n_jobs=-1,
            verbosity=0
        )
        self.rf_long = RandomForestRegressor(
            n_estimators=150, 
            max_depth=8, 
            random_state=42, 
            n_jobs=-1
        )

        self.scaler_short = StandardScaler()
        self.scaler_long = StandardScaler()
        self.is_trained = False
        self.training_history = {
            'short': {'samples': 0, 'features': 0},
            'long': {'samples': 0, 'features': 0}
        }

        # قائمة الميزات الشاملة والمحمية
        self.feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD', 'VWAP', 'CMF', 'Momentum',
            'Daily_Return', 'EMA_50', 'EMA_200', 
            'Dist_From_EMA200', 'Volume_Shock', 'OBV_Slope',
            'BB_Upper', 'BB_Lower',
            'News_Sentiment', 'ATR'  # إضافة ATR للميزات
        ]

        # معايير إضافية للتحكم في جودة الإشارة
        self.quality_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'cmf_bullish': 0.05,
            'cmf_bearish': -0.05,
            'volume_spike': 1.5,
            'min_news_sentiment': -0.2,
            'max_news_sentiment': 0.2
        }

    def prepare_features(self, df, mode="short"):
        """
        تجهيز البيانات والميزات الفنية مع عزل الأهداف بناءً على وضع التداول
        
        Parameters:
        -----------
        df : pd.DataFrame
            البيانات الخام مع المؤشرات
        mode : str
            "short" للتداول القصير، "long" للاستثمار الطويل
            
        Returns:
        --------
        tuple: (X_features, y_target, X_full)
        """
        df_clean = df.copy()

        # 🛡️ التأكد من وجود كافة الأعمدة
        for col in self.feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        X = df_clean[self.feature_cols]

        # تحديد الهدف بناءً على وضع التداول
        if mode == "short":
            y = df_clean['Close'].shift(-1)  # توقع سعر الغد
        else:
            y = df_clean['Close'].shift(-45)  # توقع سعر بعد 45 جلسة

        # إزالة القيم الفارغة
        valid_idx = y.notna()
        
        # تسجيل معلومات التدريب
        self.training_history[mode] = {
            'samples': len(X[valid_idx]),
            'features': len(self.feature_cols)
        }

        return X[valid_idx], y[valid_idx], X

    def train_all(self, df):
        """
        تدريب المحركين القصير والبعيد المدى بالتوازي من نفس البيانات الحية
        
        Parameters:
        -----------
        df : pd.DataFrame
            البيانات التاريخية مع جميع المؤشرات
        """
        if len(df) < 60:
            print(f"⚠️ بيانات غير كافية للتدريب. المطلوب: 60 يوم، المتوفر: {len(df)} يوم")
            return

        try:
            # أ. تدريب المحرك القصير (مضاربة)
            X_s, y_s, _ = self.prepare_features(df, mode="short")
            
            if len(X_s) > 10:
                X_s_scaled = self.scaler_short.fit_transform(X_s)
                self.xgb_short.fit(X_s_scaled, y_s)
                self.rf_short.fit(X_s_scaled, y_s)
                print(f"✅ تم تدريب النموذج القصير على {len(X_s)} عينة")
            else:
                print(f"⚠️ بيانات غير كافية للتدريب القصير: {len(X_s)} عينة")

            # ب. تدريب المحرك الطويل (استثمار وموجات كبرى)
            X_l, y_l, _ = self.prepare_features(df, mode="long")
            
            if len(X_l) > 10:
                X_l_scaled = self.scaler_long.fit_transform(X_l)
                self.xgb_long.fit(X_l_scaled, y_l)
                self.rf_long.fit(X_l_scaled, y_l)
                print(f"✅ تم تدريب النموذج الطويل على {len(X_l)} عينة")
            else:
                print(f"⚠️ بيانات غير كافية للتدريب الطويل: {len(X_l)} عينة")

            self.is_trained = True
            print("✅ اكتمل تدريب محرك الذكاء الاصطناعي بنجاح!")

        except Exception as e:
            print(f"❌ خطأ أثناء تدريب النماذج: {e}")
            self.is_trained = False

    def predict_next_price(self, df, strategy_mode="مضاربة سريعة"):
        """
        حساب التنبؤ الفوري وصياغة القرار الميكانيكي بالتوازي مع قياس نبرة الشارع المالي
        
        Parameters:
        -----------
        df : pd.DataFrame
            البيانات الحالية مع المؤشرات
        strategy_mode : str
            "مضاربة سريعة" أو "قناص الموجات الاستثمارية"
            
        Returns:
        --------
        tuple: (direction, predicted_target, best_entry, best_exit, decision_score)
        """
        # تدريب النموذج إذا لم يكن مدرباً
        if not self.is_trained:
            self.train_all(df)
            
            # إذا فشل التدريب، استخدم قيم افتراضية
            if not self.is_trained:
                current_price = df['Close'].iloc[-1]
                return "⚠️ النموذج غير مدرب - استخدم القيم الافتراضية", current_price, current_price, current_price, 0

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
        volume_ratio = s_df['Volume'].iloc[-1] / (s_df['Volume'].rolling(20).mean().iloc[-1] + 1e-10) if len(s_df) > 20 else 1.0

        # تجهيز آخر جلسة للتنبؤ
        last_session = s_df[self.feature_cols].iloc[[-1]]
        
        # القيم الافتراضية
        decision_score = 50
        predicted_target = current_price
        best_entry = current_price
        best_exit = current_price
        direction = "مراقبة / انتظار إشارة السيولة ⏳"

        try:
            if "مضاربة" in strategy_mode:
                # ------ وضع المضاربة السريعة ------
                last_session_scaled = self.scaler_short.transform(last_session)
                pred_xgb = self.xgb_short.predict(last_session_scaled)[0]
                pred_rf = self.rf_short.predict(last_session_scaled)[0]
                predicted_target = (pred_xgb * 0.6) + (pred_rf * 0.4)

                # حساب نسبة الربح المتوقعة
                expected_gain_pct = ((predicted_target - current_price) / current_price) * 100

                # شروط الشراء القوية
                buy_conditions = (
                    predicted_target > current_price * 1.005 and
                    cmf > self.quality_thresholds['cmf_bullish'] and
                    rsi < self.quality_thresholds['rsi_overbought'] and
                    news_sentiment >= self.quality_thresholds['min_news_sentiment']
                )

                # شروط البيع القوية
                sell_conditions = (
                    predicted_target < current_price * 0.995 or
                    rsi > self.quality_thresholds['rsi_overbought'] + 8 or
                    news_sentiment <= self.quality_thresholds['min_news_sentiment'] - 0.2
                )

                if buy_conditions:
                    # شراء مضاربي
                    direction = f"شراء مضاربي لقطة 🟢 (متوقع +{expected_gain_pct:.1f}%)"
                    best_entry = min(current_price, current_price - (0.2 * atr))
                    best_exit = predicted_target + (0.5 * atr)
                    
                    # حساب درجة الثقة
                    confidence = 70
                    if cmf > 0.15:
                        confidence += 10
                    if news_sentiment > 0.2:
                        confidence += 10
                    if volume_ratio > 1.5:
                        confidence += 5
                    decision_score = min(95, confidence)

                elif sell_conditions:
                    # خروج / تجنب
                    direction = "خروج / تجنب السهم 🔴"
                    best_entry = current_price
                    best_exit = current_price - (1.5 * atr)
                    decision_score = 20

                else:
                    # منطقة انتظار
                    direction = "مراقبة / انتظار إشارة السيولة ⏳"
                    best_entry = current_price
                    best_exit = current_price
                    
                    # تقييم إمكانية الدخول لاحقاً
                    if cmf > 0 and rsi < 60:
                        decision_score = 55
                    elif cmf < 0 and rsi > 60:
                        decision_score = 45
                    else:
                        decision_score = 50

            else:
                # ------ وضع قناص الموجات الاستثمارية ------
                last_session_scaled = self.scaler_long.transform(last_session)
                pred_xgb = self.xgb_long.predict(last_session_scaled)[0]
                pred_rf = self.rf_long.predict(last_session_scaled)[0]
                predicted_target = (pred_xgb * 0.5) + (pred_rf * 0.5)

                expected_gain_pct = ((predicted_target - current_price) / current_price) * 100

                # شروط الاستثمار القوي
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
                    
                    if expected_gain_pct < -10:
                        decision_score = 10
                    elif expected_gain_pct < 0:
                        decision_score = 20
                    else:
                        decision_score = 30

        except Exception as e:
            print(f"⚠️ خطأ في التنبؤ: {e}")
            direction = f"⚠️ خطأ في التنبؤ: {str(e)[:30]}"
            predicted_target = current_price
            best_entry = current_price
            best_exit = current_price
            decision_score = 0

        # التأكد من أن best_exit أكبر من best_entry للشراء
        if "شراء" in direction and best_exit <= best_entry:
            best_exit = best_entry + (atr * 0.5)

        return direction, predicted_target, best_entry, best_exit, decision_score

    def get_model_info(self):
        """
        الحصول على معلومات عن النماذج المدربة
        
        Returns:
        --------
        dict: معلومات التدريب
        """
        return {
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_count': len(self.feature_cols),
            'features': self.feature_cols
        }

    def evaluate_prediction_quality(self, predicted_price, current_price, indicators):
        """
        تقييم جودة التنبؤ بناءً على المؤشرات الحالية
        
        Parameters:
        -----------
        predicted_price : float
            السعر المتوقع
        current_price : float
            السعر الحالي
        indicators : dict
            المؤشرات الحالية (RSI, CMF, ATR, etc.)
            
        Returns:
        --------
        dict: تقييم الجودة
        """
        quality_score = 50
        reasons = []

        # 1. تقييم بناءً على نسبة الربح المتوقعة
        gain_pct = ((predicted_price - current_price) / current_price) * 100
        if gain_pct > 2:
            quality_score += 15
            reasons.append(f"ربح متوقع عالي: {gain_pct:.1f}%")
        elif gain_pct > 1:
            quality_score += 10
            reasons.append(f"ربح متوقع جيد: {gain_pct:.1f}%")
        elif gain_pct < -2:
            quality_score -= 15
            reasons.append(f"خسارة متوقعة: {gain_pct:.1f}%")

        # 2. تقييم بناءً على RSI
        rsi = indicators.get('RSI', 50)
        if 40 < rsi < 60:
            quality_score += 10
            reasons.append(f"RSI متعادل: {rsi:.1f}")
        elif 30 < rsi < 70:
            quality_score += 5
            reasons.append(f"RSI مقبول: {rsi:.1f}")
        else:
            quality_score -= 10
            reasons.append(f"RSI متطرف: {rsi:.1f}")

        # 3. تقييم بناءً على CMF
        cmf = indicators.get('CMF', 0)
        if cmf > 0.1:
            quality_score += 10
            reasons.append(f"تدفق سيولة إيجابي: {cmf:.2f}")
        elif cmf < -0.1:
            quality_score -= 10
            reasons.append(f"تدفق سيولة سلبي: {cmf:.2f}")

        # 4. تقييم بناءً على الأخبار
        news = indicators.get('News_Sentiment', 0)
        if news > 0.2:
            quality_score += 10
            reasons.append(f"أخبار إيجابية: {news:.2f}")
        elif news < -0.2:
            quality_score -= 10
            reasons.append(f"أخبار سلبية: {news:.2f}")

        # ضمان النتيجة بين 0 و 100
        quality_score = max(0, min(100, quality_score))

        return {
            'score': quality_score,
            'reasons': reasons,
            'is_reliable': quality_score >= 60,
            'gain_pct': gain_pct
        }


# دالة مساعدة للاستخدام السريع
def quick_predict(df, mode="مضاربة سريعة"):
    """
    دالة سريعة للتنبؤ بدون الحاجة لإنشاء الكلاس يدوياً
    
    Parameters:
    -----------
    df : pd.DataFrame
        البيانات مع المؤشرات
    mode : str
        "مضاربة سريعة" أو "قناص الموجات الاستثمارية"
        
    Returns:
    --------
    tuple: (direction, predicted_target, best_entry, best_exit, decision_score)
    """
    predictor = EnsemblePredictor()
    predictor.train_all(df)
    return predictor.predict_next_price(df, mode)
