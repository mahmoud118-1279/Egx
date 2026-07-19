import numpy as np
import pandas as pd


def calculate_rsi(df, period=14):
    """
    حساب مؤشر القوة النسبية RSI
    يقيس سرعة وحجم تغير حركة السعر لتحديد مناطق ذروة الشراء والبيع
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, slow=26, fast=12, smooth=9):
    """
    حساب مؤشر الماكد MACD وخط الإشارة
    يستخدم لتحديد اتجاه السوق ونقاط التحول المحتملة
    """
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=smooth, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    حساب حارات بولينجر Bollinger Bands
    تستخدم لتحديد مستويات التذبذب العلوية والسفلية للسعر
    """
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * std)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']  # عرض الحارة - مؤشر على التقلب
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)  # موقع السعر داخل الحارة
    return df


def calculate_atr(df, period=14):
    """
    حساب متوسط المدى الحقيقي ATR
    يقيس التقلبات السعرية ويستخدم في تحديد وقف الخسارة
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(period).mean()
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100  # نسبة ATR للسعر
    return df


def calculate_cmf(df, period=20):
    """
    حساب مؤشر تدفق الأموال Chaikin Money Flow
    يقيس ضغط الشراء والبيع بناءً على السعر والحجم
    """
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    money_flow_volume = money_flow_multiplier * df['Volume']
    df['CMF'] = money_flow_volume.rolling(period).sum() / (df['Volume'].rolling(period).sum() + 1e-10)
    return df


def calculate_mfi(df, period=14):
    """
    حساب مؤشر تدفق الأموال MFI (Money Flow Index)
    يشبه الـ RSI ولكن مع إضافة عامل الحجم لتأكيد قوة الاتجاه
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
    neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
    mfi_ratio = pos_flow / (neg_flow + 1e-10)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    return df


def calculate_vwap(df):
    """
    حساب المتوسط المرجح بالحجم VWAP
    يستخدم من قبل المؤسسات لتحديد متوسط سعر التداول العادل
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # محاولة حساب VWAP يومياً إذا كان هناك تاريخ
    if hasattr(df.index, 'date') or isinstance(df.index, pd.DatetimeIndex):
        try:
            # إضافة عمود اليوم
            df['day'] = df.index.date if hasattr(df.index, 'date') else pd.to_datetime(df.index).date
            
            # حساب VWAP لكل يوم على حدة
            vwap_series = df.groupby('day').apply(
                lambda x: (x['Volume'] * (x['High'] + x['Low'] + x['Close']) / 3).cumsum() / (x['Volume'].cumsum() + 1e-10)
            )
            df['VWAP'] = vwap_series.reset_index(level=0, drop=True)
            df.drop('day', axis=1, inplace=True)
        except Exception:
            # إذا فشل، استخدم النسخة التراكمية
            df['VWAP'] = (df['Volume'] * typical_price).cumsum() / (df['Volume'].cumsum() + 1e-10)
    else:
        # النسخة التراكمية
        df['VWAP'] = (df['Volume'] * typical_price).cumsum() / (df['Volume'].cumsum() + 1e-10)
    
    df['Dist_From_VWAP'] = (df['Close'] - df['VWAP']) / (df['VWAP'] + 1e-10)
    return df


def calculate_choppiness_index(df, period=14):
    """
    حساب مؤشر الفوضى Choppiness Index
    يحدد ما إذا كان السوق في اتجاه (قيمة منخفضة) أم في نطاق تذبذب (قيمة مرتفعة)
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr_sum = true_range.rolling(period).sum()
    max_high = df['High'].rolling(period).max()
    min_low = df['Low'].rolling(period).min()
    
    df['Chop_Index'] = 100 * (np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(period))
    
    # إضافة تفسير مبسط
    df['Market_Type'] = np.where(df['Chop_Index'] > 61.8, 'تذبذب ⚡', 
                                 np.where(df['Chop_Index'] < 38.2, 'اتجاه قوي 📈', 'متوسط ⚖️'))
    return df


def calculate_divergence_signals(df):
    """
    حساب إشارات الدايفرنس بين السعر ومؤشر RSI
    تكشف عن انعكاسات الاتجاه المحتملة قبل حدوثها
    """
    df['Price_Slope'] = df['Close'].diff(5)
    df['RSI_Slope'] = df['RSI'].diff(5)
    
    # دايفرنس إيجابي (انعكاس صعودي): السعر يهبط وRSI يصعد
    df['Divergence_Signal'] = np.where(
        (df['Price_Slope'] < 0) & (df['RSI_Slope'] > 0), 
        1.0,  # إشارة شراء
        np.where(
            (df['Price_Slope'] > 0) & (df['RSI_Slope'] < 0), 
            -1.0,  # إشارة بيع
            0.0   # لا توجد إشارة
        )
    )
    
    # قوة الإشارة (كلما زادت القيمة المطلقة، زادت قوة الإشارة)
    df['Divergence_Strength'] = np.abs(df['Price_Slope'] - df['RSI_Slope']) / (df['Close'] + 1e-10)
    return df


def calculate_obv(df):
    """
    حساب مؤشر حجم التداول التراكمي OBV (On-Balance Volume)
    يربط تغيرات السعر مع حجم التداول لتأكيد الاتجاهات
    """
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = df['OBV'].diff(10)  # ميل الـ OBV
    df['OBV_MA'] = df['OBV'].rolling(20).mean()  # متوسط الـ OBV
    df['OBV_RSI'] = calculate_rsi(pd.DataFrame({'Close': df['OBV']}), 14)['RSI']  # RSI للـ OBV
    return df


def calculate_ema_signals(df):
    """
    حساب المتوسطات المتحركة الأسية وإشارات التقاطع
    """
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # المسافة من المتوسطات
    df['Dist_From_EMA50'] = (df['Close'] - df['EMA_50']) / (df['EMA_50'] + 1e-10)
    df['Dist_From_EMA200'] = (df['Close'] - df['EMA_200']) / (df['EMA_200'] + 1e-10)
    
    # إشارات التقاطع
    df['Golden_Cross'] = np.where((df['EMA_50'] > df['EMA_200']) & (df['EMA_50'].shift(1) <= df['EMA_200'].shift(1)), 1, 0)
    df['Death_Cross'] = np.where((df['EMA_50'] < df['EMA_200']) & (df['EMA_50'].shift(1) >= df['EMA_200'].shift(1)), 1, 0)
    
    # ترتيب المتوسطات (لتحديد الاتجاه)
    df['EMA_Alignment'] = np.where(
        (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_100']) & (df['EMA_100'] > df['EMA_200']),
        'ترتيب صاعد 📈',
        np.where(
            (df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_100']) & (df['EMA_100'] < df['EMA_200']),
            'ترتيب هابط 📉',
            'متعارض ⚖️'
        )
    )
    return df


def calculate_adx(df, period=14):
    """
    حساب مؤشر الاتجاه المتوسط ADX (Average Directional Index)
    يقيس قوة الاتجاه بغض النظر عن اتجاهه
    """
    # حساب الـ True Range
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    
    # حساب +DM و -DM
    df['+DM'] = np.where(
        (df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
        np.maximum(df['High'] - df['High'].shift(), 0), 
        0
    )
    df['-DM'] = np.where(
        (df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
        np.maximum(df['Low'].shift() - df['Low'], 0),
        0
    )
    
    # حساب الـ ATR
    df['ATR_ADX'] = df['TR'].rolling(period).mean()
    
    # حساب الـ DI
    df['+DI'] = 100 * (df['+DM'].rolling(period).mean() / (df['ATR_ADX'] + 1e-10))
    df['-DI'] = 100 * (df['-DM'].rolling(period).mean() / (df['ATR_ADX'] + 1e-10))
    
    # حساب الـ DX و ADX
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10)
    df['ADX'] = df['DX'].rolling(period).mean()
    
    # تفسير قوة الاتجاه
    df['Trend_Strength'] = np.where(
        df['ADX'] > 50, 'قوي جداً 💪',
        np.where(
            df['ADX'] > 25, 'قوي ✅',
            np.where(
                df['ADX'] > 20, 'متوسط ⚖️',
                'ضعيف ❌'
            )
        )
    )
    
    # اتجاه الـ ADX (متزايد أم متناقص)
    df['ADX_Trend'] = np.where(df['ADX'] > df['ADX'].shift(1), 'تزايد 📈', 'تناقص 📉')
    
    return df


def calculate_ichimoku(df, tenkan=9, kijun=26, senkou=52):
    """
    حساب سحابة إيشيموكو Ichimoku Cloud
    نظام شامل لتحديد الدعم والمقاومة والاتجاه وزخم السوق
    """
    # Tenkan-sen (Conversion Line) - خط التحويل
    period1_max = df['High'].rolling(tenkan).max()
    period1_min = df['Low'].rolling(tenkan).min()
    df['Ichimoku_Tenkan'] = (period1_max + period1_min) / 2
    
    # Kijun-sen (Base Line) - الخط الأساسي
    period2_max = df['High'].rolling(kijun).max()
    period2_min = df['Low'].rolling(kijun).min()
    df['Ichimoku_Kijun'] = (period2_max + period2_min) / 2
    
    # Senkou Span A (Leading Span A) - النطاق المتقدم أ
    df['Ichimoku_SenkouA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B) - النطاق المتقدم ب
    period3_max = df['High'].rolling(senkou).max()
    period3_min = df['Low'].rolling(senkou).min()
    df['Ichimoku_SenkouB'] = ((period3_max + period3_min) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span) - النطاق المتأخر
    df['Ichimoku_Chikou'] = df['Close'].shift(-kijun)
    
    # إشارات إيشيموكو
    # 1. السعر فوق/تحت السحابة
    df['Ichimoku_Cloud_Position'] = np.where(
        df['Close'] > df['Ichimoku_SenkouA'],
        np.where(df['Close'] > df['Ichimoku_SenkouB'], 'فوق السحابة ☀️', 'داخل السحابة 🌥️'),
        np.where(df['Close'] < df['Ichimoku_SenkouB'], 'تحت السحابة 🌧️', 'داخل السحابة 🌥️')
    )
    
    # 2. تقاطع Tenkan و Kijun
    df['Ichimoku_TK_Cross'] = np.where(
        (df['Ichimoku_Tenkan'] > df['Ichimoku_Kijun']) & (df['Ichimoku_Tenkan'].shift(1) <= df['Ichimoku_Kijun'].shift(1)),
        'تقاطع صاعد 🟢',
        np.where(
            (df['Ichimoku_Tenkan'] < df['Ichimoku_Kijun']) & (df['Ichimoku_Tenkan'].shift(1) >= df['Ichimoku_Kijun'].shift(1)),
            'تقاطع هابط 🔴',
            'لا يوجد تقاطع ⏳'
        )
    )
    
    return df


def calculate_fibonacci_levels(df, lookback=14):
    """
    حساب مستويات فيبوناتشي للدعم والمقاومة
    تستخدم لتحديد نقاط الارتداد المحتملة
    """
    # استخدام أعلى وأدنى سعر خلال الفترة المحددة
    high = df['High'].rolling(lookback).max().iloc[-1]
    low = df['Low'].rolling(lookback).min().iloc[-1]
    diff = high - low
    
    fib_levels = {
        'Fib_0': round(high, 2),                    # 0%
        'Fib_23.6': round(high - 0.236 * diff, 2),   # 23.6%
        'Fib_38.2': round(high - 0.382 * diff, 2),   # 38.2%
        'Fib_50': round(high - 0.5 * diff, 2),       # 50%
        'Fib_61.8': round(high - 0.618 * diff, 2),   # 61.8%
        'Fib_78.6': round(high - 0.786 * diff, 2),   # 78.6%
        'Fib_100': round(low, 2)                     # 100%
    }
    
    # إضافة مستويات فيبوناتشي كأعمدة في الـ DataFrame
    for key, value in fib_levels.items():
        df[key] = value
    
    return df, fib_levels


def calculate_stochastic_rsi(df, period=14, smooth_k=3, smooth_d=3):
    """
    حساب مؤشر Stochastic RSI
    يجمع بين الـ RSI والـ Stochastic لتقديم إشارات أكثر دقة
    """
    # حساب RSI أولاً
    if 'RSI' not in df.columns:
        df = calculate_rsi(df, period)
    
    # حساب Stochastic من RSI
    lowest_rsi = df['RSI'].rolling(period).min()
    highest_rsi = df['RSI'].rolling(period).max()
    
    df['Stoch_RSI_K'] = 100 * (df['RSI'] - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)
    df['Stoch_RSI_D'] = df['Stoch_RSI_K'].rolling(smooth_d).mean()
    
    # إشارات ذروة الشراء والبيع
    df['Stoch_RSI_Signal'] = np.where(
        (df['Stoch_RSI_K'] < 20) & (df['Stoch_RSI_K'] > df['Stoch_RSI_D']),
        'شراء 🟢',
        np.where(
            (df['Stoch_RSI_K'] > 80) & (df['Stoch_RSI_K'] < df['Stoch_RSI_D']),
            'بيع 🔴',
            'محايد ⚖️'
        )
    )
    
    return df


def calculate_volume_profile(df):
    """
    تحليل حجم التداول المتقدم
    """
    # حجم التداول النسبي
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
    
    # صدمة الحجم (تغير مفاجئ في الحجم)
    df['Volume_Shock'] = df['Volume'] / (df['Volume'].rolling(22).mean() + 1e-10)
    df['Volume_Shock_Flag'] = np.where(df['Volume_Shock'] > 2, 'صدمة حجم 🔥', 'طبيعي ✅')
    
    # توزيع الحجم مع تغير السعر
    df['Volume_Price_Change'] = df['Volume'] * df['Daily_Return']
    df['Volume_Price_Change_MA'] = df['Volume_Price_Change'].rolling(10).mean()
    
    return df


def calculate_candlestick_patterns(df):
    """
    حساب أنماط الشموع اليابانية الأساسية
    """
    # Doji (الدوجي) - شمعة ذات جسم صغير جداً
    body_size = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low'] + 1e-10
    df['Doji'] = np.where(body_size / total_range < 0.1, 1, 0)
    
    # Hammer (المطرقة) - شمعة انعكاسية صاعدة
    lower_shadow = df['Low'].rolling(2).min()  # تبسيط
    upper_shadow = df['High'].rolling(2).max()  # تبسيط
    df['Hammer'] = np.where(
        (lower_shadow > 2 * body_size) & 
        (upper_shadow < body_size * 0.5) & 
        (df['Close'] > df['Open']),
        1, 0
    )
    
    # Shooting Star (النجمة الهابطة) - شمعة انعكاسية هابطة
    df['Shooting_Star'] = np.where(
        (upper_shadow > 2 * body_size) & 
        (lower_shadow < body_size * 0.5) & 
        (df['Close'] < df['Open']),
        1, 0
    )
    
    # Bullish Engulfing (الابتلاع الصاعد)
    df['Bullish_Engulfing'] = np.where(
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1)),
        1, 0
    )
    
    # Bearish Engulfing (الابتلاع الهابط)
    df['Bearish_Engulfing'] = np.where(
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1)),
        1, 0
    )
    
    return df


def calculate_pivot_points(df):
    """
    حساب نقاط البيفوت (الدعم والمقاومة) الكلاسيكية والمطورة
    تستخدم لتحديد مستويات الدعم والمقاومة الرئيسية
    """
    if len(df) < 2:
        return {
            'Pivot': 0, 'R1': 0, 'R2': 0, 'R3': 0,
            'S1': 0, 'S2': 0, 'S3': 0
        }
    
    # استخدام آخر شمعة مكتملة
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]
    close = df['Close'].iloc[-1]
    
    # حساب البيفوت الأساسي
    pivot = (high + low + close) / 3
    
    # مستويات المقاومة (R1, R2, R3)
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    # مستويات الدعم (S1, S2, S3)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    # حساب البيفوتات الأسبوعية (نسخة مطورة)
    pivot_week = {
        'Pivot': round(pivot, 2),
        'R1': round(r1, 2),
        'R2': round(r2, 2),
        'R3': round(r3, 2),
        'S1': round(s1, 2),
        'S2': round(s2, 2),
        'S3': round(s3, 2)
    }
    
    # إضافة نقاط البيفوت كأعمدة في الـ DataFrame
    for key, value in pivot_week.items():
        df[key] = value
    
    # تحديد موقع السعر الحالي بالنسبة للبيفوت
    df['Pivot_Position'] = np.where(
        df['Close'] > df['Pivot'],
        'فوق البيفوت 📈',
        np.where(
            df['Close'] < df['Pivot'],
            'تحت البيفوت 📉',
            'عند البيفوت ⚖️'
        )
    )
    
    return pivot_week


def add_all_indicators(df):
    """
    حقن وتطوير كافة الميزات الفنية والسيولية والذكية لتغذية عقل الآلة
    هذه هي الدالة الرئيسية التي تجمع كل المؤشرات
    """
    if len(df) < 5:
        return df
    
    # 1. المؤشرات الأساسية
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)
    
    # 2. مؤشرات السيولة والحجم
    df = calculate_cmf(df)
    df = calculate_mfi(df)
    df = calculate_vwap(df)
    df = calculate_obv(df)
    
    # 3. مؤشرات الاتجاه والقوة
    df = calculate_adx(df)
    df = calculate_choppiness_index(df)
    df = calculate_ema_signals(df)
    
    # 4. مؤشرات إيشيموكو وفيبوناتشي
    df = calculate_ichimoku(df)
    fib_levels, _ = calculate_fibonacci_levels(df)
    
    # 5. المؤشرات الذكية المتقدمة
    df = calculate_divergence_signals(df)
    df = calculate_stochastic_rsi(df)
    df = calculate_volume_profile(df)
    df = calculate_candlestick_patterns(df)
    
    # 6. المؤشرات الإضافية
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 7. حقن عمود تحليل مشاعر الأخبار (سيتم تعبئته لاحقاً)
    if 'News_Sentiment' not in df.columns:
        df['News_Sentiment'] = 0.0
    
    # 8. تنظيف وتعبئة القيم الفارغة
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].bfill().ffill().fillna(0)
    
    # 9. حساب النقاط المدمجة لتقييم جودة السهم
    df['Overall_Score'] = calculate_overall_score(df)
    
    return df


def calculate_overall_score(df):
    """
    حساب درجة شاملة لتقييم جودة السهم
    تجمع بين جميع المؤشرات في رقم واحد من 0 إلى 100
    """
    score = 0
    latest = df.iloc[-1]
    
    # 1. RSI (0-100) - نريد قراءة متوسطة
    rsi = latest.get('RSI', 50)
    if 40 < rsi < 60:
        score += 10
    elif 30 < rsi < 70:
        score += 5
    
    # 2. MACD إيجابي
    if latest.get('MACD', 0) > 0:
        score += 10
    if latest.get('MACD_Hist', 0) > 0:
        score += 5
    
    # 3. السعر فوق المتوسطات
    if latest.get('Close', 0) > latest.get('EMA_50', 0):
        score += 10
    if latest.get('Close', 0) > latest.get('EMA_200', 0):
        score += 10
    
    # 4. قوة الاتجاه
    adx = latest.get('ADX', 0)
    if adx > 25:
        score += 10
    elif adx > 20:
        score += 5
    
    # 5. تدفق الأموال
    cmf = latest.get('CMF', 0)
    if cmf > 0.1:
        score += 10
    elif cmf > 0:
        score += 5
    
    # 6. MFI
    mfi = latest.get('MFI', 50)
    if 40 < mfi < 60:
        score += 10
    elif 30 < mfi < 70:
        score += 5
    
    # 7. الحجم
    volume_ratio = latest.get('Volume_Ratio', 1)
    if volume_ratio > 1.5:
        score += 10
    elif volume_ratio > 1.2:
        score += 5
    
    # 8. إشارات الدايفرنس
    divergence = latest.get('Divergence_Signal', 0)
    if divergence > 0:
        score += 10
    elif divergence < 0:
        score -= 5
    
    # 9. تقاطع إيشيموكو
    tk_cross = latest.get('Ichimoku_TK_Cross', 'لا يوجد تقاطع ⏳')
    if 'صاعد' in tk_cross:
        score += 10
    elif 'هابط' in tk_cross:
        score -= 5
    
    # 10. موقع السعر بالنسبة للبيفوت
    pivot_pos = latest.get('Pivot_Position', 'عند البيفوت ⚖️')
    if 'فوق' in pivot_pos:
        score += 5
    elif 'تحت' in pivot_pos:
        score -= 5
    
    # ضمان أن النتيجة بين 0 و 100
    return max(0, min(100, score))


def get_market_summary(df):
    """
    توليد ملخص شامل لحالة السوق الحالية بناءً على جميع المؤشرات
    """
    if len(df) < 2:
        return "بيانات غير كافية للتحليل"
    
    latest = df.iloc[-1]
    
    summary = {
        'السعر_الحالي': round(latest.get('Close', 0), 2),
        'RSI': round(latest.get('RSI', 50), 2),
        'MACD': round(latest.get('MACD', 0), 4),
        'ADX': round(latest.get('ADX', 0), 2),
        'CMF': round(latest.get('CMF', 0), 4),
        'MFI': round(latest.get('MFI', 50), 2),
        'الدرجة_الكلية': round(latest.get('Overall_Score', 0), 1),
        'الاتجاه': latest.get('EMA_Alignment', 'غير محدد'),
        'قوة_الاتجاه': latest.get('Trend_Strength', 'غير محدد'),
        'نوع_السوق': latest.get('Market_Type', 'غير محدد'),
        'موقع_السحابة': latest.get('Ichimoku_Cloud_Position', 'غير محدد'),
        'إشارة_ستوكاستك': latest.get('Stoch_RSI_Signal', 'محايد ⚖️'),
        'الدايفرنس': 'إيجابي 🟢' if latest.get('Divergence_Signal', 0) > 0 else ('سلبي 🔴' if latest.get('Divergence_Signal', 0) < 0 else 'لا يوجد ⚖️')
    }
    
    return summary
