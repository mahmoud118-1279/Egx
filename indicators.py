import numpy as np
import pandas as pd


def calculate_rsi(df, period=14):
    """حساب مؤشر القوة النسبية RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, slow=26, fast=12, smooth=9):
    """حساب مؤشر الماكد MACD وخط الإشارة"""
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=smooth, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """حساب حارات بولينجر Bollinger Bands وضمان حقن الأعمدة بالأسماء المطلوبة للذكاء الاصطناعي"""
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * std)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * std)
    return df


def add_all_indicators(df):
    """حقن وتطوير كافة الميزات الفنية والسيولية والذكية لتغذية عقل الآلة"""
    if len(df) < 5:
        return df

    # 1. استدعاء الدوال الأساسية القديمة (للحفاظ على استقرار الكود)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)

    # 2. حساب مؤشر ATR التقليدي لحساب وقف الخسارة
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 3. حساب مؤشرات السيولة وحجم التداول القديمة
    df['CMF'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['CMF'] = (df['CMF'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Daily_Return'] = df['Close'].pct_change()

    # 4. الميزات الاستثمارية الكبرى القديمة
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Dist_From_EMA200'] = (df['Close'] - df['EMA_200']) / (df['EMA_200'] + 1e-10)
    df['Volume_Shock'] = df['Volume'] / (df['Volume'].rolling(window=22).mean() + 1e-10)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = df['OBV'].diff(10)

    # 🧠 5. حزمة التحديث الفوقية الجديدة (تحليل بيئة السوق والسيولة المؤسسية):

    # أ. مؤشر تدفق الأموال الذكية MFI (يرصد البيع والشراء المخفي مدمجاً بالحجم)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = pos_flow / (neg_flow + 1e-10)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # ب. سعر المتوسط المرجح بحجم التداول VWAP (مؤشر كبار المؤسسات والصناديق في مصر)
    df['VWAP'] = (df['Volume'] * typical_price).cumsum() / (df['Volume'].cumsum() + 1e-10)
    df['Dist_From_VWAP'] = (df['Close'] - df['VWAP']) / (df['VWAP'] + 1e-10)

    # ج. مؤشر الفوضى والتذبذب العرضي (Choppiness Index)
    # يساعد الذكاء الاصطناعي على معرفة ما إذا كان السهم يسير في اتجاه حاد أم يتذبذب لجمع السيولة
    atr_sum = true_range.rolling(14).sum()
    max_high = df['High'].rolling(14).max()
    min_low = df['Low'].rolling(14).min()
    df['Chop_Index'] = 100 * (np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14))

    # د. ميزة الدايفرنس الفوري (RSI Divergence Proxy)
    # يقيس الفارق بين قمم السعر وقمم الـ RSI لتنبيه النموذج بالانعكاسات الانفجارية مبكراً
    df['Price_Slope'] = df['Close'].diff(5)
    df['RSI_Slope'] = df['RSI'].diff(5)
    df['Divergence_Signal'] = np.where((df['Price_Slope'] < 0) & (df['RSI_Slope'] > 0), 1.0,
                                       np.where((df['Price_Slope'] > 0) & (df['RSI_Slope'] < 0), -1.0, 0.0))

    # 6. حقن وتهيئة عمود تحليل مشاعر الأخبار الجديد ليتم تعبئته حركياً لاحقاً
    if 'News_Sentiment' not in df.columns:
        df['News_Sentiment'] = 0.0

    # 7. تنظيف وتعبئة القيم الفارغة بحرفية لعدم إفساد أوزان الـ Scaler
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].bfill().ffill().fillna(0)

    return df