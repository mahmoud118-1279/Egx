import pandas as pd
import requests
import yfinance as yf
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import API_KEY, DEFAULT_TIME_FRAME, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


def send_telegram_alert(message):
    """إرسال تقرير مجمع أو إشعار فوري إلى هاتف المستخدم عبر تليجرام"""
    token = str(TELEGRAM_TOKEN).strip()
    chat_id = str(TELEGRAM_CHAT_ID).strip()

    if not token or not chat_id:
        print("⚠️ خطأ: إعدادات التليجرام TELEGRAM_TOKEN أو TELEGRAM_CHAT_ID فارغة في config.py")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ تم إرسال تقرير التليجرام بنجاح!")
            return True
        else:
            print(f"⚠️ فشل التليجرام. كود الاستجابة: {response.status_code}")
    except Exception as e:
        print(f"💥 خطأ أثناء الاتصال بتليجرام: {e}")
    return False


def fetch_from_eodhd(symbol):
    """جلب البيانات التاريخية النظيفة من منصة EODHD باستخدام كود .EGX المخصص لمصر"""
    try:
        # تنظيف الرمز تماماً من أي .CA أو زيادات قادمة من الملف لضمان عمل EODHD بنسبة 100%
        clean_symbol = symbol.split('.')[0].strip().upper()
        url = f"https://eodhd.com/api/eod/{clean_symbol}.EGX?api_token={API_KEY}&fmt=json"

        response = requests.get(url, timeout=7)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        pass
    return pd.DataFrame()


def fetch_stock_data(symbol, yahoo_symbol=None):
    """
    المحرك التبادلي المطور:
    1. يبدأ بـ EODHD كأولوية قصوى كمصدر رئيسي دقيق ومطابق للشاشة.
    2. يتوجه لـ Yahoo Finance كخيار احتياطي أخير فقط في حال تعذر التحديث.
    """
    # الخطوة 1: محاولة جلب الداتا الدقيقة من EODHD
    try:
        df = fetch_from_eodhd(symbol)
        if not df.empty and len(df) > 5:
            return df, "EODHD (بيانات دقيقة)"
    except Exception:
        pass

    # الخطوة 2: الخيار الاحتياطي اللحظي من ياهو فاينانس بالرمز الأساسي
    if yahoo_symbol:
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period="1y")
            if not df.empty and len(df) > 5:
                return df, "Yahoo Finance (الأساسي)"
        except Exception:
            pass

    # الخطوة 3: المحاولة الذكية للأكواد المركبة بياهو
    alternative_symbols = [f"{symbol.split('.')[0].strip().upper()}.CA", symbol]
    for alt_sym in alternative_symbols:
        if alt_sym == yahoo_symbol:
            continue
        try:
            ticker = yf.Ticker(alt_sym)
            df = ticker.history(period="1y")
            if not df.empty and len(df) > 5:
                return df, f"Yahoo Finance (مصحح: {alt_sym})"
        except Exception:
            continue

    return pd.DataFrame(), "No Source Available"


def fetch_company_news_sentiment(company_name, symbol):
    """تحليل النبرة والمشاعر العامة لآخر أخبار الشركة"""
    positive_words = ['أرباح', 'نمو', 'صعود', 'ارتفاع', 'إيجابي', 'استحواذ', 'شراء', 'توسع', 'قفزة', 'تفوق']
    negative_words = ['خسائر', 'تراجع', 'هبوط', 'انخفاض', 'سلبي', 'غرامة', 'نزاع', 'بيع', 'انهيار', 'انكماش']

    search_query = f'\"{company_name}\" أخبار البورصة المصرية'
    encoded_query = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ar&gl=EG&ceid=EG:ar"

    try:
        response = requests.get(url, timeout=7)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            sentiment_score = 0.0
            articles_count = 0

            for item in root.findall('.//item')[:5]:
                title = item.find('title').text or ""
                articles_count += 1
                pos_hits = sum(1 for w in positive_words if w in title)
                neg_hits = sum(1 for w in negative_words if w in title)

                if pos_hits > neg_hits:
                    sentiment_score += 0.5
                elif neg_hits > pos_hits:
                    sentiment_score -= 0.5

            if articles_count > 0:
                return round(max(min(sentiment_score / articles_count, 1.0), -1.0), 2)
    except Exception:
        pass
    return 0.0


def run_market_scanner(df_symbols, add_indicators_func, predictor_instance, strategy_mode="مضاربة سريعة",
                       max_workers=8):
    """ماسح السوق الشامل والمتوازي لتجميع فرص الشراء وإصدار التقرير"""
    buy_opportunities = []

    def scan_core(row):
        try:
            news_score = fetch_company_news_sentiment(row['name'], row['symbol'])
            s_df, src = fetch_stock_data(row['symbol'], row['y_symbol'])

            if s_df.empty or len(s_df) < 30:
                return None

            s_df = add_indicators_func(s_df)
            s_df['News_Sentiment'] = news_score

            dir_out, pred_target, entry_out, exit_out = predictor_instance.predict_next_price(s_df, strategy_mode)

            # إذا كانت التوصية إيجابية وتحتوي على مؤشرات صعود
            if any(keyword in dir_out for keyword in ["🟢", "🚀", "📈"]):
                return {
                    "السهم": row['symbol'],
                    "السعر الحالي": round(s_df['Close'].iloc[-1], 2),
                    "السعر المتوقع": round(pred_target, 2),
                    "أفضل دخول": round(entry_out, 2),
                    "المستهدف القريب": round(exit_out, 2),
                    "نبرة الأخبار": news_score,
                    "الإشارة": dir_out,
                    "المصدر": src
                }
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_core, row): row for _, row in df_symbols.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                buy_opportunities.append(res)

    return pd.DataFrame(buy_opportunities)