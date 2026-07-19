import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from data_manager import data_manager, get_stock_data_with_cache


def send_telegram_alert(message):
    """إرسال إشعار تليجرام"""
    token = str(TELEGRAM_TOKEN).strip()
    chat_id = str(TELEGRAM_CHAT_ID).strip()

    if not token or not chat_id:
        print("⚠️ خطأ: إعدادات التليجرام فارغة")
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
            print(f"⚠️ فشل التليجرام. كود: {response.status_code}")
    except Exception as e:
        print(f"💥 خطأ في التليجرام: {e}")
    return False


def fetch_from_eodhd_live(symbol):
    """جلب السعر اللحظي من EODHD"""
    try:
        clean_symbol = symbol.split('.')[0].strip().upper()
        url = f"https://eodhd.com/api/real-time/{clean_symbol}.EGX?api_token={API_KEY}&fmt=json"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                return {
                    'price': data.get('close', 0),
                    'high': data.get('high', 0),
                    'low': data.get('low', 0),
                    'volume': data.get('volume', 0),
                    'change': data.get('change', 0),
                    'change_pct': data.get('change_pct', 0)
                }
    except Exception as e:
        print(f"⚠️ فشل جلب السعر اللحظي لـ {symbol}: {e}")
    
    return None


def fetch_stock_data(symbol, yahoo_symbol=None):
    """
    جلب بيانات السهم باستخدام المدير الذكي
    """
    df, source = get_stock_data_with_cache(symbol)
    
    if df.empty:
        # محاولة Yahoo كحل أخير
        if yahoo_symbol:
            try:
                import yfinance as yf
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(period="1y")
                if not df.empty:
                    df.rename(columns={
                        'Open': 'Open',
                        'High': 'High',
                        'Low': 'Low',
                        'Close': 'Close',
                        'Volume': 'Volume'
                    }, inplace=True)
                    return df, "Yahoo Finance ❌ (نسخة احتياطية)"
            except:
                pass
    
    return df, source


def fetch_company_news_sentiment(company_name, symbol):
    """تحليل نبرة الأخبار"""
    positive_words = ['أرباح', 'نمو', 'صعود', 'ارتفاع', 'إيجابي', 'استحواذ', 'شراء', 'توسع']
    negative_words = ['خسائر', 'تراجع', 'هبوط', 'انخفاض', 'سلبي', 'غرامة', 'نزاع', 'بيع']
    
    sentiment_score = 0.0
    articles_count = 0
    
    try:
        import urllib.parse
        import xml.etree.ElementTree as ET
        
        search_query = f'"{company_name}" أخبار البورصة المصرية'
        encoded_query = urllib.parse.quote(search_query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ar&gl=EG&ceid=EG:ar"
        
        response = requests.get(url, timeout=7)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall('.//item')[:5]:
                title = item.find('title').text or ""
                articles_count += 1
                pos_hits = sum(1 for w in positive_words if w in title)
                neg_hits = sum(1 for w in negative_words if w in title)
                
                if pos_hits > neg_hits:
                    sentiment_score += 0.5
                elif neg_hits > pos_hits:
                    sentiment_score -= 0.5
    except:
        pass
    
    if articles_count > 0:
        return round(max(min(sentiment_score / articles_count, 1.0), -1.0), 2)
    
    return 0.0


def run_market_scanner(df_symbols, add_indicators_func, predictor_instance, 
                       strategy_mode="مضاربة سريعة", max_workers=8):
    """ماسح السوق باستخدام البيانات المخزنة محلياً"""
    buy_opportunities = []
    scanned_count = 0

    def scan_core(row):
        nonlocal scanned_count
        try:
            # استخدام البيانات المخزنة محلياً
            s_df, src = fetch_stock_data(row['symbol'], row.get('y_symbol'))
            
            if s_df.empty or len(s_df) < 30:
                return None
            
            # جلب نبرة الأخبار
            news_score = fetch_company_news_sentiment(row['name'], row['symbol'])
            
            # حساب المؤشرات
            s_df = add_indicators_func(s_df)
            s_df['News_Sentiment'] = news_score
            
            # التنبؤ
            dir_out, pred_target, entry_out, exit_out, score_out = predictor_instance.predict_next_price(
                s_df, strategy_mode
            )
            
            scanned_count += 1
            
            if any(keyword in dir_out for keyword in ["🟢", "🚀", "📈"]):
                # جلب السعر اللحظي من EODHD (استخدام محدود)
                live_price = fetch_from_eodhd_live(row['symbol'])
                current_price = live_price['price'] if live_price else s_df['Close'].iloc[-1]
                
                return {
                    "السهم": row['symbol'],
                    "السعر الحالي": round(current_price, 2),
                    "السعر المتوقع": round(pred_target, 2),
                    "أفضل دخول": round(entry_out, 2),
                    "المستهدف": round(exit_out, 2),
                    "نبرة الأخبار": news_score,
                    "الإشارة": dir_out,
                    "المصدر": src,
                    "درجة الأمان": score_out,
                    "الحالة": "✅ بيانات محدثة" if data_manager.is_data_fresh(row['symbol']) else "⚠️ بيانات قديمة"
                }
        except Exception as e:
            print(f"⚠️ خطأ في {row.get('symbol', 'unknown')}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_core, row): row for _, row in df_symbols.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                buy_opportunities.append(res)

    return pd.DataFrame(buy_opportunities)
