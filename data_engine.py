import pandas as pd
import requests
import yfinance as yf
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import API_KEY, DEFAULT_TIME_FRAME, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
import time
import json


def send_telegram_alert(message):
    """إرسال تقرير مجمع أو إشعار فوري إلى هاتف المستخدم عبر تليجرام"""
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


def fetch_from_eodhd(symbol, days=365):
    """
    جلب البيانات التاريخية من EODHD - المصدر الأكثر دقة للسوق المصري
    
    EODHD يوفر:
    - بيانات دقيقة للبورصة المصرية (.EGX)
    - تحديثات يومية موثوقة
    - بيانات تاريخية كاملة
    - مؤشرات فنية مدمجة
    """
    try:
        clean_symbol = symbol.split('.')[0].strip().upper()
        
        # استخدام EODHD API مع .EGX
        url = f"https://eodhd.com/api/eod/{clean_symbol}.EGX?api_token={API_KEY}&fmt=json&order=d"
        
        print(f"📡 جلب بيانات {clean_symbol} من EODHD...")
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 10:
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
                
                # التأكد من وجود جميع الأعمدة المطلوبة
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                # ترتيب البيانات من الأقدم للأحدث
                df = df.sort_index()
                
                print(f"✅ تم جلب {len(df)} يوم من EODHD لـ {clean_symbol}")
                return df[required_cols]
            else:
                print(f"⚠️ بيانات EODHD غير كافية لـ {clean_symbol}")
        else:
            print(f"⚠️ EODHD فشل لـ {clean_symbol}: {response.status_code}")
            
    except Exception as e:
        print(f"❌ خطأ في EODHD لـ {symbol}: {e}")
    
    return pd.DataFrame()


def fetch_from_investing(symbol):
    """
    مصدر بديل: Investing.com (مجاني جزئياً)
    """
    try:
        # Investing.com API (تحتاج إلى مكتبة investingpy)
        import investingpy as ip
        df = ip.get_stock_historical_data(
            symbol=symbol,
            country='egypt',
            from_date=datetime.now() - timedelta(days=365),
            to_date=datetime.now()
        )
        if not df.empty:
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        pass
    return pd.DataFrame()


def fetch_from_egx_official(symbol):
    """
    مصدر رسمي: البورصة المصرية (محدود)
    """
    try:
        # البورصة المصرية توفر ملفات CSV يومية
        url = f"https://www.egx.com/data/companies/{symbol}.csv"
        df = pd.read_csv(url)
        if not df.empty:
            # تنظيف البيانات حسب تنسيق البورصة
            return df
    except:
        pass
    return pd.DataFrame()


def fetch_stock_data(symbol, yahoo_symbol=None, force_eodhd=True):
    """
    المحرك المتطور لجلب البيانات:
    1. EODHD (الأولوية القصوى - الدقة المطلوبة)
    2. Investing.com (بديل مجاني)
    3. Yahoo Finance (آخر حل)
    """
    
    # 1. المحاولة الأولى: EODHD (الأكثر دقة)
    if force_eodhd or API_KEY:
        df = fetch_from_eodhd(symbol)
        if not df.empty and len(df) > 30:
            return df, "EODHD ✅ (بيانات دقيقة)"
    
    # 2. المحاولة الثانية: Investing.com
    try:
        df = fetch_from_investing(symbol)
        if not df.empty and len(df) > 30:
            return df, "Investing.com ⚠️ (بديل مجاني)"
    except:
        pass
    
    # 3. المحاولة الثالثة: البورصة المصرية
    try:
        df = fetch_from_egx_official(symbol)
        if not df.empty and len(df) > 30:
            return df, "EGX Official ⚠️ (محدود)"
    except:
        pass
    
    # 4. المحاولة الأخيرة: Yahoo Finance (ضعيف جداً للمصري)
    if yahoo_symbol:
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period="1y")
            if not df.empty and len(df) > 30:
                return df, "Yahoo Finance ❌ (ضعيف)"
        except:
            pass
    
    return pd.DataFrame(), "No Source Available ❌"


def fetch_company_news_sentiment(company_name, symbol):
    """تحليل نبرة الأخبار من مصادر متعددة"""
    
    positive_words = ['أرباح', 'نمو', 'صعود', 'ارتفاع', 'إيجابي', 'استحواذ', 'شراء', 
                      'توسع', 'قفزة', 'تفوق', 'توزيعات', 'أرباح', 'مضاعفة']
    negative_words = ['خسائر', 'تراجع', 'هبوط', 'انخفاض', 'سلبي', 'غرامة', 'نزاع', 
                      'بيع', 'انهيار', 'انكماش', 'تجميد', 'شطب', 'تحذير']
    
    sentiment_score = 0.0
    articles_count = 0
    
    # 1. محاولة جلب الأخبار من Google News
    try:
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
    
    # 2. محاولة جلب الأخبار من مصادر أخرى
    try:
        # استخدام API بديل للأخبار
        news_api_url = f"https://newsapi.org/v2/everything?q={company_name}&language=ar&apiKey=YOUR_NEWS_API_KEY"
        # (يحتاج إلى مفتاح API من newsapi.org)
    except:
        pass
    
    if articles_count > 0:
        return round(max(min(sentiment_score / articles_count, 1.0), -1.0), 2)
    
    return 0.0


def get_market_status():
    """
    التحقق من حالة السوق (مفتوح/مغلق)
    """
    now = datetime.now()
    # البورصة المصرية تعمل من الأحد للخميس 10:00 صباحاً - 2:30 ظهراً
    is_weekday = now.weekday() < 5  # الأحد=0, الخميس=4
    is_trading_hours = 10 <= now.hour < 14 or (now.hour == 14 and now.minute <= 30)
    
    if is_weekday and is_trading_hours:
        return "🟢 السوق مفتوح"
    elif is_weekday:
        return "🟡 خارج ساعات التداول"
    else:
        return "🔴 السوق مغلق (عطلة نهاية الأسبوع)"


def run_market_scanner(df_symbols, add_indicators_func, predictor_instance, strategy_mode="مضاربة سريعة",
                       max_workers=8):
    """ماسح السوق الشامل والمتوازي"""
    buy_opportunities = []
    scanned_count = 0
    failed_count = 0

    def scan_core(row):
        nonlocal scanned_count, failed_count
        try:
            news_score = fetch_company_news_sentiment(row['name'], row['symbol'])
            s_df, src = fetch_stock_data(row['symbol'], row['y_symbol'])

            if s_df.empty or len(s_df) < 30:
                failed_count += 1
                return None

            s_df = add_indicators_func(s_df)
            s_df['News_Sentiment'] = news_score

            dir_out, pred_target, entry_out, exit_out, score_out = predictor_instance.predict_next_price(s_df, strategy_mode)

            scanned_count += 1

            if any(keyword in dir_out for keyword in ["🟢", "🚀", "📈"]):
                return {
                    "السهم": row['symbol'],
                    "السعر الحالي": round(s_df['Close'].iloc[-1], 2),
                    "السعر المتوقع": round(pred_target, 2),
                    "أفضل دخول": round(entry_out, 2),
                    "المستهدف": round(exit_out, 2),
                    "نبرة الأخبار": news_score,
                    "الإشارة": dir_out,
                    "المصدر": src,
                    "درجة الأمان": score_out
                }
        except Exception as e:
            print(f"⚠️ خطأ في {row.get('symbol', 'unknown')}: {e}")
            failed_count += 1
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_core, row): row for _, row in df_symbols.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                buy_opportunities.append(res)

    print(f"📊 ملخص الفحص: {scanned_count} نجاح, {failed_count} فشل")
    return pd.DataFrame(buy_opportunities)
