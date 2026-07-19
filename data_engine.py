import pandas as pd
import requests
import yfinance as yf
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config import API_KEY, DEFAULT_TIME_FRAME, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


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


def fetch_from_investing(symbol):
    """
    جلب البيانات التاريخية من Investing.com
    المصدر الأكثر دقة للبورصة المصرية
    """
    try:
        import investingpy as ip
        
        print(f"📡 جلب بيانات {symbol} من Investing.com...")
        
        # محاولة جلب البيانات
        df = ip.get_stock_historical_data(
            symbol=symbol,
            country='egypt',
            from_date=datetime.now() - timedelta(days=365),
            to_date=datetime.now()
        )
        
        if not df.empty:
            df = df.reset_index()
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            print(f"✅ تم جلب {len(df)} يوم من Investing.com لـ {symbol}")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
    except ImportError:
        print("⚠️ مكتبة investingpy غير مثبتة، جارٍ التثبيت...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'investingpy'])
        # محاولة مرة أخرى
        try:
            import investingpy as ip
            df = ip.get_stock_historical_data(
                symbol=symbol,
                country='egypt',
                from_date=datetime.now() - timedelta(days=365),
                to_date=datetime.now()
            )
            if not df.empty:
                df = df.reset_index()
                df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except:
            pass
            
    except Exception as e:
        print(f"⚠️ فشل جلب {symbol} من Investing.com: {e}")
    
    return pd.DataFrame()


def fetch_from_eodhd(symbol):
    """جلب البيانات من EODHD (نسخة احتياطية)"""
    try:
        clean_symbol = symbol.split('.')[0].strip().upper()
        url = f"https://eodhd.com/api/eod/{clean_symbol}.EGX?api_token={API_KEY}&fmt=json"
        
        response = requests.get(url, timeout=10)
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
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df.sort_index()
                print(f"✅ تم جلب {len(df)} يوم من EODHD لـ {clean_symbol}")
                return df[required_cols]
    except Exception as e:
        print(f"⚠️ EODHD فشل لـ {symbol}: {e}")
    
    return pd.DataFrame()


def fetch_stock_data(symbol, yahoo_symbol=None):
    """
    المحرك المتطور لجلب البيانات:
    1. Investing.com (الأولوية القصوى - الأكثر دقة)
    2. EODHD (نسخة احتياطية)
    3. Yahoo Finance (آخر حل - ضعيف)
    """
    
    # ✅ 1. المحاولة الأولى: Investing.com
    try:
        df = fetch_from_investing(symbol)
        if not df.empty and len(df) > 30:
            return df, "Investing.com ✅ (مصدر رئيسي)"
    except Exception as e:
        print(f"⚠️ Investing.com فشل لـ {symbol}: {e}")
    
    # ✅ 2. المحاولة الثانية: EODHD
    try:
        df = fetch_from_eodhd(symbol)
        if not df.empty and len(df) > 30:
            return df, "EODHD ✅ (نسخة احتياطية)"
    except Exception as e:
        print(f"⚠️ EODHD فشل لـ {symbol}: {e}")
    
    # ❌ 3. المحاولة الأخيرة: Yahoo Finance (ضعيف جداً)
    if yahoo_symbol:
        try:
            import yfinance as yf
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period="1y")
            if not df.empty and len(df) > 30:
                return df, "Yahoo Finance ❌ (ضعيف - استخدم Investing)"
        except:
            pass
    
    return pd.DataFrame(), "No Source Available ❌"


def fetch_company_news_sentiment(company_name, symbol):
    """تحليل نبرة الأخبار من مصادر متعددة"""
    
    positive_words = ['أرباح', 'نمو', 'صعود', 'ارتفاع', 'إيجابي', 'استحواذ', 'شراء', 
                      'توسع', 'قفزة', 'تفوق', 'توزيعات', 'مضاعفة']
    negative_words = ['خسائر', 'تراجع', 'هبوط', 'انخفاض', 'سلبي', 'غرامة', 'نزاع', 
                      'بيع', 'انهيار', 'انكماش', 'تجميد', 'شطب', 'تحذير']
    
    sentiment_score = 0.0
    articles_count = 0
    
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
    
    if articles_count > 0:
        return round(max(min(sentiment_score / articles_count, 1.0), -1.0), 2)
    
    return 0.0


def run_market_scanner(df_symbols, add_indicators_func, predictor_instance, 
                       strategy_mode="مضاربة سريعة", max_workers=8):
    """ماسح السوق الشامل والمتوازي"""
    buy_opportunities = []
    scanned_count = 0
    failed_count = 0
    source_stats = {"Investing.com": 0, "EODHD": 0, "Yahoo Finance": 0, "No Source": 0}

    def scan_core(row):
        nonlocal scanned_count, failed_count
        try:
            news_score = fetch_company_news_sentiment(row['name'], row['symbol'])
            s_df, src = fetch_stock_data(row['symbol'], row.get('y_symbol'))

            if s_df.empty or len(s_df) < 30:
                failed_count += 1
                return None

            s_df = add_indicators_func(s_df)
            s_df['News_Sentiment'] = news_score

            dir_out, pred_target, entry_out, exit_out, score_out = predictor_instance.predict_next_price(s_df, strategy_mode)

            scanned_count += 1
            
            # تسجيل مصدر البيانات
            if "Investing" in src:
                source_stats["Investing.com"] += 1
            elif "EODHD" in src:
                source_stats["EODHD"] += 1
            elif "Yahoo" in src:
                source_stats["Yahoo Finance"] += 1
            else:
                source_stats["No Source"] += 1

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

    print(f"\n📊 ملخص مصادر البيانات:")
    for src, count in source_stats.items():
        if count > 0:
            print(f"   {src}: {count} سهم")
    
    print(f"\n📊 ملخص الفحص: {scanned_count} نجاح, {failed_count} فشل")
    return pd.DataFrame(buy_opportunities)


# دالة لاختبار المصادر
def test_data_source(symbol):
    """اختبار أي مصدر بيانات يعمل لسهم معين"""
    print(f"\n🔍 اختبار مصادر البيانات لـ {symbol}...")
    print("-" * 40)
    
    # اختبار Investing.com
    df1, src1 = fetch_from_investing(symbol)
    print(f"Investing.com: {'✅' if not df1.empty else '❌'} ({len(df1)} يوم)")
    
    # اختبار EODHD
    df2, src2 = fetch_from_eodhd(symbol)
    print(f"EODHD: {'✅' if not df2.empty else '❌'} ({len(df2)} يوم)")
    
    # اختبار Yahoo
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.CA")
        df3 = ticker.history(period="1y")
        print(f"Yahoo Finance: {'✅' if not df3.empty else '❌'} ({len(df3)} يوم)")
    except:
        print(f"Yahoo Finance: ❌")
    
    print("-" * 40)
