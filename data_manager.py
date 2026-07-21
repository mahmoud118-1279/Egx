import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, HISTORICAL_DIR, DAILY_UPDATES_DIR, 
    CACHE_FILE, CACHE_EXPIRY_DAYS, MAX_STOCKS_PER_DAY, API_KEY
)


class DataManager:
    """مدير البيانات المحلية - يخزن ويحدث البيانات من مصادر متعددة"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.historical_dir = HISTORICAL_DIR
        self.daily_updates_dir = DAILY_UPDATES_DIR
        self.cache_file = CACHE_FILE
        self.cache = self._load_cache()
        
        # ✅ إنشاء المجلدات إذا لم تكن موجودة
        for dir_path in [self.data_dir, self.historical_dir, self.daily_updates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self):
        """تحميل ذاكرة التخزين المؤقت"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ خطأ في تحميل الكاش: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """حفظ ذاكرة التخزين المؤقت"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ خطأ في حفظ الكاش: {e}")
    
    def _get_historical_file(self, symbol):
        """الحصول على مسار ملف البيانات التاريخية للسهم"""
        return self.historical_dir / f"{symbol}.csv"
    
    def _get_daily_update_file(self, symbol, date=None):
        """الحصول على مسار ملف التحديث اليومي"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        daily_dir = self.daily_updates_dir / date
        daily_dir.mkdir(parents=True, exist_ok=True)
        return daily_dir / f"{symbol}.csv"
    
    # ============================================================
    # ✅ الدالة المحسنة لجلب البيانات من Investing.com
    # ============================================================
    def fetch_from_investing(self, symbol, days=365):
        """
        جلب البيانات التاريخية من Investing.com مع محاولات متعددة
        """
        try:
            from_date = datetime.now() - timedelta(days=days)
            
            print(f"📡 جلب بيانات {symbol} من Investing.com...")
            
            # ✅ المحاولة الأولى: باستخدام investingpy
            try:
                import investingpy as ip
                
                # محاولة جلب البيانات
                df = ip.get_stock_historical_data(
                    symbol=symbol,
                    country='egypt',
                    from_date=from_date.strftime('%d/%m/%Y'),
                    to_date=datetime.now().strftime('%d/%m/%Y')
                )
                
                if not df.empty and len(df) > 10:
                    return self._process_investing_data(df, symbol)
                    
            except ImportError:
                print(f"⚠️ investingpy غير مثبت، جاري محاولة التثبيت...")
                try:
                    import subprocess
                    subprocess.check_call(['pip', 'install', 'investingpy', '--quiet'])
                    import investingpy as ip
                    
                    df = ip.get_stock_historical_data(
                        symbol=symbol,
                        country='egypt',
                        from_date=from_date.strftime('%d/%m/%Y'),
                        to_date=datetime.now().strftime('%d/%m/%Y')
                    )
                    
                    if not df.empty and len(df) > 10:
                        return self._process_investing_data(df, symbol)
                except Exception as e:
                    print(f"⚠️ فشل تثبيت investingpy: {e}")
            
            except Exception as e:
                print(f"⚠️ فشل جلب {symbol} من Investing.com: {e}")
            
            # ✅ المحاولة الثانية: EODHD كبديل
            print(f"🔄 محاولة EODHD كبديل لـ {symbol}...")
            return self.fetch_from_eodhd(symbol)
            
        except Exception as e:
            print(f"⚠️ فشل جلب {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_investing_data(self, df, symbol):
        """معالجة بيانات Investing.com وتنسيقها"""
        try:
            df = df.reset_index()
            
            # ✅ تحديد أسماء الأعمدة الصحيحة
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or 'datetime' in col_lower:
                    column_mapping[col] = 'Date'
                elif 'open' in col_lower:
                    column_mapping[col] = 'Open'
                elif 'high' in col_lower:
                    column_mapping[col] = 'High'
                elif 'low' in col_lower:
                    column_mapping[col] = 'Low'
                elif 'close' in col_lower:
                    column_mapping[col] = 'Close'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'Volume'
            
            df.rename(columns=column_mapping, inplace=True)
            
            # ✅ التأكد من وجود عمود التاريخ
            if 'Date' not in df.columns:
                if 'index' in df.columns:
                    df.rename(columns={'index': 'Date'}, inplace=True)
                else:
                    df['Date'] = pd.to_datetime(df.index)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # ✅ التأكد من وجود جميع الأعمدة المطلوبة
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            # ✅ ترتيب البيانات
            df = df.sort_index()
            
            print(f"✅ تم جلب {len(df)} يوم من Investing.com لـ {symbol}")
            return df[required_cols]
            
        except Exception as e:
            print(f"⚠️ خطأ في معالجة بيانات {symbol}: {e}")
            return pd.DataFrame()
    
    # ============================================================
    # ✅ الدالة المحسنة لجلب البيانات من EODHD
    # ============================================================
    def fetch_from_eodhd(self, symbol, days=365):
        """
        جلب البيانات التاريخية من EODHD (نسخة احتياطية)
        """
        try:
            clean_symbol = symbol.split('.')[0].strip().upper()
            
            # ✅ محاولة جلب البيانات التاريخية
            url = f"https://eodhd.com/api/eod/{clean_symbol}.EGX?api_token={API_KEY}&fmt=json"
            
            # إذا كان هناك تاريخ محدد، نضيفه
            if days:
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                url += f"&from={from_date}"
            
            print(f"📡 جلب بيانات {symbol} من EODHD...")
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 5:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # ✅ تحديد أسماء الأعمدة الصحيحة
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }
                    df.rename(columns=column_mapping, inplace=True)
                    
                    # ✅ التأكد من وجود جميع الأعمدة
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = 0.0
                    
                    df = df.sort_index()
                    print(f"✅ تم جلب {len(df)} يوم من EODHD لـ {clean_symbol}")
                    return df[required_cols]
                else:
                    print(f"⚠️ لا توجد بيانات كافية من EODHD لـ {symbol}")
            else:
                print(f"⚠️ خطأ في EODHD: كود {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"⚠️ انتهى وقت EODHD لـ {symbol}")
        except Exception as e:
            print(f"⚠️ EODHD فشل لـ {symbol}: {e}")
        
        return pd.DataFrame()
    
    # ============================================================
    # ✅ الدالة المحسنة لجلب الأسعار اللحظية
    # ============================================================
    def fetch_from_eodhd_live(self, symbol):
        """
        جلب آخر سعر من EODHD (استخدام محدود)
        """
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
                        'timestamp': data.get('timestamp', datetime.now().isoformat())
                    }
        except Exception as e:
            print(f"⚠️ فشل جلب السعر اللحظي لـ {symbol}: {e}")
        
        return None
    
    # ============================================================
    # ✅ الدالة المحسنة لحفظ البيانات
    # ============================================================
    def save_historical_data(self, symbol, df):
        """حفظ البيانات التاريخية محلياً"""
        if df.empty:
            return False
        
        try:
            file_path = self._get_historical_file(symbol)
            df.to_csv(file_path)
            print(f"💾 تم حفظ بيانات {symbol} في {file_path}")
            
            # تحديث الكاش
            self.cache[symbol] = {
                'last_updated': datetime.now().isoformat(),
                'source': 'investing',
                'rows': len(df),
                'file': str(file_path)
            }
            self._save_cache()
            return True
            
        except Exception as e:
            print(f"⚠️ فشل حفظ بيانات {symbol}: {e}")
            return False
    
    # ============================================================
    # ✅ الدالة المحسنة لتحميل البيانات
    # ============================================================
    def load_historical_data(self, symbol):
        """تحميل البيانات التاريخية من الملف المحلي"""
        file_path = self._get_historical_file(symbol)
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                print(f"📂 تم تحميل {len(df)} يوم من الملف المحلي لـ {symbol}")
                return df
            except Exception as e:
                print(f"⚠️ فشل تحميل {symbol} من الملف المحلي: {e}")
        
        return pd.DataFrame()
    
    # ============================================================
    # ✅ الدالة المحسنة للتحقق من حداثة البيانات
    # ============================================================
    def is_data_fresh(self, symbol):
        """التحقق من حداثة البيانات"""
        if symbol not in self.cache:
            return False
        
        try:
            last_updated = datetime.fromisoformat(self.cache[symbol]['last_updated'])
            days_old = (datetime.now() - last_updated).days
            return days_old < CACHE_EXPIRY_DAYS
        except:
            return False
    
    # ============================================================
    # ✅ المحرك الرئيسي لجلب البيانات (الأكثر استخداماً)
    # ============================================================
    def get_stock_data(self, symbol, force_update=False):
        """
        الحصول على بيانات السهم - محرك ذكي متعدد المصادر:
        1. تحميل من الملف المحلي إذا كان محدثاً
        2. جلب من Investing.com إذا كانت قديمة أو غير موجودة
        3. جلب من EODHD كبديل نهائي
        """
        
        # ✅ 1. محاولة تحميل من الملف المحلي
        if not force_update and self.is_data_fresh(symbol):
            df = self.load_historical_data(symbol)
            if not df.empty and len(df) > 30:
                print(f"✅ {symbol}: من التخزين المحلي (محدث)")
                return df, "Local Cache ✅"
        
        # ✅ 2. محاولة جلب من Investing.com
        print(f"🔄 جلب بيانات {symbol} من Investing.com...")
        df = self.fetch_from_investing(symbol)
        
        if not df.empty and len(df) > 20:
            self.save_historical_data(symbol, df)
            return df, "Investing.com ✅"
        
        # ✅ 3. محاولة جلب من EODHD (بديل)
        print(f"🔄 جلب بيانات {symbol} من EODHD...")
        df = self.fetch_from_eodhd(symbol)
        
        if not df.empty and len(df) > 20:
            self.save_historical_data(symbol, df)
            return df, "EODHD ✅ (نسخة احتياطية)"
        
        # ✅ 4. محاولة تحميل حتى لو كانت قديمة
        df = self.load_historical_data(symbol)
        if not df.empty and len(df) > 10:
            print(f"⚠️ {symbol}: بيانات قديمة من الملف المحلي ({len(df)} يوم)")
            return df, "Local Cache ⚠️ (قديم)"
        
        return pd.DataFrame(), "No Source ❌"
    
    # ============================================================
    # ✅ دالة تحديث جميع الأسهم
    # ============================================================
    def update_all_stocks(self, symbols_list, max_stocks=10):
        """
        تحديث جميع الأسهم (يتم تنفيذها مرة واحدة يومياً)
        """
        updated = []
        failed = []
        
        total = min(len(symbols_list), max_stocks)
        print(f"🔄 بدء تحديث {total} سهماً...")
        print("=" * 50)
        
        for i, symbol in enumerate(symbols_list[:max_stocks]):
            try:
                print(f"  [{i+1}/{total}] تحديث {symbol}...")
                df, source = self.get_stock_data(symbol, force_update=True)
                
                if not df.empty and len(df) > 20:
                    updated.append(symbol)
                    print(f"    ✅ تم تحديث {symbol} ({len(df)} يوم) من {source}")
                else:
                    failed.append(symbol)
                    print(f"    ❌ فشل تحديث {symbol} (بيانات غير كافية)")
                
                # ✅ تأخير لتجنب حظر الـ API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ فشل تحديث {symbol}: {e}")
                failed.append(symbol)
        
        print("=" * 50)
        print(f"✅ تم تحديث {len(updated)} سهماً بنجاح")
        if failed:
            print(f"⚠️ فشل تحديث {len(failed)} سهماً: {failed[:5]}...")
        
        return updated, failed
    
    # ============================================================
    # ✅ دالة الحصول على السعر اللحظي
    # ============================================================
    def get_live_price(self, symbol):
        """
        الحصول على السعر اللحظي من EODHD
        """
        # ✅ محاولة جلب السعر اللحظي
        live_data = self.fetch_from_eodhd_live(symbol)
        if live_data and live_data.get('price', 0) > 0:
            return live_data
        
        # ✅ إذا فشل، استخدم آخر سعر من الملف المحلي
        df = self.load_historical_data(symbol)
        if not df.empty:
            return {
                'price': df['Close'].iloc[-1],
                'high': df['High'].iloc[-1],
                'low': df['Low'].iloc[-1],
                'volume': df['Volume'].iloc[-1],
                'timestamp': df.index[-1].isoformat()
            }
        
        return None
    
    # ============================================================
    # ✅ دالة الحصول على معلومات السهم
    # ============================================================
    def get_stock_info(self, symbol):
        """
        الحصول على معلومات السهم
        """
        df = self.load_historical_data(symbol)
        if df.empty:
            return None
        
        info = {
            'symbol': symbol,
            'last_price': df['Close'].iloc[-1],
            'high_52w': df['High'].max(),
            'low_52w': df['Low'].min(),
            'avg_volume': df['Volume'].mean(),
            'date_range': f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}",
            'rows': len(df)
        }
        
        return info
    
    # ============================================================
    # ✅ دالة اختبار المصادر
    # ============================================================
    def test_sources(self, symbol):
        """
        اختبار جميع مصادر البيانات لسهم معين
        """
        print(f"\n🔍 اختبار مصادر البيانات لـ {symbol}")
        print("-" * 40)
        
        # 1. اختبار Investing.com
        df_inv = self.fetch_from_investing(symbol)
        print(f"Investing.com: {'✅' if not df_inv.empty else '❌'} ({len(df_inv)} يوم)")
        
        # 2. اختبار EODHD
        df_eod = self.fetch_from_eodhd(symbol)
        print(f"EODHD: {'✅' if not df_eod.empty else '❌'} ({len(df_eod)} يوم)")
        
        # 3. اختبار التخزين المحلي
        df_local = self.load_historical_data(symbol)
        print(f"Local Cache: {'✅' if not df_local.empty else '❌'} ({len(df_local)} يوم)")
        
        print("-" * 40)
        
        return {
            'investing': len(df_inv),
            'eodhd': len(df_eod),
            'local': len(df_local)
        }


# ============================================================
# ✅ إنشاء مدير البيانات (Singleton)
# ============================================================
data_manager = DataManager()


# ============================================================
# ✅ دوال مساعدة للاستخدام من خارج الملف
# ============================================================
def get_stock_data_with_cache(symbol):
    """
    دالة مساعدة لجلب البيانات باستخدام التخزين المؤقت
    """
    return data_manager.get_stock_data(symbol)


def update_all_stocks(symbols, max_stocks=10):
    """
    دالة مساعدة لتحديث جميع الأسهم
    """
    return data_manager.update_all_stocks(symbols, max_stocks)


def get_live_price(symbol):
    """
    دالة مساعدة للحصول على السعر اللحظي
    """
    return data_manager.get_live_price(symbol)


def test_data_sources(symbol):
    """
    دالة مساعدة لاختبار مصادر البيانات
    """
    return data_manager.test_sources(symbol)


# ============================================================
# ✅ كود اختبار سريع (يعمل إذا تم تشغيل الملف مباشرة)
# ============================================================
if __name__ == "__main__":
    print("🧪 اختبار مدير البيانات...")
    print("=" * 40)
    
    # اختبار سهم COMI
    symbol = "COMI"
    print(f"\n📊 اختبار السهم: {symbol}")
    
    df, source = get_stock_data_with_cache(symbol)
    
    if not df.empty:
        print(f"\n✅ نجح جلب البيانات من: {source}")
        print(f"📈 عدد الأيام: {len(df)}")
        print(f"💰 آخر سعر: {df['Close'].iloc[-1]:.2f}")
        print(f"📅 من {df.index[0]} إلى {df.index[-1]}")
        print("\nآخر 5 أيام:")
        print(df.tail(5))
    else:
        print("❌ فشل جلب البيانات")
