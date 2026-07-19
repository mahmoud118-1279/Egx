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
    
    def _load_cache(self):
        """تحميل ذاكرة التخزين المؤقت"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """حفظ ذاكرة التخزين المؤقت"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
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
    
    def fetch_from_investing(self, symbol, days=365):
        """
        جلب البيانات التاريخية من Investing.com
        """
        try:
            import investingpy as ip
            from_date = datetime.now() - timedelta(days=days)
            
            print(f"📡 جلب بيانات {symbol} من Investing.com...")
            
            df = ip.get_stock_historical_data(
                symbol=symbol,
                country='egypt',
                from_date=from_date,
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
                
        except Exception as e:
            print(f"⚠️ فشل جلب {symbol} من Investing.com: {e}")
        
        return pd.DataFrame()
    
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
    
    def save_historical_data(self, symbol, df):
        """حفظ البيانات التاريخية محلياً"""
        if df.empty:
            return
        
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
    
    def is_data_fresh(self, symbol):
        """التحقق من حداثة البيانات"""
        if symbol not in self.cache:
            return False
        
        last_updated = datetime.fromisoformat(self.cache[symbol]['last_updated'])
        days_old = (datetime.now() - last_updated).days
        
        return days_old < CACHE_EXPIRY_DAYS
    
    def get_stock_data(self, symbol, force_update=False):
        """
        الحصول على بيانات السهم - محرك ذكي:
        1. تحميل من الملف المحلي إذا كان محدثاً
        2. جلب من Investing.com إذا كانت قديمة أو غير موجودة
        3. تحديث من EODHD للأسعار اللحظية (استخدام محدود)
        """
        
        # 1. محاولة تحميل من الملف المحلي
        if not force_update and self.is_data_fresh(symbol):
            df = self.load_historical_data(symbol)
            if not df.empty:
                return df, "Local Cache ✅"
        
        # 2. جلب من Investing.com
        print(f"🔄 جلب بيانات {symbol} من Investing.com...")
        df = self.fetch_from_investing(symbol)
        
        if not df.empty:
            self.save_historical_data(symbol, df)
            return df, "Investing.com ✅"
        
        # 3. جلب من EODHD (نسخة احتياطية)
        try:
            from data_engine import fetch_from_eodhd
            df = fetch_from_eodhd(symbol)
            if not df.empty:
                self.save_historical_data(symbol, df)
                return df, "EODHD ✅ (نسخة احتياطية)"
        except:
            pass
        
        return pd.DataFrame(), "No Source ❌"
    
    def update_all_stocks(self, symbols_list, max_stocks=MAX_STOCKS_PER_DAY):
        """
        تحديث جميع الأسهم (يتم تنفيذها مرة واحدة يومياً)
        """
        updated = []
        failed = []
        
        print(f"🔄 بدء تحديث {min(len(symbols_list), max_stocks)} سهماً...")
        
        for i, symbol in enumerate(symbols_list[:max_stocks]):
            try:
                print(f"  [{i+1}/{min(len(symbols_list), max_stocks)}] تحديث {symbol}...")
                df, source = self.get_stock_data(symbol, force_update=True)
                
                if not df.empty:
                    updated.append(symbol)
                else:
                    failed.append(symbol)
                
                # تأخير بسيط لتجنب حظر الـ API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ فشل تحديث {symbol}: {e}")
                failed.append(symbol)
        
        print(f"\n✅ تم تحديث {len(updated)} سهماً")
        if failed:
            print(f"⚠️ فشل تحديث {len(failed)} سهماً: {failed}")
        
        return updated, failed
    
    def get_live_price(self, symbol):
        """
        الحصول على السعر اللحظي من EODHD (استخدام محدود)
        """
        # قراءة آخر سعر من EODHD
        live_data = self.fetch_from_eodhd_live(symbol)
        if live_data:
            return live_data
        
        # إذا فشل EODHD، استخدم آخر سعر من الملف المحلي
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
            'date_range': f"{df.index[0]} → {df.index[-1]}",
            'rows': len(df)
        }
        
        return info


# إنشاء مدير البيانات
data_manager = DataManager()


def get_stock_data_with_cache(symbol):
    """
    دالة مساعدة لجلب البيانات باستخدام التخزين المؤقت
    """
    df, source = data_manager.get_stock_data(symbol)
    return df, source


def update_all_stocks(symbols):
    """
    دالة مساعدة لتحديث جميع الأسهم
    """
    return data_manager.update_all_stocks(symbols)
