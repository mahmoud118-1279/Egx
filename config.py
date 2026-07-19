import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent

# إعدادات API
API_KEY = '68d3dcb6a42019.91266752'

# مسارات الملفات
SYMBOL_FILE = BASE_DIR / "egx_symbols.csv"
LEARNING_DATA_FILE = BASE_DIR / "ai_learning_data.json"

# مسارات تخزين البيانات المحلية
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
DAILY_UPDATES_DIR = DATA_DIR / "daily_updates"
CACHE_FILE = DATA_DIR / "stock_cache.json"

# إنشاء المجلدات
for dir_path in [DATA_DIR, HISTORICAL_DIR, DAILY_UPDATES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# إعدادات المؤشرات
DEFAULT_TIME_FRAME = 365

# إعدادات التليجرام
TELEGRAM_TOKEN = "8870808709:AAEtcxwZmerbIbbBgwcPASV3s3PSBULOx9s"
TELEGRAM_CHAT_ID = "1687227481"

# إعدادات التخزين المؤقت
CACHE_EXPIRY_DAYS = 1  # عدد الأيام قبل تحديث البيانات
MAX_STOCKS_PER_DAY = 10  # الحد الأقصى للأسهم المحدثة يومياً (لتوفير API)
