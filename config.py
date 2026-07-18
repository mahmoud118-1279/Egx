import warnings
from pathlib import Path
import streamlit as st

warnings.filterwarnings('ignore')

# تحديد مسار الفولدر الحالي على الفلاشة ديناميكياً (ليعمل على ويندوز/ماك/لينكس)
BASE_DIR = Path(__file__).resolve().parent

# الإعدادات العامة

SYMBOL_FILE = BASE_DIR / "egx_symbols.csv"
LEARNING_DATA_FILE = BASE_DIR / "ai_learning_data.json"

# إعدادات المؤشرات الفنية
DEFAULT_TIME_FRAME = 365  # عدد أيام البيانات التاريخية

# إعدادات تحليل الأخبار
NEWS_MAX_RESULTS = 5  # عدد الأخبار التي يتم جلبها لكل شركة

# إعدادات إشعارات تليجرام


# قراءة البيانات السرية بأمان من إعدادات السيرفر السحابي
API_KEY = st.secrets["API_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
TELEGRAM_CHAT_ID = "1687227481"
