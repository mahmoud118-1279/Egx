#!/usr/bin/env python
"""
برنامج تحديث البيانات - يتم تشغيله يومياً لتحديث الأسهم
"""

import pandas as pd
from datetime import datetime
from config import SYMBOL_FILE
from data_manager import data_manager, update_all_stocks

def main():
    print("=" * 60)
    print("🔄 بدء تحديث بيانات البورصة المصرية")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # تحميل قائمة الأسهم
    if not SYMBOL_FILE.exists():
        print("❌ ملف الأسهم غير موجود!")
        return
    
    df_symbols = pd.read_csv(SYMBOL_FILE)
    symbols = df_symbols['symbol'].tolist()
    
    print(f"📊 عدد الأسهم المطلوب تحديثها: {len(symbols)}")
    
    # تحديث الأسهم
    updated, failed = update_all_stocks(symbols)
    
    print("\n" + "=" * 60)
    print(f"✅ تم تحديث {len(updated)} سهماً بنجاح")
    if failed:
        print(f"⚠️ فشل تحديث {len(failed)} سهماً: {failed[:10]}...")
    print("=" * 60)

if __name__ == "__main__":
    main()
