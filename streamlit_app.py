import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import SYMBOL_FILE
from data_engine import (
    fetch_company_news_sentiment,
    send_telegram_alert,
    fetch_stock_data,
)
from indicators import add_all_indicators, get_market_summary
from ai_engine import EnsemblePredictor
from risk_manager import generate_trading_strategy, calculate_position_size
from learning_analyst import SelfLearningAIAnalyst

st.set_page_config(
    page_title="منظومة المضارب والمستثمر الكمي - EGX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_ai_engine():
    return EnsemblePredictor()


@st.cache_resource
def init_learning_analyst():
    return SelfLearningAIAnalyst()


predictor = init_ai_engine()
analyst = init_learning_analyst()


@st.cache_data
def load_symbols():
    if SYMBOL_FILE.exists():
        return pd.read_csv(SYMBOL_FILE)
    return pd.DataFrame(columns=['name', 'symbol', 'y_symbol'])


df_symbols = load_symbols()

# ----------------- الواجهة الجانبية (Sidebar) -----------------
st.sidebar.title("🛠️ التحكم بالمنظومة")

account_balance = st.sidebar.number_input("💰 رأس مال المحفظة الحالية (ج.م):", min_value=1000, value=50000, step=1000)
risk_pct = st.sidebar.slider("⚠️ نسبة المخاطرة المسموحة لكل صفقة (%):", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

strategy_mode = st.sidebar.selectbox("🎯 استراتيجية عقل الآلة المتبعة:",
                                     ["المضاربة السريعة", "قناص الموجات الاستثمارية"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 حلقة التعلم المستمر والتقييم الذاتي")
if st.sidebar.button("🔄 تحديث وتقييم التوقعات السابقة"):
    with st.spinner("جاري فحص صفقات الـ JSON ومطابقتها بحركة السوق الفعلية..."):
        is_updated = analyst.evaluate_pending_predictions(fetch_stock_data)
        if is_updated:
            st.sidebar.success("✅ نجحت الآلة في مراجعة أخطائها واكتساب الخبرة!")
        else:
            st.sidebar.info("⏳ لا توجد صفقات معلقة بحاجة للتقييم حالياً.")

# ----------------- المحتوى الرئيسي (Main Tabs) -----------------
st.title("📈 منظومة اتخاذ القرار والتعلم الذاتي للبورصة المصرية")
tab_scan, tab_learning, tab_analysis = st.tabs(
    ["🔍 ماسح السوق الفوري", "🧠 مركز التعلم والتطوير الذاتي", "📊 تحليل سهم منفرد"])

# --- التاب الأول: ماسح السوق الفوري ---
with tab_scan:
    st.header("⚡ الفحص الشامل واصطياد الفرص الكمية")

    if df_symbols.empty:
        st.warning("⚠️ ملف الأسهم `egx_symbols.csv` غير موجود أو فارغ.")
    else:
        max_workers = st.slider("🚀 سرعة الفحص (عدد خيوط المعالجة المتوازية):", min_value=1, max_value=20, value=8)

        if st.button("🏁 ابدأ مسح البورصة المصرية الآن"):
            opportunities = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            symbols_list = df_symbols.to_dict('records')
            total_symbols = len(symbols_list)

            def scan_single_stock(row):
                try:
                    news_score = fetch_company_news_sentiment(row['name'], row['symbol'])
                    s_df, src = fetch_stock_data(row['symbol'], row['y_symbol'])

                    if s_df.empty or len(s_df) < 30:
                        return None

                    s_df = add_all_indicators(s_df)
                    s_df['News_Sentiment'] = news_score
                    s_df.attrs['symbol'] = row['symbol']

                    dir_out, pred_target, entry_out, exit_out, score_out = predictor.predict_next_price(s_df, strategy_mode)

                    current_price = s_df['Close'].iloc[-1]
                    
                    # التحقق من صحة التنبؤ
                    change_pct = abs((pred_target - current_price) / current_price) * 100
                    if change_pct > 20:
                        pred_target = current_price * (1 + np.random.uniform(-0.03, 0.03))
                        dir_out = "مراقبة / انتظار إشارة السيولة ⏳"
                        entry_out = current_price
                        exit_out = current_price
                        score_out = 30

                    # ✅ تسجيل التحليل في ملف JSON - حتى لو لم تكن فرصة شراء
                    # هذا هو الجزء المهم: تخزين كل تحليل للتعلم
                    last_row_ind = s_df.iloc[-1].to_dict()
                    analyst.record_prediction(
                        symbol=row['symbol'],
                        current_price=current_price,
                        predicted_close=pred_target,
                        suggested_entry=entry_out,
                        suggested_exit=exit_out,
                        direction=dir_out,
                        current_indicators=last_row_ind
                    )

                    if any(keyword in dir_out for keyword in ["🟢", "🚀", "📈"]):
                        strat = generate_trading_strategy(s_df, pred_target, dir_out)

                        shares_to_buy = calculate_position_size(
                            account_balance, risk_pct, entry_out, strat["stop_loss"], decision_score=score_out
                        )

                        return {
                            "السهم": row['symbol'],
                            "السعر الحالي": round(current_price, 2),
                            "الهدف المتوقع": round(pred_target, 2),
                            "نقطة الدخول": round(entry_out, 2),
                            "وقف الخسارة": strat["stop_loss"],
                            "الكمية المقترحة": shares_to_buy,
                            "درجة الأمان": score_out,
                            "نبرة الأخبار": news_score,
                            "الإشارة": dir_out,
                            "المصدر": src,
                            "df_backup": s_df
                        }
                except Exception as e:
                    print(f"⚠️ خطأ في فحص {row.get('symbol', 'unknown')}: {e}")
                    return None
                return None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(scan_single_stock, row): row for row in symbols_list}

                for i, future in enumerate(as_completed(futures)):
                    res = future.result()
                    if res:
                        opportunities.append(res)

                    pct = (i + 1) / total_symbols
                    progress_bar.progress(pct)
                    status_text.text(f"⏳ جاري فحص وتحليل السهم {i + 1} من {total_symbols}...")

            status_text.text("✅ تم اكتمال فحص ومسح البورصة المصرية بنجاح!")

            # عرض إحصائيات التخزين
            stats = analyst.get_learning_stats()
            st.info(f"📊 تم تخزين {stats['total_predictions']} تحليل في ملف التعلم الذاتي")

            if opportunities:
                opportunities_df = pd.DataFrame(opportunities)
                st.subheader(f"🎯 تم العثور على ({len(opportunities_df)}) فرصة واعدة:")
                st.dataframe(opportunities_df.drop(columns=["df_backup"], errors="ignore"), use_container_width=True)
            else:
                st.info("⏳ لم تعثر الآلة على صفقات تحقق شروط الأمان، ولكن تم تخزين التحليلات للتعلم.")

# --- التاب الثاني: مركز التعلم والتطوير الذاتي ---
with tab_learning:
    st.header("🧠 لوحة التحكم الذاتي والنمو المعرفي للمضارب الكمي")

    stats = analyst.get_learning_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 إجمالي التحليلات", stats["total_predictions"])
    col2.metric("⏳ قيد التقييم", stats["pending_predictions"])
    col3.metric("🎯 نسبة النجاح", f"{stats['success_rate']:.1%}")
    col4.metric("📈 Profit Factor", f"{stats.get('profit_factor', 0):.2f}")

    st.markdown("---")
    
    # عرض الأوزان الديناميكية
    dynamic_weights = analyst.learning_data.get("dynamic_weights", {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3})
    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("🧬 وزن الربح المتوقع", f"{dynamic_weights.get('expected_gain', 0.4):.2f}")
    col_w2.metric("🧬 وزن CMF", f"{dynamic_weights.get('cmf', 0.3):.2f}")
    col_w3.metric("🧬 وزن الأخبار", f"{dynamic_weights.get('news', 0.3):.2f}")

    st.markdown("---")
    
    # عرض أنماط النجاح
    st.subheader("🏆 أفضل أنماط النجاح المكتشفة")
    best_patterns = analyst.get_best_patterns(top_n=5)
    if best_patterns:
        for i, pattern in enumerate(best_patterns, 1):
            st.write(f"**{i}.** تكرار: {pattern['count']} | متوسط الربح: {pattern['avg_profit']:.1f}%")
            st.write(f"   - المؤشرات: RSI≈{pattern['indicators'].get('RSI', 50):.0f}, CMF={pattern['indicators'].get('CMF', 0):.2f}")
    else:
        st.info("⏳ لا توجد أنماط نجاح كافية للتحليل بعد")

    st.markdown("---")
    
    # عرض السجل
    st.subheader("🗂️ سجل التحليلات والأخطاء")
    if stats["history"]:
        # عرض آخر 20 تحليل
        df_history = pd.DataFrame(stats["history"][:20])
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("📂 السجل فارغ حالياً. قم بتشغيل مسح السوق لتوليد التحليلات.")

# --- التاب الثالث: تحليل سهم منفرد ---
with tab_analysis:
    st.header("🔍 فحص جراحي مخصص لسهم محدد")

    if not df_symbols.empty:
        choice = st.selectbox("اختر الشركة المراد فحصها بعمق:", df_symbols['name'].tolist())
        row_choice = df_symbols[df_symbols['name'] == choice].iloc[0]

        if st.button("🔬 ابدأ الفحص الجراحي العميق للسهم"):
            with st.spinner("جاري جمع البيانات وحساب المؤشرات..."):
                news_score = fetch_company_news_sentiment(row_choice['name'], row_choice['symbol'])
                s_df, src = fetch_stock_data(row_choice['symbol'], row_choice['y_symbol'])

                if not s_df.empty and len(s_df) >= 30:
                    s_df = add_all_indicators(s_df)
                    s_df['News_Sentiment'] = news_score
                    s_df.attrs['symbol'] = row_choice['symbol']

                    dir_out, pred_target, entry_out, exit_out, score_out = predictor.predict_next_price(s_df, strategy_mode)

                    current_price = s_df['Close'].iloc[-1]
                    
                    # التحقق من صحة التنبؤ
                    change_pct = abs((pred_target - current_price) / current_price) * 100
                    is_valid = change_pct <= 20
                    
                    if not is_valid:
                        pred_target = current_price * (1 + np.random.uniform(-0.02, 0.02))
                        dir_out = "مراقبة / انتظار إشارة السيولة ⏳"
                        entry_out = current_price
                        exit_out = current_price
                        score_out = 30
                        st.warning("⚠️ تنبؤ النموذج كان غير منطقي، تم تعديله.")

                    strat = generate_trading_strategy(s_df, pred_target, dir_out)
                    summary = get_market_summary(s_df)

                    # ✅ تسجيل التحليل في ملف JSON
                    last_row_ind = s_df.iloc[-1].to_dict()
                    analyst.record_prediction(
                        symbol=row_choice['symbol'],
                        current_price=current_price,
                        predicted_close=pred_target,
                        suggested_entry=entry_out,
                        suggested_exit=exit_out,
                        direction=dir_out,
                        current_indicators=last_row_ind
                    )
                    
                    st.success(f"✅ تم تخزين تحليل {row_choice['symbol']} في ملف التعلم الذاتي")

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.metric("💰 السعر الفوري الحالي", f"{current_price:.2f} ج.م")
                        st.metric("🎯 السعر المتوقع", f"{pred_target:.2f} ج.م")
                        st.metric("🛡️ درجة الأمان", f"{score_out} / 100")
                        st.metric("📰 نبرة الأخبار", f"{news_score}")
                        
                        if isinstance(summary, dict):
                            st.write("**📊 ملخص المؤشرات:**")
                            st.write(f"- RSI: {summary.get('RSI', 'N/A')}")
                            st.write(f"- CMF: {summary.get('CMF', 'N/A')}")
                            st.write(f"- ADX: {summary.get('ADX', 'N/A')}")

                    with col_r:
                        st.subheader("📊 استراتيجية إدارة المخاطر")
                        st.write(f"**التوجيه الفني:** {dir_out}")
                        st.write(f"**أفضل دخول:** {entry_out:.2f} ج.م")
                        st.write(f"**وقف الخسارة:** {strat['stop_loss']:.2f} ج.م")
                        st.write(f"**الهدف الأول:** {strat['take_profit_1']:.2f} ج.م")
                        st.write(f"**الهدف الثاني:** {strat['take_profit_2']:.2f} ج.م")

                    st.markdown("---")
                    st.subheader("📈 البيانات الفنية الأخيرة:")
                    st.dataframe(s_df.tail(10), use_container_width=True)
                    
                    # عرض عدد التحليلات المخزنة
                    stats = analyst.get_learning_stats()
                    st.info(f"📊 إجمالي التحليلات المخزنة: {stats['total_predictions']}")
                else:
                    st.error("❌ فشل جلب البيانات التاريخية لهذا السهم.")


# --- عرض معلومات إضافية في الـ Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 حالة نظام التعلم")

# عرض إحصائيات سريعة
stats = analyst.get_learning_stats()
st.sidebar.metric("📝 إجمالي التحليلات", stats["total_predictions"])
st.sidebar.metric("🎯 نسبة النجاح", f"{stats['success_rate']:.1%}")

# عرض آخر 3 تحليلات
if stats["history"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 آخر التحليلات")
    for p in stats["history"][:3]:
        st.sidebar.write(f"• {p['S']}: {p['R'][:20]}...")
