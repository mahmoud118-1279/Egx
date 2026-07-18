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
from indicators import add_all_indicators
from ai_engine import EnsemblePredictor
from risk_manager import generate_trading_strategy, calculate_position_size
from learning_analyst import SelfLearningAIAnalyst

st.set_page_config(
    page_title="منظومة المضارب والمستثمر الكمي والنيوز-إنتلجنس - EGX",
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
risk_pct = st.sidebar.slider("⚠️ نسبة المخاطرة المسموحة لكل صفقة (%):", min_value=0.5, max_value=10.0, value=2.0,
                             step=0.5)

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
        st.warning("⚠️ ملف الأسهم `egx_symbols.csv` غير موجود أو فارغ. يرجى تهيئته أولاً.")
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

                    # [التعديل الفني الأول لحظر الـ KeyError]: حقن المؤشرات أولاً بالكامل في الـ df قبل التمرير للموديل
                    s_df = add_all_indicators(s_df)
                    s_df['News_Sentiment'] = news_score
                    s_df.attrs['symbol'] = row['symbol']

                    # [التعديل الفني الثاني]: استقبال المتغيرات الخمسة بالكامل شاملة الـ score_out المطور
                    dir_out, pred_target, entry_out, exit_out, score_out = predictor.predict_next_price(s_df,
                                                                                                        strategy_mode)

                    if any(keyword in dir_out for keyword in ["🟢", "🚀", "📈"]):
                        strat = generate_trading_strategy(s_df, pred_target, dir_out)

                        # [التعديل الفني الثالث]: تمرير درجة ثقة ومجموع نقاط الآلة لحساب حجم الصفقة الديناميكي
                        shares_to_buy = calculate_position_size(
                            account_balance, risk_pct, entry_out, strat["stop_loss"], decision_score=score_out
                        )

                        return {
                            "السهم": row['symbol'],
                            "السعر الحالي": round(s_df['Close'].iloc[-1], 2),
                            "الهدف المتوقع": round(pred_target, 2),
                            "نقطة الدخول": round(entry_out, 2),
                            "وقف الخسارة": strat["stop_loss"],
                            "الكمية المقترحة": shares_to_buy,
                            "درجة أمان الإشارة (100)": score_out,
                            "نبرة الأخبار": news_score,
                            "الإشارة الفورية للآلة": dir_out,
                            "المصدر": src,
                            "df_backup": s_df
                        }
                except Exception:
                    pass
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

            if opportunities:
                opportunities_df = pd.DataFrame(opportunities)

                st.subheader(f"🎯 تم العثور على ({len(opportunities_df)}) فرصة واعدة متوافقة مع شروط الأمان:")
                st.dataframe(opportunities_df.drop(columns=["df_backup"]), use_container_width=True)

                st.markdown("### 💾 أرشفة وتسجيل الفرص داخل عقل الآلة")
                if st.button("🧠 حفظ هذه الفرص وتفعيل مراقبتها حركياً"):
                    saved_count = 0
                    for _, opp in opportunities_df.iterrows():
                        backup_df = opp["df_backup"]
                        last_row_ind = backup_df.iloc[-1].to_dict()

                        analyst.record_prediction(
                            symbol=opp["السهم"],
                            current_price=opp["السعر الحالي"],
                            predicted_close=opp["الهدف المتوقع"],
                            suggested_entry=opp["نقطة الدخول"],
                            suggested_exit=opp["الهدف المتوقع"],
                            direction=opp["الإشارة الفورية للآلة"],
                            current_indicators=last_row_ind
                        )
                        saved_count += 1
                    st.success(f"✅ تم بنجاح حقن وتسجيل {saved_count} توصية داخل ملف الـ JSON لتبدأ الآلة بتعلمها!")

                st.markdown("### 📲 إرسال التقرير الميداني لهاتفك")
                if st.button("📣 إرسال هذه الفرص المكتشفة كرسالة مجمعة على تليجرام"):
                    telegram_msg = f"🔔 *تقرير مسح البورصة المصرية المطور* 🔔\n"
                    telegram_msg += f"📅 *التاريخ:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    telegram_msg += f"🎯 *الاستراتيجية المتبعة:* {strategy_mode}\n"
                    telegram_msg += "═\n"

                    for _, row in opportunities_df.iterrows():
                        telegram_msg += f"📊 *سهم:* {row['السهم']} | {row['الإشارة الفورية للآلة']}\n"
                        telegram_msg += f"💰 *السعر الحالي:* {row['السعر الحالي']} ج.م\n"
                        telegram_msg += f"🟢 *أفضل دخول:* {row['نقطة الدخول']} ج.م | *الوقف:* {row['وقف الخسارة']} ج.م\n"
                        telegram_msg += f"🛡️ *درجة الأمان والتصويت:* {row['درجة أمان الإشارة (100)']} / 100\n"
                        telegram_msg += f"🎯 *الهدف المتوقع:* {row['الهدف المتوقع']} ج.م\n"
                        telegram_msg += f"📰 *نبرة الأخبار:* {row['نبرة الأخبار']}\n"
                        telegram_msg += "───────────────────\n"

                    telegram_msg += f"📢 _إجمالي الفرص الواعدة المرسلة: {len(opportunities_df)} سهم_"

                    try:
                        send_telegram_alert(telegram_msg)
                        st.success("✅ تم إرسال التقرير المجمع بنجاح! تفقد تطبيق تليجرام على هاتفك الآن.")
                    except Exception as e:
                        st.error(f"❌ حدث خطأ أثناء الإرسال: {e}")
            else:
                st.info("⏳ السوق يسير في بيئة غير مستقرة حالياً، لم تعثر الآلة على صفقات تحقق شروط الأمان.")

# --- التاب الثاني: مركز التعلم والتطوير الذاتي ---
with tab_learning:
    st.header("🧠 لوحة التحكم الذاتي والنمو المعرفي للمضارب الكمي")

    stats = analyst.get_learning_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 إجمالي الصفقات المخزنة بالـ JSON", stats["total_predictions"])
    col2.metric("⏳ الصفقات الجارية وقيد التقييم", stats["pending_predictions"])
    col3.metric("🎯 معدل نجاح وتصحيح المنظومة الحالي", f"{stats['success_rate']:.1%}")

    st.markdown("---")
    dynamic_weights = analyst.learning_data.get("dynamic_weights", {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3})
    st.subheader("🧬 التوزيع الحالي لكتل وأوزان التصويت الجينية للذكاء الاصطناعي:")
    st.json(dynamic_weights)

    st.subheader("🗂️ السجل والأرشيف الكامل لتقييم ومعالجة أخطاء الآلة")
    if stats["history"]:
        st.table(pd.DataFrame(stats["history"]))
    else:
        st.info("📂 السجل فارغ تماماً حالياً، سيتم ملء هذه اللوحة بمجرد بدء عمليات الحفظ التلقائي.")

# --- التاب الثالث: تحليل سهم منفرد ---
with tab_analysis:
    st.header("🔍 فحص جراحي مخصص لسهم محدد")

    if not df_symbols.empty:
        choice = st.selectbox("اختر الشركة المراد فحصها بعمق:", df_symbols['name'].tolist())
        row_choice = df_symbols[df_symbols['name'] == choice].iloc[0]

        if st.button("🔬 ابدأ الفحص الجراحي العميق للسهم"):
            with st.spinner("جاري جمع بيانات السيولة وحساب المؤشرات التراكمية ومسح نبرة الأخبار..."):
                news_score = fetch_company_news_sentiment(row_choice['name'], row_choice['symbol'])
                s_df, src = fetch_stock_data(row_choice['symbol'], row_choice['y_symbol'])

                if not s_df.empty and len(s_df) >= 30:
                    # [التعديل الفني الأول لحظر الـ KeyError]: حساب المؤشرات أولاً لتوليد حارات بولينجر
                    s_df = add_all_indicators(s_df)
                    s_df['News_Sentiment'] = news_score
                    s_df.attrs['symbol'] = row_choice['symbol']

                    # [التعديل الفني الثاني]: استقبال الخمسة مخرجات بالكامل
                    dir_out, pred_target, entry_out, exit_out, score_out = predictor.predict_next_price(s_df,
                                                                                                        strategy_mode)
                    strat = generate_trading_strategy(s_df, pred_target, dir_out)

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.metric("💰 السعر الفوري الحالي بالسوق", f"{s_df['Close'].iloc[-1]:.2f} ج.م")
                        st.metric("🎯 السعر المتوقع والمستهدف للآلة", f"{pred_target:.2f} ج.م")
                        st.metric("🛡️ مجموع نقاط الأمان والوزن (100)", f"{score_out} / 100")
                    with col_r:
                        st.subheader("📊 استراتيجية إدارة المخاطر المقترحة")
                        st.write(f"**التوجيه الفني النهائي:** {dir_out}")
                        st.write(f"**أفضل نقطة دخول:** {entry_out:.2f} ج.م")
                        st.write(f"**نقطة وقف الخسارة الآمنة (ATR):** {strat['stop_loss']:.2f} ج.م")
                        st.write(f"**المستهدف الأول لجني الأرباح:** {strat['take_profit_1']:.2f} ج.م")
                        st.write(f"**المستهدف الثاني لجني الأرباح:** {strat['take_profit_2']:.2f} ج.م")

                    st.markdown("---")
                    st.subheader("📰 تحليل نبرة ومشاعر الأخبار اللحظية المتجمعة:")
                    st.info(f"معدل مشاعر الأخبار الحالي للشركة هو: ({news_score})")

                    st.subheader("📈 البيانات الفنية الأخيرة المحقونة لعقل الآلة:")
                    st.dataframe(s_df.tail(10), use_container_width=True)
                else:
                    st.error("❌ فشل جلب البيانات التاريخية لهذا السهم، أو أن تاريخ تداولاته قصير جداً بالبورصة.")