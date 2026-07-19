import os
import sys
from pathlib import Path

# 1. تحديد مسار بايثون الحالي اللي الـ PyCharm مشغل بيه المشروع
PYTHON_EXE_PATH = Path(sys.executable)

# 2. إذا كنا شغالين جوه بيئة افتراضية (.venv)، ملفات الـ Tcl بتكون في البايثون الأساسي للسيستم
# الكود ده بيكتشف مكان بايثون الأساسي المنبثق منه الـ .venv ويحدد مسار الـ tcl تلقائياً
if hasattr(sys, 'base_prefix'):
    base_python_dir = Path(sys.base_prefix)
else:
    base_python_dir = PYTHON_EXE_PATH.parent

# 3. بناء المسارات الديناميكية بناءً على الجهاز الحالي
tcl_dir = base_python_dir / "tcl" / "tcl8.6"
tk_dir = base_python_dir / "tcl" / "tk8.6"

# 4. حقن المسارات في الـ Environment Variables فقط إذا كانت موجودة
if tcl_dir.exists():
    os.environ['TCL_LIBRARY'] = str(tcl_dir)
if tk_dir.exists():
    os.environ['TK_LIBRARY'] = str(tk_dir)

# الآن يمكنك استكمال الـ imports بأمان
import pandas as pd
import numpy as np
import tkinter as tk
import customtkinter as ctk
# ... باقي الـ imports الخاصة بك ...
import pandas as pd
import numpy as np
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# استيراد محركات المنظومة الكمية والمضاربة والتعلم
from config import SYMBOL_FILE
from data_engine import fetch_stock_data
from indicators import add_all_indicators, calculate_pivot_points
from ai_engine import EnsemblePredictor
from learning_analyst import SelfLearningAIAnalyst
from risk_manager import generate_trading_strategy, calculate_position_size

# إعداد المظهر العام للواجهة الرسومية (تيم داكن احترافي)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EGXQuantumDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EGX Quantum System - محرك المضاربة والتعلم الذاتي")
        self.geometry("1150x700")  # أو المقاس المناسب لشاشتك

        # تهيئة المحركات
        self.predictor = EnsemblePredictor()
        self.analyst = SelfLearningAIAnalyst()

        # قراءة ملف الشركات والأسهم المتاحة
        if os.path.exists(SYMBOL_FILE):
            self.symbols_df = pd.read_csv(SYMBOL_FILE)
            self.stock_list = [f"{row['symbol']} - {row['name']}" for _, row in self.symbols_df.iterrows()]
        else:
            self.symbols_df = pd.DataFrame()
            self.stock_list = ["لا توجد أسهم متاحة حالياً"]

        # --- بناء الهيكل الرئيسي للواجهة (Sidebar & Main Frame) ---
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#0F172A")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.main_frame = ctk.CTkFrame(self, fg_color="#0F172A")
        self.main_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.build_sidebar_widgets()
        self.build_welcome_screen()

    def build_sidebar_widgets(self):
        """بناء عناصر التحكم في القائمة الجانبية"""
        lbl_logo = ctk.CTkLabel(self.sidebar, text="📈 EGX QUANTUM", font=ctk.CTkFont(size=20, weight="bold"),
                                text_color="#3B82F6")
        lbl_logo.pack(pady=(20, 5))

        lbl_subtitle = ctk.CTkLabel(self.sidebar, text="نظام مضاربة وتعلّم مستمر", font=ctk.CTkFont(size=12),
                                    text_color="#64748B")
        lbl_subtitle.pack(pady=(0, 20))

        lbl_select = ctk.CTkLabel(self.sidebar, text="اختر السهم من البورصة المصرية:", font=ctk.CTkFont(size=13),
                                  text_color="#94A3B8")
        lbl_select.pack(pady=5, padx=15, anchor="w")

        self.stock_dropdown = ctk.CTkOptionMenu(self.sidebar, values=self.stock_list, width=220, fg_color="#1E293B",
                                                button_color="#3B82F6")
        self.stock_dropdown.pack(pady=10, padx=15)

        self.btn_analyze = ctk.CTkButton(self.sidebar, text="⚡ تشغيل التحليل الفوري والمضاربة",
                                         command=self.execute_analysis, fg_color="#2563EB", hover_color="#1D4ED8",
                                         height=40, font=ctk.CTkFont(weight="bold"))
        self.btn_analyze.pack(pady=20, padx=15)

        self.status_label = ctk.CTkLabel(self.sidebar, text="البرنامج جاهز.. بانتظار اختيار السهم",
                                         font=ctk.CTkFont(size=11), text_color="#94A3B8", wraplength=220)
        self.status_label.pack(side="bottom", pady=20, padx=15)

    def build_welcome_screen(self):
        """بناء الشاشة الترحيبية الافتتاحية"""
        self.welcome_label = ctk.CTkLabel(self.main_frame,
                                          text="🚀 أهلاً بك في نظام المضاربة والتعلم الذاتي للبورصة المصرية\n\nقم باختيار السهم الذي ترغب بمضاربته من القائمة اليسرى\nثم اضغط على زر 'تشغيل التحليل الفوري' لبدء اقتناص الفرص وتقييم الأداء تلقائياً.",
                                          font=ctk.CTkFont(size=14), text_color="#94A3B8", justify="center")
        self.welcome_label.pack(expand=True)

    def execute_analysis(self):
        """استدعاء الحسابات وتدريب عقل المضارب وتسجيل الصفقات للتعلم من الأخطاء"""
        self.status_label.configure(text="جاري مراجعة صفقات السوق وتدريب المضارب... 🧠", text_color="#F59E0B")
        self.update_idletasks()

        # 1. الحصول على السهم المختار
        selected_text = self.stock_dropdown.get()
        selected_symbol = selected_text.split(" - ")[0]

        if self.symbols_df.empty or selected_symbol not in self.symbols_df['symbol'].values:
            self.status_label.configure(text="❌ رمز السهم غير صحيح", text_color="#EF4444")
            return

        selected_row = self.symbols_df[self.symbols_df['symbol'] == selected_symbol].iloc[0]

        # 2. جلب البيانات
        yahoo_sym = selected_row['y_symbol'] if 'y_symbol' in selected_row else None
        df, source = fetch_stock_data(selected_symbol, yahoo_symbol=yahoo_sym)

        if df.empty or len(df) < 25:
            self.status_label.configure(text="❌ فشل في جلب بيانات كافية للتحليل", text_color="#EF4444")
            return

        # 3. حساب المؤشرات
        df_indicators = add_all_indicators(df.copy())
        pivots = calculate_pivot_points(df_indicators)

        # 4. تحديث سجل الأخطاء والتعلم الذاتي أولاً بناءً على الأسعار الجديدة الحالية
        self.analyst.update_and_evaluate_predictions(selected_symbol, df_indicators)

        # 5. تدريب محرك المضاربة واستخراج الأهداف الحالية لشمعة الغد
        self.predictor.train_all(df_indicators)
        direction, predicted_close, best_entry, best_exit, score = self.predictor.predict_next_price(df_indicators)
        # 6. تسجيل التوصية الحالية في ملف الـ JSON لكي يقيمها البرنامج بنفسه في المرات القادمة
        self.analyst.record_prediction(selected_symbol, df_indicators['Close'].iloc[-1], predicted_close, best_entry,
                                       best_exit, direction)

        # 7. توليد استراتيجية إدارة المخاطر والمحفظة
        strategy = generate_trading_strategy(df_indicators, predicted_close, direction)
        # ... الكود الحالي داخل دالة execute_analysis ...
        # بعد السطر الخاص بـ strategy = generate_trading_strategy(...)

        # استيراد دالة الإرسال (ضعه في أعلى الملف أو هنا مباشرة)
        from data_engine import send_telegram_alert

        # التحقق لو الإشارة هي "شراء مضاربي لقطة" لإرسال الإشعار فوراً
        if "شراء" in direction or strategy.get("action") == "شراء 🟢":
            alert_msg = (
                f"🚨 *إشارة مضاربة كمية جديدة من منظومة بصرة!* 🚨\n\n"
                f"📈 *السهم:* {selected_symbol}\n"
                f"💵 *السعر الحالي:* {df_indicators['Close'].iloc[-1]:.2f} ج.م\n"
                f"🔮 *السعر المستهدف للآلة:* {predicted_close:.2f} ج.م\n\n"
                f"🎯 *نقطة الدخول المقترحة:* {strategy['entry_price']:.2f} ج.م\n"
                f"🛡️ *وقف الخسارة (ATR):* {strategy['stop_loss']:.2f} ج.م\n"
                f"💰 *الهدف الأول:* {strategy['take_profit_1']:.2f} ج.م\n"
                f"🚀 *الهدف الثاني:* {strategy['take_profit_2']:.2f} ج.م\n\n"
                f"⚖️ *نسبة المخاطرة للمكسب:* {strategy['risk_reward_ratio']}\n"
                f"🤖 _تم الفحص والتحليل آلياً بواسطة الهجين الكمي_"
            )
            send_telegram_alert(alert_msg)

        # ... باقي كود دالة execute_analysis الحالي لتحديث الواجهة والشاشات ...
        strategy['entry_price'] = best_entry
        strategy['take_profit_1'] = best_exit

        # 8. تنظيف الواجهة لبناء التبويبات الأربعة الجديدة
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.display_dashboard_results(selected_symbol, df_indicators, pivots, predicted_close, best_entry, best_exit,
                                       direction, strategy)
        self.status_label.configure(text="✅ تم تحديث الأهداف ومراجعة الأخطاء بنجاح", text_color="#10B981")

    def display_dashboard_results(self, symbol, df, pivots, pred_price, best_entry, best_exit, direction, strategy):
        # --- صف الكروت العلوية الرقمية الفورية ---
        metrics_frame = ctk.CTkFrame(self.main_frame, height=90, fg_color="#1E293B", corner_radius=8)
        metrics_frame.pack(fill="x", padx=10, pady=10)
        metrics_frame.pack_propagate(False)

        last_close = df['Close'].iloc[-1]
        current_cmf = df['CMF'].iloc[-1] if 'CMF' in df.columns else 0.0

        lbl_price = ctk.CTkLabel(metrics_frame, text=f"آخر إغلاق\n{last_close:.2f} ج.م",
                                 font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC")
        lbl_price.pack(side="right", expand=True, padx=5)

        lbl_entry = ctk.CTkLabel(metrics_frame, text=f"🎯 أنسب دخول\n{best_entry:.2f} ج.م",
                                 font=ctk.CTkFont(size=14, weight="bold"), text_color="#10B981")
        lbl_entry.pack(side="right", expand=True, padx=5)

        lbl_exit = ctk.CTkLabel(metrics_frame, text=f"🚀 أنسب خروج\n{best_exit:.2f} ج.م",
                                font=ctk.CTkFont(size=14, weight="bold"), text_color="#3B82F6")
        lbl_exit.pack(side="right", expand=True, padx=5)

        cmf_text = "تجمع قوي 🔥" if current_cmf > 0.1 else ("تصريف ⚠️" if current_cmf < -0.1 else "متعادل ⚖️")
        lbl_cmf = ctk.CTkLabel(metrics_frame, text=f"تدفق السيولة\n{cmf_text}",
                               font=ctk.CTkFont(size=13, weight="bold"),
                               text_color="#10B981" if current_cmf > 0.0 else "#EF4444")
        lbl_cmf.pack(side="right", expand=True, padx=5)

        # بناء تبويبات العرض (أصبح لدينا 3 تبويبات مخصصة للتشارت والإدارة وتكامل التعلم)
        tab_view = ctk.CTkTabview(self.main_frame)
        tab_view.pack(fill="both", expand=True, padx=10, pady=5)

        tab_chart = tab_view.add("📊 شاشة المضارب التفاعلية")
        tab_strategy = tab_view.add("🛡️ خطة إدارة الصفقة والمحفظة")
        tab_learning = tab_view.add("🧠 مراقبة نظام التعلم الذاتي العام")

        # ---- بناء تشارت المضارب الإحترافي ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), gridspec_kw={'height_ratios': [3, 1]}, facecolor='#0F172A')
        ax1.set_facecolor('#1E293B')
        ax2.set_facecolor('#1E293B')

        recent_df = df.tail(60).copy()
        ax1.plot(recent_df.index, recent_df['Close'], color='#3B82F6', label='سعر الإغلاق', linewidth=2.5)

        if 'BB_Upper' in recent_df.columns:
            ax1.plot(recent_df.index, recent_df['BB_Upper'], color='#64748B', linestyle='--', alpha=0.5)
            ax1.plot(recent_df.index, recent_df['BB_Lower'], color='#64748B', linestyle='--', alpha=0.5)

        ax1.axhline(best_entry, color='#10B981', linestyle='-', linewidth=1.5, label=f"دخول ({best_entry:.2f})")
        ax1.axhline(best_exit, color='#EF4444', linestyle='-', linewidth=1.5, label=f"خروج ({best_exit:.2f})")
        ax1.legend(loc='upper left', fontsize=8, facecolor='#0F172A', labelcolor='white')
        ax1.grid(True, color='#334155', alpha=0.4)
        ax1.tick_params(colors='white', labelsize=9)

        colors = ['#10B981' if recent_df['Close'].iloc[i] >= recent_df['Open'].iloc[i] else '#EF4444' for i in
                  range(len(recent_df))]
        ax2.bar(recent_df.index, recent_df['Volume'], color=colors, alpha=0.7)
        ax2.grid(True, color='#334155', alpha=0.3)
        ax2.tick_params(colors='white', labelsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab_chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # ---- تبويب خطة الصفقات وإدارة الصفقات ----
        strat_panel = ctk.CTkFrame(tab_strategy, fg_color="transparent")
        strat_panel.pack(fill="both", expand=True, padx=15, pady=15)

        rec_box = ctk.CTkTextbox(strat_panel, height=130, font=ctk.CTkFont(size=13))
        rec_box.pack(fill="x", pady=5)
        potential_profit_pct = ((best_exit - best_entry) / best_entry) * 100
        rec_box.insert("0.0", f"🎯 توصية المضارب الكمي الذكي لسهم ({symbol}):\n"
                              f"----------------------------------------------------------------------\n"
                              f"• قرار المنظومة الحالي: {direction}\n"
                              f"• السعر الأمثل لبناء مركز شراء (الاقتناص): {best_entry:.2f} ج.م\n"
                              f"• هدف جني الأرباح القريب الصارم (الهروب): {best_exit:.2f} ج.م  (عائد متوقع: +{potential_profit_pct:.1f}%)\n"
                              f"• مستوى وقف الخسارة الفوري لحماية رأسمالك: {strategy['stop_loss']:.2f} ج.م")
        rec_box.configure(state="disabled")

        # ---- تبويب نظام التعلم الذاتي من الأخطاء ----
        learn_panel = ctk.CTkFrame(tab_learning, fg_color="transparent")
        learn_panel.pack(fill="both", expand=True, padx=15, pady=15)

        stats = self.analyst.get_learning_stats()

        # كارت عرض نسبة النجاح الحالية للمنظومة بناءً على الأخطاء السابقة
        lbl_success_box = ctk.CTkLabel(learn_panel,
                                       text=f"🧠 معدل نجاح توصيات منظومة الذكاء الاصطناعي الحالي: {stats['success_rate']:.1%}",
                                       font=ctk.CTkFont(size=15, weight="bold"), text_color="#F59E0B")
        lbl_success_box.pack(pady=5)

        lbl_details = ctk.CTkLabel(learn_panel,
                                   text=f"إجمالي الصفقات المخزنة بالـ JSON ومراقبتها حركياً: {stats['total_predictions']}  |  الصفقات بانتظار إغلاق جلسة الغد: {stats['pending_predictions']}",
                                   font=ctk.CTkFont(size=12), text_color="#94A3B8")
        lbl_details.pack(pady=2)

        # جدول عرض العمليات السابقة (نصي تفصيلي مرتب داخل عقل الآلة)
        table_box = ctk.CTkTextbox(learn_panel, height=200, font=ctk.CTkFont(size=12))
        table_box.pack(fill="both", expand=True, pady=10)

        table_text = "🗂️ سجل ومراجعة صفقات المضارب التاريخية لمعالجة الأخطاء:\n"
        table_text += "--------------------------------------------------------------------------------------------\n"
        for row in stats["history"]:
            table_text += f"[{row['التاريخ']}] سهم: {row['السهم']} | القرار: {row['القرار']} | دخول: {row['سعر الدخول المقترح']} | هدف: {row['الهدف المقترح']} | النتيجة الحركية: {row['الحالة']}\n"

        table_box.insert("0.0", table_text)
        table_box.configure(state="disabled")


if __name__ == "__main__":
    app = EGXQuantumDesktopApp()
    app.mainloop() # <--- الإغلاق الصحيح للدالة
