import os
import sys
from pathlib import Path

# 1. تحديد مسار بايثون الحالي
PYTHON_EXE_PATH = Path(sys.executable)

# 2. اكتشاف مسار الـ tcl تلقائياً
if hasattr(sys, 'base_prefix'):
    base_python_dir = Path(sys.base_prefix)
else:
    base_python_dir = PYTHON_EXE_PATH.parent

# 3. بناء المسارات الديناميكية
tcl_dir = base_python_dir / "tcl" / "tcl8.6"
tk_dir = base_python_dir / "tcl" / "tk8.6"

# 4. حقن المسارات في الـ Environment Variables
if tcl_dir.exists():
    os.environ['TCL_LIBRARY'] = str(tcl_dir)
if tk_dir.exists():
    os.environ['TK_LIBRARY'] = str(tk_dir)

# 5. استيراد المكتبات
import pandas as pd
import numpy as np
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 6. استيراد محركات المنظومة
from config import SYMBOL_FILE
from data_engine import fetch_stock_data, send_telegram_alert
from indicators import add_all_indicators
from ai_engine import EnsemblePredictor
from learning_analyst import SelfLearningAIAnalyst
from risk_manager import generate_trading_strategy, calculate_position_size

# 7. إعداد المظهر العام للواجهة
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EGXQuantumDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EGX Quantum System - محرك المضاربة والتعلم الذاتي")
        self.geometry("1200x750")

        # تهيئة المحركات
        self.predictor = EnsemblePredictor()
        self.analyst = SelfLearningAIAnalyst()

        # قراءة ملف الشركات
        if os.path.exists(SYMBOL_FILE):
            self.symbols_df = pd.read_csv(SYMBOL_FILE)
            self.stock_list = [f"{row['symbol']} - {row['name']}" for _, row in self.symbols_df.iterrows()]
        else:
            self.symbols_df = pd.DataFrame()
            self.stock_list = ["لا توجد أسهم متاحة حالياً"]

        # --- بناء الهيكل الرئيسي ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#0F172A")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.main_frame = ctk.CTkFrame(self, fg_color="#0F172A")
        self.main_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        # متغير لتخزين السهم الحالي
        self.current_symbol = None

        self.build_sidebar_widgets()
        self.build_welcome_screen()

    def build_sidebar_widgets(self):
        """بناء عناصر التحكم في القائمة الجانبية"""
        lbl_logo = ctk.CTkLabel(
            self.sidebar, 
            text="📈 EGX QUANTUM", 
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#3B82F6"
        )
        lbl_logo.pack(pady=(20, 5))

        lbl_subtitle = ctk.CTkLabel(
            self.sidebar, 
            text="نظام مضاربة وتعلّم مستمر", 
            font=ctk.CTkFont(size=12),
            text_color="#64748B"
        )
        lbl_subtitle.pack(pady=(0, 20))

        lbl_select = ctk.CTkLabel(
            self.sidebar, 
            text="اختر السهم من البورصة المصرية:", 
            font=ctk.CTkFont(size=13),
            text_color="#94A3B8"
        )
        lbl_select.pack(pady=5, padx=15, anchor="w")

        self.stock_dropdown = ctk.CTkOptionMenu(
            self.sidebar, 
            values=self.stock_list, 
            width=230, 
            fg_color="#1E293B",
            button_color="#3B82F6"
        )
        self.stock_dropdown.pack(pady=10, padx=15)

        self.btn_analyze = ctk.CTkButton(
            self.sidebar, 
            text="⚡ تشغيل التحليل الفوري والمضاربة",
            command=self.execute_analysis, 
            fg_color="#2563EB", 
            hover_color="#1D4ED8",
            height=45, 
            font=ctk.CTkFont(weight="bold")
        )
        self.btn_analyze.pack(pady=20, padx=15)

        # زر تحديث التقييم الذاتي
        self.btn_evaluate = ctk.CTkButton(
            self.sidebar,
            text="🔄 تحديث وتقييم التوقعات السابقة",
            command=self.evaluate_predictions,
            fg_color="#7C3AED",
            hover_color="#6D28D9",
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.btn_evaluate.pack(pady=10, padx=15)

        # ✅ زر تنظيف البيانات القديمة
        self.btn_clean = ctk.CTkButton(
            self.sidebar,
            text="🧹 تنظيف البيانات القديمة (90 يوم)",
            command=self.clean_old_data,
            fg_color="#EF4444",
            hover_color="#DC2626",
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.btn_clean.pack(pady=5, padx=15)

        # ✅ زر تصدير بيانات التعلم
        self.btn_export = ctk.CTkButton(
            self.sidebar,
            text="📤 تصدير بيانات التعلم",
            command=self.export_learning_data,
            fg_color="#10B981",
            hover_color="#059669",
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.btn_export.pack(pady=5, padx=15)

        self.status_label = ctk.CTkLabel(
            self.sidebar, 
            text="✅ البرنامج جاهز.. بانتظار اختيار السهم",
            font=ctk.CTkFont(size=11), 
            text_color="#94A3B8", 
            wraplength=240
        )
        self.status_label.pack(side="bottom", pady=20, padx=15)

    def build_welcome_screen(self):
        """بناء الشاشة الترحيبية الافتتاحية"""
        self.welcome_label = ctk.CTkLabel(
            self.main_frame,
            text="🚀 أهلاً بك في نظام المضاربة والتعلم الذاتي للبورصة المصرية\n\n"
                 "قم باختيار السهم الذي ترغب بمضاربته من القائمة اليسرى\n"
                 "ثم اضغط على زر 'تشغيل التحليل الفوري' لبدء اقتناص الفرص\n\n"
                 "📊 يمكنك أيضاً تحديث تقييم الصفقات السابقة باستخدام الزر المخصص\n"
                 "🧹 تنظيف البيانات القديمة لإدارة حجم الملف\n"
                 "📤 تصدير بيانات التعلم لعمل نسخة احتياطية",
            font=ctk.CTkFont(size=14), 
            text_color="#94A3B8", 
            justify="center"
        )
        self.welcome_label.pack(expand=True)

    # ============================================================
    # ✅ دالة تنظيف البيانات القديمة
    # ============================================================
    def clean_old_data(self):
        """تنظيف البيانات القديمة من ملف التعلم"""
        self.status_label.configure(text="🧹 جاري تنظيف البيانات القديمة...", text_color="#F59E0B")
        self.update_idletasks()
        
        try:
            self.analyst.clear_old_predictions(days=90)
            self.status_label.configure(text="✅ تم تنظيف البيانات القديمة بنجاح!", text_color="#10B981")
            
            # تحديث الواجهة
            self.refresh_current_view()
            
        except Exception as e:
            self.status_label.configure(text=f"❌ خطأ في التنظيف: {str(e)[:50]}", text_color="#EF4444")

    # ============================================================
    # ✅ دالة تصدير بيانات التعلم
    # ============================================================
    def export_learning_data(self):
        """تصدير بيانات التعلم إلى ملف"""
        self.status_label.configure(text="📤 جاري تصدير بيانات التعلم...", text_color="#F59E0B")
        self.update_idletasks()
        
        try:
            file_path = self.analyst.export_learning_data()
            if file_path:
                self.status_label.configure(text=f"✅ تم التصدير إلى {file_path}", text_color="#10B981")
            else:
                self.status_label.configure(text="❌ فشل في التصدير", text_color="#EF4444")
        except Exception as e:
            self.status_label.configure(text=f"❌ خطأ في التصدير: {str(e)[:50]}", text_color="#EF4444")

    # ============================================================
    # ✅ دالة تقييم التوقعات
    # ============================================================
    def evaluate_predictions(self):
        """تحديث وتقييم التوقعات السابقة"""
        self.status_label.configure(text="🔄 جاري تحديث وتقييم الصفقات السابقة...", text_color="#F59E0B")
        self.update_idletasks()
        
        try:
            is_updated = self.analyst.evaluate_pending_predictions(fetch_stock_data)
            if is_updated:
                self.status_label.configure(text="✅ تم تحديث وتقييم الصفقات بنجاح!", text_color="#10B981")
                # تحديث الواجهة الحالية إذا كانت مفتوحة
                self.refresh_current_view()
            else:
                self.status_label.configure(text="ℹ️ لا توجد صفقات جديدة بحاجة للتقييم", text_color="#94A3B8")
        except Exception as e:
            self.status_label.configure(text=f"❌ خطأ في التقييم: {str(e)[:50]}", text_color="#EF4444")

    def refresh_current_view(self):
        """تحديث العرض الحالي"""
        # إذا كانت هناك شاشة تحليل مفتوحة، نعيد تحميلها
        if hasattr(self, 'current_symbol') and self.current_symbol:
            self.execute_analysis()

    # ============================================================
    # ✅ دالة تنفيذ التحليل (المعدلة)
    # ============================================================
    def execute_analysis(self):
        """استدعاء الحسابات وتدريب عقل المضارب وتسجيل الصفقات للتعلم من الأخطاء"""
        self.status_label.configure(text="🧠 جاري مراجعة صفقات السوق وتدريب المضارب...", text_color="#F59E0B")
        self.update_idletasks()

        # 1. الحصول على السهم المختار
        selected_text = self.stock_dropdown.get()
        selected_symbol = selected_text.split(" - ")[0]
        self.current_symbol = selected_symbol

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

        # 4. تحديث سجل الأخطاء والتعلم الذاتي أولاً
        try:
            self.analyst.evaluate_pending_predictions(fetch_stock_data)
        except Exception as e:
            print(f"⚠️ خطأ في تقييم التوقعات: {e}")

        # 5. تدريب محرك المضاربة واستخراج الأهداف
        self.predictor.train_all(df_indicators)
        
        # ✅ استقبال 5 قيم من الدالة
        direction, predicted_close, best_entry, best_exit, decision_score = self.predictor.predict_next_price(
            df_indicators
        )

        # 6. ✅ تسجيل التوصية (مع تحويل إلى dict)
        try:
            last_row = df_indicators.iloc[-1]
            indicators_dict = {
                "RSI": float(last_row.get('RSI', 50)),
                "CMF": float(last_row.get('CMF', 0)),
                "News_Sentiment": float(last_row.get('News_Sentiment', 0))
            }
            
            self.analyst.record_prediction(
                selected_symbol,
                df_indicators['Close'].iloc[-1],
                predicted_close,
                best_entry,
                best_exit,
                direction,
                indicators_dict  # ✅ dict
            )
        except Exception as e:
            print(f"⚠️ خطأ في تسجيل التوصية: {e}")

        # 7. توليد استراتيجية إدارة المخاطر والمحفظة
        strategy = generate_trading_strategy(df_indicators, predicted_close, direction)
        strategy['entry_price'] = best_entry
        strategy['take_profit_1'] = best_exit

        # 8. ✅ إرسال إشعار تليجرام إذا كانت إشارة شراء
        if "شراء" in direction or strategy.get("action") == "شراء 🟢":
            alert_msg = (
                f"🚨 *إشارة مضاربة كمية جديدة من منظومة بصرة!* 🚨\n\n"
                f"📈 *السهم:* {selected_symbol}\n"
                f"💵 *السعر الحالي:* {df_indicators['Close'].iloc[-1]:.2f} ج.م\n"
                f"🔮 *السعر المستهدف للآلة:* {predicted_close:.2f} ج.م\n\n"
                f"🎯 *نقطة الدخول المقترحة:* {best_entry:.2f} ج.م\n"
                f"🛡️ *وقف الخسارة (ATR):* {strategy.get('stop_loss', best_entry * 0.95):.2f} ج.م\n"
                f"💰 *الهدف الأول:* {strategy.get('take_profit_1', best_exit):.2f} ج.م\n"
                f"🚀 *الهدف الثاني:* {strategy.get('take_profit_2', best_exit * 1.1):.2f} ج.م\n\n"
                f"⚖️ *نسبة المخاطرة للمكسب:* {strategy.get('risk_reward_ratio', '1:1.5')}\n"
                f"🤖 *درجة ثقة الآلة:* {decision_score}/100\n"
                f"_تم الفحص والتحليل آلياً بواسطة الهجين الكمي_"
            )
            try:
                send_telegram_alert(alert_msg)
                print("✅ تم إرسال إشعار التليجرام")
            except Exception as e:
                print(f"⚠️ خطأ في إرسال التليجرام: {e}")

        # 9. تنظيف الواجهة لبناء التبويبات الجديدة
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # 10. عرض النتائج
        self.display_dashboard_results(
            selected_symbol, 
            df_indicators, 
            predicted_close, 
            best_entry, 
            best_exit, 
            direction, 
            strategy,
            decision_score
        )
        
        self.status_label.configure(text="✅ تم تحديث الأهداف ومراجعة الأخطاء بنجاح", text_color="#10B981")

    # ============================================================
    # ✅ دالة عرض النتائج (المعدلة)
    # ============================================================
    def display_dashboard_results(self, symbol, df, pred_price, best_entry, best_exit, direction, strategy, score=50):
        """عرض النتائج في لوحة التحكم مع تبويبات متعددة"""
        
        # --- صف الكروت العلوية الرقمية الفورية ---
        metrics_frame = ctk.CTkFrame(self.main_frame, height=100, fg_color="#1E293B", corner_radius=10)
        metrics_frame.pack(fill="x", padx=10, pady=10)
        metrics_frame.pack_propagate(False)

        last_close = df['Close'].iloc[-1]
        current_cmf = df['CMF'].iloc[-1] if 'CMF' in df.columns else 0.0
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50.0

        # بطاقة السعر الحالي
        lbl_price = ctk.CTkLabel(
            metrics_frame, 
            text=f"💰 السعر الحالي\n{last_close:.2f} ج.م",
            font=ctk.CTkFont(size=14, weight="bold"), 
            text_color="#F8FAFC"
        )
        lbl_price.pack(side="left", expand=True, padx=10)

        # بطاقة أفضل دخول
        entry_color = "#10B981" if best_entry < last_close else "#F59E0B"
        lbl_entry = ctk.CTkLabel(
            metrics_frame, 
            text=f"🎯 أفضل دخول\n{best_entry:.2f} ج.م",
            font=ctk.CTkFont(size=14, weight="bold"), 
            text_color=entry_color
        )
        lbl_entry.pack(side="left", expand=True, padx=10)

        # بطاقة أفضل خروج
        exit_color = "#EF4444" if best_exit > last_close else "#3B82F6"
        lbl_exit = ctk.CTkLabel(
            metrics_frame, 
            text=f"🎯 أفضل خروج\n{best_exit:.2f} ج.م",
            font=ctk.CTkFont(size=14, weight="bold"), 
            text_color=exit_color
        )
        lbl_exit.pack(side="left", expand=True, padx=10)

        # بطاقة RSI
        rsi_color = "#10B981" if 30 < current_rsi < 70 else ("#EF4444" if current_rsi > 70 else "#F59E0B")
        lbl_rsi = ctk.CTkLabel(
            metrics_frame, 
            text=f"📊 RSI\n{current_rsi:.1f}",
            font=ctk.CTkFont(size=14, weight="bold"), 
            text_color=rsi_color
        )
        lbl_rsi.pack(side="left", expand=True, padx=10)

        # بطاقة CMF
        cmf_text = "🔥 تجمع قوي" if current_cmf > 0.1 else ("⚠️ تصريف" if current_cmf < -0.1 else "⚖️ متعادل")
        cmf_color = "#10B981" if current_cmf > 0.05 else ("#EF4444" if current_cmf < -0.05 else "#F59E0B")
        lbl_cmf = ctk.CTkLabel(
            metrics_frame, 
            text=f"💧 تدفق السيولة\n{cmf_text}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=cmf_color
        )
        lbl_cmf.pack(side="left", expand=True, padx=10)

        # بطاقة درجة الثقة
        lbl_score = ctk.CTkLabel(
            metrics_frame,
            text=f"🤖 ثقة الآلة\n{score}/100",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8B5CF6"
        )
        lbl_score.pack(side="left", expand=True, padx=10)

        # --- بناء التبويبات ---
        tab_view = ctk.CTkTabview(self.main_frame)
        tab_view.pack(fill="both", expand=True, padx=10, pady=5)

        tab_chart = tab_view.add("📊 شاشة المضارب التفاعلية")
        tab_strategy = tab_view.add("🛡️ خطة إدارة الصفقة")
        tab_learning = tab_view.add("🧠 نظام التعلم الذاتي")

        # ---- تبويب 1: تشارت المضارب ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5), gridspec_kw={'height_ratios': [3, 1]}, facecolor='#0F172A')
        ax1.set_facecolor('#1E293B')
        ax2.set_facecolor('#1E293B')

        recent_df = df.tail(80).copy()
        
        # خط السعر
        ax1.plot(recent_df.index, recent_df['Close'], color='#3B82F6', label='سعر الإغلاق', linewidth=2.5)

        # حارات بولينجر
        if 'BB_Upper' in recent_df.columns:
            ax1.plot(recent_df.index, recent_df['BB_Upper'], color='#64748B', linestyle='--', alpha=0.5, linewidth=1)
            ax1.plot(recent_df.index, recent_df['BB_Lower'], color='#64748B', linestyle='--', alpha=0.5, linewidth=1)
            ax1.fill_between(recent_df.index, recent_df['BB_Upper'], recent_df['BB_Lower'], alpha=0.1, color='#64748B')

        # متوسطات EMA
        if 'EMA_20' in recent_df.columns:
            ax1.plot(recent_df.index, recent_df['EMA_20'], color='#F59E0B', linestyle='-', alpha=0.7, linewidth=1.5, label='EMA 20')
        if 'EMA_50' in recent_df.columns:
            ax1.plot(recent_df.index, recent_df['EMA_50'], color='#10B981', linestyle='-', alpha=0.7, linewidth=1.5, label='EMA 50')

        # خطوط الدخول والخروج
        ax1.axhline(best_entry, color='#10B981', linestyle='-', linewidth=2, label=f"دخول ({best_entry:.2f})")
        ax1.axhline(best_exit, color='#EF4444', linestyle='-', linewidth=2, label=f"خروج ({best_exit:.2f})")
        
        # إضافة نقطة السعر المتوقع
        if pred_price:
            ax1.scatter(recent_df.index[-1:], [pred_price], color='#8B5CF6', s=100, zorder=5, label=f"متوقع ({pred_price:.2f})")

        ax1.legend(loc='upper left', fontsize=9, facecolor='#0F172A', labelcolor='white')
        ax1.grid(True, color='#334155', alpha=0.4)
        ax1.tick_params(colors='white', labelsize=9)
        ax1.set_title(f'📈 تحليل سهم {symbol}', color='white', fontsize=12)

        # الحجم
        colors = ['#10B981' if recent_df['Close'].iloc[i] >= recent_df['Open'].iloc[i] else '#EF4444' 
                  for i in range(len(recent_df))]
        ax2.bar(recent_df.index, recent_df['Volume'], color=colors, alpha=0.7)
        ax2.grid(True, color='#334155', alpha=0.3)
        ax2.tick_params(colors='white', labelsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.set_ylabel('حجم التداول', color='white', fontsize=9)

        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab_chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # ---- تبويب 2: خطة الصفقات ----
        strat_panel = ctk.CTkFrame(tab_strategy, fg_color="transparent")
        strat_panel.pack(fill="both", expand=True, padx=15, pady=15)

        rec_box = ctk.CTkTextbox(strat_panel, height=160, font=ctk.CTkFont(size=13))
        rec_box.pack(fill="x", pady=5)

        potential_profit_pct = ((best_exit - best_entry) / best_entry * 100) if best_entry > 0 else 0
        stop_loss = strategy.get('stop_loss', best_entry * 0.95)
        take_profit_2 = strategy.get('take_profit_2', best_exit * 1.1)

        rec_box.insert("0.0", 
            f"🎯 توصية المضارب الكمي الذكي لسهم ({symbol}):\n"
            f"{'='*70}\n"
            f"• 📌 قرار المنظومة الحالي: {direction}\n"
            f"• 🟢 السعر الأمثل لبناء مركز شراء: {best_entry:.2f} ج.م\n"
            f"• 🎯 هدف جني الأرباح القريب: {best_exit:.2f} ج.م (عائد متوقع: +{potential_profit_pct:.1f}%)\n"
            f"• 🛡️ وقف الخسارة الفوري: {stop_loss:.2f} ج.م\n"
            f"• 🚀 الهدف الثاني (الممتد): {take_profit_2:.2f} ج.م\n"
            f"• ⚖️ نسبة المخاطرة/المكسب: {strategy.get('risk_reward_ratio', '1:1.5')}\n"
            f"• 🤖 درجة ثقة الذكاء الاصطناعي: {score}/100\n"
            f"{'='*70}\n"
            f"📊 الإشارة: {'🟢 شراء' if 'شراء' in direction else '🔴 تجنب' if 'تجنب' in direction else '⏳ انتظار'}"
        )
        rec_box.configure(state="disabled")

        # معلومات إضافية عن المخاطرة
        risk_frame = ctk.CTkFrame(strat_panel, fg_color="#1E293B", corner_radius=8)
        risk_frame.pack(fill="x", pady=10)

        risk_info = (
            f"📋 معلومات إدارة المخاطر:\n"
            f"• نسبة المخاطرة الموصى بها: 2% من رأس المال\n"
            f"• المخاطرة لكل سهم: {abs(best_entry - stop_loss):.2f} ج.م\n"
            f"• عدد الأسهم الموصى به (لـ 50,000 ج.م): ~{int(50000 * 0.02 / max(abs(best_entry - stop_loss), 0.01))}"
        )
        lbl_risk = ctk.CTkLabel(risk_frame, text=risk_info, font=ctk.CTkFont(size=12), text_color="#94A3B8", justify="left")
        lbl_risk.pack(pady=10, padx=15, anchor="w")

        # ---- تبويب 3: نظام التعلم الذاتي ----
        learn_panel = ctk.CTkFrame(tab_learning, fg_color="transparent")
        learn_panel.pack(fill="both", expand=True, padx=15, pady=15)

        stats = self.analyst.get_learning_stats()

        # كارت نسبة النجاح
        success_color = "#10B981" if stats['success_rate'] > 0.6 else ("#F59E0B" if stats['success_rate'] > 0.3 else "#EF4444")
        lbl_success_box = ctk.CTkLabel(
            learn_panel,
            text=f"🧠 معدل نجاح توصيات الذكاء الاصطناعي: {stats['success_rate']:.1%}",
            font=ctk.CTkFont(size=16, weight="bold"), 
            text_color=success_color
        )
        lbl_success_box.pack(pady=5)

        # ✅ إضافة Profit Factor
        lbl_profit_factor = ctk.CTkLabel(
            learn_panel,
            text=f"📈 Profit Factor: {stats.get('profit_factor', 0):.2f}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#8B5CF6"
        )
        lbl_profit_factor.pack(pady=2)

        lbl_details = ctk.CTkLabel(
            learn_panel,
            text=f"📊 إجمالي الصفقات: {stats['total_predictions']}  |  ⏳ قيد التقييم: {stats['pending_predictions']}",
            font=ctk.CTkFont(size=13), 
            text_color="#94A3B8"
        )
        lbl_details.pack(pady=2)

        # عرض الأوزان الديناميكية
        weights = self.analyst.learning_data.get("dynamic_weights", {})
        weights_text = f"🧬 أوزان التصويت الجينية:   الربح المتوقع: {weights.get('expected_gain', 0.4):.1f}  |  CMF: {weights.get('cmf', 0.3):.1f}  |  الأخبار: {weights.get('news', 0.3):.1f}"
        lbl_weights = ctk.CTkLabel(
            learn_panel,
            text=weights_text,
            font=ctk.CTkFont(size=12),
            text_color="#8B5CF6"
        )
        lbl_weights.pack(pady=2)

        # ✅ عرض أفضل أنماط النجاح
        try:
            best_patterns = self.analyst.get_best_patterns(top_n=3)
            if best_patterns:
                patterns_text = "🏆 أفضل أنماط النجاح:\n"
                for i, p in enumerate(best_patterns[:3], 1):
                    patterns_text += f"  {i}. RSI: {p['rsi_range']} | CMF: {p['cmf_type']} | نجاح: {p['win_rate']}%\n"
                lbl_patterns = ctk.CTkLabel(
                    learn_panel,
                    text=patterns_text,
                    font=ctk.CTkFont(size=11),
                    text_color="#94A3B8",
                    justify="left"
                )
                lbl_patterns.pack(pady=5, anchor="w")
        except Exception as e:
            print(f"⚠️ خطأ في عرض الأنماط: {e}")

        # جدول السجل
        table_box = ctk.CTkTextbox(learn_panel, height=180, font=ctk.CTkFont(size=11))
        table_box.pack(fill="both", expand=True, pady=10)

        table_text = "🗂️ سجل ومراجعة صفقات المضارب التاريخية:\n"
        table_text += "=" * 90 + "\n"
        
        if stats["history"]:
            # عرض آخر 20 صفقة فقط
            for row in stats["history"][:20]:
                status_icon = "✅" if "نجاح" in row['الحالة'] else ("❌" if "فشل" in row['الحالة'] else "⏳")
                table_text += (
                    f"[{row['التاريخ']}] {row['S']} | "
                    f"القرار: {row['R'][:20]}... | "
                    f"دخول: {row['سعر الدخول']} | "
                    f"هدف: {row['الهدف']} | "
                    f"أعلى: {row['أعلى سعر']} | "
                    f"{status_icon} {row['الحالة']}\n"
                )
        else:
            table_text += "📂 السجل فارغ حالياً، سيتم ملء هذه اللوحة بمجرد بدء عمليات الحفظ التلقائي."

        table_box.insert("0.0", table_text)
        table_box.configure(state="disabled")

        # حفظ إشارة للسهم الحالي
        self.current_symbol = symbol


# ============================================================
# ✅ تشغيل التطبيق
# ============================================================
if __name__ == "__main__":
    app = EGXQuantumDesktopApp()
    app.mainloop()
