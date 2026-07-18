import os
import json
from datetime import datetime
import pandas as pd

from config import LEARNING_DATA_FILE


class SelfLearningAIAnalyst:
    """نظام التعليم العميق والتطور الذاتي الحركي لتحليل الأخطاء وإعادة توزين الاستراتيجيات كمياً"""

    def __init__(self):
        self.learning_file = LEARNING_DATA_FILE
        self.learning_data = self._load_learning_data()

    def _load_learning_data(self):
        """تحميل سجل التوقعات المطور مع تهيئة مصفوفات الأوزان المتطورة"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "failure_patterns" not in data: data["failure_patterns"] = []
                    if "success_patterns" not in data: data["success_patterns"] = []
                    # أوزان تصويت ديناميكية يبدأ النظام بها ثم يعدلها بنفسه يومياً
                    if "dynamic_weights" not in data:
                        data["dynamic_weights"] = {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3}
                    return data
            except Exception:
                return {"predictions": [], "success_rate": 0.0, "failure_patterns": [], "success_patterns": [],
                        "dynamic_weights": {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3}}
        return {"predictions": [], "success_rate": 0.0, "failure_patterns": [], "success_patterns": [],
                "dynamic_weights": {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3}}

    def _save_learning_data(self):
        """حفظ السجلات والتطورات الناتجة عن التفكير الذاتي للآلة"""
        try:
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"خطأ أثناء حفظ سجل التعلم المتطور: {e}")

    def record_prediction(self, symbol, current_price, predicted_close, suggested_entry, suggested_exit, direction,
                          current_indicators=None):
        """تسجيل البصمة الفنية المؤشرية الكاملة لحظة اتخاذ القرار لعقد المقارنات لاحقاً"""
        new_pred = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symbol": symbol,
            "current_price": float(current_price),
            "predicted_close": float(predicted_close),
            "suggested_entry": float(suggested_entry),
            "suggested_exit": float(suggested_exit),
            "direction": direction,
            "actual_max_reached": float(current_price),
            "status": "Pending ⏳"
        }

        if current_indicators is not None:
            new_pred["indicators_snapshot"] = {
                "RSI": float(current_indicators.get("RSI", 50.0)),
                "CMF": float(current_indicators.get("CMF", 0.0)),
                "News_Sentiment": float(current_indicators.get("News_Sentiment", 0.0))
            }

        self.learning_data["predictions"].append(new_pred)
        self._save_learning_data()
        return True

    def evaluate_pending_predictions(self, fetch_stock_data_func):
        """تحديث الصفقات وتشغيل محرك التفكير المقارن (Evolutionary Core) لتحديث أوزان التصويت تلقائياً"""
        updated = False

        for pred in self.learning_data.get("predictions", []):
            if pred["status"] != "Pending ⏳":
                continue

            symbol = pred["symbol"]
            try:
                df, _ = fetch_stock_data_func(symbol)
                if df.empty: continue

                recent_high = float(df['High'].max())
                recent_close = float(df['Close'].iloc[-1])

                if recent_high > pred["actual_max_reached"]:
                    pred["actual_max_reached"] = recent_high
                    updated = True

                # فحص النجاح أو الفشل
                if pred["actual_max_reached"] >= pred["suggested_exit"]:
                    pred["status"] = "نجاح باهر 🎯 (حقق الهدف كاملاً)"
                    if "indicators_snapshot" in pred:
                        self.learning_data["success_patterns"].append(
                            {"symbol": symbol, "indicators": pred["indicators_snapshot"]})
                    updated = True
                elif recent_close < pred["suggested_entry"] * 0.95:
                    pred["status"] = "فشل ❌ (ضرب وقف الخسارة)"
                    if "indicators_snapshot" in pred:
                        self.learning_data["failure_patterns"].append(
                            {"symbol": symbol, "indicators": pred["indicators_snapshot"]})
                    updated = True

            except Exception as e:
                print(f"خطأ في مراجعة السهم {symbol}: {e}")

        if updated:
            # 🧠 تشغيل التفكير والتحليل الذاتي: موازنة الأوزان بناءً على الأنماط الأكثر تكراراً في النجاح
            self._evolve_voting_weights()
            self._recalculate_success_rate()
            self._save_learning_data()

        return updated

    def _evolve_voting_weights(self):
        """محرك التطور الذاتي: يحلل أي المعايير كانت سبباً في النجاح لرفع وزنها مستقبلاً وعقاب المؤشرات الفاشلة"""
        success_list = self.learning_data.get("success_patterns", [])
        failure_list = self.learning_data.get("failure_patterns", [])

        if len(success_list) < 3:
            return  # الانتظار حتى تجميع خبرة كافية للتطور

        current_weights = self.learning_data["dynamic_weights"]

        # إذا وجدنا أن الصفقات الناجحة كانت تتميز دائماً بنبرة أخبار قوية وعالية الإيجابية
        avg_success_news = sum(p["indicators"].get("News_Sentiment", 0.0) for p in success_list) / len(success_list)
        avg_failed_news = sum(p["indicators"].get("News_Sentiment", 0.0) for p in failure_list) / max(len(failure_list),
                                                                                                      1)

        # التعديل الوراثي الديناميكي للأوزان
        if avg_success_news > 0.1 and avg_failed_news < 0.0:
            # نبرة الأخبار فارقة جداً في السوق حالياً، نرفع وزنها حركياً بنسبة 5% على حساب بقية المؤشرات
            current_weights["news"] = min(current_weights["news"] + 0.05, 0.5)
            current_weights["cmf"] = max(current_weights["cmf"] - 0.025, 0.15)
            current_weights["expected_gain"] = max(current_weights["expected_gain"] - 0.025, 0.2)

        self.learning_data["dynamic_weights"] = current_weights

    def _recalculate_success_rate(self):
        total_evaluated = 0
        total_correct = 0
        for pred in self.learning_data.get("predictions", []):
            if pred["status"] != "Pending ⏳":
                total_evaluated += 1
                if "نجاح" in pred["status"]:
                    total_correct += 1
        if total_evaluated > 0:
            self.learning_data["success_rate"] = total_correct / total_evaluated

    def get_learning_stats(self):
        total = len(self.learning_data.get("predictions", []))
        pending = sum(1 for p in self.learning_data.get("predictions", []) if p["status"] == "Pending ⏳")
        success_rate = self.learning_data.get("success_rate", 0.0)
        history = []
        for p in reversed(self.learning_data.get("predictions", [])):
            history.append({
                "التاريخ": p["date"], "S": p["symbol"], "R": p["direction"],
                "سعر الدخول": f"{p['suggested_entry']:.2f}", "الهدف": f"{p['suggested_exit']:.2f}",
                "أعلى سعر": f"{p['actual_max_reached']:.2f}" if p['actual_max_reached'] else "⏳", "الحالة": p["status"]
            })
        return {"total_predictions": total, "pending_predictions": pending, "success_rate": success_rate,
                "history": history}