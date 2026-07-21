import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

from config import LEARNING_DATA_FILE


class SelfLearningAIAnalyst:
    """نظام التعليم العميق والتطور الذاتي الحركي لتحليل الأخطاء وإعادة توزين الاستراتيجيات كمياً"""

    def __init__(self):
        self.learning_file = LEARNING_DATA_FILE
        self.learning_data = self._load_learning_data()

    def _get_default_data(self):
        """الحصول على البيانات الافتراضية"""
        return {
            "predictions": [],
            "success_rate": 0.0,
            "failure_patterns": [],
            "success_patterns": [],
            "dynamic_weights": {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3},
            "profit_factor": 0.0
        }

    def _load_learning_data(self):
        """تحميل سجل التوقعات المطور مع تهيئة مصفوفات الأوزان المتطورة"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # التأكد من وجود جميع المفاتيح المطلوبة
                    if "failure_patterns" not in data:
                        data["failure_patterns"] = []
                    if "success_patterns" not in data:
                        data["success_patterns"] = []
                    if "dynamic_weights" not in data:
                        data["dynamic_weights"] = {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3}
                    if "profit_factor" not in data:
                        data["profit_factor"] = 0.0
                    return data
            except Exception as e:
                print(f"⚠️ خطأ في تحميل ملف التعلم: {e}")
                return self._get_default_data()
        return self._get_default_data()

    def _save_learning_data(self):
        """حفظ السجلات والتطورات الناتجة عن التفكير الذاتي للآلة"""
        try:
            # ✅ التأكد من وجود المجلد
            os.makedirs(os.path.dirname(self.learning_file), exist_ok=True)

            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=4)

            total_predictions = len(self.learning_data.get('predictions', []))
            print(f"✅ تم حفظ {total_predictions} تحليل في {self.learning_file}")
            return True
        except Exception as e:
            print(f"❌ خطأ أثناء حفظ سجل التعلم: {e}")
            return False

    def record_prediction(self, symbol, current_price, predicted_close, suggested_entry, suggested_exit, direction,
                          current_indicators=None):
        """
        تسجيل البصمة الفنية المؤشرية الكاملة لحظة اتخاذ القرار

        Parameters:
        -----------
        symbol : str
            رمز السهم
        current_price : float
            السعر الحالي
        predicted_close : float
            السعر المتوقع
        suggested_entry : float
            نقطة الدخول المقترحة
        suggested_exit : float
            نقطة الخروج المقترحة
        direction : str
            اتجاه التوصية
        current_indicators : dict or pd.Series
            المؤشرات الفنية الحالية
        """
        try:
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
                # ✅ التحويل إلى dict إذا كان Series
                if hasattr(current_indicators, 'to_dict'):
                    current_indicators = current_indicators.to_dict()

                # ✅ استخراج القيم بشكل آمن
                new_pred["indicators_snapshot"] = {
                    "RSI": float(current_indicators.get("RSI", 50.0)),
                    "CMF": float(current_indicators.get("CMF", 0.0)),
                    "News_Sentiment": float(current_indicators.get("News_Sentiment", 0.0))
                }

            self.learning_data["predictions"].append(new_pred)

            # ✅ حفظ فوري
            success = self._save_learning_data()

            if success:
                print(f"✅ تم تسجيل تحليل {symbol} (إجمالي: {len(self.learning_data['predictions'])})")
            return success

        except Exception as e:
            print(f"❌ خطأ في تسجيل التحليل: {e}")
            return False

    def evaluate_pending_predictions(self, fetch_stock_data_func):
        """
        تحديث الصفقات وتشغيل محرك التفكير المقارن

        Parameters:
        -----------
        fetch_stock_data_func : function
            دالة لجلب بيانات السهم
        """
        updated = False
        total_checked = 0
        total_success = 0
        total_fail = 0

        for pred in self.learning_data.get("predictions", []):
            if pred["status"] != "Pending ⏳":
                continue

            symbol = pred["symbol"]
            try:
                df, _ = fetch_stock_data_func(symbol)
                if df.empty:
                    continue

                recent_high = float(df['High'].max())
                recent_close = float(df['Close'].iloc[-1])

                if recent_high > pred["actual_max_reached"]:
                    pred["actual_max_reached"] = recent_high
                    updated = True

                # ✅ حساب الربح الفعلي
                entry_price = pred["suggested_entry"]
                target_price = pred["suggested_exit"]

                # إذا تجاوز السعر الهدف
                if pred["actual_max_reached"] >= target_price:
                    pred["status"] = "نجاح باهر 🎯 (حقق الهدف كاملاً)"
                    profit_pct = ((target_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                    if "indicators_snapshot" in pred:
                        self.learning_data["success_patterns"].append({
                            "symbol": symbol,
                            "indicators": pred["indicators_snapshot"],
                            "profit": profit_pct,
                            "entry_price": entry_price,
                            "target_price": target_price,
                            "max_reached": pred["actual_max_reached"]
                        })
                    total_success += 1
                    updated = True

                # إذا كسر وقف الخسارة (5% تحت سعر الدخول)
                elif recent_close < entry_price * 0.95:
                    pred["status"] = "فشل ❌ (ضرب وقف الخسارة)"
                    if "indicators_snapshot" in pred:
                        self.learning_data["failure_patterns"].append({
                            "symbol": symbol,
                            "indicators": pred["indicators_snapshot"],
                            "entry_price": entry_price,
                            "stop_loss": entry_price * 0.95,
                            "recent_close": recent_close
                        })
                    total_fail += 1
                    updated = True

                total_checked += 1

            except Exception as e:
                print(f"خطأ في مراجعة السهم {symbol}: {e}")

        if updated:
            self._evolve_voting_weights()
            self._recalculate_success_rate()
            self._calculate_profit_factor()
            self._save_learning_data()
            print(f"✅ تم تحديث التقييم: {total_success} نجاح, {total_fail} فشل")

        return updated

    def _evolve_voting_weights(self):
        """محرك التطور الذاتي: يحلل أي المعايير كانت سبباً في النجاح لرفع وزنها"""
        success_list = self.learning_data.get("success_patterns", [])
        failure_list = self.learning_data.get("failure_patterns", [])

        if len(success_list) < 3:
            return

        current_weights = self.learning_data["dynamic_weights"]

        # حساب متوسط المؤشرات في حالات النجاح والفشل
        avg_success_news = sum(p["indicators"].get("News_Sentiment", 0.0) for p in success_list) / len(success_list)
        avg_failed_news = sum(p["indicators"].get("News_Sentiment", 0.0) for p in failure_list) / max(len(failure_list), 1)

        avg_success_cmf = sum(p["indicators"].get("CMF", 0.0) for p in success_list) / len(success_list)
        avg_failed_cmf = sum(p["indicators"].get("CMF", 0.0) for p in failure_list) / max(len(failure_list), 1)

        # ✅ تعديل الأوزان بناءً على الأداء
        if avg_success_news > 0.1 and avg_failed_news < 0.0:
            current_weights["news"] = min(current_weights["news"] + 0.05, 0.5)
            current_weights["cmf"] = max(current_weights["cmf"] - 0.025, 0.15)
            current_weights["expected_gain"] = max(current_weights["expected_gain"] - 0.025, 0.2)

        if avg_success_cmf > 0.1 and avg_failed_cmf < -0.05:
            current_weights["cmf"] = min(current_weights["cmf"] + 0.05, 0.5)
            current_weights["news"] = max(current_weights["news"] - 0.025, 0.15)

        self.learning_data["dynamic_weights"] = current_weights

    def _recalculate_success_rate(self):
        """إعادة حساب نسبة النجاح"""
        total_evaluated = 0
        total_correct = 0

        for pred in self.learning_data.get("predictions", []):
            if pred["status"] != "Pending ⏳":
                total_evaluated += 1
                if "نجاح" in pred["status"]:
                    total_correct += 1

        if total_evaluated > 0:
            self.learning_data["success_rate"] = total_correct / total_evaluated
        else:
            self.learning_data["success_rate"] = 0.0

    def _calculate_profit_factor(self):
        """حساب Profit Factor (إجمالي الأرباح / إجمالي الخسائر)"""
        total_profit = 0.0
        total_loss = 0.0

        for pred in self.learning_data.get("predictions", []):
            if pred["status"] == "Pending ⏳":
                continue

            entry = pred["suggested_entry"]
            target = pred["suggested_exit"]

            if "نجاح" in pred["status"]:
                profit = ((target - entry) / entry) * 100 if entry > 0 else 0
                if profit > 0:
                    total_profit += profit
            elif "فشل" in pred["status"]:
                loss = ((entry * 0.95 - entry) / entry) * 100 if entry > 0 else 0
                total_loss += abs(loss)

        if total_loss > 0:
            self.learning_data["profit_factor"] = total_profit / total_loss
        else:
            self.learning_data["profit_factor"] = total_profit

    def get_learning_stats(self):
        """الحصول على إحصائيات التعلم"""
        total = len(self.learning_data.get("predictions", []))
        pending = sum(1 for p in self.learning_data.get("predictions", []) if p["status"] == "Pending ⏳")
        success_rate = self.learning_data.get("success_rate", 0.0)
        profit_factor = self.learning_data.get("profit_factor", 0.0)

        history = []
        for p in reversed(self.learning_data.get("predictions", [])):
            history.append({
                "التاريخ": p["date"],
                "S": p["symbol"],
                "R": p["direction"][:25] + "..." if len(p["direction"]) > 25 else p["direction"],
                "سعر الدخول": f"{p['suggested_entry']:.2f}",
                "الهدف": f"{p['suggested_exit']:.2f}",
                "أعلى سعر": f"{p['actual_max_reached']:.2f}" if p['actual_max_reached'] else "⏳",
                "الحالة": p["status"]
            })

        return {
            "total_predictions": total,
            "pending_predictions": pending,
            "success_rate": success_rate,
            "profit_factor": profit_factor,
            "history": history
        }

    def get_best_patterns(self, top_n=5):
        """
        الحصول على أفضل أنماط النجاح

        Parameters:
        -----------
        top_n : int
            عدد الأنماط المطلوبة

        Returns:
        --------
        list: أفضل الأنماط
        """
        success_patterns = self.learning_data.get("success_patterns", [])
        if not success_patterns:
            return []

        # تجميع الأنماط حسب المؤشرات
        patterns = {}
        for p in success_patterns:
            ind = p.get("indicators", {})
            rsi = ind.get('RSI', 50)
            cmf = ind.get('CMF', 0)
            news = ind.get('News_Sentiment', 0)

            # تصنيف المؤشرات إلى فئات
            rsi_bucket = int(rsi / 10) * 10
            cmf_bucket = 'إيجابي' if cmf > 0 else 'سلبي'
            news_bucket = 'إيجابي' if news > 0 else 'سلبي'

            key = f"RSI_{rsi_bucket}_CMF_{cmf_bucket}_News_{news_bucket}"

            if key not in patterns:
                patterns[key] = {
                    "count": 0,
                    "profits": [],
                    "rsi_range": f"{rsi_bucket}-{rsi_bucket + 10}",
                    "cmf_type": cmf_bucket,
                    "news_type": news_bucket,
                    "avg_rsi": 0,
                    "avg_cmf": 0,
                    "avg_news": 0,
                    "total_profit": 0,
                    "wins": 0
                }

            patterns[key]["count"] += 1
            profit = p.get("profit", 0)
            patterns[key]["profits"].append(profit)
            patterns[key]["total_profit"] += profit
            if profit > 0:
                patterns[key]["wins"] += 1

            # تحديث المتوسطات
            count = patterns[key]["count"]
            patterns[key]["avg_rsi"] = (patterns[key]["avg_rsi"] * (count - 1) + rsi) / count
            patterns[key]["avg_cmf"] = (patterns[key]["avg_cmf"] * (count - 1) + cmf) / count
            patterns[key]["avg_news"] = (patterns[key]["avg_news"] * (count - 1) + news) / count

        # ترتيب الأنماط حسب التكرار
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True)

        result = []
        for key, data in sorted_patterns[:top_n]:
            avg_profit = data["total_profit"] / data["count"] if data["count"] > 0 else 0
            win_rate = data["wins"] / data["count"] if data["count"] > 0 else 0

            result.append({
                "pattern": key,
                "count": data["count"],
                "avg_profit": round(avg_profit, 2),
                "win_rate": round(win_rate * 100, 1),
                "rsi_range": data["rsi_range"],
                "cmf_type": data["cmf_type"],
                "news_type": data["news_type"],
                "avg_rsi": round(data["avg_rsi"], 1),
                "avg_cmf": round(data["avg_cmf"], 3),
                "avg_news": round(data["avg_news"], 2)
            })

        return result

    def get_dynamic_weights(self):
        """الحصول على الأوزان الديناميكية الحالية"""
        return self.learning_data.get("dynamic_weights", {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3})

    def clear_old_predictions(self, days=90):
        """
        حذف التوقعات القديمة

        Parameters:
        -----------
        days : int
            عدد الأيام للاحتفاظ بالبيانات
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")

            predictions = self.learning_data.get("predictions", [])
            new_predictions = []

            for p in predictions:
                try:
                    pred_date = p.get("date", "").split(" ")[0]  # استخراج التاريخ فقط
                    if pred_date >= cutoff_str:
                        new_predictions.append(p)
                except:
                    # في حالة وجود خطأ في التاريخ، نحتفظ بالعنصر
                    new_predictions.append(p)

            self.learning_data["predictions"] = new_predictions
            self._save_learning_data()
            print(f"✅ تم تنظيف البيانات: الاحتفاظ بـ {len(new_predictions)} تحليل")

        except Exception as e:
            print(f"⚠️ خطأ في تنظيف البيانات: {e}")

    def export_learning_data(self, file_path=None):
        """
        تصدير بيانات التعلم إلى ملف

        Parameters:
        -----------
        file_path : str
            مسار الملف للتصدير (اختياري)
        """
        if file_path is None:
            file_path = f"learning_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=4)
            print(f"✅ تم تصدير بيانات التعلم إلى {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ خطأ في تصدير البيانات: {e}")
            return None
