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

    def _load_learning_data(self):
        """تحميل سجل التوقعات المطور مع تهيئة مصفوفات الأوزان المتطورة"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "failure_patterns" not in data: 
                        data["failure_patterns"] = []
                    if "success_patterns" not in data: 
                        data["success_patterns"] = []
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
        """تسجيل البصمة الفنية المؤشرية الكاملة لحظة اتخاذ القرار"""
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
        """تحديث الصفقات وتشغيل محرك التفكير المقارن"""
        updated = False

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

                if pred["actual_max_reached"] >= pred["suggested_exit"]:
                    pred["status"] = "نجاح باهر 🎯 (حقق الهدف كاملاً)"
                    if "indicators_snapshot" in pred:
                        self.learning_data["success_patterns"].append({
                            "symbol": symbol, 
                            "indicators": pred["indicators_snapshot"],
                            "profit": ((pred["suggested_exit"] - pred["suggested_entry"]) / pred["suggested_entry"]) * 100
                        })
                    updated = True
                elif recent_close < pred["suggested_entry"] * 0.95:
                    pred["status"] = "فشل ❌ (ضرب وقف الخسارة)"
                    if "indicators_snapshot" in pred:
                        self.learning_data["failure_patterns"].append({
                            "symbol": symbol, 
                            "indicators": pred["indicators_snapshot"]
                        })
                    updated = True

            except Exception as e:
                print(f"خطأ في مراجعة السهم {symbol}: {e}")

        if updated:
            self._evolve_voting_weights()
            self._recalculate_success_rate()
            self._save_learning_data()

        return updated

    def _evolve_voting_weights(self):
        """محرك التطور الذاتي: يحلل أي المعايير كانت سبباً في النجاح لرفع وزنها"""
        success_list = self.learning_data.get("success_patterns", [])
        failure_list = self.learning_data.get("failure_patterns", [])

        if len(success_list) < 3:
            return

        current_weights = self.learning_data["dynamic_weights"]

        avg_success_news = sum(p["indicators"].get("News_Sentiment", 0.0) for p in success_list) / len(success_list)
        avg_failed_news = sum(p["indicators"].get("News_Sentiment", 0.0) for p in failure_list) / max(len(failure_list), 1)

        if avg_success_news > 0.1 and avg_failed_news < 0.0:
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
                "التاريخ": p["date"], 
                "S": p["symbol"], 
                "R": p["direction"],
                "سعر الدخول": f"{p['suggested_entry']:.2f}", 
                "الهدف": f"{p['suggested_exit']:.2f}",
                "أعلى سعر": f"{p['actual_max_reached']:.2f}" if p['actual_max_reached'] else "⏳", 
                "الحالة": p["status"]
            })
        return {
            "total_predictions": total, 
            "pending_predictions": pending, 
            "success_rate": success_rate,
            "history": history
        }

    # ============================================================
    # ✅ الدالة المضافة لحل مشكلة AttributeError
    # ============================================================
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
                    "rsi_range": f"{rsi_bucket}-{rsi_bucket+10}",
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
