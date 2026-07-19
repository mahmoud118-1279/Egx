import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config import LEARNING_DATA_FILE


class SelfLearningAIAnalyst:
    """
    نظام التعليم العميق والتطور الذاتي الحركي لتحليل الأخطاء وإعادة توزين الاستراتيجيات كمياً
    
    المميزات:
    - تسجيل جميع الصفقات مع بصمة المؤشرات الفنية
    - تقييم تلقائي للصفقات ومقارنتها بالنتائج الفعلية
    - تطور ذاتي للأوزان بناءً على أنماط النجاح والفشل
    - تحليل إحصائي لأداء النموذج
    """

    def __init__(self):
        self.learning_file = LEARNING_DATA_FILE
        self.learning_data = self._load_learning_data()

    def _load_learning_data(self):
        """تحميل سجل التوقعات المطور مع تهيئة مصفوفات الأوزان المتطورة"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # التأكد من وجود جميع المفاتيح
                    if "failure_patterns" not in data: 
                        data["failure_patterns"] = []
                    if "success_patterns" not in data: 
                        data["success_patterns"] = []
                    if "dynamic_weights" not in data:
                        data["dynamic_weights"] = {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3}
                    if "performance_history" not in data:
                        data["performance_history"] = []
                    if "total_trades" not in data:
                        data["total_trades"] = 0
                    if "winning_trades" not in data:
                        data["winning_trades"] = 0
                    if "losing_trades" not in data:
                        data["losing_trades"] = 0
                    if "average_profit" not in data:
                        data["average_profit"] = 0.0
                    if "average_loss" not in data:
                        data["average_loss"] = 0.0
                    return data
            except Exception as e:
                print(f"⚠️ خطأ في تحميل البيانات: {e}")
                return self._get_default_data()
        return self._get_default_data()

    def _get_default_data(self):
        """إرجاع البيانات الافتراضية"""
        return {
            "predictions": [],
            "success_rate": 0.0,
            "failure_patterns": [],
            "success_patterns": [],
            "dynamic_weights": {"expected_gain": 0.4, "cmf": 0.3, "news": 0.3},
            "performance_history": [],
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "average_profit": 0.0,
            "average_loss": 0.0
        }

    def _save_learning_data(self):
        """حفظ السجلات والتطورات الناتجة عن التفكير الذاتي للآلة"""
        try:
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"❌ خطأ أثناء حفظ سجل التعلم المتطور: {e}")

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
            سعر الدخول المقترح
        suggested_exit : float
            سعر الخروج المقترح
        direction : str
            اتجاه التوصية
        current_indicators : dict, optional
            المؤشرات الفنية الحالية
        """
        new_pred = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symbol": symbol,
            "current_price": float(current_price),
            "predicted_close": float(predicted_close),
            "suggested_entry": float(suggested_entry),
            "suggested_exit": float(suggested_exit),
            "direction": direction,
            "actual_max_reached": float(current_price),
            "status": "Pending ⏳",
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "exit_date": None,
            "profit_pct": 0.0,
            "holding_days": 0
        }

        if current_indicators is not None:
            new_pred["indicators_snapshot"] = {
                "RSI": float(current_indicators.get("RSI", 50.0)),
                "CMF": float(current_indicators.get("CMF", 0.0)),
                "News_Sentiment": float(current_indicators.get("News_Sentiment", 0.0)),
                "ATR": float(current_indicators.get("ATR", 0.0)),
                "Volume": float(current_indicators.get("Volume", 0.0)),
                "MACD": float(current_indicators.get("MACD", 0.0)),
                "ADX": float(current_indicators.get("ADX", 0.0))
            }

        self.learning_data["predictions"].append(new_pred)
        self.learning_data["total_trades"] += 1
        self._save_learning_data()
        print(f"✅ تم تسجيل توصية جديدة لـ {symbol}")
        return True

    def evaluate_pending_predictions(self, fetch_stock_data_func):
        """
        تحديث الصفقات وتشغيل محرك التفكير المقارن
        
        Parameters:
        -----------
        fetch_stock_data_func : callable
            دالة لجلب بيانات السهم
        
        Returns:
        --------
        bool: تم التحديث أم لا
        """
        updated = False
        evaluated_count = 0

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
                entry_price = pred["suggested_entry"]
                exit_target = pred["suggested_exit"]
                current_price = pred["current_price"]

                # تحديث أعلى سعر
                if recent_high > pred["actual_max_reached"]:
                    pred["actual_max_reached"] = recent_high
                    updated = True

                # حساب عدد أيام الاحتفاظ
                entry_date = datetime.strptime(pred["entry_date"], "%Y-%m-%d")
                days_held = (datetime.now() - entry_date).days
                pred["holding_days"] = days_held

                # شروط النجاح
                if pred["actual_max_reached"] >= exit_target:
                    profit_pct = ((exit_target - entry_price) / entry_price) * 100
                    pred["status"] = f"نجاح 🎯 (ربح {profit_pct:.1f}%)"
                    pred["profit_pct"] = profit_pct
                    pred["exit_date"] = datetime.now().strftime("%Y-%m-%d")
                    
                    self.learning_data["winning_trades"] += 1
                    self.learning_data["average_profit"] = (
                        (self.learning_data["average_profit"] * (self.learning_data["winning_trades"] - 1) + profit_pct) 
                        / self.learning_data["winning_trades"]
                    )
                    
                    if "indicators_snapshot" in pred:
                        self.learning_data["success_patterns"].append({
                            "symbol": symbol, 
                            "indicators": pred["indicators_snapshot"],
                            "profit": profit_pct,
                            "date": pred["date"]
                        })
                    updated = True
                    evaluated_count += 1

                # شروط الفشل - ضرب وقف الخسارة
                elif recent_close < entry_price * 0.95:
                    loss_pct = ((recent_close - entry_price) / entry_price) * 100
                    pred["status"] = f"فشل ❌ (خسارة {loss_pct:.1f}%)"
                    pred["profit_pct"] = loss_pct
                    pred["exit_date"] = datetime.now().strftime("%Y-%m-%d")
                    
                    self.learning_data["losing_trades"] += 1
                    self.learning_data["average_loss"] = (
                        (self.learning_data["average_loss"] * (self.learning_data["losing_trades"] - 1) + abs(loss_pct)) 
                        / self.learning_data["losing_trades"]
                    )
                    
                    if "indicators_snapshot" in pred:
                        self.learning_data["failure_patterns"].append({
                            "symbol": symbol, 
                            "indicators": pred["indicators_snapshot"],
                            "loss": loss_pct,
                            "date": pred["date"]
                        })
                    updated = True
                    evaluated_count += 1

                # إذا مر وقت طويل ولم يتم تحقيق الهدف ولا وقف الخسارة
                elif days_held > 30:
                    profit_pct = ((recent_close - entry_price) / entry_price) * 100
                    if profit_pct > 0:
                        pred["status"] = f"انتهاء المدة ⏰ (ربح {profit_pct:.1f}%)"
                    else:
                        pred["status"] = f"انتهاء المدة ⏰ (خسارة {profit_pct:.1f}%)"
                    pred["profit_pct"] = profit_pct
                    pred["exit_date"] = datetime.now().strftime("%Y-%m-%d")
                    updated = True
                    evaluated_count += 1

            except Exception as e:
                print(f"⚠️ خطأ في مراجعة السهم {symbol}: {e}")

        if evaluated_count > 0:
            print(f"✅ تم تقييم {evaluated_count} صفقة جديدة")
            self._recalculate_success_rate()
            self._update_performance_history()
            self._evolve_voting_weights()
            self._save_learning_data()

        return updated

    def _update_performance_history(self):
        """تحديث سجل الأداء الشهري"""
        now = datetime.now()
        month_key = now.strftime("%Y-%m")
        
        # البحث عن الشهر الحالي في السجل
        found = False
        for entry in self.learning_data.get("performance_history", []):
            if entry["month"] == month_key:
                entry["trades"] = self.learning_data["total_trades"]
                entry["wins"] = self.learning_data["winning_trades"]
                entry["losses"] = self.learning_data["losing_trades"]
                entry["win_rate"] = self.learning_data["success_rate"]
                found = True
                break
        
        if not found:
            self.learning_data["performance_history"].append({
                "month": month_key,
                "trades": self.learning_data["total_trades"],
                "wins": self.learning_data["winning_trades"],
                "losses": self.learning_data["losing_trades"],
                "win_rate": self.learning_data["success_rate"]
            })

    def _evolve_voting_weights(self):
        """
        محرك التطور الذاتي المتقدم
        يحلل الأنماط الناجحة والفاشلة ويعدل الأوزان ديناميكياً
        """
        success_list = self.learning_data.get("success_patterns", [])
        failure_list = self.learning_data.get("failure_patterns", [])

        if len(success_list) < 3 or len(failure_list) < 2:
            return

        # تحليل أنماط النجاح
        success_indicators = {
            "news": [p["indicators"].get("News_Sentiment", 0.0) for p in success_list],
            "cmf": [p["indicators"].get("CMF", 0.0) for p in success_list],
            "rsi": [p["indicators"].get("RSI", 50.0) for p in success_list]
        }

        failure_indicators = {
            "news": [p["indicators"].get("News_Sentiment", 0.0) for p in failure_list],
            "cmf": [p["indicators"].get("CMF", 0.0) for p in failure_list],
            "rsi": [p["indicators"].get("RSI", 50.0) for p in failure_list]
        }

        # حساب المتوسطات
        avg_success_news = np.mean(success_indicators["news"])
        avg_failed_news = np.mean(failure_indicators["news"])
        avg_success_cmf = np.mean(success_indicators["cmf"])
        avg_failed_cmf = np.mean(failure_indicators["cmf"])

        current_weights = self.learning_data["dynamic_weights"]

        # تعديل وزن الأخبار
        news_diff = avg_success_news - avg_failed_news
        if news_diff > 0.2:
            # الأخبار مؤثر جداً
            current_weights["news"] = min(current_weights["news"] + 0.05, 0.5)
            current_weights["cmf"] = max(current_weights["cmf"] - 0.025, 0.15)
            current_weights["expected_gain"] = max(current_weights["expected_gain"] - 0.025, 0.2)
        elif news_diff < -0.1:
            # الأخبار غير موثوقة
            current_weights["news"] = max(current_weights["news"] - 0.03, 0.1)
            current_weights["cmf"] = min(current_weights["cmf"] + 0.015, 0.4)

        # تعديل وزن CMF
        cmf_diff = avg_success_cmf - avg_failed_cmf
        if cmf_diff > 0.1:
            current_weights["cmf"] = min(current_weights["cmf"] + 0.03, 0.4)
            current_weights["news"] = max(current_weights["news"] - 0.015, 0.15)

        # التأكد من أن مجموع الأوزان = 1
        total = sum(current_weights.values())
        if total > 0:
            for key in current_weights:
                current_weights[key] = current_weights[key] / total

        self.learning_data["dynamic_weights"] = current_weights
        print(f"🧬 تم تحديث الأوزان: {current_weights}")

    def _recalculate_success_rate(self):
        """إعادة حساب نسبة النجاح الكلية"""
        total_evaluated = 0
        total_correct = 0
        
        for pred in self.learning_data.get("predictions", []):
            if pred["status"] != "Pending ⏳":
                total_evaluated += 1
                if "نجاح" in pred["status"] or "ربح" in pred["status"]:
                    total_correct += 1
        
        if total_evaluated > 0:
            self.learning_data["success_rate"] = total_correct / total_evaluated
        
        # تحديث إحصائيات إضافية
        self.learning_data["total_evaluated"] = total_evaluated

    def get_learning_stats(self):
        """
        الحصول على إحصائيات التعلم
        
        Returns:
        --------
        dict: إحصائيات التعلم
        """
        total = len(self.learning_data.get("predictions", []))
        pending = sum(1 for p in self.learning_data.get("predictions", []) if p["status"] == "Pending ⏳")
        success_rate = self.learning_data.get("success_rate", 0.0)
        winning_trades = self.learning_data.get("winning_trades", 0)
        losing_trades = self.learning_data.get("losing_trades", 0)
        avg_profit = self.learning_data.get("average_profit", 0.0)
        avg_loss = self.learning_data.get("average_loss", 0.0)
        
        # حساب الـ Profit Factor
        profit_factor = (winning_trades * avg_profit) / (losing_trades * avg_loss + 1e-10) if losing_trades > 0 else float('inf')
        
        history = []
        for p in reversed(self.learning_data.get("predictions", [])):
            history.append({
                "التاريخ": p["date"],
                "S": p["symbol"],
                "R": p["direction"][:30] + "..." if len(p["direction"]) > 30 else p["direction"],
                "سعر الدخول": f"{p['suggested_entry']:.2f}",
                "الهدف": f"{p['suggested_exit']:.2f}",
                "أعلى سعر": f"{p['actual_max_reached']:.2f}" if p['actual_max_reached'] else "⏳",
                "الربح %": f"{p.get('profit_pct', 0):.1f}%" if p.get('profit_pct', 0) != 0 else "⏳",
                "الحالة": p["status"]
            })

        return {
            "total_predictions": total,
            "pending_predictions": pending,
            "success_rate": success_rate,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "history": history,
            "dynamic_weights": self.learning_data.get("dynamic_weights", {}),
            "performance_history": self.learning_data.get("performance_history", [])
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
            key = f"RSI_{int(ind.get('RSI', 50)/10)*10}_CMF_{'pos' if ind.get('CMF', 0) > 0 else 'neg'}_News_{'pos' if ind.get('News_Sentiment', 0) > 0 else 'neg'}"
            
            if key not in patterns:
                patterns[key] = {"count": 0, "profits": [], "indicators": ind}
            patterns[key]["count"] += 1
            patterns[key]["profits"].append(p.get("profit", 0))
        
        # ترتيب الأنماط حسب التكرار
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True)
        
        result = []
        for key, data in sorted_patterns[:top_n]:
            avg_profit = np.mean(data["profits"]) if data["profits"] else 0
            result.append({
                "pattern": key,
                "count": data["count"],
                "avg_profit": avg_profit,
                "indicators": data["indicators"]
            })
        
        return result

    def get_learning_summary(self):
        """
        الحصول على ملخص التعلم
        
        Returns:
        --------
        str: ملخص التعلم
        """
        stats = self.get_learning_stats()
        
        summary = f"""
📊 **ملخص أداء نظام التعلم الذاتي**
{'='*40}

📈 **الإحصائيات العامة:**
- إجمالي الصفقات: {stats['total_predictions']}
- الصفقات المعلقة: {stats['pending_predictions']}
- نسبة النجاح: {stats['success_rate']:.1%}

💰 **الربح والخسارة:**
- الصفقات الرابحة: {stats['winning_trades']}
- الصفقات الخاسرة: {stats['losing_trades']}
- متوسط الربح: {stats['average_profit']:.2f}%
- متوسط الخسارة: {stats['average_loss']:.2f}%
- Profit Factor: {stats['profit_factor']:.2f}

🧬 **الأوزان الحالية:**
- الربح المتوقع: {stats['dynamic_weights'].get('expected_gain', 0.4):.2f}
- CMF: {stats['dynamic_weights'].get('cmf', 0.3):.2f}
- الأخبار: {stats['dynamic_weights'].get('news', 0.3):.2f}
"""
        return summary
