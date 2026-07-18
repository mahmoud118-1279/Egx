import numpy as np


def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss, decision_score=70):
    """حساب حجم الصفقة الديناميكي المطور بناءً على المخاطرة المحددة ودرجة ثقة الآلة الكمية"""
    try:
        # 🧠 ترقية إدارة المخاطر: تطويع حجم المخاطرة بناءً على مجموع نقاط التصويت الفني والخبري
        if decision_score >= 85:
            adjusted_risk = risk_percentage * 1.2  # زيادة المحفظة المخصصة لقوة التوافق الفني والخبري
        elif decision_score < 60:
            adjusted_risk = risk_percentage * 0.5  # تقليص المخاطرة للنصف لعدم تناسق المؤشرات
        else:
            adjusted_risk = risk_percentage

        risk_amount = account_balance * (adjusted_risk / 100)
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return 0

        num_shares = risk_amount / risk_per_share
        return int(num_shares)
    except Exception:
        return 0


def generate_trading_strategy(df, final_prediction, direction):
    """توليد توصية كاملة تشمل سعر الدخول، وقف الخسارة، وأهداف الربح بناءً على تقلبات السهم (الكود القديم)"""
    last_row = df.iloc[-1]
    current_price = last_row['Close']

    atr = last_row['ATR'] if 'ATR' in df.columns and not np.isnan(last_row['ATR']) else (current_price * 0.02)

    strategy = {
        "action": "انتظار ⏳",
        "entry_price": current_price,
        "stop_loss": current_price,
        "take_profit_1": current_price,
        "take_profit_2": current_price,
        "risk_reward_ratio": "0.0"
    }

    if "شراء" in direction and final_prediction > current_price:
        strategy["action"] = "شراء 🟢"
        strategy["entry_price"] = current_price
        strategy["stop_loss"] = round(current_price - (1.5 * atr), 2)
        strategy["take_profit_1"] = round(current_price + (1.5 * atr), 2)
        strategy["take_profit_2"] = round(current_price + (3.0 * atr), 2)
        strategy["risk_reward_ratio"] = "1:2"

    elif "خروج" in direction or "تجنب" in direction:
        strategy["action"] = "تجنب/انتظار ❌"

    return strategy