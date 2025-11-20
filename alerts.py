from utils import get_current_price

def get_price_alert(ticker, threshold):
    price = get_current_price(ticker)

    if price is None:
        return "⚠️ Could not fetch price."

    try:
        threshold = float(threshold)
    except:
        return "❌ Threshold must be a number."

    if price >= threshold:
        return f"🚨 ALERT: {ticker} has crossed ₹{threshold}! Current Price = ₹{price}"
    else:
        return f"📉 {ticker} is below threshold. Current Price = ₹{price}"
