"""
========================================================================
 TRX Predictor — نسخة خفيفة (Lite) — توقع + دقة آخر 100 توقع فقط
========================================================================
مصمم عمداً بأقل استهلاك موارد ممكن (مناسب لـ Streamlit Community Cloud
المجاني): موديل واحد بس، بدون شارت، بدون طلبات إنترنت إضافية (لا BTC/ETH،
لا أخبار، لا Tronscan، لا دفتر طلبات). فقط: بيانات TRX + توقع + دقة تاريخية.
========================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_ml_predictor as core

import streamlit as st
from sklearn.metrics import accuracy_score

SYMBOL = "TRX/USDT"

st.set_page_config(page_title="TRX Predictor Lite", layout="centered", page_icon="🔺")

# ------------------------------------------------------------------------
# تنسيق نظيف بخط بيانات/أرقام واضح (Google Fonts: JetBrains Mono + Tajawal)
# ------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Tajawal:wght@400;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; }
.num { font-family: 'JetBrains Mono', monospace; }

.big-signal {
    font-family: 'JetBrains Mono', monospace;
    font-size: 42px; font-weight: 800; text-align: center;
    padding: 18px; border-radius: 14px; letter-spacing: 1px;
    margin-bottom: 6px;
}
.buy   { background: #0d3320; color: #4ade80; border: 2px solid #4ade80; }
.sell  { background: #3a0d12; color: #f87171; border: 2px solid #f87171; }

.conf-box {
    text-align:center; font-family:'JetBrains Mono', monospace;
    font-size: 20px; color:#ccc; margin-bottom: 20px;
}

.grid-symbols {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px; line-height: 1.9; letter-spacing: 6px;
    text-align: center; background:#111; padding: 16px;
    border-radius: 10px; border: 1px solid #333;
}

.stat-num {
    font-family: 'JetBrains Mono', monospace; font-weight: 800;
}

hr { border-color: #333; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# الشريط الجانبي — إعدادات مبسّطة وخفيفة
# ------------------------------------------------------------------------
st.sidebar.title("🔺 TRX Predictor — Lite")
st.sidebar.caption("نسخة خفيفة: توقع + دقة آخر 100 توقع فقط")

timeframe = st.sidebar.selectbox(
    "الفريم الزمني", ["1h", "2h", "4h", "6h", "8h", "12h", "1d"], index=4,
)
years_back = st.sidebar.slider("عدد سنين البيانات", 0.5, 2.0, 1.0, step=0.5,
                                help="أقل عدد سنين = أسرع تحميل وتحليل. غير محتاج 5 سنين لموديل خفيف.")
horizon = st.sidebar.number_input("التنبؤ بعد كام شمعة", min_value=1, max_value=20, value=1)
min_move_pct = st.sidebar.slider("الحد الأدنى للحركة (Dead Zone) %", 0.0, 1.0, 0.2, step=0.05) / 100.0

run_btn = st.sidebar.button("🔄 تحديث وتشغيل", type="primary", use_container_width=True)

st.markdown("<h2 style='text-align:center;'>🔺 TRX / USDT</h2>", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# التشغيل — موديل واحد فقط (LightGBM) لأقل استهلاك موارد وأسرع نتيجة
# ------------------------------------------------------------------------
if run_btn:
    with st.spinner("جاري تحميل بيانات TRX من Binance..."):
        raw = core.fetch_binance_ohlcv(SYMBOL, timeframe, years_back)

    if len(raw) < 200:
        st.error("بيانات قليلة جداً. جرّب فريم زمني أكبر أو مدة أطول.")
        st.stop()

    with st.spinner("جاري بناء المؤشرات..."):
        featured = core.build_features(raw)
        targeted = core.build_target(featured, horizon, "direction", min_move_pct)

    feature_cols = core.get_feature_columns(targeted)
    train_df, test_df = core.time_split(targeted, 0.85, None, None)

    if len(train_df) < 100 or len(test_df) < 30:
        st.error("بيانات غير كافية بعد التصفية. صغّر الـ Dead Zone أو كبّر عدد السنين.")
        st.stop()

    with st.spinner("جاري تدريب الموديل (LightGBM)..."):
        trained = core.train_models(train_df, feature_cols, "direction", ["lightgbm"])

    if not trained:
        st.error("فشل تدريب الموديل.")
        st.stop()

    model = trained["lightgbm"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"].values
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    correct = (preds == y_test).astype(int)

    st.session_state["model"] = model
    st.session_state["feature_cols"] = feature_cols
    st.session_state["featured"] = featured
    st.session_state["preds"] = preds
    st.session_state["correct"] = correct
    st.session_state["probs"] = probs
    st.session_state["last_ts"] = featured["timestamp"].iloc[-1]

# ------------------------------------------------------------------------
# العرض — نص وأرقام ورموز فقط، بدون أي شارت
# ------------------------------------------------------------------------
if "model" in st.session_state:
    model = st.session_state["model"]
    feature_cols = st.session_state["feature_cols"]
    featured = st.session_state["featured"]
    preds = st.session_state["preds"]
    correct = st.session_state["correct"]

    # --- التوقع الحالي ---
    last_row = featured.tail(1)[feature_cols]
    p_up = float(model.predict_proba(last_row)[:, 1][0])
    is_buy = p_up >= 0.5
    confidence = p_up * 100 if is_buy else (1 - p_up) * 100

    signal_class = "buy" if is_buy else "sell"
    signal_text = "شراء ▲ BUY" if is_buy else "بيع ▼ SELL"
    st.markdown(f"<div class='big-signal {signal_class}'>{signal_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='conf-box'>نسبة الثقة: <b class='stat-num'>{confidence:.1f}%</b></div>",
                unsafe_allow_html=True)
    st.caption(f"⏱️ آخر بيانات حتى: {st.session_state['last_ts']}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- دقة آخر 100 توقع ---
    last_n = min(100, len(correct))
    last_correct = correct[-last_n:]
    n_right = int(last_correct.sum())
    acc_pct = (n_right / last_n) * 100

    st.markdown(f"<h4 style='text-align:center;'>📊 دقة آخر {last_n} توقع</h4>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div style='text-align:center;'><span class='stat-num' style='font-size:28px;color:#4ade80;'>{n_right}</span><br>صحيح ✅</div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='text-align:center;'><span class='stat-num' style='font-size:28px;color:#f87171;'>{last_n - n_right}</span><br>خطأ ❌</div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='text-align:center;'><span class='stat-num' style='font-size:28px;'>{acc_pct:.1f}%</span><br>الدقة</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- شبكة رموز: ✅ لكل توقع صحيح، ❌ لكل توقع خطأ (بالترتيب الزمني) ---
    symbols = "".join("✅" if c == 1 else "❌" for c in last_correct)
    rows = [symbols[i:i + 10] for i in range(0, len(symbols), 10)]
    grid_html = "<br>".join(rows)
    st.markdown(f"<div class='grid-symbols'>{grid_html}</div>", unsafe_allow_html=True)
    st.caption("كل رمز = توقع شمعة واحدة، بالترتيب من الأقدم (أعلى) للأحدث (تحت)، 10 في كل صف.")

    if acc_pct < 55:
        st.warning(f"⚠️ الدقة ({acc_pct:.1f}%) قريبة من العشوائية (50%) — الموديل بالإعدادات الحالية مش عنده ميزة واضحة.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("⚠️ توقعات إحصائية تعليمية وليست نصيحة استثمارية.")

else:
    st.info("اضبط الإعدادات من الشريط الجانبي، وبعدين دوس «تحديث وتشغيل».")
