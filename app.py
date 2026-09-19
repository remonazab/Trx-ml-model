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

st.sidebar.markdown("---")
compare_btn = st.sidebar.button("🔍 قارن كل قيم Dead Zone تلقائياً", use_container_width=True,
                                 help="يجرب 7 قيم مختلفة على نفس البيانات (أسرع من التجربة اليدوية)، ويوريك أقوى نتيجة.")
st.sidebar.caption("⚠️ الميزة دي بتدرب الموديل 7 مرات — استخدمها بحكمة عشان متستهلكش حصة المعالجة.")

st.markdown("<h2 style='text-align:center;'>🔺 TRX / USDT</h2>", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# مقارنة تلقائية لكل قيم Dead Zone على نفس البيانات (بيانات تتحمّل مرة واحدة بس)
# ------------------------------------------------------------------------
if compare_btn:
    try:
        with st.spinner("جاري تحميل بيانات TRX من Binance..."):
            raw_cmp = core.fetch_binance_ohlcv(SYMBOL, timeframe, years_back)
    except Exception as e:
        st.error(f"❌ فشل تحميل البيانات من Binance:\n\n{e}")
        st.stop()

    if len(raw_cmp) < 200:
        st.error("بيانات قليلة جداً. جرّب فريم زمني أكبر أو مدة أطول.")
        st.stop()

    with st.spinner("جاري بناء المؤشرات..."):
        featured_cmp = core.build_features(raw_cmp)
        featured_cmp = core.add_swing_points(featured_cmp, order=5)
        featured_cmp = core.add_liquidity_features(featured_cmp, window=20)

    candidate_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]  # بالنسبة المئوية
    results = []
    progress = st.progress(0, text="بدء المقارنة...")

    for i, dz_pct in enumerate(candidate_values):
        progress.progress((i) / len(candidate_values), text=f"جاري تجربة Dead Zone = {dz_pct}%...")
        dz = dz_pct / 100.0
        targeted_cmp = core.build_target(featured_cmp, horizon, "direction", dz)
        feature_cols_cmp = core.get_feature_columns(targeted_cmp)
        train_df_cmp, test_df_cmp = core.time_split(targeted_cmp, 0.85, None, None)

        if len(train_df_cmp) < 100 or len(test_df_cmp) < 30:
            results.append({"dz": dz_pct, "n_test": len(test_df_cmp), "acc": None, "n_used": 0})
            continue

        trained_cmp = core.train_models(train_df_cmp, feature_cols_cmp, "direction", ["lightgbm"])
        if not trained_cmp:
            results.append({"dz": dz_pct, "n_test": len(test_df_cmp), "acc": None, "n_used": 0})
            continue

        model_cmp = trained_cmp["lightgbm"]
        X_test_cmp = test_df_cmp[feature_cols_cmp]
        y_test_cmp = test_df_cmp["target"].values
        probs_cmp = model_cmp.predict_proba(X_test_cmp)[:, 1]
        preds_cmp = (probs_cmp >= 0.5).astype(int)
        correct_cmp = (preds_cmp == y_test_cmp).astype(int)

        last_n_cmp = min(100, len(correct_cmp))
        acc_cmp = correct_cmp[-last_n_cmp:].mean() * 100
        results.append({"dz": dz_pct, "n_test": len(test_df_cmp), "acc": acc_cmp, "n_used": last_n_cmp})

    progress.progress(1.0, text="اكتملت المقارنة!")
    progress.empty()

    valid_results = [r for r in results if r["acc"] is not None]
    if not valid_results:
        st.error("مفيش أي قيمة عطت بيانات كافية للمقارنة. جرّب فريم زمني أكبر أو مدة أطول.")
    else:
        best = max(valid_results, key=lambda r: r["acc"])

        st.markdown("<h3 style='text-align:center;'>🏆 أقوى نتيجة</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='big-signal buy' style='font-size:26px;'>
        Dead Zone = {best['dz']}%<br>
        <span style='font-size:34px;'>{best['acc']:.1f}%</span> دقة
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"محسوبة على آخر {best['n_used']} توقع من أصل {best['n_test']} في بيانات الاختبار.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;'>📊 جدول المقارنة الكامل</h4>", unsafe_allow_html=True)

        table_rows = []
        for r in sorted(results, key=lambda x: (x["acc"] is None, -(x["acc"] or 0))):
            table_rows.append({
                "Dead Zone %": r["dz"],
                "الدقة": f"{r['acc']:.1f}%" if r["acc"] is not None else "بيانات غير كافية",
                "عدد التوقعات المستخدمة": r["n_used"] if r["acc"] is not None else "-",
                "🏆": "✅" if r["acc"] == best["acc"] else "",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.info(f"💡 استخدم **Dead Zone = {best['dz']}%** في الإعداد العادي من الشريط الجانبي "
                f"عشان تاخد أقوى دقة ممكنة بالإعدادات الحالية (فريم {timeframe}، {years_back} سنة بيانات).")

        if best["acc"] < 55:
            st.warning("⚠️ حتى أفضل نتيجة قريبة من العشوائية (50%) — الموديل بالإعدادات دي (الفريم/المدة) "
                       "مش لاقي ميزة حقيقية واضحة، جرّب تغيّر الفريم الزمني أو المدة بدل الـ Dead Zone بس.")

    st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# التشغيل — موديل واحد فقط (LightGBM) لأقل استهلاك موارد وأسرع نتيجة
# ------------------------------------------------------------------------
if run_btn:
    try:
        with st.spinner("جاري تحميل بيانات TRX من Binance..."):
            raw = core.fetch_binance_ohlcv(SYMBOL, timeframe, years_back)
    except Exception as e:
        st.error(f"❌ فشل تحميل البيانات من Binance:\n\n{e}")
        st.info("جرّب تدوس «تحديث وتشغيل» تاني بعد شوية، أو غيّر الفريم الزمني.")
        st.stop()

    if len(raw) < 200:
        st.error("بيانات قليلة جداً. جرّب فريم زمني أكبر أو مدة أطول.")
        st.stop()

    with st.spinner("جاري بناء المؤشرات..."):
        featured = core.build_features(raw)
        featured = core.add_swing_points(featured, order=5)
        featured = core.add_liquidity_features(featured, window=20)
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

    # مدة كل شمعة، عشان نحسب "من الساعة كام لحد كام" لكل توقع
    timeframe_minutes = {
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440,
    }
    candle_duration = pd.Timedelta(minutes=timeframe_minutes.get(timeframe, 480))
    candle_start = test_df["timestamp"].values
    candle_end = test_df["timestamp"] + candle_duration

    st.session_state["model"] = model
    st.session_state["feature_cols"] = feature_cols
    st.session_state["featured"] = featured
    st.session_state["preds"] = preds
    st.session_state["correct"] = correct
    st.session_state["probs"] = probs
    st.session_state["candle_start"] = candle_start
    st.session_state["candle_end"] = candle_end.values
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

    # --- القمم والقيعان والسيولة (بلغة مبسّطة) ---
    last_r = featured.iloc[-1]
    st.markdown("<div style='text-align:center; margin-top:8px;'>", unsafe_allow_html=True)

    dist_high = last_r.get("dist_to_last_swing_high", None)
    dist_low = last_r.get("dist_to_last_swing_low", None)
    vz = last_r.get("volume_zscore", None)

    lines = []
    if dist_high is not None and pd.notna(dist_high):
        lines.append(f"📍 السعر الحالي أبعد من آخر **قمة** بمقدار **{dist_high*100:.2f}%**")
    if dist_low is not None and pd.notna(dist_low):
        lines.append(f"📍 السعر الحالي أبعد من آخر **قاعدة** بمقدار **{dist_low*100:.2f}%**")
    if vz is not None and pd.notna(vz):
        if vz > 2:
            lines.append("💧 **سيولة قوية دخلت السوق الآن** (حجم تداول أعلى من الطبيعي بوضوح)")
        elif vz < -1.5:
            lines.append("💧 **السيولة ضعيفة حالياً** (حجم تداول أقل من الطبيعي)")
        else:
            lines.append("💧 السيولة طبيعية حالياً")

    for line in lines:
        st.markdown(f"<div style='color:#ccc; font-size:15px; margin:4px 0;'>{line}</div>",
                     unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🧠 ده موديل ML حقيقي؟ اضغط هنا للشرح المبسّط"):
        st.markdown("""
- **نعم، LightGBM موديل تعلم آلي حقيقي** — مش قواعد ثابتة "لو حصل X اعمل Y".
  هو عبارة عن **مئات من شجرات القرار** بتتدرب على البيانات التاريخية،
  وكل شجرة بتصحّح أخطاء اللي قبلها، وفي الآخر بتتجمع كل الشجرات في تصويت واحد نهائي.
- الموديل شايف كل شمعة من زوايا كتير مع بعض: مؤشرات فنية (RSI, MACD...)،
  **أقرب قمة وقاعدة (Swing Points)**، **حجم السيولة الحالي**، وأنماط سعرية سابقة —
  وبيتعلم لوحده أي زاوية من دول أهم في كل لحظة، من غير ما أي حد يكتبله قاعدة يدوية.
- "دقة آخر 100 توقع" اللي تحت هي **الاختبار الحقيقي** — بتوريك أداء الموديل ده
  على بيانات حقيقية ماشافهاش وقت التدريب، مش مجرد وعد نظري.
        """)

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

    # --- جدول تفصيلي: كل توقع بتاريخه ووقته (من الساعة كام لحد كام) ---
    st.markdown("<h4 style='text-align:center;'>📋 جدول تفاصيل التوقعات</h4>", unsafe_allow_html=True)

    candle_start = st.session_state["candle_start"][-last_n:]
    candle_end = st.session_state["candle_end"][-last_n:]
    preds_slice = preds[-last_n:]

    detail_rows = []
    for i in range(last_n):
        ts_start = pd.Timestamp(candle_start[i])
        ts_end = pd.Timestamp(candle_end[i])
        detail_rows.append({
            "التاريخ": ts_start.strftime("%Y-%m-%d"),
            "من الساعة": ts_start.strftime("%H:%M"),
            "إلى الساعة": ts_end.strftime("%H:%M"),
            "التوقع": "شراء ⬆️" if preds_slice[i] == 1 else "بيع ⬇️",
            "النتيجة": "✅ صحيح" if last_correct[i] == 1 else "❌ خطأ",
        })

    # الأحدث فوق عشان يبقى أسهل تلاقي آخر توقع
    detail_df = pd.DataFrame(detail_rows).iloc[::-1].reset_index(drop=True)
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=400)
    st.caption("الجدول مرتب من الأحدث (فوق) للأقدم (تحت). كل صف = توقع شمعة واحدة بالكامل.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("⚠️ توقعات إحصائية تعليمية وليست نصيحة استثمارية.")

else:
    st.info("اضبط الإعدادات من الشريط الجانبي، وبعدين دوس «تحديث وتشغيل».")
