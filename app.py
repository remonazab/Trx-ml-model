"""
========================================================================
 TRX ML Predictor — واجهة ويب شاملة (Streamlit) — مخصصة لعملة TRON (TRX)
========================================================================
تشغيل:  streamlit run app.py
كل الإعدادات (فريم زمني / تواريخ / موديلات / تحسينات) من الشريط الجانبي.
========================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_ml_predictor as core  # يشغّل ensure_packages() تلقائياً عند الاستيراد

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score

SYMBOL = "TRX/USDT"  # مثبّتة عمداً — السكربت كله مخصّص لـ TRX فقط

st.set_page_config(page_title="TRX ML Predictor", layout="wide", page_icon="🔺")

# ------------------------------------------------------------------------
# الشريط الجانبي — كل الإعدادات من هنا
# ------------------------------------------------------------------------
st.sidebar.title("🔺 TRX ML Predictor")
st.sidebar.caption("سكربت مخصّص بالكامل لعملة TRON (TRX/USDT) على Binance")

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ الفريم الزمني والبيانات")
timeframe = st.sidebar.selectbox(
    "الفريم الزمني (تحكم كامل)",
    ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
    index=9,  # "8h" افتراضياً
)
years_back = st.sidebar.slider("عدد سنين البيانات (Training)", 0.5, 5.0, 3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 فترة الـ Backtest (اختياري)")
use_custom_range = st.sidebar.checkbox("تحديد تاريخ اختبار يدوي", value=False)
test_start, test_end = None, None
if use_custom_range:
    col_a, col_b = st.sidebar.columns(2)
    test_start = col_a.date_input("من تاريخ")
    test_end = col_b.date_input("إلى تاريخ")
    test_start = str(test_start) if test_start else None
    test_end = str(test_end) if test_end else None
else:
    split_ratio = st.sidebar.slider("نسبة بيانات التدريب %", 60, 95, 85) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 إعدادات الموديل")
target_type = st.sidebar.radio("نوع التنبؤ", ["direction", "price"], index=0,
                                format_func=lambda x: "اتجاه (صعود/نزول)" if x == "direction" else "سعر فعلي")
horizon = st.sidebar.number_input("التنبؤ بعد كام شمعة (= الفريم الزمني المختار فوق)", min_value=1, max_value=50, value=1)
models_selected = st.sidebar.multiselect(
    "الموديلات المستخدمة", ["lightgbm", "xgboost"], default=["lightgbm", "xgboost"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 تحسينات الذكاء")
min_move_pct = st.sidebar.slider(
    "الحد الأدنى للحركة المعتبرة (Dead Zone) %", 0.0, 1.0, 0.2, step=0.05,
    help="الحركات الأصغر من النسبة دي تُعتبر نويز وتُستبعد من التدريب والاختبار.",
) / 100.0
use_htf_trend = st.sidebar.checkbox("إضافة اتجاه فريم زمني أعلى (يومي)", value=True)
use_cross_asset = st.sidebar.checkbox(
    "إضافة حركة BTC و ETH كميزات", value=True,
    help="TRX بيتحرك غالباً تبع البيتكوين — ده بيضيف عوائد BTC/ETH ومعامل الارتباط المتحرك.",
)
use_walk_forward = st.sidebar.checkbox("تقييم Walk-Forward (أدق وأكتر مصداقية)", value=True)
wf_splits = st.sidebar.slider("عدد فترات Walk-Forward", 3, 10, 5) if use_walk_forward else 5

st.sidebar.markdown("---")
fetch_news_toggle = st.sidebar.checkbox("جلب آخر أخبار TRX/Tron", value=True)

run_btn = st.sidebar.button("🔄 تحديث البيانات وتشغيل التحليل الكامل", type="primary", use_container_width=True)

st.title(f"🔺 التحليل الشامل لعملة TRX/USDT — فريم {timeframe}")

# ------------------------------------------------------------------------
# تشغيل خط الأنابيب الكامل عند الضغط على الزر
# ------------------------------------------------------------------------
if run_btn:
    with st.spinner("جاري تحميل بيانات Binance لـ TRX..."):
        raw = core.fetch_binance_ohlcv(SYMBOL, timeframe, years_back)

    if len(raw) < 300:
        st.error("بيانات قليلة جداً — جرّب فريم زمني أكبر أو عدد سنين أكبر.")
        st.stop()

    with st.spinner("جاري بناء المؤشرات الفنية (RSI, MACD, Ichimoku...)..."):
        featured = core.build_features(raw)

    if use_htf_trend:
        with st.spinner("جاري إضافة اتجاه الفريم الزمني الأعلى (يومي)..."):
            featured = core.add_higher_timeframe_trend(featured, SYMBOL, timeframe, years_back, "1d")

    if use_cross_asset:
        with st.spinner("جاري إضافة ميزات BTC/ETH ومعامل الارتباط..."):
            featured = core.add_cross_asset_features(featured, timeframe, years_back, ["BTC/USDT", "ETH/USDT"])
            featured = core.add_rolling_correlation_with_btc(featured, window=30)

    featured = core.add_tron_onchain_features(featured, SYMBOL)

    with st.spinner("جاري كشف القمم والقيعان وأنماط الشموع ومراقبة السيولة..."):
        featured = core.add_swing_points(featured, order=5)
        featured = core.add_candlestick_patterns(featured)
        featured = core.add_liquidity_features(featured, window=20)

    news_list = []
    if fetch_news_toggle:
        with st.spinner("جاري جلب آخر أخبار TRX/Tron..."):
            news_list = core.fetch_trx_news(limit=12)

    targeted = core.build_target(featured, horizon, target_type, min_move_pct)
    feature_cols = core.get_feature_columns(targeted)

    wf_results = None
    if use_walk_forward and target_type == "direction":
        with st.spinner(f"جاري تقييم Walk-Forward على {wf_splits} فترات..."):
            wf_results = core.walk_forward_validation(targeted, feature_cols, target_type, models_selected, wf_splits)

    train_df, test_df = core.time_split(
        targeted,
        split_ratio if not use_custom_range else 0.85,
        test_start, test_end,
    )

    if len(train_df) < 100 or len(test_df) < 20:
        st.error("بيانات التدريب/الاختبار غير كافية — راجع الفترة الزمنية المحددة أو صغّر الـ Dead Zone.")
        st.stop()

    with st.spinner("جاري تدريب الموديلات النهائية..."):
        trained_models = core.train_models(train_df, feature_cols, target_type, models_selected)

    if not trained_models:
        st.error("لم يتم تدريب أي موديل — تأكد من اختيار موديل واحد على الأقل.")
        st.stop()

    y_test = test_df["target"]
    predictions = {}
    metrics_rows = []
    X_test = test_df[feature_cols]

    for name, model in trained_models.items():
        if target_type == "direction":
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            predictions[name] = pred
            metrics_rows.append({"الموديل": name, "Accuracy": round(accuracy_score(y_test, pred), 4)})
        else:
            pred = model.predict(X_test)
            predictions[name] = pred
            mae = np.mean(np.abs(y_test - pred))
            metrics_rows.append({"الموديل": name, "MAE": round(mae, 4)})

    if len(trained_models) > 1 and target_type == "direction":
        stacked = np.mean([predictions[n] for n in predictions], axis=0)
        ensemble_pred = (stacked >= 0.5).astype(int)
        predictions["ensemble"] = ensemble_pred
        metrics_rows.append({"الموديل": "ensemble", "Accuracy": round(accuracy_score(y_test, ensemble_pred), 4)})

    # حفظ كل حاجة في session_state عشان الصفحة ماتفضلش تعيد التحميل كل تفاعل
    st.session_state.update({
        "featured": featured, "test_df": test_df, "predictions": predictions,
        "metrics_rows": metrics_rows, "trained_models": trained_models,
        "feature_cols": feature_cols, "target_type": target_type,
        "timeframe": timeframe, "wf_results": wf_results, "news_list": news_list,
    })

# ------------------------------------------------------------------------
# عرض النتائج (لو موجودة في session_state) — منظمة في Tabs
# ------------------------------------------------------------------------
if "featured" in st.session_state:
    featured = st.session_state["featured"]
    test_df = st.session_state["test_df"]
    predictions = st.session_state["predictions"]
    metrics_rows = st.session_state["metrics_rows"]
    trained_models = st.session_state["trained_models"]
    feature_cols = st.session_state["feature_cols"]
    target_type = st.session_state["target_type"]
    wf_results = st.session_state.get("wf_results")
    news_list = st.session_state.get("news_list", [])

    tab_rec, tab_chart, tab_backtest, tab_wf, tab_news = st.tabs(
        ["🎯 التوصية والملخص", "📊 الشارت والمؤشرات", "💰 Backtest والصفقات", "🔁 Walk-Forward", "📰 الأخبار"]
    )

    # ================= التبويب 1: التوصية والملخص =================
    with tab_rec:
        st.subheader("🎯 التوصية الحالية (بناءً على آخر شمعة)")
        if target_type == "direction":
            last_row = featured.tail(1)[feature_cols]
            cols = st.columns(len(trained_models) + (1 if len(trained_models) > 1 else 0))
            probs = []
            for i, (name, model) in enumerate(trained_models.items()):
                p_up = float(model.predict_proba(last_row)[:, 1][0])
                probs.append(p_up)
                is_buy = p_up >= 0.5
                shown_pct = p_up * 100 if is_buy else (1 - p_up) * 100
                with cols[i]:
                    if is_buy:
                        st.success(f"**{name.upper()}**\n\n### 🟢 شراء (BUY)\n\nنسبة التوقع: {shown_pct:.1f}%")
                    else:
                        st.error(f"**{name.upper()}**\n\n### 🔴 بيع (SELL)\n\nنسبة التوقع: {shown_pct:.1f}%")

            if len(trained_models) > 1:
                avg_p = sum(probs) / len(probs)
                is_buy = avg_p >= 0.5
                shown_pct = avg_p * 100 if is_buy else (1 - avg_p) * 100
                with cols[-1]:
                    if is_buy:
                        st.success(f"**ENSEMBLE**\n\n### 🟢 شراء (BUY)\n\nنسبة التوقع: {shown_pct:.1f}%")
                    else:
                        st.error(f"**ENSEMBLE**\n\n### 🔴 بيع (SELL)\n\nنسبة التوقع: {shown_pct:.1f}%")

            st.caption(f"آخر بيانات متاحة حتى: {featured['timestamp'].iloc[-1]}")

            if "corr_with_btc_30" in featured.columns:
                last_corr = featured["corr_with_btc_30"].iloc[-1]
                if pd.notna(last_corr):
                    corr_desc = "مرتفع (تابع للسوق)" if abs(last_corr) > 0.6 else (
                        "متوسط" if abs(last_corr) > 0.3 else "منخفض (مستقل نسبياً عن BTC)"
                    )
                    st.caption(f"📎 معامل الارتباط الحالي مع BTC (آخر 30 شمعة): **{last_corr:.2f}** — {corr_desc}")

            # --- مؤشرات السيولة والقمم/القيعان الحالية ---
            st.markdown("---")
            st.markdown("**💧 حالة السيولة والبنية السعرية الحالية:**")
            last_r = featured.iloc[-1]
            lq1, lq2, lq3, lq4 = st.columns(4)
            vz = last_r.get("volume_zscore", np.nan)
            lq1.metric("Volume Z-Score", f"{vz:.2f}" if pd.notna(vz) else "-",
                       help="أعلى من 2 = نشاط تداول غير عادي (سيولة قوية دخلت السوق)")
            lq2.metric("الانحراف عن VWAP", f"{last_r.get('dist_from_vwap_pct', 0)*100:.2f}%")
            lq3.metric("بُعد السعر عن آخر قمة مؤكدة", f"{last_r.get('dist_to_last_swing_high', 0)*100:.2f}%")
            lq4.metric("بُعد السعر عن آخر قاعدة مؤكدة", f"{last_r.get('dist_to_last_swing_low', 0)*100:.2f}%")

            active_patterns = [c.replace("pattern_", "") for c in featured.columns
                                if c.startswith("pattern_") and last_r.get(c, 0) == 1]
            if active_patterns:
                st.info(f"🕯️ نمط شمعة مكتشف في آخر شمعة: **{', '.join(active_patterns)}**")

            # --- لقطة سيولة لحظية من دفتر الطلبات الفعلي على Binance ---
            with st.spinner("جاري جلب لقطة سيولة لحظية من دفتر الطلبات..."):
                ob = core.fetch_live_orderbook_snapshot(SYMBOL)
            if ob:
                st.markdown("**📖 دفتر الطلبات اللحظي (Order Book) الآن:**")
                ob1, ob2, ob3, ob4 = st.columns(4)
                ob1.metric("أفضل Bid", f"{ob['best_bid']:.5f}" if ob["best_bid"] else "-")
                ob2.metric("أفضل Ask", f"{ob['best_ask']:.5f}" if ob["best_ask"] else "-")
                ob3.metric("Spread", f"{ob['spread_pct']:.3f}%" if ob["spread_pct"] is not None else "-")
                imbalance_desc = "ضغط شراء" if ob["imbalance"] > 0.1 else ("ضغط بيع" if ob["imbalance"] < -0.1 else "متوازن")
                ob4.metric("توازن السيولة", f"{ob['imbalance']*100:.1f}%", help=imbalance_desc)
        else:
            last_row = featured.tail(1)[feature_cols]
            for name, model in trained_models.items():
                p = model.predict(last_row)[0]
                st.info(f"**{name.upper()}** — السعر المتوقع: {p:.5f}")

        st.markdown("---")
        st.subheader("📋 ملخص أداء الموديلات (على بيانات الاختبار)")
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)
        st.warning("⚠️ توصيات إحصائية تعليمية وليست نصيحة استثمارية. الأداء الفعلي قد يختلف تماماً.")

    # ================= التبويب 2: الشارت والمؤشرات + جدول التوقعات =================
    with tab_chart:
        st.subheader("📊 شارت السعر والمؤشرات")
        plot_df = featured
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03,
            subplot_titles=("السعر + Bollinger + EMA", "الحجم", "RSI", "MACD"),
        )
        fig.add_trace(go.Candlestick(x=plot_df["timestamp"], open=plot_df["open"], high=plot_df["high"],
                                      low=plot_df["low"], close=plot_df["close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["bb_high"],
                                  line=dict(color="rgba(150,150,255,0.5)", width=1), name="BB High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["bb_low"],
                                  line=dict(color="rgba(150,150,255,0.5)", width=1), name="BB Low",
                                  fill="tonexty", fillcolor="rgba(150,150,255,0.08)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["ema_50"],
                                  line=dict(color="orange", width=1.3), name="EMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["ema_200"],
                                  line=dict(color="purple", width=1.3), name="EMA 200"), row=1, col=1)

        if "is_swing_high" in plot_df.columns:
            sh = plot_df[plot_df["is_swing_high"] == 1]
            sl = plot_df[plot_df["is_swing_low"] == 1]
            fig.add_trace(go.Scatter(x=sh["timestamp"], y=sh["high"], mode="markers", name="قمة (Swing High)",
                                      marker=dict(symbol="triangle-down", size=9, color="red")), row=1, col=1)
            fig.add_trace(go.Scatter(x=sl["timestamp"], y=sl["low"], mode="markers", name="قاعدة (Swing Low)",
                                      marker=dict(symbol="triangle-up", size=9, color="lime")), row=1, col=1)

        fig.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["volume"], name="Volume",
                              marker_color="rgba(100,149,237,0.6)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["rsi_14"],
                                  line=dict(color="teal"), name="RSI 14"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["macd"],
                                  line=dict(color="blue"), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["macd_signal"],
                                  line=dict(color="orange"), name="Signal"), row=4, col=1)
        fig.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["macd_diff"], name="MACD Diff",
                              marker_color="grey"), row=4, col=1)
        fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False,
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        if target_type == "direction" and len(test_df) > 0:
            st.markdown("---")
            st.subheader("📑 جدول استراتيجية Backtest — كل فترة والتوقع ونسبة صحته")

            best_name = "ensemble" if "ensemble" in predictions else list(trained_models.keys())[0]
            bt_table_df = test_df.reset_index(drop=True).copy()
            X_for_table = bt_table_df[feature_cols]

            if best_name == "ensemble":
                all_probs = [m.predict_proba(X_for_table)[:, 1] for m in trained_models.values()]
                probs = np.mean(all_probs, axis=0)
            else:
                probs = trained_models[best_name].predict_proba(X_for_table)[:, 1]

            bt_table_df["prob_up"] = probs
            bt_table_df["predicted"] = (probs >= 0.5).astype(int)
            bt_table_df["actual"] = bt_table_df["target"]
            bt_table_df["correct"] = bt_table_df["predicted"] == bt_table_df["actual"]

            n_rows_view = st.slider("عدد الفترات المعروضة في الجدول", 20, min(500, len(bt_table_df)), 50, step=10)

            table_rows = []
            for _, r in bt_table_df.tail(n_rows_view).iterrows():
                confidence = r["prob_up"] * 100 if r["predicted"] == 1 else (1 - r["prob_up"]) * 100
                table_rows.append({
                    "الفترة (بداية الشمعة)": r["timestamp"],
                    "السعر (Close)": round(r["close"], 5),
                    "التوقع": "شراء ⬆️" if r["predicted"] == 1 else "بيع ⬇️",
                    "نسبة الثقة في التوقع": f"{confidence:.1f}%",
                    "الاتجاه الفعلي بعد الفترة": "صعود ⬆️" if r["actual"] == 1 else "نزول ⬇️",
                    "صحة التوقع": "✅ صحيح" if r["correct"] else "❌ خطأ",
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            overall_acc = bt_table_df["correct"].mean() * 100
            st.caption(
                f"📊 نسبة صحة التوقع الإجمالية على كل فترات الاختبار ({len(bt_table_df)} فترة، "
                f"كل فترة = {st.session_state['timeframe']}): **{overall_acc:.1f}%**"
            )

    # ================= التبويب 3: Backtest والصفقات =================
    with tab_backtest:
        if target_type == "direction":
            st.subheader("💰 Backtest: الاستراتيجية مقابل Buy & Hold")
            key = "ensemble" if "ensemble" in predictions else list(predictions.keys())[0]
            bt_df = test_df.reset_index(drop=True).copy()
            bt_df["signal"] = predictions[key]
            bt_df["market_return"] = bt_df["close"].pct_change().shift(-1)
            bt_df["strategy_return"] = bt_df["market_return"] * bt_df["signal"]
            bt_df["cum_market"] = (1 + bt_df["market_return"].fillna(0)).cumprod()
            bt_df["cum_strategy"] = (1 + bt_df["strategy_return"].fillna(0)).cumprod()

            c1, c2 = st.columns(2)
            c1.metric("عائد Buy & Hold (تراكمي)", f"{(bt_df['cum_market'].iloc[-1]-1)*100:.2f}%")
            c2.metric("عائد استراتيجية الموديل (تراكمي)", f"{(bt_df['cum_strategy'].iloc[-1]-1)*100:.2f}%")

            trades = bt_df[bt_df["signal"] == 1].dropna(subset=["strategy_return"])
            n_trades = len(trades)
            if n_trades > 0:
                win_rate = (trades["strategy_return"] > 0).mean() * 100
                avg_trade_return = trades["strategy_return"].mean() * 100
                best_trade = trades["strategy_return"].max() * 100
                worst_trade = trades["strategy_return"].min() * 100
                period_days = (bt_df["timestamp"].iloc[-1] - bt_df["timestamp"].iloc[0]).total_seconds() / 86400
                trades_per_year_est = n_trades / max(period_days, 1) * 365 if period_days > 0 else 0

                st.markdown(f"**📌 إحصائيات لكل صفقة (كل صفقة = كل {st.session_state['timeframe']}):**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("عدد الصفقات", f"{n_trades}")
                m2.metric("Win Rate", f"{win_rate:.1f}%")
                m3.metric("متوسط الربح/الخسارة لكل صفقة", f"{avg_trade_return:.3f}%")
                m4.metric("أفضل / أسوأ صفقة", f"{best_trade:.2f}% / {worst_trade:.2f}%")
                st.caption(f"عدد الصفقات المقدّر سنوياً: ~{trades_per_year_est:.0f} صفقة/سنة "
                           f"(فترة الاختبار الفعلية {period_days:.1f} يوم).")
            else:
                st.info("لا توجد صفقات شراء في فترة الاختبار بالإعدادات الحالية.")

            bt_fig = go.Figure()
            bt_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["cum_market"], name="Buy & Hold"))
            bt_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["cum_strategy"], name="Model Strategy"))
            bt_fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(bt_fig, use_container_width=True)
            st.caption("بدون عمولات/سبريد — الأداء الفعلي غالباً أضعف من ده.")
        else:
            st.info("الـ Backtest متاح فقط عند اختيار نوع التنبؤ 'اتجاه (صعود/نزول)'.")

    # ================= التبويب 4: Walk-Forward =================
    with tab_wf:
        if wf_results:
            st.subheader("🔁 Walk-Forward Validation — الاستقرار عبر فترات متعددة")
            st.caption("كل فترة يتدرب الموديل على البيانات اللي قبلها بس، ويُختبر على فترة جديدة — أدق من split واحد.")

            wf_table = []
            for r in wf_results:
                row = {"الفترة #": r["fold"], "من": r["test_start"], "إلى": r["test_end"], "عدد الشمعات": r["n_test"]}
                for name in trained_models.keys():
                    key = f"{name}_accuracy"
                    if key in r:
                        row[f"Accuracy ({name})"] = round(r[key], 4)
                wf_table.append(row)
            st.dataframe(pd.DataFrame(wf_table), use_container_width=True, hide_index=True)

            cols_wf = st.columns(len(trained_models))
            for i, name in enumerate(trained_models.keys()):
                key = f"{name}_accuracy"
                accs = [r[key] for r in wf_results if key in r]
                if accs:
                    cols_wf[i].metric(f"متوسط {name}", f"{np.mean(accs)*100:.1f}%",
                                       delta=f"± {np.std(accs)*100:.1f}% تذبذب", delta_color="off")
            st.info(
                "لو المتوسط قريب من 50% أو التذبذب بين الفترات كبير جداً، الموديل مش عنده ميزة حقيقية "
                "ومستقرة، حتى لو فترة واحدة طلعت نتيجتها كويسة."
            )
        else:
            st.info("فعّل خيار 'تقييم Walk-Forward' من الشريط الجانبي عشان يظهر هنا.")

    # ================= التبويب 5: الأخبار =================
    with tab_news:
        st.subheader("📰 آخر الأخبار المؤثرة على TRX/Tron")
        if news_list:
            for n in news_list:
                with st.container(border=True):
                    st.markdown(f"**[{n['title']}]({n['url']})**")
                    st.caption(f"🗞️ {n['source']} — {n['published_on'].strftime('%Y-%m-%d %H:%M UTC')}")
                    if n["body"]:
                        st.write(n["body"] + "...")
        else:
            st.info("لا توجد أخبار متاحة حالياً (فعّل خيار 'جلب آخر أخبار TRX/Tron' من الشريط الجانبي وشغّل التحليل تاني).")

        st.warning(
            "⚠️ الأخبار دي معلوماتية بس — الموديل الإحصائي **لا يقرأها ولا يفهم تأثيرها**. "
            "قرارات Justin Sun أو تغييرات تنظيمية مفاجئة على USDT-TRC20 ممكن تحرك السعر فوراً "
            "بشكل لا يقدر أي موديل تقني يتوقعه. راجع الأخبار دايماً بعينك قبل أي قرار."
        )

else:
    st.info("👈 اضبط الإعدادات من الشريط الجانبي، وبعدين دوس على زر «تحديث البيانات وتشغيل التحليل الكامل».")
