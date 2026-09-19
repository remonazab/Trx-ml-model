"""
========================================================================
 TRX Paper Trading — متصل بـ Pionex API حقيقي (وضع تجربة آمن تماماً)
========================================================================
الخطوة 1 فقط دلوقتي: التأكد من الاتصال الناجح بحسابك على Pionex
(قراءة الرصيد فقط - Read-Only) قبل ما نضيف أي منطق تنبؤ أو تسجيل صفقات.

المفتاحين (API Key & Secret) بيتقروا من Streamlit Secrets، مش مكتوبين
هنا في الكود خالص، عشان يفضلوا محميين حتى لو الكود اتنشر على GitHub.
========================================================================
"""

import os
import sys
import time
import hmac
import hashlib
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_ml_predictor as core

st.set_page_config(page_title="TRX Paper Trading — Pionex", layout="centered", page_icon="🧪")

PIONEX_BASE_URL = "https://api.pionex.com"
SYMBOL_BINANCE = "TRX/USDT"
SYMBOL_PIONEX = "TRX_USDT"
TIMEFRAME = "8h"  # ثابتة عشان تطابق دورة المداولة كل 8 ساعات اللي طلبتها
TRADES_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.csv")

TRADE_COLUMNS = [
    "trade_id", "side", "status", "leverage",
    "open_time", "entry_price",
    "close_time", "close_price", "close_reason",
    "pnl_pct",
]

# ------------------------------------------------------------------------
# قراءة المفاتيح من Streamlit Secrets (آمن — مش مكتوب في الكود)
# ------------------------------------------------------------------------
def get_pionex_credentials():
    try:
        api_key = st.secrets["PIONEX_API_KEY"]
        api_secret = st.secrets["PIONEX_API_SECRET"]
        return api_key, api_secret
    except Exception:
        return None, None


# ------------------------------------------------------------------------
# توقيع الطلبات الخاصة (Private Endpoints) حسب مواصفات Pionex الرسمية
# ------------------------------------------------------------------------
def sign_request(method, path, params, api_secret):
    """
    يبني التوقيع بالضبط حسب خطوات Pionex الرسمية:
    1. رتّب الباراميترات أبجدياً (ASCII) واربطهم بـ &
    2. اعمل PATH_URL = المسار + ? + الباراميترات المرتبة
    3. اعمل METHOD + PATH_URL
    4. وقّع النتيجة بـ HMAC-SHA256 باستخدام الـ Secret، وحوّلها Hex
    """
    params = dict(params)
    params["timestamp"] = str(int(time.time() * 1000))

    sorted_items = sorted(params.items())
    query_string = "&".join(f"{k}={v}" for k, v in sorted_items)
    path_url = f"{path}?{query_string}"
    message = f"{method}{path_url}"

    signature = hmac.new(
        api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return path_url, signature


def pionex_get(path, params, api_key, api_secret):
    """طلب GET موقّع لأي endpoint خاص (Private) في Pionex"""
    path_url, signature = sign_request("GET", path, params, api_secret)
    url = f"{PIONEX_BASE_URL}{path_url}"
    headers = {
        "PIONEX-KEY": api_key,
        "PIONEX-SIGNATURE": signature,
    }
    resp = requests.get(url, headers=headers, timeout=10)
    return resp


# ------------------------------------------------------------------------
# الصفحة
# ------------------------------------------------------------------------
st.title("🧪 TRX Paper Trading")
st.caption("الخطوة 1: التأكد من الاتصال بحساب Pionex (قراءة فقط، بدون أي تداول حقيقي)")

api_key, api_secret = get_pionex_credentials()

if not api_key or not api_secret:
    st.error(
        "❌ مفيش مفاتيح Pionex متسجّلة في Secrets.\n\n"
        "لازم تضيف `PIONEX_API_KEY` و `PIONEX_API_SECRET` في إعدادات الـ App "
        "على Streamlit Cloud (Settings → Secrets)."
    )
    st.stop()

st.success("✅ تم العثور على مفاتيح Pionex في Secrets.")

if st.button("🔌 اختبار الاتصال (جلب رصيد الحساب - Read Only)", type="primary", use_container_width=True):
    with st.spinner("جاري الاتصال بـ Pionex..."):
        try:
            resp = pionex_get("/api/v1/account/balances", {}, api_key, api_secret)
        except Exception as e:
            st.error(f"❌ فشل الاتصال بالشبكة: {e}")
            st.stop()

    if resp.status_code == 200:
        data = resp.json()
        if data.get("result"):
            st.success("✅ الاتصال نجح! دلوقتي متأكدين إن المفتاحين شغالين صح.")
            balances = data.get("data", {}).get("balances", [])
            non_zero = [b for b in balances if float(b.get("free", 0)) > 0 or float(b.get("frozen", 0)) > 0]
            if non_zero:
                st.markdown("**أرصدتك الحالية (قراءة فقط):**")
                for b in non_zero:
                    st.write(f"- **{b['coin']}**: متاح {b['free']} | مجمّد {b['frozen']}")
            else:
                st.info("الحساب متصل، لكن مفيش أرصدة ظاهرة حالياً (طبيعي لو الحساب فاضي أو جديد).")
        else:
            st.error(f"❌ الطلب رجع بدون نجاح: {data}")
    elif resp.status_code == 401:
        st.error(
            "❌ فشل التوثيق (401). الأسباب المحتملة:\n\n"
            "- المفتاح أو الـ Secret متكتوب غلط في Secrets\n"
            "- صلاحية Read مش مفعّلة على المفتاح في Pionex\n"
            "- فيه IP Whitelist مفعّل على المفتاح ومحدّد IP تاني غير سيرفر Streamlit"
        )
        st.code(resp.text)
    else:
        st.error(f"❌ خطأ غير متوقع (status {resp.status_code})")
        st.code(resp.text)


# ========================================================================
# الخطوة 2: محاكاة صفقات Long/Short بالرافعة على فريم 8 ساعات
# ========================================================================
st.markdown("---")
st.header("📈 محاكاة تداول Long/Short (Paper — بدون فلوس حقيقية)")
st.caption(
    "⚠️ التطبيق بيشتغل بس وقت ما تفتحه وتدوس تحديث — مش سيرفر شغال 24 ساعة. "
    "يعني فحص TP/SL وتبديل الصفقات بيحصل وقت الضغط بس، مش لحظياً بالثانية."
)


def get_pionex_price(symbol=SYMBOL_PIONEX):
    """السعر الحالي الفعلي من Pionex (Public endpoint - بدون مفتاح)"""
    resp = requests.get(
        f"{PIONEX_BASE_URL}/api/v1/market/tickers",
        params={"symbol": symbol}, timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    tickers = data.get("data", {}).get("tickers", [])
    if not tickers:
        raise RuntimeError("مفيش بيانات سعر راجعة من Pionex")
    return float(tickers[0]["close"])


def load_trades_log():
    if os.path.exists(TRADES_LOG_PATH):
        return pd.read_csv(TRADES_LOG_PATH, parse_dates=["open_time", "close_time"])
    return pd.DataFrame(columns=TRADE_COLUMNS)


def save_trades_log(df):
    df.to_csv(TRADES_LOG_PATH, index=False)


def get_latest_model_signal():
    """يدرب موديل خفيف بسرعة ويرجع آخر إشارة (long/short) + نسبة الثقة"""
    raw = core.fetch_binance_ohlcv(SYMBOL_BINANCE, TIMEFRAME, 1.0)
    featured = core.build_features(raw)
    targeted = core.build_target(featured, 1, "direction", 0.002)
    feature_cols = core.get_feature_columns(targeted)
    train_df, _ = core.time_split(targeted, 0.95, None, None)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    model = core.lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=-1, class_weight="balanced",
    )
    model.fit(X_train, y_train)

    last_row = featured.tail(1)[feature_cols]
    p_up = float(model.predict_proba(last_row)[:, 1][0])
    side = "long" if p_up >= 0.5 else "short"
    confidence = p_up * 100 if side == "long" else (1 - p_up) * 100
    return side, confidence


# --- إعدادات المخاطرة في الشريط الجانبي ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ إعدادات المحاكاة")
leverage = st.sidebar.number_input("الرافعة (Leverage)", min_value=1, max_value=50, value=5)
tp_pct = st.sidebar.number_input("نسبة جني الأرباح Take-Profit % (بعد الرافعة)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
sl_pct = st.sidebar.number_input("نسبة وقف الخسارة Stop-Loss % (بعد الرافعة)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)

run_sim_btn = st.button("🔄 تحديث ومراقبة الصفقات", type="primary", use_container_width=True)

if run_sim_btn:
    trades = load_trades_log()

    with st.spinner("جاري جلب السعر الحالي من Pionex..."):
        try:
            current_price = get_pionex_price()
        except Exception as e:
            st.error(f"❌ فشل جلب السعر من Pionex: {e}")
            st.stop()

    # --- 1) فحص الصفقة المفتوحة (لو موجودة) على TP/SL ---
    open_mask = trades["status"] == "open"
    if open_mask.any():
        idx = trades[open_mask].index[0]
        side = trades.loc[idx, "side"]
        entry_price = trades.loc[idx, "entry_price"]
        lev = trades.loc[idx, "leverage"]

        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100 * lev
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100 * lev

        if pnl_pct >= tp_pct:
            trades.loc[idx, ["status", "close_time", "close_price", "close_reason", "pnl_pct"]] = \
                ["closed", datetime.now(timezone.utc), current_price, "take_profit", pnl_pct]
            st.success(f"🎯 الصفقة ({side}) قفلت بربح +{pnl_pct:.2f}% (Take-Profit)")
        elif pnl_pct <= -sl_pct:
            trades.loc[idx, ["status", "close_time", "close_price", "close_reason", "pnl_pct"]] = \
                ["closed", datetime.now(timezone.utc), current_price, "stop_loss", pnl_pct]
            st.error(f"🛑 الصفقة ({side}) قفلت بخسارة {pnl_pct:.2f}% (Stop-Loss)")

    # --- 2) جيب إشارة الموديل الحالية ---
    with st.spinner("جاري تشغيل الموديل للحصول على آخر إشارة..."):
        try:
            signal_side, confidence = get_latest_model_signal()
        except Exception as e:
            st.error(f"❌ فشل تشغيل الموديل: {e}")
            st.stop()

    st.info(f"🧠 إشارة الموديل الحالية: **{'شراء (Long)' if signal_side=='long' else 'بيع (Short)'}** — ثقة {confidence:.1f}%")

    # --- 3) قرار التداول بناءً على الإشارة ---
    open_mask = trades["status"] == "open"

    if not open_mask.any():
        # مفيش صفقة مفتوحة → افتح واحدة جديدة
        new_id = int(trades["trade_id"].max() + 1) if len(trades) > 0 else 1
        new_row = {
            "trade_id": new_id, "side": signal_side, "status": "open", "leverage": leverage,
            "open_time": datetime.now(timezone.utc), "entry_price": current_price,
            "close_time": pd.NaT, "close_price": None, "close_reason": None, "pnl_pct": None,
        }
        trades = pd.concat([trades, pd.DataFrame([new_row])], ignore_index=True)
        st.success(f"🆕 اتفتحت صفقة **{signal_side}** جديدة بسعر {current_price}")
    else:
        idx = trades[open_mask].index[0]
        current_side = trades.loc[idx, "side"]
        if current_side == signal_side:
            st.info(f"↔️ الإشارة نفسها ({signal_side}) — الصفقة المفتوحة فضلت زي ما هي (اتمددت لفريم تاني).")
        else:
            # قلب الصفقة: اقفل الحالية بسبب انعكاس الإشارة، وافتح عكسها
            entry_price = trades.loc[idx, "entry_price"]
            lev = trades.loc[idx, "leverage"]
            if current_side == "long":
                pnl_pct = (current_price - entry_price) / entry_price * 100 * lev
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100 * lev

            trades.loc[idx, ["status", "close_time", "close_price", "close_reason", "pnl_pct"]] = \
                ["closed", datetime.now(timezone.utc), current_price, "signal_flip", pnl_pct]

            new_id = int(trades["trade_id"].max() + 1)
            new_row = {
                "trade_id": new_id, "side": signal_side, "status": "open", "leverage": leverage,
                "open_time": datetime.now(timezone.utc), "entry_price": current_price,
                "close_time": pd.NaT, "close_price": None, "close_reason": None, "pnl_pct": None,
            }
            trades = pd.concat([trades, pd.DataFrame([new_row])], ignore_index=True)
            st.warning(f"🔄 الإشارة اتقلبت! اتقفلت صفقة {current_side} (نتيجتها {pnl_pct:.2f}%) واتفتحت صفقة {signal_side} جديدة.")

    save_trades_log(trades)

# --- عرض سجل الصفقات ---
if os.path.exists(TRADES_LOG_PATH):
    trades = load_trades_log()

    st.markdown("---")
    st.subheader("📋 سجل الصفقات")

    open_trade = trades[trades["status"] == "open"]
    if len(open_trade) > 0:
        r = open_trade.iloc[0]
        st.markdown(f"**🔵 صفقة مفتوحة حالياً:** {r['side']} — دخول عند {r['entry_price']} — رافعة x{r['leverage']}")

    closed_trades = trades[trades["status"] == "closed"]
    if len(closed_trades) > 0:
        n_total = len(closed_trades)
        n_win = (closed_trades["pnl_pct"] > 0).sum()
        total_pnl = closed_trades["pnl_pct"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("عدد الصفقات المقفولة", n_total)
        c2.metric("نسبة الصفقات الرابحة", f"{n_win/n_total*100:.1f}%")
        c3.metric("إجمالي الـ PnL (تراكمي، غير مركّب)", f"{total_pnl:+.2f}%")

    st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True)

    # --- تحميل كملف Excel ---
    excel_path = TRADES_LOG_PATH.replace(".csv", ".xlsx")
    trades.to_excel(excel_path, index=False, engine="openpyxl")
    with open(excel_path, "rb") as f:
        st.download_button(
            "⬇️ تحميل السجل كملف Excel", data=f,
            file_name="trx_paper_trades.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
else:
    st.info("لسه مفيش صفقات مسجّلة. دوس «تحديث ومراقبة الصفقات» عشان تبدأ.")

st.caption(
    "⚠️ محاكاة تعليمية بالكامل (Paper Trading) — لا يتم تنفيذ أي أوامر حقيقية على Pionex. "
    "الأداء هنا لا يضمن نفس الأداء بفلوس حقيقية بسبب العمولات والـ Slippage والتنفيذ الفعلي."
)

