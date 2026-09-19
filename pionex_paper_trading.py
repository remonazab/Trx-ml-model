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

import requests
import streamlit as st

st.set_page_config(page_title="TRX Paper Trading — Pionex", layout="centered", page_icon="🧪")

PIONEX_BASE_URL = "https://api.pionex.com"

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
