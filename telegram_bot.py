import requests
from datetime import datetime
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


def _zaman():
    return datetime.now().strftime("%H:%M:%S")


def mesaj_gonder(metin, chat_id=None):
    """Telegram'a mesaj gönderir"""
    hedef = chat_id or TELEGRAM_CHAT_ID
    url   = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {
        "chat_id"                  : hedef,
        "text"                     : metin,
        "parse_mode"               : "HTML",
        "disable_web_page_preview" : True
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print(f"[{_zaman()}] ✅ Telegram: Mesaj gönderildi.")
        return True
    except requests.exceptions.Timeout:
        print(f"[{_zaman()}] ❌ Telegram: Zaman aşımı.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"[{_zaman()}] ❌ HTTP Hata: {e.response.text}")
        return False
    except Exception as e:
        print(f"[{_zaman()}] ❌ Genel Hata: {e}")
        return False


def sabah_raporu_olustur(tum_sonuclar):
    """Sabah 09:00 detaylı raporu"""
    simdi     = datetime.now().strftime("%d.%m.%Y %H:%M")
    guclu_al  = [s for s in tum_sonuclar if s.get("sinyal") == "GÜÇLÜ AL"]
    al        = [s for s in tum_sonuclar if s.get("sinyal") == "AL"]
    guclu_sat = [s for s in tum_sonuclar if s.get("sinyal") == "GÜÇLÜ SAT"]
    sat_list  = [s for s in tum_sonuclar if s.get("sinyal") == "SAT"]

    msg = (
        f"⚡ <b>MIDAS TERMINAL — SABAH RAPORU</b>\n"
        f"<code>📅 {simdi}</code>\n"
        f"{'─'*30}\n\n"
    )

    # Güçlü Al
    if guclu_al:
        msg += f"🚀 <b>GÜÇLÜ AL ({len(guclu_al)})</b>\n"
        for s in guclu_al[:10]:
            msg += (
                f"  ├ <code>{s['hisse']:<12}</code>"
                f" ${s['fiyat']:.2f}"
                f" | %{s['guven']*100:.0f} güven"
                f" | RSI {s['rsi']:.0f}\n"
                f"  │  🎯 Hedef: ${s['risk']['hedef_1']:.2f}"
                f"  🛑 Stop: ${s['risk']['stop_loss']:.2f}\n"
            )
        msg += "\n"

    # Al
    if al:
        msg += f"🟢 <b>AL ({len(al)})</b>\n"
        for s in al[:8]:
            msg += (
                f"  ├ <code>{s['hisse']:<12}</code>"
                f" ${s['fiyat']:.2f}"
                f" | %{s['guven']*100:.0f} güven\n"
            )
        msg += "\n"

    # Satışlar
    if guclu_sat or sat_list:
        msg += f"🔴 <b>SAT ({len(guclu_sat)+len(sat_list)})</b>\n"
        for s in (guclu_sat + sat_list)[:5]:
            msg += (
                f"  ├ <code>{s['hisse']:<12}</code>"
                f" {s['sinyal']}"
                f" | %{s['guven']*100:.0f}\n"
            )
        msg += "\n"

    # Özet
    msg += (
        f"{'─'*30}\n"
        f"📊 Toplam taranan: <b>{len(tum_sonuclar)} hisse</b>\n"
        f"🚀 Güçlü Al: <b>{len(guclu_al)}</b>  "
        f"🟢 Al: <b>{len(al)}</b>  "
        f"🔴 Sat: <b>{len(guclu_sat)+len(sat_list)}</b>\n\n"
        f"⚠️ <i>AI analizi — yatırım tavsiyesi değildir.</i>"
    )
    return msg


def saatlik_rapor_olustur(tum_sonuclar, saat):
    """Saat başı kısa güncelleme"""
    simdi  = datetime.now().strftime("%H:%M")
    onemli = [
        s for s in tum_sonuclar
        if s.get("sinyal") in ["GÜÇLÜ AL", "GÜÇLÜ SAT"]
        and s.get("guven", 0) >= 0.70
    ]

    if not onemli:
        return (
            f"⚡ <b>MIDAS {simdi} GÜNCELLEME</b>\n"
            f"Güçlü sinyal bulunamadı 🔍\n"
            f"Tarama devam ediyor..."
        )

    msg = f"⚡ <b>MIDAS {simdi} — GÜÇLÜ SİNYALLER</b>\n{'─'*28}\n"
    for s in onemli[:10]:
        ok = "🚀" if "AL" in s["sinyal"] else "🔻"
        msg += (
            f"{ok} <code>{s['hisse']:<12}</code>"
            f" {s['sinyal']}"
            f" | %{s['guven']*100:.0f}\n"
        )
    return msg


def hata_bildirimi(hata_mesaji):
    """Sistem hatalarını Telegram'a bildir"""
    simdi = datetime.now().strftime("%d.%m.%Y %H:%M")
    msg   = (
        f"🔴 <b>MIDAS SİSTEM HATASI</b>\n"
        f"<code>{simdi}</code>\n\n"
        f"<code>{str(hata_mesaji)[:300]}</code>"
    )
    mesaj_gonder(msg)