"""
scheduler.py — Her saat başı otomatik çalışır

Çalıştırmak için:
  python scheduler.py
"""

import schedule
import time
import json
import os
import sys
import traceback
from datetime import datetime

# ─── Aynı klasördeki dosyaları bul ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Kendi modüllerin ─────────────────────────────
from config import (
    CALISMA_SAATLERI, SABAH_RAPOR_SAATI,
    MIN_GUVEN, MAX_BILDIRIM, BILDIRIM_SINYALLER
)
from telegram_bot import (
    mesaj_gonder, sabah_raporu_olustur,
    saatlik_rapor_olustur, hata_bildirimi
)

# ─── app.py'deki analiz fonksiyonları ─────────────
from app import tam_tarama, TUM

# ─── Dosya yolları ────────────────────────────────
TARAMA_DOSYA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tarama_cache.json")
LOG_DOSYA    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scheduler.log")


# ══════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════

def log(mesaj):
    """Ekrana ve dosyaya log yaz"""
    simdi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    satir = f"[{simdi}] {mesaj}"
    print(satir)
    try:
        with open(LOG_DOSYA, "a", encoding="utf-8") as f:
            f.write(satir + "\n")
    except Exception:
        pass


def cache_guncelle(sonuclar):
    """Sonuçları diske yaz — Streamlit buradan okur"""
    try:
        temiz = []
        for r in sonuclar:
            rc = dict(r)
            # Pandas Series → dict'e çevir
            kap = rc.get("kapanis")
            if kap is not None and hasattr(kap, "to_dict"):
                rc["kapanis"] = {str(k): float(v) for k, v in kap.items()}
            temiz.append(rc)

        with open(TARAMA_DOSYA, "w", encoding="utf-8") as f:
            json.dump(
                {"zaman": datetime.now().isoformat(), "sonuclar": temiz},
                f, ensure_ascii=False, indent=2, default=str
            )
        log(f"✅ Cache güncellendi ({len(temiz)} hisse)")
    except Exception as e:
        log(f"❌ Cache yazma hatası: {e}")


def sinyal_filtrele(sinyaller):
    """Config'e göre filtrele"""
    sonuc = []
    for s in sinyaller:
        if s.get("guven", 0) < MIN_GUVEN:
            continue
        if BILDIRIM_SINYALLER and s.get("sinyal") not in BILDIRIM_SINYALLER:
            continue
        sonuc.append(s)
    sonuc.sort(key=lambda x: x.get("guven", 0), reverse=True)
    return sonuc[:MAX_BILDIRIM]


# ══════════════════════════════════════════════════
# TARAMA FONKSİYONLARI
# ══════════════════════════════════════════════════

def sabah_tarama():
    """Sabah 09:00 — Tam tarama + detaylı rapor"""
    log("━"*40)
    log("🌅 SABAH TARAMASI BAŞLIYOR")
    log("━"*40)

    try:
        log(f"📊 {len(TUM)} hisse analiz ediliyor...")
        tum_sonuclar = tam_tarama()
        log(f"✅ Tamamlandı → {len(tum_sonuclar)} sonuç")

        # Cache'e yaz
        cache_guncelle(tum_sonuclar)

        # Telegram mesajı oluştur ve gönder
        mesaj = sabah_raporu_olustur(tum_sonuclar)
        ok    = mesaj_gonder(mesaj)

        if ok:
            log("✅ Sabah raporu Telegram'a gönderildi")
        else:
            log("❌ Telegram gönderilemedi")

    except Exception as e:
        log(f"❌ SABAH HATA: {traceback.format_exc()}")
        hata_bildirimi(f"Sabah tarama hatası: {e}")


def saatlik_tarama():
    """Her saat başı çalışır"""
    simdi = datetime.now()
    saat  = simdi.hour

    # Çalışma saati kontrolü
    if saat not in CALISMA_SAATLERI:
        log(f"⏭️ Saat {saat:02d}:00 — çalışma saati dışı, atlandı")
        return

    # Sabah saatiyse sabah raporuna yönlendir
    if saat == SABAH_RAPOR_SAATI:
        sabah_tarama()
        return

    log(f"🔄 SAATLİK TARAMA — {saat:02d}:00")

    try:
        tum_sonuclar = tam_tarama()
        cache_guncelle(tum_sonuclar)

        mesaj = saatlik_rapor_olustur(tum_sonuclar, saat)
        mesaj_gonder(mesaj)

        log(f"✅ Saatlik rapor gönderildi")

    except Exception as e:
        log(f"❌ SAATLİK HATA: {traceback.format_exc()}")
        hata_bildirimi(f"Saat {saat:02d}:00 hatası: {e}")


# ══════════════════════════════════════════════════
# BAŞLANGIÇ
# ══════════════════════════════════════════════════

def sistem_testi():
    """Telegram bağlantısını test et"""
    log("🧪 Telegram bağlantısı test ediliyor...")
    simdi = datetime.now().strftime("%d.%m.%Y %H:%M")

    test_msg = (
        f"⚡ <b>MIDAS TERMINAL BAŞLADI</b>\n"
        f"<code>{simdi}</code>\n\n"
        f"✅ Scheduler aktif\n"
        f"⏰ Çalışma saatleri: "
        f"{CALISMA_SAATLERI[0]:02d}:00 – {CALISMA_SAATLERI[-1]:02d}:00\n"
        f"🌅 Sabah raporu: {SABAH_RAPOR_SAATI:02d}:00\n"
        f"📊 Min güven: %{MIN_GUVEN*100:.0f}\n"
        f"🎯 Sinyal filtre: "
        f"{', '.join(BILDIRIM_SINYALLER) if BILDIRIM_SINYALLER else 'Hepsi'}"
    )

    ok = mesaj_gonder(test_msg)
    if ok:
        log("✅ Telegram bağlantısı başarılı!")
    else:
        log("❌ Telegram BAĞLANAMADI — config.py'yi kontrol et!")
        log("   TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID doğru mu?")


if __name__ == "__main__":
    log("="*50)
    log("⚡ MIDAS SCHEDULER BAŞLIYOR")
    log("="*50)

    # 1) Test
    sistem_testi()

    # 2) Zamanlamayı kur — her saat başı :00'da çalış
    schedule.every().hour.at(":00").do(saatlik_tarama)
    log("⏰ Zamanlayıcı kuruldu → Her saat başı (:00)")

    # 3) Hemen bir tarama yap
    log("🚀 Başlangıç taraması yapılıyor...")
    saatlik_tarama()

    # 4) Sonsuz döngü
    log("♾️  Döngüye girildi. Durdurmak için CTRL+C")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log("🛑 Scheduler durduruldu.")
    except Exception as e:
        log(f"❌ KRİTİK HATA: {e}")
        hata_bildirimi(f"Scheduler çöktü: {e}")