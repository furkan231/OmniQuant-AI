import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json, os
from datetime import date, datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

OTOMATIK_YENILEME_SN = 3600
st.set_page_config(page_title="Midas Terminal v10.2", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f'<meta http-equiv="refresh" content="{OTOMATIK_YENILEME_SN}">', unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#0b0c0e !important;color:#c9cdd4 !important;font-family:'Syne',sans-serif;}
[data-testid="stHeader"]{background:transparent !important;}[data-testid="stSidebar"]{display:none;}
.block-container{padding:0 !important;max-width:100% !important;}
[data-testid="stTabs"]>div:first-child{background:#0b0c0e;border-bottom:1px solid #1e2028;padding:0 24px;gap:0;}
button[data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace !important;font-size:12px !important;color:#555 !important;letter-spacing:.8px !important;padding:10px 20px !important;background:transparent !important;border:none !important;border-bottom:2px solid transparent !important;}
button[data-baseweb="tab"][aria-selected="true"]{color:#e8c97a !important;border-bottom:2px solid #e8c97a !important;}
[data-testid="stTabPanel"]{background:#0b0c0e;padding:20px 24px;}
[data-testid="stMetric"]{background:#111318 !important;border:1px solid #1e2028 !important;border-radius:6px !important;padding:14px !important;}
[data-testid="stMetricLabel"] p{font-family:'IBM Plex Mono',monospace !important;font-size:10px !important;color:#444 !important;letter-spacing:1px !important;text-transform:uppercase;}
[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace !important;font-size:20px !important;color:#e8c97a !important;}
[data-testid="stButton"] button{font-family:'IBM Plex Mono',monospace !important;font-size:12px !important;background:#111318 !important;color:#e8c97a !important;border:1px solid #2a2c35 !important;border-radius:4px !important;letter-spacing:.8px !important;}
[data-testid="stButton"] button:hover{background:#1a1c22 !important;border-color:#e8c97a !important;}
[data-testid="stButton"] button[kind="primary"]{background:#1a2d10 !important;color:#3fcb7e !important;border-color:#2a4a1e !important;}
[data-testid="stTextInput"] input{font-family:'IBM Plex Mono',monospace !important;font-size:13px !important;background:#111318 !important;color:#e8c97a !important;border:1px solid #2a2c35 !important;border-radius:4px !important;}
[data-testid="stTextInput"] label{font-family:'IBM Plex Mono',monospace !important;font-size:11px !important;color:#555 !important;}
[data-testid="stRadio"] label{font-family:'IBM Plex Mono',monospace !important;font-size:12px !important;color:#888 !important;}
[data-testid="stDataFrame"]{background:#111318 !important;border:1px solid #1e2028 !important;border-radius:6px !important;font-family:'IBM Plex Mono',monospace !important;font-size:12px !important;}
[data-testid="stNumberInput"] input{font-family:'IBM Plex Mono',monospace !important;font-size:13px !important;background:#111318 !important;color:#e8c97a !important;border:1px solid #2a2c35 !important;border-radius:4px !important;}
hr{border-color:#1e2028 !important;}
::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-track{background:#0b0c0e;}::-webkit-scrollbar-thumb{background:#2a2c35;border-radius:3px;}
@keyframes mt_pulse{0%,100%{opacity:1}50%{opacity:.2}}@keyframes mt_scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
</style>""", unsafe_allow_html=True)

ABD = ["PLTR","MU","NVDA","QS","TSLA","AAPL","MSFT","AMZN","GOOGL","META","AMD","AVGO","ARM","SMCI","ASML","INTC","CRM","ADBE","NFLX","UBER","COIN","MSTR","HOOD","PATH","SNOW","PANW","CRWD","ZS","DDOG","NET","ROKU","SHOP","SE","SQ","PYPL","DIS","V","MA","JPM","COST","WMT","GE","CAT","LLY","PFE","JNJ","BAC","XOM","CVX","F"]
TR  = ["THYAO.IS","ASELS.IS","EREGL.IS","AKBNK.IS","YKBNK.IS","ISCTR.IS","GARAN.IS","KCHOL.IS","SAHOL.IS","TUPRS.IS","BIMAS.IS","SISE.IS","FROTO.IS","TOASO.IS","ARCLK.IS","PETKM.IS","TKFEN.IS","ENKAI.IS","PGSUS.IS","TAVHL.IS","KOZAL.IS","KOZAA.IS","MGROS.IS","EKGYO.IS","HALKB.IS","VAKBN.IS","TTKOM.IS","TCELL.IS","DOHOL.IS","KRDMD.IS","HEKTS.IS","SASA.IS","GUBRF.IS","VESBE.IS","VESTL.IS","OYAKC.IS","KONTR.IS","SMRTG.IS","ASTOR.IS","ALFAS.IS","CWENE.IS","ENJSA.IS","BRYAT.IS","ALARK.IS","CIMSA.IS","KONYA.IS","MIATK.IS","REEDR.IS","DOAS.IS","EGEEN.IS"]
TUM = ABD + TR

SEKTORLER = {
    "🔬 Yarı İletken":["NVDA","AMD","AVGO","MU","ASML","ARM","SMCI","INTC"],
    "🤖 Yapay Zeka":  ["PLTR","PATH","SNOW","DDOG","NET","CRWD","PANW","ZS"],
    "💳 Fintech":     ["COIN","MSTR","HOOD","SQ","PYPL","V","MA"],
    "☁️ Bulut/SaaS":  ["CRM","ADBE","SHOP","SE","ROKU"],
    "📱 Büyük Tek.":  ["AAPL","MSFT","AMZN","GOOGL","META","NFLX","TSLA","UBER","DIS"],
    "🏦 Banka-TR":    ["AKBNK.IS","YKBNK.IS","ISCTR.IS","GARAN.IS","HALKB.IS","VAKBN.IS"],
    "✈️ Ulaşım-TR":   ["THYAO.IS","PGSUS.IS","TAVHL.IS"],
    "⚡ Enerji-TR":   ["TUPRS.IS","ENJSA.IS","CWENE.IS","PETKM.IS"],
    "🏗️ Sanayi-TR":   ["ASELS.IS","EREGL.IS","FROTO.IS","TOASO.IS","ARCLK.IS","TKFEN.IS","ENKAI.IS"],
    "🏦 Holding-TR":  ["KCHOL.IS","SAHOL.IS","DOHOL.IS"],
}

TAKIP_DOSYA  = "tahmin_takip.json"
TARAMA_DOSYA = "tarama_cache.json"
PORTFOY_DOSYA= "portfoy.json"

# FIX 2+3: TEK FEATURE LİSTESİ — analiz ve backtest aynıyı kullanır
MODEL_FT = ['SMA_20','EMA_50','RSI_14','MACD','MACD_Signal','MACD_Hist',
            'BB_Width','Price_Pos','ATR_14','Vol_20','R1','R5','R10','R20','VR','Regime']

def mono(t,c="#c9cdd4",s="12px"):
    return f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:{s};color:{c};">{t}</span>'

def takip_yukle():
    if os.path.exists(TAKIP_DOSYA):
        with open(TAKIP_DOSYA,"r",encoding="utf-8") as f: return json.load(f)
    return {}

def takip_kaydet(v):
    with open(TAKIP_DOSYA,"w",encoding="utf-8") as f: json.dump(v,f,ensure_ascii=False,indent=2)

def portfoy_yukle():
    if os.path.exists(PORTFOY_DOSYA):
        with open(PORTFOY_DOSYA,"r",encoding="utf-8") as f: return json.load(f)
    return {}

def portfoy_kaydet(v):
    with open(PORTFOY_DOSYA,"w",encoding="utf-8") as f: json.dump(v,f,ensure_ascii=False,indent=2)

# FIX 1: DISK CACHE — sayfa her açıldığında önce diske bakar
def tarama_cache_yukle():
    if not os.path.exists(TARAMA_DOSYA): return None
    try:
        with open(TARAMA_DOSYA,"r",encoding="utf-8") as f: veri=json.load(f)
        kayit_z = datetime.fromisoformat(veri["zaman"])
        yas = (datetime.now()-kayit_z).total_seconds()
        if yas > 3600: return None
        return veri["sonuclar"], kayit_z, yas
    except: return None

def tarama_cache_kaydet(sonuclar):
    try:
        temiz=[]
        for r in sonuclar:
            rc=dict(r)
            if isinstance(rc.get("kapanis"),pd.Series):
                rc["kapanis"]={str(k):float(v) for k,v in rc["kapanis"].items()}
            temiz.append(rc)
        with open(TARAMA_DOSYA,"w",encoding="utf-8") as f:
            json.dump({"zaman":datetime.now().isoformat(),"sonuclar":temiz},f,ensure_ascii=False,indent=2)
    except: pass

def tarama_cache_onar(sonuclar):
    for r in sonuclar:
        if isinstance(r.get("kapanis"),dict):
            r["kapanis"]=pd.Series(list(r["kapanis"].values()),index=pd.to_datetime(list(r["kapanis"].keys())))
    return sonuclar

def sinyal_hesapla(tahmin,guven,rsi,macd,hx):
    if tahmin==1:
        if guven>0.78 and rsi<50 and macd>0 and hx>1.3: return "GÜÇLÜ AL","#00ff88","#0d2a1a","#1a4a2e"
        elif guven>0.63: return "AL","#3fcb7e","#0d2018","#1a3d26"
        else: return "BEKLE","#e8c97a","#1a1a0d","#2a2a1a"
    else:
        if guven>0.78 and rsi>60 and macd<0 and hx<0.8: return "GÜÇLÜ SAT","#ff3355","#2a0d12","#4a1a22"
        elif guven>0.63: return "SAT","#e05050","#200d0d","#3d1a1a"
        else: return "BEKLE","#e8c97a","#1a1a0d","#2a2a1a"

def risk_hesapla(fiyat,atr,tahmin):
    if tahmin==1: stop=fiyat-atr*1.5; h1=fiyat+atr*2.0; h2=fiyat+atr*3.5
    else: stop=fiyat+atr*1.5; h1=fiyat-atr*2.0; h2=fiyat-atr*3.5
    risk=abs(fiyat-stop); odul=abs(h1-fiyat)
    return {"stop_loss":round(stop,2),"hedef_1":round(h1,2),"hedef_2":round(h2,2),
            "risk_odul":round(odul/risk,2) if risk>0 else 0,
            "stop_yuzde":round((stop-fiyat)/fiyat*100,1),"h1_yuzde":round((h1-fiyat)/fiyat*100,1),"h2_yuzde":round((h2-fiyat)/fiyat*100,1)}

def veri_hazirla(df):
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
    gerekli=['Open','High','Low','Close','Volume']
    for s in gerekli:
        if s not in df.columns: return None
    df=df[gerekli].copy()
    for col in df.columns: df[col]=pd.to_numeric(df[col],errors='coerce')
    df.dropna(subset=['Close','Volume'],inplace=True)
    if len(df)<100: return None
    df['SMA_20']=df['Close'].rolling(20).mean(); df['SMA_50']=df['Close'].rolling(50).mean()
    df['SMA_200']=df['Close'].rolling(200).mean(); df['EMA_50']=df['Close'].ewm(span=50,adjust=False).mean()
    delta=df['Close'].diff(); gain=delta.clip(lower=0).ewm(com=13,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(com=13,adjust=False).mean()
    df['RSI_14']=100-(100/(1+gain/loss.replace(0,np.nan)))
    e12=df['Close'].ewm(span=12,adjust=False).mean(); e26=df['Close'].ewm(span=26,adjust=False).mean()
    df['MACD']=e12-e26; df['MACD_Signal']=df['MACD'].ewm(span=9,adjust=False).mean(); df['MACD_Hist']=df['MACD']-df['MACD_Signal']
    bm=df['Close'].rolling(20).mean(); bs=df['Close'].rolling(20).std()
    df['BB_Upper']=bm+2*bs; df['BB_Lower']=bm-2*bs
    df['BB_Width']=(df['BB_Upper']-df['BB_Lower'])/bm.replace(0,np.nan)
    df['Price_Pos']=(df['Close']-df['BB_Lower'])/(df['BB_Upper']-df['BB_Lower']).replace(0,np.nan)
    hl=df['High']-df['Low']; hc=(df['High']-df['Close'].shift()).abs(); lc=(df['Low']-df['Close'].shift()).abs()
    df['ATR_14']=pd.concat([hl,hc,lc],axis=1).max(axis=1).ewm(com=13,adjust=False).mean()
    df['Vol_20']=df['Close'].pct_change().rolling(20).std()
    df['R1']=df['Close'].pct_change(1); df['R5']=df['Close'].pct_change(5)
    df['R10']=df['Close'].pct_change(10); df['R20']=df['Close'].pct_change(20)
    df['VR']=df['Volume']/df['Volume'].rolling(20).mean().replace(0,np.nan)
    df['Regime']=np.where(df['Close']>df['SMA_200'],1,0)
    df['Target']=(df['Close'].shift(-1)>df['Close']).astype(int)
    df.replace([np.inf,-np.inf],np.nan,inplace=True); df.dropna(inplace=True)
    return df

# FIX 4: DETAYLI AI ANALİZ METNİ
def detayli_yorum(hisse, tahmin, guven, sinyal, rsi, macd, macd_hist, hx, r5, r20, fiyat, atr, bw, pp, regime, risk):
    s=[]
    karar_map={"GÜÇLÜ AL":"Tüm önemli göstergeler aynı anda pozitif sinyal veriyor — nadir görülen güçlü fırsat.","AL":"Teknik veriler toplamda yükselişi destekliyor, birkaç onaylayıcı sinyal mevcut.","GÜÇLÜ SAT":"Birden fazla gösterge aynı anda baskı altında — riskten kaçınma öncelik almalı.","SAT":"Teknik tablo düşüşe işaret ediyor, pozisyon yönetimi kritik.","BEKLE":"Sinyaller karışık, net bir yön oluşmadı — konfirmasyon beklenmeli."}
    s.append(f"📌 {karar_map.get(sinyal,'Veri değerlendiriliyor.')}")
    if rsi<25: s.append(f"⚡ RSI {rsi:.1f} — Aşırı satım (kritik alt sınır). Paniğe kapılmış satıcılar fiyatı gerçek değerin çok altına itmiş. Tarihsel olarak bu seviyelerde güçlü tepki alımı görülüyor.")
    elif rsi<35: s.append(f"📉 RSI {rsi:.1f} — Satım baskısı yoğun ama dip arayışı başlayabilir. Kısa vadeli tepki potansiyeli arttı, trend hâlâ aşağı.")
    elif rsi<45: s.append(f"🔄 RSI {rsi:.1f} — Nörden hafif zayıfa kayıyor. Satıcılar kontrolde ama tükeniyor olabilir.")
    elif rsi<55: s.append(f"⚖️ RSI {rsi:.1f} — Nötr denge bölgesi. Alıcı ve satıcılar güç mücadelesinde.")
    elif rsi<65: s.append(f"📈 RSI {rsi:.1f} — Momentum alıcılar lehine güçleniyor. Trend devam sinyali.")
    elif rsi<75: s.append(f"🔥 RSI {rsi:.1f} — Aşırı alıma yaklaşıyor. Kar satışı riski belirgin.")
    else: s.append(f"🚨 RSI {rsi:.1f} — Aşırı alım kritik sınırda. Kısa vadeli düzeltme neredeyse kaçınılmaz.")
    if macd>0 and macd_hist>0: s.append(f"✅ MACD {macd:.3f} — Sinyal hattının üzerinde + histogram pozitif: çift onaylı yükseliş momentumu. Kurumsal alım devrede olabilir.")
    elif macd>0 and macd_hist<0: s.append(f"⚠️ MACD {macd:.3f} — Pozitif bölgede ama histogram negatife döndü: momentum yavaşlıyor.")
    elif macd<0 and macd_hist<0: s.append(f"❌ MACD {macd:.3f} — Negatif bölgede + histogram aşağı: çift onaylı düşüş baskısı. Satıcılar kontrolde.")
    else: s.append(f"🔄 MACD {macd:.3f} — Negatif bölgede ama histogram pozitife döndü: toparlanma çabası başlıyor.")
    if hx>2.0: s.append(f"🚀 Hacim ortalamanın {hx:.1f} katı — olağanüstü para akışı. Büyük oyuncuların (kurumlar/fonlar) hamlesi olabilir. Güçlü hacim trendi teyit ediyor.")
    elif hx>1.3: s.append(f"📊 Hacim {hx:.1f}x — normalin belirgin üzerinde. Kırılımlar hacimle desteklendiğinde daha güvenilir.")
    elif hx>0.8: s.append(f"📊 Hacim {hx:.1f}x — normal seviye. Büyük oyuncu hareketi yok.")
    else: s.append(f"⚠️ Hacim {hx:.1f}x — normalin altında. Alıcı iştahı zayıf; düşük hacimli hareketler genellikle sürdürülebilir değil.")
    if pp<0.1: s.append(f"📐 Fiyat Bollinger alt bandına çok yakın ({pp:.2f}). Aşırı satım + bant baskısı — sıkışma sonrası tepki formasyonu.")
    elif pp<0.3: s.append(f"📐 Bollinger alt bölgesi ({pp:.2f}). Alıcılar devreye girebilir ama trend onayı beklenmeli.")
    elif pp>0.9: s.append(f"📐 Bollinger üst bandına çok yakın ({pp:.2f}). Bant içine geri çekilme riski yüksek.")
    elif pp>0.7: s.append(f"📐 Bollinger üst bölgesi ({pp:.2f}). Güçlü momentum var ama kar satışına dikkat.")
    else: s.append(f"📐 Bollinger orta bölgesi ({pp:.2f}). Net bir yön baskısı yok.")
    bw_y="Dar bant → güçlü kırılım yaklaşıyor olabilir." if bw<0.15 else ("Çok geniş bant → yüksek volatilite, risk yönetimine dikkat." if bw>0.40 else "Normal volatilite aralığı.")
    s.append(f"↔️ Bollinger genişliği {bw:.2f}: {bw_y}")
    if r5>0.08: s.append(f"🏃 5 günlük getiri: +%{r5*100:.1f} — güçlü kısa vadeli trend. Momentum alıcıları pozisyon artırıyor.")
    elif r5>0.03: s.append(f"📈 5 günlük getiri: +%{r5*100:.1f} — pozitif momentum devam ediyor.")
    elif r5>-0.03: s.append(f"➡️ 5 günlük getiri: %{r5*100:.1f} — yatay seyir.")
    elif r5>-0.08: s.append(f"📉 5 günlük getiri: %{r5*100:.1f} — kısa vadeli satış baskısı mevcut.")
    else: s.append(f"🔴 5 günlük getiri: %{r5*100:.1f} — sert düşüş. Panik satışı olabilir; dip avcıları dikkat.")
    ayl=f"+%{r20*100:.1f} → aylık trend güçlü." if r20>0.05 else (f"%{r20*100:.1f} → aylık baskı devam ediyor." if r20<-0.05 else f"%{r20*100:.1f} → aylık baz nötr.")
    s.append(f"📅 20 günlük getiri: {ayl}")
    s.append("🌳 Piyasa rejimi: BOĞA — Fiyat 200 günlük ortalamanın üzerinde. Yükseliş tahminleri daha güvenilir." if regime==1 else
              "🐻 Piyasa rejimi: AYI — Fiyat 200 günlük ortalamanın altında. Düşüş tahminleri daha güvenilir, riskten sakın.")
    atr_p=(atr/fiyat)*100
    if atr_p>4: s.append(f"⚡ ATR %{atr_p:.1f} — yüksek volatilite. Günlük ${atr:.2f} salınım. Geniş stop, küçük pozisyon gerekli.")
    elif atr_p>2: s.append(f"↕️ ATR %{atr_p:.1f} — orta volatilite. Günlük ${atr:.2f} salınım. Normal risk yönetimi yeterli.")
    else: s.append(f"😴 ATR %{atr_p:.1f} — düşük volatilite. Büyük hareketler için dışsal katalizör (haber/rapor) gerekebilir.")
    rr=risk['risk_odul']
    if rr>=2.5: s.append(f"⚖️ Risk/Ödül 1:{rr:.1f} — mükemmel asimetrik fırsat. Stop: ${risk['stop_loss']:.2f} ({risk['stop_yuzde']:+.1f}%) → Hedef: ${risk['hedef_1']:.2f} ({risk['h1_yuzde']:+.1f}%).")
    elif rr>=1.5: s.append(f"⚖️ Risk/Ödül 1:{rr:.1f} — iyi. Stop: ${risk['stop_loss']:.2f} → Hedef: ${risk['hedef_1']:.2f}. Kabul edilebilir profil.")
    else: s.append(f"⚠️ Risk/Ödül 1:{rr:.1f} — zayıf. Potansiyel kazanç riski karşılamıyor. Daha iyi giriş noktası beklenmeli.")
    if guven>0.78: s.append(f"🤖 AI güven %{guven*100:.1f} — modelin yüksek kesinlikte verdiği tahmin. Bu güven seviyesi tarihsel olarak daha yüksek isabet oranıyla ilişkilendirilmiş.")
    elif guven>0.65: s.append(f"🤖 AI güven %{guven*100:.1f} — yeterli güven. Diğer göstergelerle birlikte değerlendirin.")
    else: s.append(f"🤖 AI güven %{guven*100:.1f} — sınırda güven. Pozisyon boyutunu küçük tutun, onay bekleyin.")
    return s

def analiz(hisse):
    try:
        raw=yf.download(hisse,start="2020-01-01",progress=False,auto_adjust=True)
        if raw is None or raw.empty: return None
        df=veri_hazirla(raw)
        if df is None: return None
        bugun=df.iloc[[-1]].copy(); egitim=df.iloc[:-1].dropna(subset=MODEL_FT+['Target'])
        bugun=bugun.dropna(subset=MODEL_FT)
        if len(egitim)<50 or bugun.empty: return None
        pipe=Pipeline([('sc',StandardScaler()),('clf',GradientBoostingClassifier(n_estimators=150,learning_rate=0.05,max_depth=4,min_samples_leaf=20,subsample=0.8,random_state=42))])
        pipe.fit(egitim[MODEL_FT],egitim['Target'])
        t=pipe.predict(bugun[MODEL_FT])[0]; ih=pipe.predict_proba(bugun[MODEL_FT])[0]
        fp=float(df['Close'].iloc[-1]); rsi=float(df['RSI_14'].iloc[-1]); hx=float(df['VR'].iloc[-1])
        mc=float(df['MACD'].iloc[-1]); mch=float(df['MACD_Hist'].iloc[-1]); atr=float(df['ATR_14'].iloc[-1])
        r5=float(df['R5'].iloc[-1]); r20=float(df['R20'].iloc[-1]); bw=float(df['BB_Width'].iloc[-1])
        pp=float(df['Price_Pos'].iloc[-1]); reg=int(df['Regime'].iloc[-1]); guven=float(ih[1] if t==1 else ih[0])
        sinyal,s_r,s_b,s_bd=sinyal_hesapla(t,guven,rsi,mc,hx); risk=risk_hesapla(fp,atr,t)
        yorum_s=detayli_yorum(hisse,t,guven,sinyal,rsi,mc,mch,hx,r5,r20,fp,atr,bw,pp,reg,risk)
        return {"hisse":hisse,"tahmin":int(t),"guven":guven,"fiyat":fp,"rsi":rsi,"hx":hx,"macd":mc,
                "macd_hist":mch,"atr":atr,"r5":r5,"r20":r20,"bb_width":bw,"price_pos":pp,"regime":reg,
                "yorum":yorum_s[0] if yorum_s else "","yorum_detay":yorum_s,
                "tarih":str(df.index[-1].date()),"kapanis":df['Close'].tail(90),
                "sinyal":sinyal,"sinyal_renk":s_r,"sinyal_bg":s_b,"sinyal_border":s_bd,"risk":risk}
    except Exception as e:
        print(f"Analiz hatası {hisse}: {e}"); return None

def piyasa_rejimi_algila(df):
    df['SMA_200']=df['Close'].rolling(200).mean(); df['Regime']=np.where(df['Close']>df['SMA_200'],1,0); return df

def adaptif_pencere(df,min_p=120,maks_p=250):
    vol=df['Close'].pct_change().rolling(60).std().iloc[-1]
    return min_p if vol>0.03 else (maks_p if vol<0.015 else int(min_p+(maks_p-min_p)*(0.03-vol)/0.015))

def ensemble_model():
    return VotingClassifier([
        ('gbm',GradientBoostingClassifier(n_estimators=120,learning_rate=0.05,max_depth=3,min_samples_leaf=15,subsample=0.8,random_state=42)),
        ('rf',RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=20,random_state=42,n_jobs=-1)),
        ('lr',LogisticRegression(max_iter=1000,random_state=42))
    ],voting='soft',weights=[0.4,0.35,0.25])

def backtest_standart(df,guven_esigi=0.0):
    sonuclar=[]; p=250
    for i in range(max(p,60),len(df)):
        eg=df.iloc[i-p:i].dropna(subset=MODEL_FT+['Target']); te=df.iloc[[i]].dropna(subset=MODEL_FT)
        if len(eg)<100 or te.empty: continue
        try:
            pipe=Pipeline([('sc',StandardScaler()),('clf',GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=3,min_samples_leaf=20,subsample=0.8,random_state=42))])
            pipe.fit(eg[MODEL_FT],eg['Target']); pr=pipe.predict_proba(te[MODEL_FT])[0]
            ta=1 if pr[1]>pr[0] else 0; gu=pr[ta]
            if gu<guven_esigi: continue
            sonuclar.append({"tarih":df.index[i],"tahmin":int(ta),"gercek":int(df['Target'].iloc[i]),"dogru":int(ta)==int(df['Target'].iloc[i]),"fiyat":float(df['Close'].iloc[i]),"guven":gu})
        except: continue
    return sonuclar

def backtest_iyilestirilmis(df,guven_esigi=0.55):
    sonuclar=[]
    for i in range(300,len(df)):
        pb=adaptif_pencere(df.iloc[:i]); mr=int(df['Regime'].iloc[i])
        eg=df.iloc[i-pb:i].copy(); eg=eg[eg['Regime']==mr].dropna(subset=MODEL_FT+['Target'])
        te=df.iloc[[i]].dropna(subset=MODEL_FT)
        if len(eg)<80 or te.empty: continue
        try:
            pipe=Pipeline([('sc',StandardScaler()),('clf',ensemble_model())])
            pipe.fit(eg[MODEL_FT],eg['Target']); pr=pipe.predict_proba(te[MODEL_FT])[0]
            ta=1 if pr[1]>pr[0] else 0; gu=pr[ta]
            if gu<guven_esigi: continue
            gc=int(df['Target'].iloc[i]); atr=df['ATR_14'].iloc[i]; fiy=df['Close'].iloc[i]
            sonuclar.append({"tarih":df.index[i],"tahmin":int(ta),"gercek":gc,"dogru":int(ta)==gc,
                             "fiyat":float(fiy),"guven":gu,"pozisyon_skoru":min(gu*(1/(atr/fiy if fiy>0 else 1)),2.0),"rejim":mr})
        except: continue
    return sonuclar

def backtest_calistir(hisse,test_suresi=120):
    try:
        raw=yf.download(hisse,start="2021-01-01",progress=False,auto_adjust=True)
        if raw is None or raw.empty: return None,None,None
        df=veri_hazirla(raw)
        if df is None: return None,None,None
        df=piyasa_rejimi_algila(df)
        if len(df)<test_suresi+300: return None,None,None
        return backtest_standart(df),backtest_iyilestirilmis(df),df
    except Exception as e:
        print(f"Backtest hatası: {e}"); return None,None,None

def analiz_et(sonuclar,isim="Model"):
    if not sonuclar: return None
    sdf=pd.DataFrame(sonuclar); basari=sdf['dogru'].mean()*100; toplam=len(sdf); dogru=sdf['dogru'].sum()
    precision=precision_score(sdf['gercek'],sdf['tahmin'],zero_division=0)*100
    recall=recall_score(sdf['gercek'],sdf['tahmin'],zero_division=0)*100
    f1=f1_score(sdf['gercek'],sdf['tahmin'],zero_division=0)*100
    yuk=sdf[sdf['tahmin']==1]; dus=sdf[sdf['tahmin']==0]
    ybas=yuk['dogru'].mean()*100 if len(yuk)>0 else 0; dbas=dus['dogru'].mean()*100 if len(dus)>0 else 0
    portfoy=100.0; pg=[100.0]
    for i,row in sdf.iterrows():
        snr=sdf.iloc[i+1]['fiyat'] if i+1<len(sdf) else row['fiyat']*1.001
        getiri=(snr-row['fiyat'])/row['fiyat']
        portfoy*=(1+getiri) if row['tahmin']==1 else (1-getiri*0.5); pg.append(portfoy)
    alt=(sdf['fiyat'].iloc[-1]-sdf['fiyat'].iloc[0])/sdf['fiyat'].iloc[0]*100 if len(sdf)>0 else 0
    return {"isim":isim,"basari":round(basari,1),"toplam":toplam,"dogru":int(dogru),"yanlis":toplam-int(dogru),
            "precision":round(precision,1),"recall":round(recall,1),"f1":round(f1,1),
            "yukselis_basari":round(ybas,1),"dusus_basari":round(dbas,1),
            "yukselis_sayisi":len(yuk),"dusus_sayisi":len(dus),
            "alt_getiri":round(alt,1),"model_getiri":round(portfoy-100,1),"portfoy_gecmisi":pg,"sdf":sdf}

def tam_tarama():
    return sorted([r for h in TUM if (r:=analiz(h))],key=lambda x:x['guven'],reverse=True)

# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: SAYFA AÇILIŞINDA CACHE KONTROL
# Disk'te taze cache varsa yükle → tarama yapma.
# Yoksa tara ve yaz.
# ══════════════════════════════════════════════════════════════════════════════
cache_sonuc=tarama_cache_yukle()
if cache_sonuc is not None:
    ham,cache_z,yas_sn=cache_sonuc
    tum=tarama_cache_onar(ham)
    tarama_yapildi=False; cache_kalan=int((3600-yas_sn)//60)
    cache_z_str=cache_z.strftime("%H:%M")
else:
    with st.spinner("⚡ Midas Terminal — 100 hisse analiz ediliyor..."):
        tum=tam_tarama()
    tarama_cache_kaydet(tum)
    tarama_yapildi=True; cache_kalan=60; cache_z_str=datetime.now().strftime("%H:%M")

yuk=[r for r in tum if r['tahmin']==1]; dus=[r for r in tum if r['tahmin']==0]

# ÜST BAR
durum_ic="🟢 Az önce tarandı" if tarama_yapildi else f"🟡 Cache · {cache_z_str} · {cache_kalan}dk kaldı"
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 24px;background:#0b0c0e;border-bottom:1px solid #1e2028;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:#e8c97a;letter-spacing:1px;">⚡ MIDAS <span style="color:#e8c97a;">TERMINAL</span><span style="font-size:11px;color:#444;margin-left:10px;">v10.2</span></div>
  <div style="display:flex;gap:16px;align-items:center;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#3fcb7e;margin-right:5px;vertical-align:middle;animation:mt_pulse 2s infinite;"></span>CANLI · BİST + NYSE</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;">{durum_ic}</span>
    <span style="background:#111318;border:1px solid #1e2028;border-radius:3px;padding:3px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;"><span style="color:#3fcb7e;">{len(yuk)} ▲</span><span style="color:#555;margin:0 5px;">·</span><span style="color:#e05050;">{len(dus)} ▼</span><span style="color:#555;margin:0 5px;">·</span><span style="color:#e8c97a;">{len(tum)} hisse</span></span>
  </div>
</div>""", unsafe_allow_html=True)

parcalar=["".join([f'<span style="margin-right:48px;"><b style="color:#e8c97a;font-family:\'IBM Plex Mono\',monospace;font-size:12px;">{r["hisse"]}</b> <span style="color:#555;font-family:\'IBM Plex Mono\',monospace;font-size:11px;">${r["fiyat"]:.2f}</span> <span style="color:{"#3fcb7e" if r["tahmin"]==1 else "#e05050"};font-family:\'IBM Plex Mono\',monospace;font-size:11px;">{r.get("sinyal","—")} %{r["guven"]*100:.0f}</span></span>' for r in tum[:20]])]
ic="".join(parcalar)
st.markdown(f'<div style="background:#0d0f12;border-bottom:1px solid #1a1c22;padding:8px 0;overflow:hidden;white-space:nowrap;"><div style="display:inline-block;animation:mt_scroll 50s linear infinite;">{ic}&nbsp;&nbsp;&nbsp;{ic}</div></div>',unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs(["/ TEKİL ANALİZ","/ 100 HİSSE TARAMASI","/ SEKTÖR HARİTASI","/ PORTFÖY TAKİP","/ BACKTEST v2","/ DOĞRULUK TAKİBİ","/ GEÇMİŞ PERFORMANS"])

# ══ SEKME 1 ══════════════════════════════════════════════════════════════════
with tab1:
    cA,cB=st.columns([1,3],gap="large")
    with cA:
        st.markdown(mono("/ HİSSE KODU","#444","10px"),unsafe_allow_html=True)
        secilen=st.text_input("",value="PLTR",placeholder="PLTR, THYAO.IS...",key="tekil_input").upper().strip()
        ab=st.button("ANALIZ ET →",use_container_width=True,type="primary",key="tekil_btn")
        st.markdown("""<div style="margin-top:20px;background:#111318;border:1px solid #1e2028;border-radius:6px;padding:16px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1.2px;margin-bottom:12px;">TAHMİN PENCERESİ</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666;line-height:1.8;">Tahmin üretildi<br><span style="color:#e8c97a;font-size:13px;">17:30 (kapanış)</span></div>
          <div style="height:1px;background:#1e2028;margin:10px 0;"></div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666;line-height:1.8;">Ana pencere<br><span style="color:#3fcb7e;font-size:13px;">10:00 – 17:00</span></div>
          <div style="height:1px;background:#1e2028;margin:10px 0;"></div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666;line-height:1.8;">Doğrulama<br><span style="color:#e8c97a;font-size:13px;">17:30 BİST · 23:00 NYSE</span></div>
        </div>""",unsafe_allow_html=True)

    with cB:
        sa=st.empty()
        if ab and secilen:
            with st.spinner(f"🔍 {secilen} analiz ediliyor..."):
                s=analiz(secilen)
            if s is None:
                sa.error(f"❌ {secilen} için veri alınamadı.")
            else:
                with sa.container():
                    c1,c2,c3,c4,c5=st.columns(5)
                    c1.metric("SON KAPANIS",f"${s['fiyat']:.2f}"); c2.metric("RSI (14)",f"{s['rsi']:.1f}")
                    c3.metric("ATR (14)",f"{s['atr']:.2f}"); c4.metric("HACIM",f"{s['hx']:.2f}x"); c5.metric("5G GETİRİ",f"%{s['r5']*100:.1f}")
                    st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
                    sn=s['sinyal']; sr=s['sinyal_renk']; sb=s['sinyal_bg']; sbd=s['sinyal_border']
                    st.markdown(f"""<div style="background:{sb};border:1px solid {sbd};border-radius:6px;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                      <div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1.5px;margin-bottom:4px;">AI SİNYAL — {s['hisse']}</div>
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:26px;color:{sr};font-weight:600;letter-spacing:2px;">{sn}</div></div>
                      <div style="text-align:right;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;margin-bottom:4px;">AI GÜVENİ</div>
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;color:{sr};">%{s['guven']*100:.1f}</div></div>
                    </div>""",unsafe_allow_html=True)
                    rk=s['risk']
                    st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
                      <div style="background:#1a0d0d;border:1px solid #3d1a1a;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#e05050;letter-spacing:1px;margin-bottom:4px;">🛑 STOP LOSS</div><div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:#e05050;">${rk['stop_loss']:.2f}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:3px;">{rk['stop_yuzde']:+.1f}%</div></div>
                      <div style="background:#0d1a0d;border:1px solid #1a3d1a;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3fcb7e;letter-spacing:1px;margin-bottom:4px;">🎯 HEDEF 1</div><div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:#3fcb7e;">${rk['hedef_1']:.2f}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:3px;">{rk['h1_yuzde']:+.1f}%</div></div>
                      <div style="background:#0d1a0d;border:1px solid #1a4a1a;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#00ff88;letter-spacing:1px;margin-bottom:4px;">🚀 HEDEF 2</div><div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:#00ff88;">${rk['hedef_2']:.2f}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:3px;">{rk['h2_yuzde']:+.1f}%</div></div>
                      <div style="background:#111318;border:1px solid #2a2c35;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#888;letter-spacing:1px;margin-bottom:4px;">⚖️ RİSK/ÖDÜL</div><div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:{'#3fcb7e' if rk['risk_odul']>=1.5 else '#e8c97a'};">1 : {rk['risk_odul']:.1f}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:3px;">{'✓ İyi' if rk['risk_odul']>=1.5 else '⚠ Zayıf'}</div></div>
                    </div>""",unsafe_allow_html=True)
                    rc="#3fcb7e" if s['rsi']<30 else ("#e05050" if s['rsi']>70 else "#888")
                    mc2="#3fcb7e" if s['macd']>0 else "#e05050"; rc5="#3fcb7e" if s['r5']>0 else "#e05050"
                    st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
                      <div style="background:#111318;border:1px solid #1e2028;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1px;margin-bottom:4px;">RSI (14)</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:{rc};">{s['rsi']:.1f}</div><div style="height:3px;background:#1e2028;border-radius:2px;margin-top:8px;"><div style="width:{min(s['rsi'],100):.0f}%;height:100%;background:{rc};border-radius:2px;"></div></div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;margin-top:4px;">{'Aşırı Satım' if s['rsi']<30 else ('Aşırı Alım' if s['rsi']>70 else 'Normal')}</div></div>
                      <div style="background:#111318;border:1px solid #1e2028;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1px;margin-bottom:4px;">MACD</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:{mc2};">{s['macd']:.3f}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555;margin-top:6px;">{'▲ Pozitif momentum' if s['macd']>0 else '▼ Negatif momentum'}</div></div>
                      <div style="background:#111318;border:1px solid #1e2028;border-radius:5px;padding:12px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1px;margin-bottom:4px;">5G GETİRİ</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:{rc5};">%{s['r5']*100:.1f}</div><div style="height:3px;background:#1e2028;border-radius:2px;margin-top:8px;"><div style="width:{min(abs(s['r5'])*200,100):.0f}%;height:100%;background:{rc5};border-radius:2px;"></div></div></div>
                    </div>""",unsafe_allow_html=True)

                    # FIX 4: Detaylı analiz
                    st.markdown(f'<div style="background:{sb};border:1px solid {sbd};border-left:3px solid {sr};border-radius:5px;padding:16px 18px;margin-bottom:12px;"><div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:{sr};letter-spacing:1.2px;margin-bottom:12px;">🧠 DETAYLI AI ANALİZİ — {s["hisse"]}</div>',unsafe_allow_html=True)
                    for satir in s.get('yorum_detay',[]):
                        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#888;line-height:1.7;padding:6px 0;border-bottom:1px solid #1a1c22;">{satir}</div>',unsafe_allow_html=True)
                    st.markdown("</div>",unsafe_allow_html=True)
                    st.markdown(mono("/ SON 90 GÜN FİYAT","#444","10px"),unsafe_allow_html=True)
                    st.line_chart(pd.DataFrame({"Fiyat":s['kapanis'].values},index=s['kapanis'].index),height=180,use_container_width=True)
        else:
            sa.markdown("""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:40px;text-align:center;margin-top:20px;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;color:#e8c97a;margin-bottom:10px;">⚡</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#444;margin-bottom:8px;">Hisse kodu girin ve analiz edin</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#333;">PLTR · NVDA · THYAO.IS · GARAN.IS</div>
            </div>""",unsafe_allow_html=True)

# ══ SEKME 2 ══════════════════════════════════════════════════════════════════
with tab2:
    cb,cbtn=st.columns([4,1])
    with cb:
        dr="#3fcb7e" if tarama_yapildi else "#e8c97a"
        dm="Az önce tarandı · 1 saat diskde saklanacak" if tarama_yapildi else f"Diskten yüklendi · {cache_kalan} dakika sonra yenilenecek"
        st.markdown(f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:{dr};">● {dm}</span>',unsafe_allow_html=True)
    with cbtn:
        if st.button("🔄 YENİLE",key="yenile_btn",use_container_width=True):
            if os.path.exists(TARAMA_DOSYA): os.remove(TARAMA_DOSYA)
            st.rerun()
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    f1c,f2c=st.columns([2,2])
    with f1c: pazar=st.radio("PAZAR",["Hepsi","Sadece ABD","Sadece Türkiye"],horizontal=True,key="pazar_f")
    with f2c: sf=st.radio("SİNYAL",["Hepsi","GÜÇLÜ AL","AL","BEKLE","SAT","GÜÇLÜ SAT"],horizontal=True,key="sinyal_f")
    fl=tum
    if pazar=="Sadece ABD": fl=[r for r in fl if not r['hisse'].endswith('.IS')]
    elif pazar=="Sadece Türkiye": fl=[r for r in fl if r['hisse'].endswith('.IS')]
    if sf!="Hepsi": fl=[r for r in fl if r.get('sinyal')==sf]
    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("TARANAN",len(fl)); m2.metric("GÜÇLÜ AL",sum(1 for r in fl if r.get('sinyal')=="GÜÇLÜ AL"))
    m3.metric("AL",sum(1 for r in fl if r.get('sinyal')=="AL")); m4.metric("SAT",sum(1 for r in fl if r.get('sinyal') in ["SAT","GÜÇLÜ SAT"]))
    m5.metric("ORT. GÜVEN",f"%{sum(r['guven'] for r in fl)/max(len(fl),1)*100:.1f}")
    st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
    if fl:
        st.dataframe(pd.DataFrame([{"Hisse":r['hisse'],"Sinyal":r.get('sinyal','—'),"Güven %":round(r['guven']*100,1),"Fiyat":round(r['fiyat'],2),"Stop Loss":r['risk']['stop_loss'],"Hedef 1":r['risk']['hedef_1'],"Hedef 2":r['risk']['hedef_2'],"R/R":r['risk']['risk_odul'],"RSI":round(r['rsi'],1),"Hacim x":round(r['hx'],2),"5G %":round(r['r5']*100,1),"AI Özet":r['yorum'],"Tarih":r['tarih']} for r in fl]),use_container_width=True,height=520)

# ══ SEKME 3 ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(mono("/ SEKTÖR BAZLI AI ANALİZİ","#444","10px"),unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
    hm={r['hisse']:r for r in tum}; so=[]
    for sek,his in SEKTORLER.items():
        bl=[hm[h] for h in his if h in hm]
        if not bl: continue
        yuk2=sum(1 for r in bl if r['tahmin']==1); og=sum(r['guven'] for r in bl)/len(bl)
        or2=sum(r['rsi'] for r in bl)/len(bl); or5=sum(r['r5'] for r in bl)/len(bl)
        sk=(yuk2/len(bl))*og*100; du="🟢 GÜÇLÜ" if sk>65 else ("🟡 NÖTR" if sk>50 else "🔴 ZAYIF")
        so.append({"sektor":sek,"hisse_say":len(bl),"yukselis":yuk2,"dusus":len(bl)-yuk2,"ort_guven":og,"ort_rsi":or2,"ort_r5":or5,"skor":sk,"durum":du,"hisseler":bl})
    so.sort(key=lambda x:x['skor'],reverse=True)
    cols=st.columns(2)
    for idx,s in enumerate(so):
        with cols[idx%2]:
            bg="#0d1a0d" if s['skor']>65 else ("#1a1a0d" if s['skor']>50 else "#1a0d0d")
            bdr="#1a3d1a" if s['skor']>65 else ("#2a2a1a" if s['skor']>50 else "#3d1a1a")
            ren="#3fcb7e" if s['skor']>65 else ("#e8c97a" if s['skor']>50 else "#e05050")
            hl=" · ".join([f'<span style="color:{"#3fcb7e" if r["tahmin"]==1 else "#e05050"}">{r["hisse"]}</span>' for r in s['hisseler']])
            st.markdown(f"""<div style="background:{bg};border:1px solid {bdr};border-radius:6px;padding:16px;margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#c9cdd4;">{s['sektor']}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{ren};">{s['durum']}</div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:10px;">
                <div style="text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;">HİSSE</div><div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#c9cdd4;">{s['hisse_say']}</div></div>
                <div style="text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;">YÜKSELİŞ</div><div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#3fcb7e;">{s['yukselis']}</div></div>
                <div style="text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;">ORT RSI</div><div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#e8c97a;">{s['ort_rsi']:.0f}</div></div>
                <div style="text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;">5G GETİRİ</div><div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:{'#3fcb7e' if s['ort_r5']>0 else '#e05050'};">%{s['ort_r5']*100:.1f}</div></div>
              </div>
              <div style="height:3px;background:#1e2028;border-radius:2px;margin-bottom:8px;"><div style="width:{min(s['skor'],100):.0f}%;height:100%;background:{ren};border-radius:2px;"></div></div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.8;">{hl}</div>
            </div>""",unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Sektör":s['sektor'],"Durum":s['durum'],"Skor":round(s['skor'],1),"Hisse":s['hisse_say'],"▲":s['yukselis'],"▼":s['dusus'],"Ort. Güven %":round(s['ort_guven']*100,1),"Ort. RSI":round(s['ort_rsi'],1),"5G %":round(s['ort_r5']*100,1)} for s in so]),use_container_width=True,height=380)

# ══ SEKME 4 ══════════════════════════════════════════════════════════════════
with tab4:
    portfoy=portfoy_yukle(); hm={r['hisse']:r for r in tum}
    ce,cl=st.columns([1,2],gap="large")
    with ce:
        st.markdown(mono("/ HİSSE EKLE","#444","10px"),unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
        ph=st.text_input("Hisse Kodu",placeholder="NVDA",key="p_hisse").upper().strip()
        pa=st.number_input("Adet",min_value=1,value=10,key="p_adet")
        pm=st.number_input("Alış Fiyatı ($)",min_value=0.01,value=100.0,format="%.2f",key="p_maliyet")
        if st.button("➕ PORTFÖYE EKLE",use_container_width=True,type="primary",key="p_ekle"):
            if ph: portfoy[ph]={"adet":pa,"maliyet":pm}; portfoy_kaydet(portfoy); st.success(f"{ph} eklendi."); st.rerun()
        st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
        if portfoy:
            ss=st.selectbox("Hisse Sil",["—"]+list(portfoy.keys()),key="p_sil_sec")
            if st.button("🗑️ SİL",use_container_width=True,key="p_sil"):
                if ss!="—": portfoy.pop(ss,None); portfoy_kaydet(portfoy); st.rerun()
    with cl:
        if not portfoy:
            st.markdown("""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:40px;text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#444;">Portföy boş · Sol taraftan hisse ekleyin</div></div>""",unsafe_allow_html=True)
        else:
            satirlar=[]; tm2=0; td2=0
            for h2,b in portfoy.items():
                ad=b['adet']; ma=b['maliyet']
                if h2 in hm: gu2=hm[h2]['fiyat']; si2=hm[h2].get('sinyal','—'); gv2=hm[h2]['guven']
                else:
                    try:
                        tmp2=yf.download(h2,period="1d",progress=False,auto_adjust=True)
                        gu2=float(tmp2['Close'].iloc[-1]) if not tmp2.empty else ma
                    except: gu2=ma
                    si2="—"; gv2=0
                mt2=ad*ma; dt2=ad*gu2; kz2=dt2-mt2; ky2=(kz2/mt2*100) if mt2>0 else 0
                tm2+=mt2; td2+=dt2
                satirlar.append({"Hisse":h2,"Adet":ad,"Alış $":round(ma,2),"Güncel $":round(gu2,2),"Maliyet":round(mt2,2),"Değer":round(dt2,2),"K/Z $":round(kz2,2),"K/Z %":round(ky2,1),"AI Sinyal":si2,"Güven %":round(gv2*100,1)})
            tkz=td2-tm2; ty=(tkz/tm2*100) if tm2>0 else 0
            p1,p2,p3,p4=st.columns(4)
            p1.metric("TOPLAM MALİYET",f"${tm2:,.0f}"); p2.metric("GÜNCEL DEĞER",f"${td2:,.0f}")
            p3.metric("K/Z",f"${tkz:+,.0f}"); p4.metric("K/Z %",f"%{ty:+.1f}")
            st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(satirlar),use_container_width=True,height=380)

# ══ SEKME 5 ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:12px 16px;margin-bottom:16px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#e8c97a;">
      ✨ BACKTEST v2 — Standart vs İyileştirilmiş · Her iki model aynı 16 feature kullanıyor (FIX 3: tutarlı karşılaştırma)
    </div>""",unsafe_allow_html=True)
    bt1,bt2=st.columns([1,3],gap="large")
    with bt1:
        bth=st.text_input("Hisse Kodu",value="NVDA",key="bt_hisse").upper().strip()
        btg=st.selectbox("Test Süresi",[60,90,120,180],index=2,key="bt_gun")
        btb=st.button("▶ İKİ MODELİ KARŞILAŞTIR",use_container_width=True,type="primary",key="bt_btn")
    with bt2:
        if btb and bth:
            with st.spinner(f"⏳ {bth} — iki model çalıştırılıyor ({btg} gün)..."):
                sts,iys,dfh=backtest_calistir(bth,test_suresi=btg)
            if sts is None:
                st.error("Yeterli veri yok.")
            else:
                sta=analiz_et(sts,"Standart GBM"); iya=analiz_et(iys,"İyileştirilmiş v2")
                if sta and iya:
                    fark=iya['basari']-sta['basari']; fr="#3fcb7e" if fark>0 else "#e05050"
                    cs,ci,cf=st.columns(3)
                    with cs: st.markdown(f"""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:16px;text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666;margin-bottom:8px;">STANDART GBM</div><div style="font-family:'IBM Plex Mono',monospace;font-size:32px;color:#e8c97a;">%{sta['basari']}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:6px;">{sta['dogru']} ✓ / {sta['yanlis']} ✗ ({sta['toplam']})</div></div>""",unsafe_allow_html=True)
                    with ci: st.markdown(f"""<div style="background:#0d1a0d;border:1px solid #1a3d1a;border-radius:6px;padding:16px;text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#3fcb7e;margin-bottom:8px;">✨ İYİLEŞTİRİLMİŞ v2</div><div style="font-family:'IBM Plex Mono',monospace;font-size:32px;color:#3fcb7e;">%{iya['basari']}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:6px;">{iya['dogru']} ✓ / {iya['yanlis']} ✗ ({iya['toplam']})</div></div>""",unsafe_allow_html=True)
                    with cf: st.markdown(f"""<div style="background:#111318;border:1px solid #2a2c35;border-radius:6px;padding:16px;text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#666;margin-bottom:8px;">İYİLEŞME</div><div style="font-family:'IBM Plex Mono',monospace;font-size:32px;color:{fr};">{fark:+.1f}%</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555;margin-top:6px;">Filtreden geçen: %{(100*iya['toplam']/max(sta['toplam'],1)):.0f}</div></div>""",unsafe_allow_html=True)
                    st.markdown("<div style='height:14px'></div>",unsafe_allow_html=True)
                    cm1,cm2,cm3,cm4=st.columns(4)
                    cm1.metric("PRECISION (St)",f"%{sta['precision']}",delta=f"{iya['precision']-sta['precision']:+.1f}%")
                    cm2.metric("RECALL (St)",f"%{sta['recall']}",delta=f"{iya['recall']-sta['recall']:+.1f}%")
                    cm3.metric("F1 (St)",f"%{sta['f1']}",delta=f"{iya['f1']-sta['f1']:+.1f}%")
                    cm4.metric("AL&TUT GETİRİ",f"%{sta['alt_getiri']:+.1f}")
                    st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
                    y1,y2=st.columns(2)
                    with y1: st.markdown(f"""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:14px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1px;margin-bottom:10px;">YÜKSELİŞ TAHMİNİ BAŞARISI</div><div style="display:flex;justify-content:space-between;"><div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#666;">Standart</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:#e8c97a;">%{sta['yukselis_basari']} ({sta['yukselis_sayisi']})</div></div><div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3fcb7e;">İyileştirilmiş</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:#3fcb7e;">%{iya['yukselis_basari']} ({iya['yukselis_sayisi']})</div></div></div></div>""",unsafe_allow_html=True)
                    with y2: st.markdown(f"""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:14px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#444;letter-spacing:1px;margin-bottom:10px;">DÜŞÜŞ TAHMİNİ BAŞARISI</div><div style="display:flex;justify-content:space-between;"><div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#666;">Standart</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:#e8c97a;">%{sta['dusus_basari']} ({sta['dusus_sayisi']})</div></div><div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3fcb7e;">İyileştirilmiş</div><div style="font-family:'IBM Plex Mono',monospace;font-size:18px;color:#3fcb7e;">%{iya['dusus_basari']} ({iya['dusus_sayisi']})</div></div></div></div>""",unsafe_allow_html=True)
                    st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
                    st.markdown(mono("/ KÜMÜLATİF BAŞARI GRAFİĞİ","#444","10px"),unsafe_allow_html=True)
                    scu=sta['sdf']['dogru'].expanding().mean()*100; icu=iya['sdf']['dogru'].expanding().mean()*100
                    gdf=pd.DataFrame({"Standart GBM":scu.values},index=sta['sdf']['tarih'])
                    idf=pd.DataFrame({"İyileştirilmiş v2":icu.values},index=iya['sdf']['tarih'])
                    grf=gdf.join(idf,how='outer'); grf['%50 Baz']=50
                    st.line_chart(grf,height=220,use_container_width=True,color=["#e8c97a","#3fcb7e","#333"])
                    yr="#3fcb7e" if fark>2 else ("#e8c97a" if fark>=-1 else "#e05050")
                    yt=f"✅ İyileştirmeler çalışıyor! Başarı {fark:+.1f}% arttı." if fark>2 else (f"⚡ Benzer sonuç. Farklı hisseler deneyin." if fark>=-1 else f"⚠️ Bu hisse için iyileştirmeler çalışmadı.")
                    st.markdown(f"""<div style="background:#111318;border:1px solid #1e2028;border-left:3px solid {yr};border-radius:5px;padding:12px 16px;margin-top:8px;"><div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{yr};">{yt}</div><div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555;margin-top:4px;">💡 NVDA · TSLA · PLTR · AMD gibi volatil hisselerde daha iyi çalışır.</div></div>""",unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:40px;text-align:center;"><div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#444;">Hisse kodu girin ve iki modeli karşılaştırın</div><div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#333;margin-top:8px;">Önerilen: NVDA · TSLA · PLTR · AMD · THYAO.IS</div></div>""",unsafe_allow_html=True)

# ══ SEKME 6 ══════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""<div style="background:#111318;border:1px solid #1e2028;border-radius:6px;padding:14px 18px;margin-bottom:18px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#666;line-height:2;"><span style="color:#e8c97a;">Kullanım:</span> Sabah → <span style="color:#c9cdd4;">BUGÜNÜ KAYDET</span> · Kapanış sonrası → <span style="color:#c9cdd4;">SONUÇLARI GÜNCELLE</span></div>""",unsafe_allow_html=True)
    tk=takip_yukle(); bs=str(date.today())
    ck,cg,_=st.columns([1,1,2])
    with ck:
        if st.button("💾  BUGÜNÜ KAYDET",use_container_width=True,key="kaydet_btn"):
            if bs in tk: st.warning("Bugün zaten kaydedilmiş.")
            else:
                kl=[{"hisse":r['hisse'],"tahmin":r['tahmin'],"sinyal":r.get('sinyal','—'),"guven":round(r['guven'],4),"fiyat_sabah":r['fiyat'],"tarih":bs,"sonuc":None,"fiyat_aksam":None} for r in tum]
                tk[bs]=kl; takip_kaydet(tk); st.success(f"{len(kl)} tahmin kaydedildi.")
    with cg:
        if st.button("🔄  SONUÇLARI GÜNCELLE",use_container_width=True,key="guncelle_btn"):
            if bs not in tk: st.warning("Önce kaydet butonuna bas.")
            else:
                g=0; bar2=st.progress(0); kl=tk[bs]
                for i,k in enumerate(kl):
                    try:
                        tmp=yf.download(k['hisse'],period="2d",progress=False,auto_adjust=True)
                        if isinstance(tmp.columns,pd.MultiIndex): tmp.columns=tmp.columns.droplevel(1)
                        if len(tmp)>=2:
                            d2=float(tmp['Close'].iloc[-2]); b2=float(tmp['Close'].iloc[-1])
                            k['fiyat_aksam']=round(b2,2); k['sonuc']="dogru" if (1 if b2>d2 else 0)==k['tahmin'] else "yanlis"; g+=1
                    except: pass
                    bar2.progress((i+1)/len(kl))
                takip_kaydet(tk); st.success(f"{g} hisse güncellendi.")
    st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
    if bs in tk:
        kl=tk[bs]; tam=[k for k in kl if k['sonuc'] is not None]
        if tam:
            dogru=sum(1 for k in tam if k['sonuc']=="dogru")
            b1,b2,b3,b4=st.columns(4)
            b1.metric("TAMAMLANAN",len(tam)); b2.metric("✓ DOĞRU",dogru); b3.metric("✗ YANLIŞ",len(tam)-dogru); b4.metric("BAŞARI",f"%{dogru/len(tam)*100:.1f}")
            st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        st.markdown(mono(f"/ {bs} TAHMİNLERİ","#444","10px"),unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Hisse":k['hisse'],"Sinyal":k.get('sinyal','—'),"Tahmin":"▲" if k['tahmin']==1 else "▼","Güven %":round(k['guven']*100,1),"Sabah $":k['fiyat_sabah'],"Akşam $":k['fiyat_aksam'] if k['fiyat_aksam'] else "—","Sonuç":"✓ Doğru" if k['sonuc']=="dogru" else ("✗ Yanlış" if k['sonuc']=="yanlis" else "⏳ Bekl.")} for k in kl]),use_container_width=True,height=420)
    else:
        st.info("Bugün için henüz kayıt yok.")

# ══ SEKME 7 ══════════════════════════════════════════════════════════════════
with tab7:
    tk=takip_yukle()
    if not tk: st.info("Henüz geçmiş veri yok.")
    else:
        oz=[]
        for tarih,kl in sorted(tk.items()):
            tam=[k for k in kl if k['sonuc'] is not None]
            if not tam: continue
            dogru=sum(1 for k in tam if k['sonuc']=="dogru")
            oz.append({"Tarih":tarih,"Taranan":len(kl),"Tamamlanan":len(tam),"✓ Doğru":dogru,"✗ Yanlış":len(tam)-dogru,"Başarı %":round(dogru/len(tam)*100,1)})
        if oz:
            tt=sum(r['Tamamlanan'] for r in oz); td=sum(r['✓ Doğru'] for r in oz)
            p1,p2,p3,p4=st.columns(4)
            p1.metric("TOPLAM TAHMİN",tt); p2.metric("TOPLAM DOĞRU",td); p3.metric("TOPLAM YANLIŞ",tt-td); p4.metric("GENEL BAŞARI",f"%{td/tt*100:.1f}" if tt else "%0")
            st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
            st.markdown(mono("/ GÜNLÜK ÖZET","#444","10px"),unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(oz),use_container_width=True)
            if len(oz)>1:
                st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
                st.markdown(mono("/ BAŞARI GRAFİĞİ","#444","10px"),unsafe_allow_html=True)
                st.line_chart(pd.DataFrame({"Başarı %":[r['Başarı %'] for r in oz]},index=[r['Tarih'] for r in oz]),height=220)
        else: st.info("Henüz sonuçlandırılmış tahmin yok.")
