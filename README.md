**OmniQuant AI**, borsa piyasalarını otonom olarak tarayan, Yapay Zeka (AI) ve Makine Öğrenmesi (ML) ile yön tahmini yapan kapsamlı bir yatırım terminalidir. Sistem, 800+ günlük tarihsel veriyi analiz ederek teknik formasyonları öğrenir ve yatırımcıya rasyonel gerekçelerle sunar.

---

## 🌟 Key Features (Temel Özellikler)

- **Dual-Market Scanner:** Simultaneously monitors 50 US Tech Giants (NASDAQ/NYSE) and 50 Turkish Blue-Chip Stocks (BIST30/100).
- **Explainable AI (XAI):** Provides natural language reasoning for every "Buy/Sell" signal based on RSI, Volume, and Trend metrics.
- **Hybrid Architecture:**
    - **Interactive Dashboard:** Real-time analysis and manual stock deep-dives via Streamlit.
    - **Autonomous Scheduler:** Runs background tasks to identify opportunities without user intervention.
    - **Telegram Intelligence:** Delivers daily AI-generated reports and alerts directly to your mobile device.
- **Smart Tracking System:** Maintains a history of predictions in `tahmin_takip.json` to monitor AI success rates over time.
- **Live Scrolling Ticker:** Dynamic UI element showing top AI-ranked picks instantly.

---

## 🛠️ Technology Stack (Teknoloji Yığını)

- **AI Engine:** Scikit-Learn (Gradient Boosting Classifier)
- **Data Pipeline:** Yahoo Finance API (yfinance)
- **UI Framework:** Streamlit (Web Dashboard)
- **Messaging:** python-telegram-bot
- **Storage:** JSON-based local database (Cache & Tracking)

---

## 📂 Project Structure (Dosya Yapısı)

- `app.py`: The main web terminal and AI inference engine.
- `telegram_bot.py`: Interface for sending AI reports to Telegram.
- `scheduler.py`: Automation engine that triggers periodic scans.
- `config.py`: Centralized storage for API tokens and system settings.
- `tarama_cache.json`: Performance optimization via local caching.
- `tahmin_takip.json`: Database for tracking AI prediction accuracy.

---

## ⚙️ Installation & Setup (Kurulum)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/OmniQuant-AI.git](https://github.com/YOUR_USERNAME/OmniQuant-AI.git)
   cd OmniQuant-AI
