# 💻 Sales Forecasting Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mr-adonis-jimenez-chromebook-sales-forecast-dashboard.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/mr-adonis-jimenez/chromebook-sales-forecast-dashboard)

An interactive **sales forecasting dashboard** developed using Python and Streamlit.  
The app forecasts future sales for multiple products using the **Prophet** model, supports **auto-refresh retraining**, and integrates seamlessly with **Google Drive (local sync)**, **OneDrive**, **Dropbox**, **Amazon Drive**, **Mega**, **Nextcloud**, **Seafile**, or **Syncthing**

---

## 🧠 Overview

This project demonstrates how to transform a Chromebook into a full data science workstation using Linux (Beta).  
It reads real or sample sales data from a synced Google Drive folder, visualizes historical performance, and generates sales forecasts for selected products — all offline, directly from ChromeOS.

---

## 🌟 Features

✅ **Multi-Product Forecasting** — switch between product lines with dynamic Prophet models  
✅ **Auto-Retrain Every 7 Days** — background scheduler keeps forecasts fresh  
✅ **Offline Google Drive Integration** — reads data locally from Drive sync folder  
✅ **Interactive Visuals** — Plotly charts for sales and forecasts  
✅ **Export Results** — one-click CSV download of predictions  
✅ **100% Chromebook Compatible** — runs inside Linux (Beta) environment  

---

## 🧱 Tech Stack

| Tool | Purpose |
|------|----------|
| **Python 3** | Core programming language |
| **Streamlit** | Web dashboard framework |
| **Prophet** | Forecasting and time series modeling |
| **Plotly** | Interactive data visualization |
| **pandas** | Data wrangling and manipulation |
| **APScheduler** | Automatic retraining scheduler |

---

## 🚀 Live Demo

🔗 **[Launch App on Streamlit Cloud](https://mr-adonis-jimenez-chromebook-sales-forecast-dashboard.streamlit.app)**

If the live demo doesn’t load, clone and run it locally (instructions below).

---

## ⚙️ Run Locally on Chromebook

1. **Enable Linux (Beta)** on ChromeOS  
   → Settings → Developers → Turn on Linux.

2. **Clone this repository**
   ```bash
   git clone https://github.com/mr-adonis-jimenez/chromebook-sales-forecast-dashboard.git
   cd chromebook-sales-forecast-dashboard
