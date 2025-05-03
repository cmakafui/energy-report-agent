# 🏠 Energy Rating Report Generator

A Streamlit web app for analyzing building energy efficiency using RGB and thermal image pairs, powered by [Agno](https://github.com/agno-ai/agno) and Google Gemini AI.

---

## 🚀 Overview

This app lets users upload paired RGB and thermal images of buildings to identify:

- 🔥 Thermal anomalies
- 🧱 Insulation issues
- 💨 Air leaks
- 💧 Moisture problems

It then generates:

- ✅ Structured per-image analysis
- 📊 A final energy efficiency report with EU-style energy rating (A–G)
- 💡 Priority recommendations with cost and savings estimates

---

## ✨ Features

- 📂 Upload & pair multiple RGB + thermal images
- 🧠 AI-powered analysis with structured markdown outputs
- ⚙️ Customizable building metadata input (year, type, location)
- 📉 Issue severity classification (minor/moderate/severe)
- 📃 Final report generation with rating, cost breakdown, and improvements
- 📥 One-click markdown report download
- 💅 Elegant UI with custom CSS styling

---

## 🧰 Tech Stack

| Layer    | Tech          |
| -------- | ------------- |
| Frontend | Streamlit     |
| AI Agent | Agno + Gemini |
| Styling  | Custom CSS    |
| Runtime  | Python 3.10+  |

---

## 🧪 Requirements

- Python 3.10+
- A valid **Google Gemini API Key**

Install dependencies:

```bash
pip install -r requirements.txt
```
