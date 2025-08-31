# Insider Threat Detection System — AI/ML-based Multi-source User Behavior Analysis

<p align="center">
  <img src="https://www.activtrak.com/wp-content/uploads/2021/09/detection-2@2x.png" alt="System Architecture" width="700"/>
</p>

## 📌 Project Overview
This project implements an **AI/ML-powered Insider Threat Detection System** that analyzes user activity from multiple data sources to detect potential risks inside an organization.

The system leverages **multi-source CERT logs** (logon, device, email, file) to model user behavior and applies anomaly detection techniques to identify deviations that may indicate malicious or risky activity.

---

## 🚀 Features
- 🔹 **Multi-source log analysis** — integrates logon, device, email, and file access logs  
- 🔹 **Feature engineering** — temporal patterns, email indicators, network centrality metrics  
- 🔹 **Anomaly detection with Random Forest & Isolation Forest** — detects deviations from normal user behavior  
- 🔹 **Unsupervised ML** — anomaly scoring & ranking without labeled data  
- 🔹 **Interactive Streamlit dashboard** — visualize high-risk users, trends, and raw logs

## 📊 Dashboard Features
- Risky user ranking based on anomaly scores  
- Per-user temporal activity trends  
- Email & device usage insights  
- Raw log exploration for deeper investigation

## 🏠 Home
<p align="center">
  <img src="images/image1.jpg" alt="Streamlit Dashboard Screenshot" width="700"/>
</p>

## ⚠️ Risky users
<p align="center">
  <img src="images/image2.jpg" alt="Streamlit Risky User Screenshot" width="700"/>
</p>

## 🧑‍💻User Feature
<p align="center">
  <img src="images/image3.jpg" alt="Streamlit Risky User Screenshot" width="700"/>
  <img src="images/image4.jpg" alt="Streamlit Risky User Screenshot" width="700"/>
</p>

## 🧑‍💻Raw Logs
<p align="center">
  <img src="images/image5.jpg" alt="Streamlit Risky User Screenshot" width="700"/>
</p>


---

## 🛠️ Tech Stack
- **Python 3.x**  
- **Pandas, NumPy, Scikit-learn**  
- **Random Forest & Isolation Forest models**  
- **NetworkX** (for network centrality features)  
- **Streamlit** (interactive dashboard & visualization)  

---

## 📂 Dataset
- **CERT Insider Threat Dataset**  
  - Includes multi-source logs:  
    - 🖥️ Device & logon events  
    - 📧 Email communications  
    - 📁 File access  
  - Preprocessed into **user-level, per-day features** for modeling.


---

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/SunnyXcode/Ai-Insider-Threat-Detection.git
   cd Ai-Insider-Threat-Detectio
