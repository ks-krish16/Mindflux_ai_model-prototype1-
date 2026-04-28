# 🧠 Portable EEG-Based Brain State Detection System

## 🚀 Overview
This project focuses on building a **portable EEG system** combined with AI models to detect brain states such as:

- ⚡ Seizure / abnormal activity  
- 🧠 Confusion state  
- 🎯 Focused / Relaxed / Neutral states  

The system integrates **hardware (EEG acquisition)** and **machine learning models** to analyze brain signals.

---

## 🧪 Datasets Used
We trained our models using multiple public EEG datasets:

1. **EEG Seizure Dataset**
   - 5-class classification (seizure, tumor, healthy, eyes open/closed)
   - Accuracy: ~72%

2. **EEG Confusion Dataset**
   - Binary classification (confused / not confused)
   - Accuracy: ~71%

3. **Muse Brain State Dataset**
   - 3-class classification (focused, relaxed, neutral)
   - Accuracy: ~98%

---

## 🤖 Models Used
- XGBoost (main model for structured data)
- Random Forest
- CNN
- CNN + LSTM (for temporal EEG patterns)


---

## ▶️ How to Run

### 1. Install dependencies
  pip install pandas numpy scikit-learn xgboost tensorflow joblib
  
---

### 2. Run model
      python main.py
      python brain_state.py
      python mental_state.py


---

## 📊 Results
| Model | Task | Accuracy |
|------|------|---------|
| XGBoost | Brain State (3-class) | 98% |
| CNN + LSTM | Confusion Detection | 71% |
| Random Forest | Seizure Detection | 72% |

---

## ⚠️ Limitations
- EEG signals are noisy and sensitive to electrode placement  
- Limited number of subjects in datasets  
- Some datasets lack subject-wise separation  
- Labels like confusion are subjective  

---

## 🔮 Future Work
- Use **hospital-grade EEG datasets**  
- Collect real-world multi-user EEG data  
- Improve hardware with dry EEG electrodes  
- Deploy real-time EEG streaming with ESP32  
- Build personalized AI models  

---

## 🧠 Key Insight
High accuracy on controlled datasets does not guarantee real-world performance.  
Future improvements will focus on **real-device validation and personalization**.

---

## 📌 Tech Stack
- Python  
- XGBoost, Scikit-learn  
- TensorFlow / Keras  
- ESP32 (hardware integration)  
- Arduino IDE  

---

  
---

## ⚙️ Project Structure
