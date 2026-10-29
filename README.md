# Portable EEG-Based Brain State Detection System

## Overview
This project focuses on building a **portable EEG system** combined with AI models to detect brain states such as:

- Seizure / abnormal activity  
- Confusion state  
- Focused / Relaxed / Neutral states  

The system integrates hardware (EEG acquisition) and machine learning models to analyze brain signals.

---

## Datasets Used
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

## Models Used
- XGBoost (main model for structured data)
- Random Forest
- CNN
- CNN + LSTM (for temporal EEG patterns)


---

## How to Run

### 1. Install dependencies
  pip install pandas numpy scikit-learn xgboost tensorflow joblib
  
---

### 2. Run model
      python main.py
      python brain_state.py
      python mental_state.py


---

## Results
| Model | Task | Accuracy |
|------|------|---------|
| XGBoost | Brain State (3-class) | 98% |
| CNN + LSTM | Confusion Detection | 71% |
| Random Forest | Seizure Detection | 72% |

---

## Limitations
- EEG signals are noisy and sensitive to electrode placement  
- Limited number of subjects in datasets  
- Some datasets lack subject-wise separation  
- Labels like confusion are subjective  

---

## Future Work
- Use **hospital-grade EEG datasets**  
- Collect real-world multi-user EEG data  
- Improve hardware with dry EEG electrodes  
- Deploy real-time EEG streaming with ESP32  
- Build personalized AI models  

---

## Key Insight
High accuracy on controlled datasets does not guarantee real-world performance.  
Future improvements will focus on **real-device validation and personalization**.

---

## Tech Stack
- Python  
- XGBoost, Scikit-learn  
- TensorFlow / Keras  
- ESP32 (hardware integration)  
- Arduino IDE  

---

  
---

## ESP32–ADS1115 EEG Signal Acquisition
This module implements the analog signal acquisition stage of a portable EEG monitoring system using an ESP32 microcontroller and an ADS1115 16-bit Analog-to-Digital Converter (ADC).

---

## System Architecture

       EEG Electrodes
              |
              v
     Analog Front-End
   (Amplification / Filtering)
              |
              v
       ADS1115 ADC
      (Analog to Digital)
              |
        I2C: SDA / SCL
              |
              v
        ESP32 MCU
       (Firmware)
              |
       Serial Communication
              |
              v
       Computer / PC
      (Serial Monitor)
              |
              v
    Data Logging / Analysis

The analog front end conditions the EEG signal before it reaches the ADC. The ADS1115 converts the analog input into a digital value, and the ESP32 reads that value over I²C.

The ESP32 then calculates the corresponding voltage, attaches a timestamp, and sends the result to the connected computer.

---

## I²C Pin Configuration

The firmware initializes I²C using:

Wire.begin(8, 9);
SDA (Serial Data): Carries data between the ESP32 and ADS1115.
SCL (Serial Clock): Synchronizes communication between the devices.

---

## Communication Protocols

1 I²C Communication

I²C stands for Inter-Integrated Circuit. It is a synchronous serial communication protocol that enables a controller to communicate with peripheral devices using two signal lines.
In this project, I²C is used for communication between the ESP32 and ADS1115.


2 UART / Serial Communication

Serial communication is used to transmit the acquired readings from the ESP32 to a computer for monitoring and debugging.
The firmware initializes the serial interface with:
Serial.begin(115200);
The configured baud rate is 115200 bits per second.

---

## EEG Signal Visualization and Artifact Observations

The following figure shows a five-second segment of the acquired signal and its filtered frequency spectrum.
![EEG waveform and filtered spectrum](images/eegWaveform.png)

Several large, sharp deflections are visible.
Possible contributors include:
- Eye movement and blinks (EOG artifacts): Eye movements and blinks can produce large voltage changes, particularly with electrodes positioned near the forehead.
- Muscle activity (EMG artifacts): Facial and scalp muscle activity can introduce irregular, high-frequency components into an EEG recording.
- Other recording artifacts: Electrode movement, cable movement, poor electrode contact, and electrical interference may also affect the waveform.

## Frequency-Domain Spectrum
The lower plot shows the filtered signal spectrum, with shaded regions marking common EEG frequency bands:
Band	Approximate frequency range
Delta	0.5–4 Hz
Theta	4–8 Hz
Alpha	8–13 Hz
Beta	13–30 Hz

Elevated power in the beta range is because of muscle activity around the eye region captured by the ecg electrodes. Using proper dry comb eeg electrodes would filter out the high muscle signals.

---

## Image of our first hardware prototype

![Prototype hardware of portable eeg](images/portableEEG.png)
