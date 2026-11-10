# 🏎️ AI101 Mini-Project: F1 Winner Prediction

Predicting Formula 1 race winners based on circuit conditions, driver performance, and grid positions using **Logistic Regression**.

---

## 👥 Team and Roles

| Member Name | Entry Number | Role | GitHub Handle |
| :--- | :--- | :--- | :--- |
| **Monika Singh** | `2026AIB1059` | Documentation Lead | [@2026AIB1059](https://github.com/2026AIB1059) |
| **Naitik Goyal** | `2026EEB1382` | Execution Lead | [@NaitikGo-Yal](https://github.com/NaitikGo-Yal) |
| **Princy Satapathy** | `2026EEB1395` | Video Lead | [@2026eeb1395](https://github.com/2026eeb1395) |
| **Ashmita Gupta** | `2026MCB1518` | PPT Lead | [@AshmitaGupta164](https://github.com/AshmitaGupta164) |

---

## 📁 Project Structure

```text
AI101-team-81/
├── data/              # 14 Raw F1 CSV Dataset files
├── outputs/           # Output visualizations and confusion matrix plots
├── src/
│   ├── main.py        # Main execution entry point & data verification
│   ├── model.py       # ML Model building, training & evaluation
│   ├── plots.py       # Visualization generation scripts
│   └── utils.py       # Helper functions and metric formatters
├── requirements.txt   # Project dependencies
└── README.md          # Comprehensive project documentation
```

## ⚙️ How to Setup

### 1. Prerequisites
Make sure you have Python installed on your system.

### 2. Installation
Clone the repository and install all required libraries:
```bash
git clone https://github.com/2026AIB1059/AI101-team-81.git
cd AI101-team-81
pip install -r requirements.txt
```

## 🚀 How to Run

Execute the main entry script from the root folder:
```bash
python src/main.py
```


## 📊 Summary of Results

- **Machine Learning Model:** Logistic Regression
- **Primary Features:** Starting Grid Position, Driver Experience, Constructor Points, Circuit Attributes
- **Evaluation Metrics:** Classification Accuracy, Precision, Recall, and ROC-AUC
- **Key Finding:** Starting grid position combined with constructor performance provides the highest weight in predicting podium finishes.


## 🔗 Project Deliverables

* 📊 **Presentation Slides (PPT):** *[Link coming soon]*
* 🎥 **Live Demo Video (5 min):** *[Link coming soon]*
