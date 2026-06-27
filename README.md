# 🏋️ FitProtein AI - Smart Protein Requirement Predictor

## 📌 Project Overview

FitProtein AI is an Artificial Intelligence and Machine Learning-based fitness application designed to predict personalized protein requirements based on individual health details, workout patterns, and fitness goals.

The project uses Machine Learning techniques to analyze user information such as age, weight, height, gender, workout type, workout frequency, and fitness objectives to estimate the recommended protein intake required for better fitness planning.

The application provides an interactive and user-friendly interface built with Streamlit, allowing users to enter their details and receive AI-based protein requirement predictions instantly.

---

## 🎯 Problem Statement

Maintaining proper nutrition is essential for achieving fitness goals, but many people struggle to determine the right amount of protein intake based on their body composition and activity level.

General protein recommendations may not consider individual differences such as workout intensity, body weight, and personal fitness goals.

FitProtein AI solves this problem by using Machine Learning to provide personalized protein requirement predictions based on user-specific data.

---

## 🚀 Features

* ✅ AI-based protein requirement prediction
* ✅ Personalized fitness recommendations
* ✅ Machine Learning regression model
* ✅ Interactive Streamlit web application
* ✅ User-based health and workout analysis
* ✅ Data visualization and insights
* ✅ Automated prediction system

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Regression

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Application Framework

* Streamlit

---

## ⚙️ How It Works

The system follows these steps:

### 1. User Input

The user provides information such as:

* Weight
* Height
* Age
* Gender
* Workout type
* Workout frequency
* Fitness goal

---

### 2. Data Processing

The input data is processed by:

* Converting categorical values into numerical format
* Preparing features for machine learning prediction
* Cleaning and organizing the data

---

### 3. Machine Learning Model

The project uses a **Random Forest Regression model** to learn patterns from fitness data and predict the required protein intake.

The model analyzes relationships between:

* Body characteristics
* Exercise habits
* Fitness goals
* Current protein needs

---

### 4. Prediction

After processing the input, the trained AI model predicts the recommended protein requirement for the user.

---

## 📂 Project Structure

```text
FitProtein-AI
│
├── app.py
│
├── create_dataset.py
│
├── protein_fitness_dataset.csv
│
├── requirements.txt
│
└── README.md
```

---

## 🔧 Installation & Setup

### Clone Repository

```bash
git clone https://github.com/your-username/FitProtein-AI.git
```

### Navigate to Project Folder

```bash
cd FitProtein-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Machine Learning Workflow

```text
User Fitness Data
        |
        ↓
Data Preprocessing
        |
        ↓
Feature Encoding
        |
        ↓
Random Forest Model
        |
        ↓
Protein Requirement Prediction
        |
        ↓
Personalized Recommendation
```

---

## 🌍 Applications

This project can be useful for:

* Fitness tracking applications
* Personal nutrition planning
* Gym and workout platforms
* AI-based health assistants
* Sports nutrition analysis

---

## 🔮 Future Improvements

Future enhancements:

* Integration with wearable fitness devices
* Real-time health data tracking
* Deep learning-based recommendations
* Meal planning suggestions
* Mobile application development
* Cloud deployment

---

## 🤖 Machine Learning Concepts Used

* Supervised Learning
* Regression Models
* Feature Engineering
* Data Preprocessing
* Predictive Analytics

---

## 👨‍💻 Author

Developed as an AI and Machine Learning project focused on personalized fitness and nutrition recommendations.

---

## 📜 License

This project is open-source and available for educational and research purposes.
