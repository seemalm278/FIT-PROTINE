# app.py
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================
# APP CONFIGURATION
# ============================================
st.set_page_config(
    page_title="FitProtein AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .feature-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🏋️ FitProtein AI</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Smart Protein Requirement & Intake Predictor</h2>', unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("protein_fitness_dataset.csv")
        st.success(f"✅ Dataset loaded successfully! ({len(df)} records)")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please run create_dataset.py first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

# Load the data
df = load_data()

if df is None:
    st.stop()

# ============================================
# DATA PREPARATION
# ============================================
st.sidebar.header("⚙️ Data & Model Settings")

# Show dataset info in sidebar
with st.sidebar.expander("📊 Dataset Info", expanded=True):
    st.write(f"**Total Samples:** {len(df)}")
    st.write(f"**Features:** {len(df.columns)}")
    st.write(f"**Protein Range:** {df['protein_required'].min():.0f} - {df['protein_required'].max():.0f} g")
    
    if st.button("Show Sample Data"):
        st.dataframe(df.head(), use_container_width=True)

# Encode categorical features
encoder = LabelEncoder()
df_encoded = df.copy()
df_encoded['gender'] = encoder.fit_transform(df['gender'])
df_encoded['workout_type'] = encoder.fit_transform(df['workout_type'])
df_encoded['fitness_goal'] = encoder.fit_transform(df['fitness_goal'])

# Prepare features and target
X = df_encoded.drop(['protein_required'], axis=1)
y = df_encoded['protein_required']

# Split data
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ============================================
# MODEL TRAINING
# ============================================
st.sidebar.subheader("🤖 Model Configuration")
n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)

@st.cache_resource
def train_model():
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# Calculate model performance
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ============================================
# USER INPUT SECTION
# ============================================
st.header("👤 Enter Your Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70, step=1)
    height = st.number_input("Height (cm)", min_value=140, max_value=210, value=170, step=1)
    age = st.number_input("Age", min_value=15, max_value=70, value=25, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    st.subheader("Fitness Profile")
    workout_type = st.selectbox("Primary Workout Type", ["Strength", "Cardio", "HIIT"])
    workout_freq = st.slider("Workout Frequency (days/week)", 1, 7, 3)
    goal = st.selectbox("Fitness Goal", ["Gain", "Maintain", "Lose"])
    activity_level = st.select_slider(
        "Daily Activity Level",
        options=["Sedentary", "Light", "Moderate", "Active", "Very Active"]
    )

with col3:
    st.subheader("Current Nutrition")
    protein_intake = st.number_input("Current Daily Protein Intake (g)", 
                                     min_value=0, max_value=400, value=90, step=5)
    
    # Calculate BMI for additional insight
    bmi = weight / ((height/100) ** 2)
    st.metric("Your BMI", f"{bmi:.1f}")
    
    # BMI interpretation
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
    st.caption(f"*BMI Status: {bmi_status}*")

# ============================================
# PREDICTION
# ============================================
# Encode user input
gender_enc = 1 if gender == "Male" else 0
workout_enc = ["Strength", "Cardio", "HIIT"].index(workout_type)
goal_enc = ["Gain", "Maintain", "Lose"].index(goal)

# Activity level multiplier
activity_multipliers = {
    "Sedentary": 0.9,
    "Light": 1.0,
    "Moderate": 1.1,
    "Active": 1.2,
    "Very Active": 1.3
}
activity_mult = activity_multipliers[activity_level]

input_data = np.array([[weight, height, age, gender_enc,
                        workout_enc, workout_freq, goal_enc, protein_intake]])

# Make prediction
predicted_protein = model.predict(input_data)[0]
adjusted_prediction = predicted_protein * activity_mult

# Calculate intake status
protein_ratio = protein_intake / adjusted_prediction if adjusted_prediction > 0 else 0

if protein_ratio < 0.85:
    status = "❌ Insufficient"
    status_color = "red"
    recommendation = "Increase protein intake"
elif protein_ratio < 1.15:
    status = "✅ Optimal"
    status_color = "green"
    recommendation = "Maintain current intake"
else:
    status = "⚠️ Excessive"
    status_color = "orange"
    recommendation = "Consider reducing protein"

# ============================================
# RESULTS DISPLAY
# ============================================
st.markdown("---")
st.header("📊 Your Protein Analysis Results")

# Create metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Recommended Protein",
        value=f"{adjusted_prediction:.1f} g",
        help="AI-predicted daily protein requirement"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Your Intake",
        value=f"{protein_intake} g",
        delta=f"{protein_intake - adjusted_prediction:+.1f} g",
        delta_color="off"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Intake Ratio",
        value=f"{protein_ratio:.2f}",
        help="Your intake ÷ Recommended (Optimal: 0.85-1.15)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Status",
        value=status,
        help=recommendation
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Status message
st.markdown("---")
if status_color == "red":
    deficit = adjusted_prediction - protein_intake
    st.error(f"""
    **{status} Protein Intake**
    
    You need approximately **{deficit:.1f}g more protein** daily to meet your requirements.
    
    **Suggestions to increase protein:**
    - Add {deficit/7:.1f} eggs (7g protein each)
    - Add {deficit/25:.1f} servings of chicken breast (25g each)
    - Add {deficit/20:.1f} scoops of protein powder (20g each)
    """)
elif status_color == "orange":
    excess = protein_intake - adjusted_prediction
    st.warning(f"""
    **{status} Protein Intake**
    
    You're consuming **{excess:.1f}g more protein** than recommended.
    
    **Consider:**
    - Reducing protein intake by {excess:.1f}g daily
    - Ensuring balanced nutrition with carbs and healthy fats
    - Consulting with a nutritionist if consistently high
    """)
else:
    st.success(f"""
    **{status} Protein Intake!** 🎉
    
    You're meeting your protein requirements perfectly!
    
    **Maintain your current intake** and focus on:
    - Balanced nutrition with all food groups
    - Proper hydration
    - Consistent workout routine
    """)

# ============================================
# VISUALIZATIONS
# ============================================
st.markdown("---")
st.header("📈 Visualization & Insights")

tab1, tab2, tab3 = st.tabs(["Protein Analysis", "Model Insights", "Food Suggestions"])

with tab1:
    # Protein comparison chart
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    categories = ['Recommended', 'Your Intake']
    values = [adjusted_prediction, protein_intake]
    colors = ['#4CAF50', '#2196F3']
    
    bars = ax1.bar(categories, values, color=colors, width=0.6)
    ax1.set_ylabel('Protein (grams)')
    ax1.set_title('Protein Requirement vs Intake')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}g',
                ha='center', va='bottom', fontweight='bold')
    
    # Gauge chart for ratio
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    
    # Color sections
    ax2.fill_between(theta, 0, 0.7, color='red', alpha=0.3)  # Low
    ax2.fill_between(theta, 0.7, 0.9, color='orange', alpha=0.3)  # Borderline
    ax2.fill_between(theta, 0.9, 1.1, color='green', alpha=0.3)  # Optimal
    ax2.fill_between(theta, 1.1, 1.3, color='orange', alpha=0.3)  # Borderline
    ax2.fill_between(theta, 1.3, 1.5, color='red', alpha=0.3)  # High
    
    # Needle position
    ratio_clipped = min(max(protein_ratio, 0.5), 1.5)
    needle_angle = np.pi * (ratio_clipped - 0.5)
    ax2.plot([needle_angle, needle_angle], [0, 0.8], color='black', linewidth=3)
    
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title(f'Intake Ratio: {protein_ratio:.2f}', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig1)

with tab2:
    # Model performance and feature importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        metrics = ['MAE', 'R² Score']
        values = [mae, r2]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax2.bar(metrics, values, color=colors)
        ax2.set_ylabel('Score')
        ax2.set_title('Model Evaluation Metrics')
        ax2.set_ylim(0, 1 if max(values) < 1 else max(values) * 1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        st.pyplot(fig2)
        st.caption("MAE: Mean Absolute Error (lower is better)")
        st.caption("R²: Coefficient of Determination (closer to 1 is better)")
    
    with col2:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': ['Weight', 'Height', 'Age', 'Gender', 'Workout Type', 
                       'Frequency', 'Goal', 'Current Intake'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        bars = ax3.barh(feature_importance['Feature'], feature_importance['Importance'],
                       color=sns.color_palette("viridis", len(feature_importance)))
        ax3.set_xlabel('Importance')
        ax3.set_title('What Affects Protein Requirements Most?')
        
        # Add value labels
        for i, (importance, feature) in enumerate(zip(feature_importance['Importance'], 
                                                      feature_importance['Feature'])):
            ax3.text(importance, i, f' {importance:.3f}', va='center')
        
        st.pyplot(fig3)

with tab3:
    # Food suggestions based on deficit/surplus
    st.subheader("🍎 Food-Based Protein Suggestions")
    
    protein_sources = {
        'Chicken Breast (100g)': 31,
        'Eggs (1 large)': 6,
        'Greek Yogurt (100g)': 10,
        'Lentils (cooked, 100g)': 9,
        'Tofu (100g)': 8,
        'Protein Powder (1 scoop)': 24,
        'Salmon (100g)': 25,
        'Almonds (28g)': 6,
        'Milk (1 cup)': 8,
        'Quinoa (cooked, 100g)': 4
    }
    
    if protein_ratio < 0.85:
        deficit = adjusted_prediction - protein_intake
        st.info(f"**You need approximately {deficit:.0f}g more protein daily.**")
        
        st.write("**Food options to reach your goal:**")
        for food, protein in protein_sources.items():
            servings = deficit / protein
            if servings <= 5:  # Show only reasonable options
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f'<div class="feature-box">{food}</div>', unsafe_allow_html=True)
                with col2:
                    st.write(f"→ {servings:.1f} servings")
    
    elif protein_ratio > 1.15:
        excess = protein_intake - adjusted_prediction
        st.warning(f"**You're consuming {excess:.0f}g more protein than needed.**")
        
        st.write("**Consider reducing these high-protein foods:**")
        for food, protein in protein_sources.items():
            if protein > 15:  # Show only high-protein foods
                servings = excess / protein
                if servings <= 3:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f'<div class="feature-box">{food}</div>', unsafe_allow_html=True)
                    with col2:
                        st.write(f"→ Reduce by {servings:.1f} servings")

# ============================================
# DOWNLOAD & EXPORT
# ============================================
st.markdown("---")
st.header("💾 Export Results")

# Create results summary
results_df = pd.DataFrame({
    'Metric': ['Weight (kg)', 'Height (cm)', 'Age', 'Gender', 'Workout Type', 
               'Workout Frequency', 'Fitness Goal', 'Activity Level',
               'Current Protein Intake (g)', 'Recommended Protein (g)',
               'Intake Ratio', 'Status'],
    'Value': [weight, height, age, gender, workout_type, workout_freq,
              goal, activity_level, protein_intake, adjusted_prediction,
              protein_ratio, status]
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Download Results")
    
    # Convert to CSV
    csv = results_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="fitprotein_analysis.csv",
        mime="text/csv",
        help="Download your protein analysis results"
    )
    
    # Generate summary text
    summary = f"""
    FitProtein AI Analysis Report
    =============================
    
    Personal Information:
    - Weight: {weight} kg
    - Height: {height} cm
    - Age: {age}
    - Gender: {gender}
    
    Fitness Profile:
    - Workout Type: {workout_type}
    - Frequency: {workout_freq} days/week
    - Goal: {goal}
    - Activity Level: {activity_level}
    
    Protein Analysis:
    - Current Intake: {protein_intake} g/day
    - Recommended: {adjusted_prediction:.1f} g/day
    - Intake Ratio: {protein_ratio:.2f}
    - Status: {status}
    
    Recommendation: {recommendation}
    
    Generated by FitProtein AI
    """
    
    st.download_button(
        label="📄 Download Summary Report",
        data=summary,
        file_name="fitprotein_summary.txt",
        mime="text/plain"
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>FitProtein AI v2.0</strong> | Smart Protein Prediction System</p>
    <p style="font-size: 0.9rem;">
        <em>Disclaimer: This tool provides AI-based estimates. For personalized dietary advice, 
        please consult with a certified nutritionist or healthcare professional.</em>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        Built with ❤️ using Streamlit, Scikit-learn, and Matplotlib
    </p>
</div>
""", unsafe_allow_html=True)