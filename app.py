import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np

# --- Configuration ---
# Set page title and icon
st.set_page_config(page_title="Biomedical Equipment Predictive Maintenance", page_icon="🏥")

# --- Load Model and Data ---
@st.cache_resource # Cache the model loading for efficiency
def load_model():
    """Loads the pre-trained machine learning model."""
    try:
        model = joblib.load('C:/Users/ROHAN/Desktop/Biomed_Maintenance_Predictive_Model/predictive_maintenance_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'predictive_maintenance_model.joblib' not found. "
                 "Please ensure the model training script has been run and the file exists.")
        return None

@st.cache_data # Cache the data loading for efficiency
def load_data():
    """Loads the biomedical equipment dataset."""
    try:
        df = pd.read_csv('C:/Users/ROHAN/Desktop/Biomed_Maintenance_Predictive_Model/biomedical_equipment_data.csv')
        df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'])
        return df
    except FileNotFoundError:
        st.error("Error: Dataset file 'biomedical_equipment_data.csv' not found. "
                 "Please ensure the data generation script has been run and the file exists.")
        return None

model = load_model()
equipment_df = load_data()

# Check if model and data loaded successfully
if model is None or equipment_df is None:
    st.stop() # Stop the app if essential files are missing


dummy_data = pd.DataFrame({
    'Age_in_years': [0], 'Usage_hours_per_day': [0.0],
    'Last_Maintenance_Date': [datetime.now()], 'Breakdown_Count': [0],
    'Vibration_Index': [0.0], 'Temperature_Score': [0.0],
    'Error_Log_Frequency': [0], 'Type': ['Ventilator'] # Use a sample type
})
# Add all possible types to ensure all dummy columns are created
all_equipment_types = ['Ventilator', 'MRI', 'ECG', 'Ultrasound']
dummy_data_for_dummies = pd.DataFrame({'Type': all_equipment_types})
dummy_data_for_dummies = pd.get_dummies(dummy_data_for_dummies, columns=['Type'], drop_first=True)
expected_type_cols = [col for col in dummy_data_for_dummies.columns if col.startswith('Type_')]

# Combine with other numerical features
expected_features = [
    'Age_in_years', 'Usage_hours_per_day', 'Breakdown_Count',
    'Vibration_Index', 'Temperature_Score', 'Error_Log_Frequency',
    'Days_Since_Maintenance'
] + sorted(expected_type_cols) # Sort to ensure consistent order

# --- Streamlit App Layout ---
st.title("🏥 Hospital Biomedical Equipment Predictive Maintenance")
st.markdown("Predict when biomedical equipment needs maintenance to prevent breakdowns.")

st.sidebar.header("Predict Single Equipment Maintenance")

# Input fields for prediction
with st.sidebar.form("prediction_form"):
    st.subheader("Equipment Parameters")
    equipment_type = st.selectbox("Equipment Type", all_equipment_types)
    age_in_years = st.slider("Age in Years", 0, 15, 5)
    usage_hours_per_day = st.slider("Usage Hours per Day", 0.0, 24.0, 8.0, 0.1)
    last_maintenance_date = st.date_input("Last Maintenance Date", datetime.now() - timedelta(days=90))
    breakdown_count = st.slider("Breakdown Count", 0, 5, 0)
    vibration_index = st.slider("Vibration Index (0.0 = no wear, 1.0 = high wear)", 0.0, 1.0, 0.3, 0.01)
    temperature_score = st.slider("Temperature Score (20°C = low, 80°C = high)", 20.0, 80.0, 45.0, 0.1)
    error_log_frequency = st.slider("Weekly Error Log Frequency", 0, 10, 1)

    predict_button = st.form_submit_button("Predict Maintenance Need")

# --- Prediction Logic ---
if predict_button:
    # Feature Engineering for the single prediction
    days_since_maintenance = (datetime.now() - datetime.combine(last_maintenance_date, datetime.min.time())).days

    # Create a DataFrame for the input, matching the training data's structure
    input_data = pd.DataFrame({
        'Age_in_years': [age_in_years],
        'Usage_hours_per_day': [usage_hours_per_day],
        'Breakdown_Count': [breakdown_count],
        'Vibration_Index': [vibration_index],
        'Temperature_Score': [temperature_score],
        'Error_Log_Frequency': [error_log_frequency],
        'Days_Since_Maintenance': [days_since_maintenance]
    })

    # One-hot encode 'Type' for the input data
    # Initialize all expected type columns to 0
    for col in expected_type_cols:
        input_data[col] = 0
    # Set the specific type column to 1
    if equipment_type != 'ECG': # 'ECG' is the base, so its dummy column is dropped
        type_col_name = f'Type_{equipment_type}'
        if type_col_name in input_data.columns:
            input_data[type_col_name] = 1
        else:
            st.warning(f"Warning: Type '{equipment_type}' not recognized for one-hot encoding. "
                       f"Expected columns: {expected_type_cols}")

    # Ensure the input_data columns are in the same order as the model's training features
    # This is critical for correct prediction
    try:
        input_data = input_data[expected_features]
    except KeyError as e:
        st.error(f"Error: Mismatch in feature columns. Missing or extra column: {e}. "
                 f"Expected: {expected_features}, Got: {input_data.columns.tolist()}")
        st.stop()


    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Prediction Result")
    if prediction == 1:
        st.sidebar.error(f"🚨 Maintenance Needed: YES")
        st.sidebar.write(f"Confidence (Maintenance): {prediction_proba[1]*100:.2f}%")
        st.sidebar.info("This equipment shows signs of potential issues based on its current parameters.")
    else:
        st.sidebar.success(f"✅ Maintenance Needed: NO")
        st.sidebar.write(f"Confidence (No Maintenance): {prediction_proba[0]*100:.2f}%")
        st.sidebar.info("This equipment appears to be in good condition for now.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Maintenance Indicators Applied:")
    # Display which rules triggered the maintenance flag
    triggered_rules = []
    if age_in_years > 8:
        triggered_rules.append(f"Age (> 8 years): {age_in_years} years")
    if usage_hours_per_day > 16:
        triggered_rules.append(f"High Usage (> 16 hrs/day): {usage_hours_per_day} hrs/day")
    if days_since_maintenance > 180:
        triggered_rules.append(f"Overdue Maintenance (> 6 months): {days_since_maintenance} days")
    if breakdown_count > 2:
        triggered_rules.append(f"High Breakdown Count (> 2): {breakdown_count} breakdowns")
    if vibration_index > 0.6:
        triggered_rules.append(f"High Vibration Index (> 0.6): {vibration_index}")
    if temperature_score > 60:
        triggered_rules.append(f"High Temperature Score (> 60°C): {temperature_score}°C")
    if error_log_frequency > 3:
        triggered_rules.append(f"High Error Log Frequency (> 3): {error_log_frequency} errors/week")

    if triggered_rules:
        st.sidebar.markdown("**Reasons for Maintenance:**")
        for rule in triggered_rules:
            st.sidebar.write(f"- {rule}")
    else:
        st.sidebar.write("No specific maintenance indicators were triggered for this equipment.")


st.markdown("---")

# --- Overall Equipment Status ---
st.header("Overall Equipment Status")
st.write("Overview of all biomedical equipment and their predicted maintenance needs.")

# Calculate summary statistics
total_equipment = len(equipment_df)
maintenance_needed_count = equipment_df['Maintenance_Needed'].sum()
maintenance_not_needed_count = total_equipment - maintenance_needed_count

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Equipment", total_equipment)
with col2:
    st.metric("Maintenance Needed", maintenance_needed_count, delta=f"{maintenance_needed_count} more than 'No Maintenance'")
with col3:
    st.metric("No Maintenance Needed", maintenance_not_needed_count, delta=f"{maintenance_not_needed_count} less than 'Maintenance Needed'")

st.markdown("---")

st.subheader("Detailed Equipment List")

# Display the dataframe with conditional formatting
def color_maintenance_needed(val):
    """Applies color based on 'Maintenance_Needed' column."""
    if isinstance(val, (int, np.integer)) and val == 1:
        return 'background-color: #ffcccc' # Light red for maintenance needed
    return ''

# Apply styling to the DataFrame
st.dataframe(equipment_df.style.applymap(color_maintenance_needed, subset=['Maintenance_Needed']), height=500)

st.markdown("---")
st.caption("Developed using Python, Scikit-learn, and Streamlit.")
