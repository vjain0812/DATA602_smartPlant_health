"""
Streamlit Frontend for Smart Plant Health Monitor
User-friendly interface for plant disease detection and health monitoring
"""

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Smart Plant Health Monitor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_disease(image_file, top_k=5):
    """Call disease classification API"""
    try:
        files = {"file": image_file}
        data = {"top_k": top_k}
        response = requests.post(f"{API_URL}/api/v1/predict/disease", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def predict_yolo(image_file, confidence=0.25):
    """Call YOLO detection API"""
    try:
        files = {"file": image_file}
        data = {"confidence": confidence}
        response = requests.post(f"{API_URL}/api/v1/predict/yolo", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def comprehensive_analysis(image_file, temp, humidity, soil, light, nitrogen, phosphorus, potassium):
    """Call comprehensive analysis API"""
    try:
        files = {"file": image_file}
        data = {
            "temperature": temp,
            "humidity": humidity,
            "soil_moisture": soil,
            "light_intensity": light,
            "npk_nitrogen": nitrogen,
            "npk_phosphorus": phosphorus,
            "npk_potassium": potassium
        }
        response = requests.post(f"{API_URL}/api/v1/predict/comprehensive", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def plot_confidence_chart(predictions):
    """Create confidence chart for top predictions"""
    diseases = [p['disease'].replace('___', ' - ').replace('_', ' ') for p in predictions]
    confidences = [p['percentage'] for p in predictions]
    
    fig = px.bar(
        x=confidences,
        y=diseases,
        orientation='h',
        title='Top 5 Disease Predictions',
        labels={'x': 'Confidence (%)', 'y': 'Disease'},
        color=confidences,
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_health_gauge(health_score):
    """Create gauge chart for health score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Plant Health Score"},
        delta={'reference': 85},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_npk_chart(nitrogen, phosphorus, potassium):
    """Create NPK nutrient bar chart"""
    nutrients = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    values = [nitrogen, phosphorus, potassium]
    optimal = [75, 70, 80]  # Optimal values
    
    fig = go.Figure(data=[
        go.Bar(name='Current', x=nutrients, y=values, marker_color='lightblue'),
        go.Bar(name='Optimal', x=nutrients, y=optimal, marker_color='lightgreen')
    ])
    
    fig.update_layout(
        title='NPK Nutrient Levels',
        yaxis_title='Level',
        barmode='group',
        height=300
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Smart Plant Health Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Plant Disease Detection & Health Monitoring</p>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    if not api_status:
        st.error("⚠️ API Server is not running!")
        st.info("Please start the API server: `python api/main.py`")
        st.stop()
    
    st.success("✅ Connected to API Server")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        This system combines three AI models:
        - **ResNet**: Disease classification
        - **YOLOv5**: Disease localization  
        - **LSTM**: Watering prediction
        
        **Accuracy**: 89%  
        **False Alert Reduction**: 22%
        """)
        
        st.header("🎯 How to Use")
        st.markdown("""
        1. Upload a plant leaf image
        2. Enter sensor readings (optional)
        3. Enter NPK nutrient levels
        4. Click "Analyze Plant Health"
        5. View comprehensive results
        """)
        
        st.header("📈 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ResNet", "89%", "Accuracy")
            st.metric("YOLO", "85%", "mAP")
        with col2:
            st.metric("LSTM", "92%", "Accuracy")
            st.metric("System", "89%", "Overall")
    
    # Main Content
    tab1, tab2 = st.tabs(["🔬 Full Analysis", "📸 Quick Scan"])
    
    # Tab 1: Full Analysis
    with tab1:
        st.header("Comprehensive Plant Health Analysis")

        col1, col2 = st.columns([1, 1])

        # =======================
        # IMAGE UPLOAD COLUMN
        # =======================
        with col1:
            st.subheader("1. Upload Plant Image")
            uploaded_file = st.file_uploader(
                "Choose a leaf image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a plant leaf"
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

        # =======================
        # SENSOR & NUTRIENT DATA COLUMN
        # =======================
        with col2:
            st.subheader("2. Environmental Data")

            use_sensors = st.checkbox(
                "Enable environmental & watering prediction",
                value=True,
                help="Check this to get LSTM watering predictions"
            )

            if use_sensors:
                # Environmental sensors
                st.markdown("**Environmental Sensors:**")
                temp = st.slider("Temperature (°C)", 10, 40, 25)
                humidity = st.slider("Humidity (%)", 30, 90, 60)
                soil = st.slider("Soil Moisture (%)", 20, 90, 50)
                light = st.slider("Light Intensity (lux)", 0, 1000, 500)

                st.markdown("---")
                
                # NPK Nutrients
                st.markdown("**NPK Nutrient Levels:**")
                nitrogen = st.slider("Nitrogen (N)", 0, 150, 75, help="Optimal: 75")
                phosphorus = st.slider("Phosphorus (P)", 0, 150, 70, help="Optimal: 70")
                potassium = st.slider("Potassium (K)", 0, 150, 80, help="Optimal: 80")

                st.write("**Status Overview:**")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    temp_status = "🟢" if 20 <= temp <= 30 else "🟡" if 15 <= temp <= 35 else "🔴"
                    st.write(f"{temp_status} Temperature")

                with col_b:
                    hum_status = "🟢" if 50 <= humidity <= 70 else "🟡" if 40 <= humidity <= 80 else "🔴"
                    st.write(f"{hum_status} Humidity")

                with col_c:
                    soil_status = "🟢" if soil >= 40 else "🟡" if soil >= 30 else "🔴"
                    st.write(f"{soil_status} Soil Moisture")

            else:
                temp = None
                humidity = None
                soil = None
                light = None
                nitrogen = 75
                phosphorus = 70
                potassium = 80
                st.info("💡 Enable environmental data to get watering predictions from LSTM model")

        # =======================
        # ANALYZE BUTTON
        # =======================
        if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
            if not uploaded_file:
                st.warning("Please upload an image first!")
            else:
                with st.spinner("🤖 AI is analyzing your plant... Please wait..."):
                    uploaded_file.seek(0)

                    # API call
                    if use_sensors and temp is not None:
                        st.info(
                            f"📊 Using sensor data: "
                            f"Temp={temp}°C, Humidity={humidity}%, "
                            f"Soil={soil}%, Light={light} lux, "
                            f"NPK=({nitrogen}, {phosphorus}, {potassium})"
                        )
                        results = comprehensive_analysis(
                            uploaded_file,
                            temp, humidity, soil, light,
                            nitrogen, phosphorus, potassium
                        )
                    else:
                        st.info("📊 Analyzing without sensor data (LSTM skipped)")
                        results = comprehensive_analysis(
                            uploaded_file,
                            None, None, None, None,
                            nitrogen, phosphorus, potassium
                        )

                    if "error" in results or results.get("status") != "success":
                        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                    else:
                        data = results["data"]
                        st.success("✅ Analysis Complete!")

                        # =======================
                        # OVERALL HEALTH
                        # =======================
                        st.header("Overall Plant Health")
                        hcol1, hcol2, hcol3 = st.columns([2, 1, 1])

                        with hcol1:
                            health_score = data.get("overall_health_score", 0)
                            st.plotly_chart(
                                plot_health_gauge(health_score),
                                use_container_width=True
                            )

                        with hcol2:
                            st.metric("Health Score", f"{health_score}/100")
                            if health_score >= 80:
                                st.success("Excellent Health")
                            elif health_score >= 60:
                                st.warning("Needs Attention")
                            else:
                                st.error("Critical Condition")

                        with hcol3:
                            disease_data = data.get("disease_classification", {})
                            primary = disease_data.get("primary_disease", "Unknown")
                            conf = disease_data.get("confidence", 0)

                            st.metric("Disease Detected", "")
                            st.write(
                                f"**{primary.replace('___', ' - ').replace('_', ' ')}**"
                            )
                            st.write(f"Confidence: {conf:.1%}")

                        st.divider()

                        # =======================
                        # DETAILED RESULTS
                        # =======================
                        left, middle, right = st.columns(3)

                        # ---- Disease Classification ----
                        with left:
                            st.subheader("🦠 Disease Classification")
                            top_preds = disease_data.get("top_predictions", [])

                            if top_preds:
                                st.plotly_chart(
                                    plot_confidence_chart(top_preds),
                                    use_container_width=True
                                )

                                with st.expander("View All Predictions"):
                                    for i, pred in enumerate(top_preds, 1):
                                        st.write(
                                            f"{i}. "
                                            f"{pred['disease'].replace('___', ' - ').replace('_', ' ')}: "
                                            f"{pred['percentage']:.2f}%"
                                        )

                        # ---- NPK Nutrients ----
                        with middle:
                            st.subheader("🧪 Nutrient Analysis")
                            st.plotly_chart(
                                plot_npk_chart(nitrogen, phosphorus, potassium),
                                use_container_width=True
                            )
                            
                            # Nutrient status
                            st.markdown("**Nutrient Status:**")
                            n_status = "✅ Optimal" if 70 <= nitrogen <= 80 else "⚠️ Adjust" if 60 <= nitrogen <= 90 else "🔴 Critical"
                            p_status = "✅ Optimal" if 65 <= phosphorus <= 75 else "⚠️ Adjust" if 55 <= phosphorus <= 85 else "🔴 Critical"
                            k_status = "✅ Optimal" if 75 <= potassium <= 85 else "⚠️ Adjust" if 65 <= potassium <= 95 else "🔴 Critical"
                            
                            st.write(f"Nitrogen: {n_status}")
                            st.write(f"Phosphorus: {p_status}")
                            st.write(f"Potassium: {k_status}")

                        # ---- Watering Prediction ----
                        with right:
                            st.subheader("💧 Watering Prediction")
                            water_info = data.get("watering_prediction")

                            if water_info:
                                if "error" in water_info:
                                    st.warning(f"⚠️ {water_info['error']}")
                                elif "message" in water_info:
                                    st.info(water_info["message"])
                                else:
                                    needed = water_info.get("watering_needed", False)
                                    prob = water_info.get("probability", 0)
                                    hours = water_info.get("hours_until_watering", 0)
                                    rec = water_info.get("recommendation", "")

                                    if needed:
                                        st.error("### ⚠️ WATERING NEEDED")
                                        st.metric("Urgency", f"{prob:.1%}")
                                        st.write(f"Water within **{hours} hours**")
                                    else:
                                        st.success("### ✅ SOIL MOISTURE OK")
                                        st.metric("Safety Margin", f"{(1 - prob):.1%}")
                                        st.write(f"Next check in **{hours} hours**")

                                    st.info(f"💡 **Recommendation:**\n\n{rec}")

                        # =======================
                        # ACTION ITEMS
                        # =======================
                        st.divider()
                        st.subheader("📋 Action Items")
                        actions = []

                        if health_score < 70:
                            actions.append("🔴 **URGENT**: Inspect plant immediately")

                        if "healthy" not in primary.lower():
                            actions.append(
                                f"🦠 Treat for {primary.split('___')[-1].replace('_', ' ')}"
                            )

                        if water_info and water_info.get("watering_needed"):
                            actions.append("💧 **PRIORITY**: Water plant soon")
                        
                        # Nutrient recommendations
                        if nitrogen < 60 or nitrogen > 90:
                            actions.append(f"🧪 Adjust Nitrogen levels (Current: {nitrogen})")
                        if phosphorus < 55 or phosphorus > 85:
                            actions.append(f"🧪 Adjust Phosphorus levels (Current: {phosphorus})")
                        if potassium < 65 or potassium > 95:
                            actions.append(f"🧪 Adjust Potassium levels (Current: {potassium})")

                        if not actions:
                            actions.append("✅ Continue regular care routine")

                        for action in actions:
                            st.write(action)

                        
    # Tab 2: Quick Scan
    with tab2:
        st.header("Quick Disease Scan")
        st.write("Upload an image for fast disease classification")
        
        quick_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'], key="quick")
        
        if quick_file:
            col1, col2 = st.columns(2)
            
            with col1:
                quick_image = Image.open(quick_file)
                st.image(quick_image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("🚀 Quick Scan", type="primary"):
                with st.spinner("Scanning..."):
                    quick_file.seek(0)
                    results = predict_disease(quick_file, top_k=3)
                    
                    if results.get("status") == "success":
                        data = results["data"]
                        
                        with col2:
                            st.subheader("Results")
                            primary = data['primary_disease'].replace('___', ' - ').replace('_', ' ')
                            conf = data['confidence']
                            
                            st.metric("Primary Disease", primary)
                            st.metric("Confidence", f"{conf:.1%}")
                            
                            st.write("**Top 3 Predictions:**")
                            for pred in data['top_predictions'][:3]:
                                st.write(f"- {pred['disease'].replace('___', ' ').replace('_', ' ')}: {pred['percentage']:.1f}%")

    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>🎓 DATA-602 Final Project | Smart Plant Health Monitor</p>
            <p>Hybrid ML System: ResNet + YOLOv5 + LSTM</p>
            <p>System Accuracy: 89% | False Alert Reduction: 22%</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()