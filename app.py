import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import os
import joblib

# Load dataset using relative path
@st.cache_data
def load_data():
    # تحديد مسار الملف في نفس مجلد `app.py`
    file_path = os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    
    df = pd.read_csv(file_path)
    
   
    df.dropna(inplace=True)
    
    return df


df = load_data()


model_path = os.path.join(os.path.dirname(__file__), "xgboost_churn_model (1).pkl")

xgb_model = joblib.load(model_path)


# Language Selection
language = st.sidebar.radio("🌍 Select Language / اختر اللغة:", ["English", "العربية"])

# Sidebar navigation
st.sidebar.title("🔍 Churn Analysis Dashboard")

# Project Objective
st.subheader("Project Objective" if language == "English" else "هدف المشروع")
st.write("This dashboard provides insights into customer churn in the telecom industry, analyzing various factors affecting retention and suggesting strategies to reduce churn." if language == "English" else "يقدم هذا اللوحة رؤى حول فقدان العملاء في صناعة الاتصالات، من خلال تحليل العوامل المختلفة التي تؤثر على الاحتفاظ بالعملاء واقتراح استراتيجيات لتقليل الفقدان.")

if language == "English":
    page = st.sidebar.radio("Select Analysis:", [
        "📊 Data Overview",
        "📉 Contracts & Billing Impact",
        "📡 Additional Services Impact",
        "🏠 Demographic Impact",
        "💳 Payment Method Impact",
        "📈 Conclusions & Recommendations",
        "🔮 Predict Churn Probability"
    ])
else:
    page = st.sidebar.radio("اختر التحليل:", [
        "📊 نظرة عامة على البيانات",
        "📉 تأثير العقود والفواتير",
        "📡 تأثير الخدمات الإضافية",
        "🏠 التأثير الديموغرافي",
        "💳 تأثير طرق الدفع",
        "📈 الاستنتاجات والتوصيات",
        "🔮 توقع احتمالية فقدان العميل"
    ])

# 📊 Data Overview
if page in ["📊 Data Overview", "📊 نظرة عامة على البيانات"]:
    st.title("📊 Customer Churn Analysis" if language == "English" else "📊 تحليل فقدان العملاء")
    
    # Upload project image
    st.image(
    "https://raw.githubusercontent.com/abedahdy500/abedahdy500--Customer-Churn-Analysis_app/main/customer_churn_analysis.png",
    caption="Customer Churn Analysis Dashboard",
    use_container_width=True
    )
    
    st.dataframe(df.head())

# 📉 Contracts & Billing Impact
elif page in ["📉 Contracts & Billing Impact", "📉 تأثير العقود والفواتير"]:
    st.title("📉 Contracts & Billing Impact" if language == "English" else "📉 تأثير العقود والفواتير")
    fig = px.histogram(df, x="Contract", color="Churn", barmode="group", title="Churn Rate by Contract Type")
    st.plotly_chart(fig)

# 📡 Additional Services Impact
elif page in ["📡 Additional Services Impact", "📡 تأثير الخدمات الإضافية"]:
    st.title("📡 Additional Services Impact" if language == "English" else "📡 تأثير الخدمات الإضافية")
    fig = px.histogram(df, x="OnlineSecurity", color="Churn", barmode="group", title="Churn Rate by Online Security")
    st.plotly_chart(fig)

# 🏠 Demographic Impact
elif page in ["🏠 Demographic Impact", "🏠 التأثير الديموغرافي"]:
    st.title("🏠 Demographic Impact" if language == "English" else "🏠 التأثير الديموغرافي")
    fig = px.histogram(df, x="gender", color="Churn", barmode="group", title="Churn Rate by Gender")
    st.plotly_chart(fig)

# 💳 Payment Method Impact
elif page in ["💳 Payment Method Impact", "💳 تأثير طرق الدفع"]:
    st.title("💳 Payment Method Impact" if language == "English" else "💳 تأثير طرق الدفع")
    fig = px.pie(names=df["PaymentMethod"].value_counts().index, values=df["PaymentMethod"].value_counts().values, title="Churn Distribution by Payment Method")
    st.plotly_chart(fig)

# 📈 Conclusions & Recommendations
elif page in ["📈 Conclusions & Recommendations", "📈 الاستنتاجات والتوصيات"]:
    st.title("📈 Conclusions & Recommendations" if language == "English" else "📈 الاستنتاجات والتوصيات")
    st.subheader("Key Findings" if language == "English" else "النتائج الرئيسية")
    st.write("- Customers with month-to-month contracts have the highest churn rate.\n"
             "- High monthly charges correlate with increased churn probability.\n"
             "- Customers who do not use additional services (e.g., online security, tech support) are more likely to churn.\n"
             "- Senior citizens and single individuals are at a higher risk of leaving." if language == "English" else 
             "- العملاء الذين لديهم عقود شهرية لديهم أعلى معدل فقدان.\n"
             "- الرسوم الشهرية العالية مرتبطة بزيادة احتمال فقدان العملاء.\n"
             "- العملاء الذين لا يستخدمون الخدمات الإضافية (مثل الأمان عبر الإنترنت، الدعم الفني) أكثر عرضة للفقدان.\n"
             "- كبار السن والأفراد غير المتزوجين أكثر عرضة لترك الخدمة.")
    
    st.subheader("Recommendations" if language == "English" else "التوصيات")
    st.write("1. Offer discounts for long-term contracts to encourage retention.\n"
             "2. Create lower-priced service plans for budget-conscious customers.\n"
             "3. Promote additional services like security and tech support to reduce churn.\n"
             "4. Implement customer engagement strategies for high-risk groups." if language == "English" else 
             "1. تقديم خصومات للعقود طويلة الأجل لتشجيع الاحتفاظ بالعملاء.\n"
             "2. إنشاء خطط تسعير منخفضة للعملاء ذوي الميزانيات المحدودة.\n"
             "3. الترويج للخدمات الإضافية مثل الأمان والدعم الفني لتقليل الفقدان.\n"
             "4. تنفيذ استراتيجيات تفاعل مع العملاء للمجموعات الأكثر عرضة للمغادرة.")
# 🔮 Predict Churn Probability
elif page in ["🔮 Predict Churn Probability", "🔮 توقع احتمالية فقدان العميل"]:
    st.title("🔮 Predict Churn Probability" if language == "English" else "🔮 توقع احتمالية فقدان العميل")
    st.write("Enter customer details to predict churn probability." if language == "English" else "أدخل تفاصيل العميل لتوقع احتمالية فقدانه.")
    
    feature_names = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", 
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "TotalServices"
    ]
    
    input_features = np.zeros((1, 20))  # Create an array matching the number of features
    
    for i, feature in enumerate(feature_names):
        input_features[0, i] = st.number_input(f"{feature}", min_value=0.0, max_value=10000.0, value=0.0)
    
    if st.button("Predict Churn" if language == "English" else "توقع الفقدان"):
        input_features_scaled = scaler.transform(input_features)
        prediction = xgb_model.predict(input_features_scaled)[0]
        if prediction == "Yes":
            st.error("🚨 High risk of churn!" if language == "English" else "🚨 خطر فقدان العميل مرتفع!")
        else:
            st.success("✅ Low risk of churn!" if language == "English" else "✅ خطر فقدان العميل منخفض!")

st.sidebar.markdown("---")
st.sidebar.write(" BY ENG:**ABED AHDY**")


 
