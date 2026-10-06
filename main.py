import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 학습 및 평가")
st.markdown("""
과거 학습 데이터(**최근 50년**, **최근 100년**)에 따라 회귀 모델을 생성하고, 
공통 테스트 데이터(**최근 20년: 2006~2025년**)에 대한 예측 성능(MAE, MSE, R²) 및 기울기를 비교합니다.
""")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 평균기온 결측치(NaN) 제거
    df = df.dropna(subset=["평균기온"])
    
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 관측일 300일 이상, 2025년 이하 데이터만 추출
    valid_years = df.groupby("연도").filter(lambda x: len(x) >= 300 and x["연도"].iloc[0] <= 2025)
    yearly_df = valid_years.groupby("연도")["평균기온"].mean().reset_index()
    yearly_df["경과연수"] = yearly_df["연도"] - 1908
    return yearly_df

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

try:
    df = load_and_preprocess_data()

    # 데이터셋 분할
    test_df = df[(df["연도"] >= 2006) & (df["연도"] <= 2025)]
    train_50_df = df[(df["연도"] >= 1956) & (df["연도"] <= 2005)]
    train_100_df = df[(df["연도"] >= 1906) & (df["연도"] <= 2005)]

    # 1. 전체 데이터 모델
    X_all, y_all = df["경과연수"].values, df["평균기온"].values
    slope_all, intercept_all = np.polyfit(X_all, y_all, 1)
    y_pred_all = slope_all * X_all + intercept_all
    mae_all, mse_all, r2_all = evaluate_model(y_all, y_pred_all)

    # 2. 최근 50년 학습 모델 (1956~2005)
    X_train_50, y_train_
