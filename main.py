import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 학습 및 평가")
st.markdown("과거 학습 데이터(**최근 50년**, **최근 100년**)에 따라 회귀 모델을 생성하고, 최근 20년(2006~2025) 테스트 데이터의 예측 성능을 비교합니다.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8").dropna(subset=["평균기온"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    valid = df.groupby("연도").filter(lambda x: len(x) >= 300 and x["연도"].iloc[0] <= 2025)
    yearly = valid.groupby("연도")["평균기온"].mean().reset_index()
    yearly["경과연수"] = yearly["연도"] - 1908
    return yearly

def eval_model(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred),
        r2_score(y_true, y_pred)
    )

try:
    df = load_and_preprocess_data()

    # 데이터 분할
    test_df = df[(df["연도"] >= 2006) & (df["연도"] <= 2025)]
    tr50_df = df[(df["연도"] >= 1956) & (df["연도"] <= 2005)]
    tr100_df = df[(df["연도"] >= 1906) & (df["연도"] <= 2005)]

    # 1. 전체 데이터 모델
    s_all, i_all = np.polyfit(df["경과연수"], df["평균기온"], 1)
    mae_all, mse_all, r2_all = eval_model(df["평균기온"], s_all * df["경과연수"] + i_all)

    # 2. 최근 50년 (1956~2005) & 3. 최근 100년 (1906~2005)
    s_50, i_50 = np.polyfit(tr50_df["경과연수"], tr50_df["평균기온"], 1)
    s_100, i_100 = np.polyfit(tr100_df["경과연수"], tr100_df["평균기온"], 1)

    # 테스트 평가 (2006~2025)
    mae_50, mse_50, r2_50 = eval_model(test_df["평균기온"], s_50 * test_df["경과연수"] + i_50)
    mae_100, mse_100, r2_100 = eval_model(test_df["평균기온"], s_100 * test_df["경과연수"] + i_100)

    # 지표 출력
    st.subheader("1. 전체 데이터셋 모델 평가")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기울기", f"{s_all:.4f} °C/년")
    c2.metric("MAE", f"{mae_all:.4f} °C")
    c3.metric("MSE", f"{mse_all:.4f}")
    c4.metric("R² Score", f"{r2_all:.4f}")

    st.markdown("---")
    st.subheader("2. 최근 50년 vs 최근 100년 학습 모델 비교 (테스트 평가)")

    comp
