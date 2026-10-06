import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 선형회귀 모델 학습 및 평가")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8").dropna(subset=["평균기온"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    valid = df.groupby("연도").filter(lambda x: len(x) >= 300 and x["연도"].iloc[0] <= 2025)
    yearly = valid.groupby("연도")["평균기온"].mean().reset_index()
    yearly["경과연수"] = yearly["연도"] - 1908
    return yearly

df = load_data()

# 데이터 분할
test_df = df[(df["연도"] >= 2006) & (df["연도"] <= 2025)]
tr50_df = df[(df["연도"] >= 1956) & (df["연도"] <= 2005)]
tr100_df = df[(df["연도"] >= 1906) & (df["연도"] <= 2005)]

# 1. 전체 데이터 모델
s_all, i_all = np.polyfit(df["경과연수"], df["평균기온"], 1)
pred_all = s_all * df["경과연수"] + i_all
mae_all = mean_absolute_error(df["평균기온"], pred_all)
mse_all = mean_squared_error(df["평균기온"], pred_all)
r2_all = r2_score(df["평균기온"], pred_all)

# 2. 최근 50년 & 최근 100년 모델
s_50, i_50 = np.polyfit(tr50_df["경과연수"], tr50_df["평균기온"], 1)
s_100, i_100 = np.polyfit(tr100_df["경과연수"], tr100_df["평균기온"], 1)

# 테스트 평가
pred_50 = s_50 * test_df["경과연수"] + i_50
pred_100 = s_100 * test_df["경과연수"] + i_100

mae_50, mse_50, r2_50 = mean_absolute_error(test_df["평균기온"], pred_50), mean_squared_error(test_df["평균기온"], pred_50), r2_score(test_df["평균기온"], pred_50)
mae_100, mse_100, r2_100 = mean_absolute_error(test_df["평균기온"], pred_100), mean_squared_error(test_df["평균기온"], pred_100), r2_score(test_df["평균기온"], pred_100)

# 출력
st.subheader("1. 전체 데이터셋 모델 평가")
c1, c2, c3, c4 = st.columns(4)
c1.metric("기울기", f"{s_all:.4f} °C/년")
c2.metric("MAE", f"{mae_all:.4f} °C")
c3.metric("MSE", f"{mse_all:.4f}")
c4.metric("R² Score", f"{r2_all:.4f}")

st.markdown("---")
st.subheader("2. 최근 50년 vs 최근 100년 학습 모델 비교")

comp_df = pd.DataFrame({
    "구분": ["최근 50년 (1956~2005)", "최근 100년 (1906~2005)"],
    "기울기 (°C/년)": [s_50, s_100],
    "10년당 상승 폭 (°C)": [s_50 * 10, s_100 * 10],
    "테스트 MAE (°C)": [mae_50, mae_100],
    "테스트 MSE": [mse_50, mse_100],
    "테스트 R² Score": [r2_50, r2_100]
})
st.dataframe(comp_df, use_container_width=True)

st.info(f"💡 최근 50년 모델의 기울기({s_50:.4f})가 최근 100년 모델({s_100:.4f})보다 급하며, 최근 20년 테스트 데이터에 대한 MAE 오차({mae_50:.4f})도 더 적어 최신 온난화 추세를 더 잘 반영합니다.")

st.markdown("---")
st.subheader("3. 회귀선 시각화 비교")

years = np.arange(1900, 2026)
x_range = years - 1908

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["연도"], y=df["평균기온"], mode='markers', name='실제 연평균기온', marker=dict(color='gray', opacity=0.5)))
fig.add_trace(go.Scatter(x=test_df["연도"], y=test_df["평균기온"], mode='markers', name='테스트 데이터 (2006~2025)', marker=dict(color='red', size=8)))
fig.add_trace(go.Scatter(x=years, y=s_50 * x_range + i_50, mode='lines', name='최근 50년 학습 회귀선', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=years, y=s_100 * x_range + i_100, mode='lines', name='최근 100년 학습 회귀선', line=dict(color='green', width=2, dash='dash')))

fig.update_layout(title="학습 기간별 회귀선 비교", xaxis_title="연도", yaxis_title="평균기온 (°C)")
st.plotly_chart(fig, use_container_width=True)
