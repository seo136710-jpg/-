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
    X_train_50, y_train_50 = train_50_df["경과연수"].values, train_50_df["평균기온"].values
    slope_50, intercept_50 = np.polyfit(X_train_50, y_train_50, 1)

    # 3. 최근 100년 학습 모델 (1906~2005)
    X_train_100, y_train_100 = train_100_df["경과연수"].values, train_100_df["평균기온"].values
    slope_100, intercept_100 = np.polyfit(X_train_100, y_train_100, 1)

    # 테스트 데이터(2006~2025) 예측
    X_test = test_df["경과연수"].values
    y_test = test_df["평균기온"].values

    y_pred_test_50 = slope_50 * X_test + intercept_50
    mae_50, mse_50, r2_50 = evaluate_model(y_test, y_pred_test_50)

    y_pred_test_100 = slope_100 * X_test + intercept_100
    mae_100, mse_100, r2_100 = evaluate_model(y_test, y_pred_test_100)

    # 섹션 1: 전체 데이터 성능
    st.subheader("1. 전체 데이터셋 모델 평가")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기울기 (연간 상승폭)", f"{slope_all:.4f} °C/년")
    c2.metric("MAE", f"{mae_all:.4f} °C")
    c3.metric("MSE", f"{mse_all:.4f}")
    c4.metric("R² Score", f"{r2_all:.4f}")

    st.markdown("---")

    # 섹션 2: 최근 50년 vs 100년 학습 모델 비교 (테스트셋: 2006~2025)
    st.subheader("2. 최근 50년 vs 최근 100년 학습 모델 비교 (테스트 평가)")
    
    comp_df = pd.DataFrame({
        "구분": ["최근 50년 학습 (1956~2005)", "최근 100년 학습 (1906~2005)"],
        "기울기 (°C/년)": [slope_50, slope_100],
        "10년당 상승 폭 (°C)": [slope_50 * 10, slope_100 * 10],
        "테스트 MAE (°C)": [mae_50, mae_100],
        "테스트 MSE": [mse_50, mse_100],
        "테스트 R² Score": [r2_50, r2_100]
    })
    
    st.dataframe(comp_df.style.format({
        "기울기 (°C/년)": "{:.4f}",
        "10년당 상승 폭 (°C)": "{:.4f}",
        "테스트 MAE (°C)": "{:.4f}",
        "테스트 MSE": "{:.4f}",
        "테스트 R² Score": "{:.4f}"
    }), use_container_width=True)

    # 인사이트 요약 Box
    st.info(f"""
    💡 **학습 기간별 기울기 및 성능 비교 요약**
    - **기울기 비교**: 최근 50년 모델의 기울기({slope_50:.4f})가 최근 100년 모델({slope_100:.4f})보다 급합니다. 이는 최근으로 올수록 기온 상승 속도(온난화)가 가속화되었음을 보여줍니다.
    - **예측 성능 비교**: 최근 50년 모델이 테스트 데이터(2006~2025)에 대한 예측 오차가 더 적고(MAE: {mae_50:.4f} vs {mae_100:.4f}), 높은 결정계수(R²)를 보여 최신 온난화 경향성을 더 잘 반영합니다.
    """)

    st.markdown("---")

    # 섹션 3: Plotly 시각화
    st.subheader("3. 회귀선 시각화 비교")
    
    years = np.arange(1900, 2026)
    x_range = years - 1908

    fig = go.Figure()

    # 실제 데이터
    fig.add_trace(go.Scatter(
        x=df["연도"], y=df["평균기온"],
        mode='markers', name='실제 연평균기온',
        marker=dict(color='gray', opacity=0.5)
    ))
    
    # 테스트 데이터 강조
    fig.add_trace(go.Scatter(
        x=test_df["연도"], y=test_df["평균기온"],
        mode='markers', name='테스트 데이터 (2006~2025)',
        marker=dict(color='red', size=8)
    ))

    # 최근 50년 회귀선
    fig.add_trace(go.Scatter(
        x=years, y=slope_50 * x_range + intercept_50,
        mode='lines', name='최근 50년 학습 회귀선 (1956~2005)',
        line=dict(color='blue', width=2)
    ))

    # 최근 100년 회귀선
    fig.add_trace(go.Scatter(
        x=years, y=slope_100 * x_range + intercept_100,
        mode='lines', name='최근 100년 학습 회귀선 (1906~2005)',
        line=dict(color='green', width=2, dash='dash')
    ))

    fig.update_layout(
        title="학습 기간별 회귀선과 최근 20년 테스트 데이터 비교",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
