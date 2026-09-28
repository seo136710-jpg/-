import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.markdown("1908년부터의 서울 기온 데이터를 바탕으로 선형 회귀 모델을 만들어 미래/과거의 기온을 예측합니다.")

# 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df

try:
    raw_df = load_data()

    # 데이터 전처리
    # 1. 연도별 관측일수 계산 및 2025년 이하 데이터 필터링
    valid_years = raw_df.groupby("연도").filter(lambda x: len(x) >= 300 and x["연도"].iloc[0] <= 2025)
    
    # 2. 연도별 평균기온 계산
    yearly_df = valid_years.groupby("연도")["평균기온"].mean().reset_index()

    # 독립변수 (1908년부터의 경과 연수) 계산
    yearly_df["경과연수"] = yearly_df["연도"] - 1908

    X = yearly_df["경과연수"].values
    y = yearly_df["평균기온"].values

    # 선형 회귀 분석 (1차 다항식 피팅)
    slope, intercept = np.polyfit(X, y, 1)

    # 상관계수 계산
    correlation = np.corrcoef(X, y)[0, 1]

    # 회귀선 데이터를 위한 연도 범위 (1900년 ~ 2100년)
    full_years = np.arange(1900, 2101)
    full_x = full_years - 1908
    full_y_pred = slope * full_x + intercept

    # 데이터 정보 표시
    min_year = yearly_df["연도"].min()
    max_year = yearly_df["연도"].max()
    count_years = len(yearly_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 대상 연도 수", f"{count_years}개해")
    col2.metric("시작 연도", f"{min_year}년")
    col3.metric("끝 연도", f"{max_year}년")
    col4.metric("상관계수 (r)", f"{correlation:.4f}")

    st.markdown("---")

    # 슬라이더 및 예측 결과 표시
    selected_year = st.slider("예측하고 싶은 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

    predicted_temp = slope * (selected_year - 1908) + intercept

    st.metric(
        label=f"🔮 {selected_year}년 예상 연평균 기온",
        value=f"{predicted_temp:.2f} °C"
    )

    # Plotly 시각화
    fig = go.Figure()

    # 1. 실제 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=yearly_df["연도"],
        y=yearly_df["평균기온"],
        mode='markers',
        name='실제 연평균 기온',
        marker=dict(color='blue', opacity=0.7)
    ))

    # 2. 회귀 직선
    fig.add_trace(go.Scatter(
        x=full_years,
        y=full_y_pred,
        mode='lines',
        name='선형 회귀선',
        line=dict(color='red', width=2)
    ))

    # 3. 선택된 연도의 예측 지점 강조
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode='markers',
        name=f'{selected_year}년 예측점',
        marker=dict(color='green', size=12, symbol='star')
    ))

    fig.update_layout(
        title="서울 연평균 기온 추이 및 회귀선",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified",
        xaxis=dict(dtick=20)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
