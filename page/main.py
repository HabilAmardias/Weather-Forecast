import streamlit as st
from utils import fetch_data, fetch_forecast, adfuller_test, calculate_acf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config('Gunung Putri Weather',
                   layout='wide',page_icon=':sun_small_cloud:')
st.title('Gunung Putri Weather')

st.markdown("""
This Streamlit app provides insights into weather patterns in Gunung Putri, Indonesia. Weather data is taken from [Open-Meteo](https://open-meteo.com/).

- 6-Day Forecast: Utilizes a Vector Autoregression (VAR) model to predict future weather conditions (updated every 5 days).
- Historical Data: Explore Gunung Putri historical weather data.
- Statistical Analysis: Includes Dickey-Fuller tests for stationarity and autocorrelation analysis to understand the relationships and characteristics of the weather data.
""")

st.markdown('Disclaimer: Weather forecasting involves inherent uncertainty. This page provides a prediction and should not be considered absolutely accurate.')

try:
    data = fetch_data()
except Exception as e:
    st.error(f'Error when fetching data: {str(e)}')

try:
    forecast = fetch_forecast()
except Exception as e:
    st.error(f'Error when fetching forecast: {str(e)}')

st.dataframe(data.tail(),use_container_width=True)

with st.expander("Data Description",expanded=False):
    st.write("""
             temperature_2m_max: Maximum daily air temperature at 2 meters above ground\n
             temperature_2m_min: Minimum daily air temperature at 2 meters above ground\n
             apparent_temperature_max: Maximum daily apparent temperature\n
             apparent_temperature_min: Minimum daily apparent temperature\n
             daylight_duration: Number of seconds of daylight per day\n
             sunshine_duration: The number of seconds of sunshine per day is determined by calculating 
             direct normalized irradiance exceeding 120 W/m², following the WMO definition. 
             Sunshine duration will consistently be less than daylight duration due to dawn and dusk.\n
             wind_speed_10m_max: Maximum wind speed on a day\n
             wind_gusts_10m_max: Maximum wind gusts on a day\n
             """)

tab1,tab2,tab3 = st.tabs(['Forecast and Actuals', 'Past (Historical) Weathers', 'Statistical Analysis'])

with tab1:    
    fig = make_subplots(
        rows=4, cols=2,  
        subplot_titles=[f"{metric}: Forecast and Actuals" for metric in data.columns]
    )

    row, col = 1, 1
    for metric in forecast.columns:
        fig.add_trace(
            go.Scatter(x=data.index[-6:], y=data[metric].iloc[-6:], mode='lines', name=metric),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=forecast.index, y=forecast[metric], mode='lines', name=f"{metric}_forecast"),
            row=row, col=col
        )
        
        if col < 2:
            col += 1
        else:
            col = 1
            row += 1
    fig.update_layout(
        height=1200,  
        width=1200,   
        title="Forecast and Actuals",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = make_subplots(
        rows=4, cols=2,  
        subplot_titles=[f"{metric}: Past Data" for metric in data.columns]
    )
    moving_average = data.rolling(30).mean()
    row, col = 1, 1
    for metric in forecast.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[metric], mode='lines', name=metric),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=moving_average.index, y=moving_average[metric], mode='lines', name=f"{metric} 30 days moving average"),
            row=row, col=col
        )
        
        if col < 2:
            col += 1
        else:
            col = 1
            row += 1
    fig.update_layout(
        height=1200,  
        width=1200,   
        title="Past data",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""### Augmented Dickey-Fuller Unit Root Test""")
    st.markdown("""
    <small>Statistical test to test whether a time series is stationary or non-stationary. 
    The null hypothesis of Augmented Dickey-Fuller test is that time series is non-stationary.
    Alternative hypothesis of the test is that the time series is stationary.</small>
    """,unsafe_allow_html=True)
    st.markdown("""
    <small>If a time series is stationary then its variance does not change over time</small>
    """,unsafe_allow_html=True)
    adf_results = adfuller_test(data)
    col1,col2 = st.columns([1,1],border=True,vertical_alignment='center')
    with col1:
        for i in range(4):
            result = adf_results[i]
            st.markdown(f"""
                        Augmented Dickey-Fuller Test on {result[0]}
                        """)
            
            st.markdown(f"""
                        Statistic: {result[1]}\n
                        P-Value: {result[2]}\n
                        Lags Used: {result[3]}\n
                        {result[0]} is {result[4]}
                        """)
            if i != 3:
                st.markdown('------------------------------------------------')
    with col2:
        for i in range(4,8):
            result = adf_results[i]
            st.markdown(f"""
                        Augmented Dickey-Fuller Test on {result[0]}
                        """)
            
            st.markdown(f"""
                        Statistic: {result[1]}\n
                        P-Value: {result[2]}\n
                        Lags Used: {result[3]}\n
                        {result[0]} is {result[4]}
                        """)
            if i != 7:
                st.markdown('------------------------------------------------')

    st.markdown("""### Autocorrelation Plot""")
    st.markdown("""
    <small>Autocorrelation is the correlation between the given time series and a lagged version of itself.</small>
    """,unsafe_allow_html=True)
    acf_results = calculate_acf(data)

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[f"{metric}: Autocorrelation Plot" for metric in data.columns]
    )
    row, col = 1, 1
    for i,metric in enumerate(data.columns):
        result = acf_results[i]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(result))), 
                y=result, 
                mode='markers', 
                name=metric,
                marker=dict(color='red',line=dict(color='white', width=1),size=10), 
                showlegend=False),
            row=row, col=col
        )
        if col < 2:
            col += 1
        else:
            col = 1
            row += 1
    
    fig.update_layout(
        height=1200,
        width=1200,
    )
    st.plotly_chart(fig, use_container_width=True)