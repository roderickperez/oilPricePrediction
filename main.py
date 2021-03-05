import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# Create a start date
START = "2015-01-01"

TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("CL=F", "CL=G", "NG=F", "BZ=F")

selected_stocks = st.sidebar.selectbox(
    "Select a dataset for prediction", stocks)

# Slider for years of prediction

n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years*365


@st.cache  # Avoid data be reloaded everytime
# Load Atock Data
def load_data(ticker):
    # Download data in Pandas data frame format
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # Will place the date in the first column
    return data


data_load_state = st.sidebar.text("Load data...")  # Text BEFORE data is loaded
data = load_data(selected_stocks)

# Text AFTER data is loaded
data_load_state.text("Load data...done!!!")

# Show the data

st.subheader('Raw Data')
show_raw_data = st.sidebar.checkbox("Display Raw Data")

if show_raw_data:
    st.write(data.tail(10))


# Plot data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Open'], name='Open'))
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Close'], name='Close'))
    fig.layout.update(title_text="Text Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forecasting (with FB Prophet)

df_train = data[['Date', 'Close']]
# Rename columns
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Define Prophet Model
m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)

forecast = m.predict(future)

# Show the data
st.subheader('Forecast Data')

show_forecast_data = st.sidebar.checkbox("Display Forecast Data")

if show_forecast_data:
    st.write(forecast.tail(10))

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
