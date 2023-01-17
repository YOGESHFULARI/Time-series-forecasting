import streamlit as st
from prophet import Prophet
from datetime import date
from prophet.plot import plot_plotly
import pickle
from plotly import graph_objs as go
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
from streamlit_lottie import st_lottie
import requests


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets1.lottiefiles.com/temp/lf20_tfGoXW.json"
lottie_json = load_lottieurl(lottie_url)

st_lottie(lottie_json,height=150,width=150)
final_data=pd.read_csv('Gold_data.csv')
st.title('Gold Price Forecasting App')
values=st.slider('No.Of Days',1,60)
period=values
df_train=final_data[['date','price']]
df_train=df_train.rename(columns={'date':'ds','price':'y'})
m=Prophet()
my_model=m.fit(df_train)
future=m.make_future_dataframe(periods=period,freq='D')
forecast=my_model.predict(future)
fig=plot_plotly(m,forecast)
forecast=forecast.rename(columns={'yhat':'Price','yhat_lower':'Lower price','yhat_upper':'Higher price'})

fig2=m.plot_components(forecast,uncertainty=True,)

tab1, tab2,tab3 = st.tabs(['Forecasted result','Forecasted plot','Forecasting components'])

tab1.subheader("Forecasted Prices ")
tab1.write(forecast[['ds','Price','Lower price','Higher price']].tail(period))

tab2.subheader(f'Forecast plot for {period} days')
tab2.plotly_chart(fig)

tab3.subheader("Forecasting Components")
tab3.write(fig2)