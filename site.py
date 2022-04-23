import streamlit as st  
import numpy as np 
import pandas as pd 
import altair as alt 

st.title('Rethinking Money: An interactive demo')

def liq_composition():
    st.header('Liquidity Composition')
    st.text('Explore how invariant curves can be chained or composed to form new ones!')

    C1 = st.slider('Curve 1 C', min_value=1, max_value= 900)
    X1 = 10
    C2 = st.slider('Curve 2 C', min_value=1, max_value= 900)
    X2 = 10

    RESOLUTION = 500 

    x_range = np.linspace(-20, 20, RESOLUTION)
    y_range = C2/X2 - C2 /(X2 + (C1/X1 - C1 / (x_range + X1))) 

    data = pd.DataFrame({
        'X': x_range,
        'Y': y_range
    })

    chart = alt.Chart(data).mark_point(clip=True).encode(
        x = alt.X('X'),
        y = alt.Y('Y', scale=alt.Scale(domain=[-50, 50]))
    ) 

    st.altair_chart(chart)


def two_stocks_one_token():
    st.header('2 Stocks 1 Token')
    st.text('Explore the reflexive interaction between liquidity and token valuation. Assumed simple constant product formula for liquidity pools.')

    W1 = np.array(10,0,0)
    W2 = np.array(0,10,0)
    L = np.array(0,0,10)


    data = pd.DataFrame({
    })

    chart = alt.Chart(data).mark_point(clip=True).encode(
    ) 

    st.altair_chart(chart)