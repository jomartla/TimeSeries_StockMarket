"""
Create a simple stocks correlation dashboard.

Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve stocks
at your command prompt.

"""
import os
from functools import lru_cache
from os.path import dirname, join

import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.models.widgets import RadioButtonGroup, Div, Slider
from bokeh.plotting import figure, curdoc

DATA_DIR = join(dirname(__file__), 'Dataset')

DEFAULT_TICKERS = ['AMZN', 'AAPL', 'GOOGL', 'IBM']


def nix(val, lst):
    return [x for x in lst if x != val]


def load_ticker(ticker):
    file_name = join(DATA_DIR, '%s_2006-01-01_to_2018-01-01.csv' % ticker.lower())
    data = pd.read_csv(file_name, header=1, parse_dates=['date'],
                       names=['date', 'o', 'h', 'l', 'c', 'v', 'name'])

    if slider.title == "Days used for prediction":
        pred_file_name = join(DATA_DIR, "predictions_stock-market-length-%d.csv" % slider.value)
    else:
        pred_file_name = join(DATA_DIR, "predictions_stock-market-length-30.csv")

    pred = pd.read_csv(pred_file_name, header=1, parse_dates=['date'],
                       names=['date', 'AAPL_pred', 'AMZN_pred', 'GOOGL_pred', 'IBM_pred'], sep=";")

    data = pd.merge(data, pred, on='date', how='outer')

    data = data.set_index('date')

    return pd.DataFrame({ticker: data.c,
                         ticker + '_returns': data.c.diff(),
                         ticker + '_smooth_1': data.c.rolling(window=slider.value).mean(),
                         ticker + '_pred': data[ticker + '_pred']})


def get_data(t1, t2):
    df1 = load_ticker(t1)
    df2 = load_ticker(t2)

    data = pd.concat([df1, df2], axis=1)
    # data = data.dropna()

    data['t1'] = data[t1]
    data['t2'] = data[t2]

    data['t1_smooth_1'] = data[t1 + '_smooth_1']
    data['t2_smooth_1'] = data[t2 + '_smooth_1']

    data['t1_returns'] = data[t1 + '_returns']
    data['t2_returns'] = data[t2 + '_returns']

    data['t1_pred'] = data[t1 + '_pred']
    data['t2_pred'] = data[t2 + '_pred']

    return data


#
ticker1 = Select(value=DEFAULT_TICKERS[0], width=150, options=nix(DEFAULT_TICKERS[1], DEFAULT_TICKERS))
ticker2 = Select(value=DEFAULT_TICKERS[1], width=150, options=nix(DEFAULT_TICKERS[0], DEFAULT_TICKERS))

ticker3 = Select(value=DEFAULT_TICKERS[0], width=150, options=nix(DEFAULT_TICKERS[0], DEFAULT_TICKERS))

# Radio buttons group
radio_button_group = RadioButtonGroup(labels=['Normal', 'Moving Average', 'Prediction CNN'], active=0)

# Radio buttons group
radio_button_group_second = RadioButtonGroup(labels=['Prediction CNN', 'Prediction LSTM'], active=0)

# Slider
slider = Slider(start=5, end=50, value=1, step=1, title="-", visible=False)

# set up plots
source_static = ColumnDataSource(data=dict(date=[], t1=[], t2=[],
                                           t1_returns=[], t2_returns=[],
                                           t1_smooth_1=[], t2_smooth_1=[],
                                           t1_pred=[], t2_pred=[]))
tools = 'pan,wheel_zoom,reset'

ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime')
ts1_line = ts1.line('date', 't1', color="blue", source=source_static)
ts1_smooth_1 = ts1.line('date', 't1_smooth_1', color="red", visible=False, source=source_static)
ts1_pred = ts1.line('date', 't1_pred', color="red", visible=False, source=source_static)

ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime')
ts2.x_range = ts1.x_range
ts2.line('date', 't2', color="blue", source=source_static)
ts2_smooth_1 = ts2.line('date', 't2_smooth_1', color="red", visible=False, source=source_static)
ts2_pred = ts2.line('date', 't2_pred', color="red", visible=False, source=source_static)

# Figure ts3
IMAGE_TYPE = "Cnn1d"

ts3_img_path = "BokehApp/static/" + str(ticker3.value) + "_" + IMAGE_TYPE + ".png"
ts3 = Div(text="""<img src=\"""" + ts3_img_path + """\" alt="div_image">""",
          width=864, height=432)


# set up callbacks
def ticker1_change(attrname, old, new):
    ticker2.options = nix(new, DEFAULT_TICKERS)
    update()


def ticker2_change(attrname, old, new):
    ticker1.options = nix(new, DEFAULT_TICKERS)
    update()


def ticker3_change(attrname, old, new):
    update()


def slider_change(attrname, old, new):
    update()


def radio_button_group_change(attrname, old, new):
    if new == 0:
        ts1_smooth_1.visible = False
        ts2_smooth_1.visible = False
        ts1_pred.visible = False
        ts2_pred.visible = False
        slider.visible = False
    elif new == 1:
        ts1_smooth_1.visible = True
        ts2_smooth_1.visible = True
        ts1_pred.visible = False
        ts2_pred.visible = False
        slider.title = "Slide window"
        slider.visible = True
        slider.start = 5
        slider.step = 1
        slider.end = 120
        slider.value = 5
    elif new == 2:
        ts1_smooth_1.visible = False
        ts2_smooth_1.visible = False
        ts1_pred.visible = True
        ts2_pred.visible = True
        slider.title = "Days used for prediction"
        slider.visible = True
        slider.start = 10
        slider.step = 20
        slider.end = 50
        slider.value = 10

    update()

def radio_button_group_second_change(attrname, old, new):
    update()


def update(selected=None):
    t1, t2 = ticker1.value, ticker2.value

    if radio_button_group_second.active == 0:
        IMAGE_TYPE = "Cnn1d"
    elif radio_button_group_second.active == 1:
        IMAGE_TYPE = "LSTM_20days"

    ts3_img_path = "BokehApp/static/" + str(ticker3.value) + "_" + IMAGE_TYPE + ".png"
    ts3.text = """<img src=\"""" + ts3_img_path + """\" alt="div_image">"""

    df = get_data(t1, t2)

    data = df[['t1', 't2',
               't1_returns', 't2_returns',
               't1_smooth_1', 't2_smooth_1',
               't1_pred', 't2_pred']]

    source_static.data = data

    ts1.title.text, ts2.title.text = t1, t2


ticker1.on_change('value', ticker1_change)
ticker2.on_change('value', ticker2_change)
ticker3.on_change('value', ticker3_change)

slider.on_change('value', slider_change)
radio_button_group.on_change('active', radio_button_group_change)
radio_button_group_second.on_change('active', radio_button_group_second_change)

# set up layout


heading_general = Div(text="""<h2>Time series assignment</h2><style>h2, p {margin: 0;}</style>""" +
                       """&nbsp&nbsp&nbsp&nbsp""" +
                       """by Wladyslaw Eysymontt, Juan Luis Ruiz-Tagle Oriol and Jorge Martín Lasaosa</br>""")

blank_space = Div(text="""""")

heading_cnn = Div(text="""<h4>Time series presentation, smoothing and CNN prediction</h4>""")

widgets = column(ticker1, ticker2, radio_button_group, slider)
series = column(ts1, ts2, sizing_mode="stretch_width")

heading_LSTM = Div(text="""<h4>Time series binary prediction with LSTM and CNN</h4>""")

widgets_LSTM = column(ticker3, radio_button_group_second)
series_LSTM = column(ts3, sizing_mode="stretch_width")


layout = column(heading_general, blank_space,
                heading_cnn, row(widgets, series), blank_space,
                heading_LSTM, row(widgets_LSTM, series_LSTM))

# initialize
update()

curdoc().add_root(layout)
curdoc().title = "Time Series Assignment"
