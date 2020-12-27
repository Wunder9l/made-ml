import os
import pickle
import urllib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow import keras

REFERENCE = 'https://www.kaggle.com/kushleshkumar/cornell-car-rental-dataset'
DATASET_NAME = 'CarRentalData.csv'

CONTENT = []
RESOURCES = {
    'transformer': {'file': 'transformer.data', 'url': 'https://yadi.sk/d/3ghVMZH8i_hQgw'},  # extracts features
    'linear': {'file': 'linear_model.data', 'url': 'https://yadi.sk/d/CkaqAA2F6NP5xw'},
    'random_forest': {'file': 'random_forest.data', 'url': 'https://yadi.sk/d/miL3yIwbUTvL5g'},
    'catboost': {'file': 'catboost.data', 'url': 'https://yadi.sk/d/ja-OWV7SGNW31Q'},
    'nn': {'file': 'nn_model.h5', 'url': 'https://yadi.sk/d/yvMmCsCwd79xzw'},
}


# Defining MAPE function
def my_mape(y_predicted, y_actual):
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
    return mape


def clear_page():
    for item in CONTENT:
        item.empty()


def main():
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Investigate dataset", "Run models", "Show the source code"])
    if app_mode == "Show instructions":
        st.markdown(get_file_content_as_string('readme.md'))
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        st.code(read_file(__file__))
    elif app_mode == "Investigate dataset":
        investigate_dataset()
    elif app_mode == "Run models":
        run_models()


def read_file(filename):
    """Reads file from os"""
    with open(filename) as input_file:
        return input_file.read()

# This file downloader demonstrates Streamlit animation.
def download_file(resource):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(RESOURCES[resource]['file']):
        if "size" not in RESOURCES[resource]:
            return
        elif os.path.getsize(resource) == RESOURCES[resource]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % resource)
        progress_bar = st.progress(0)
        with open(RESOURCES[resource]['file'], "wb") as output_file:
            with urllib.request.urlopen(RESOURCES[resource]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                                            (resource, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


# @st.cache
def draw_map_plotly(df: pd.DataFrame):
    figures = []
    df_location = df.groupby('location.city').agg(
        lat=('location.latitude', 'mean'),
        lon=('location.longitude', 'mean'),
        cnt=('location.city', 'count'),
        rate=('rate.daily', 'mean')
    )
    df_location['text'] = df_location.index + '<br>Rentals count: ' + (df_location['cnt']).astype(str)
    df_location['text_rate'] = df_location.index + '<br>Avg(rate): ' + (df_location['rate']).astype(str)

    fig = go.Figure(data=go.Scattergeo(
        locationmode='USA-states',
        lon=df_location['lon'],
        lat=df_location['lat'],
        text=df_location['text'],
        mode='markers',
        marker=dict(
            size=3.*np.log2(df_location['cnt']+2),
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            line=dict(
                width=0.3,
                color='rgb(40,40,40)'
            ),
            colorscale='sunset',
            cmin=0,
            color=df_location['cnt'],
            cmax=df_location['cnt'].max(),
            colorbar_title='Number of cars<br>rented in city'
        )
    ))
    fig.update_layout(
        title='Car rentals by city (July 2020).<br>(Hover for cities names)',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.5,
            subunitwidth=0.5
        ),
    )
    figures.append(fig)

    fig = go.Figure(data=go.Scattergeo(
        locationmode='USA-states',
        lon=df_location['lon'],
        lat=df_location['lat'],
        text=df_location['text_rate'],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.7,
            autocolorscale=False,
            line=dict(
                width=0.3,
                color='rgb(40,40,40)'
            ),
            colorscale='sunsetdark',
            cmin=0,
            color=df_location['rate'],
            cmax=df_location['rate'].max(),
            colorbar_title='Avg price of rate<br>rented in city'
        )
    ))
    fig.update_layout(
        title='Avg price of rate<br>by city (July 2020).<br>(Hover for cities names)',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.5,
            subunitwidth=0.5
        ),
    )
    figures.append(fig)

    df_state = df.groupby('location.state').agg(count=('location.state', 'count')).reset_index()
    fig = px.choropleth(df_state,
                        locationmode="USA-states",
                        locations='location.state',
                        color='count',
                        color_continuous_scale="redor",
                        scope="usa",
                        title='Car rentals by State',
                        labels={'count': 'Number of cars<br>rented in state'}
                        )
    figures.append(fig)

    df_state = df.groupby('location.state').agg(rate=('rate.daily', 'mean')).reset_index()
    fig = px.choropleth(df_state,
                        locationmode="USA-states",
                        locations='location.state',
                        color='rate',
                        color_continuous_scale="reds",
                        scope="usa",
                        title='Avg rate per day by State',
                        labels={'rate': 'Avg price of rate<br>rented in state'}
                        )
    figures.append(fig)
    return figures


def draw_vehicles_characteristics(df):
    df_fuel_type = df.groupby('fuelType').agg(count=('fuelType', 'count')).reset_index()
    st.plotly_chart(
        px.pie(df_fuel_type, values='count', names='fuelType', title='By fuel type', color='fuelType', hole=0.25)
    )

    df_vehicle = df.groupby('vehicle.type').agg(count=('vehicle.type', 'count')).reset_index()
    st.plotly_chart(
        px.pie(df_vehicle, values='count', names='vehicle.type', title='By vehicle type', color='vehicle.type',
               hole=0.25)
    )

    df_year = df.groupby('vehicle.year').agg(count=('vehicle.year', 'count')).reset_index()
    st.plotly_chart(
        px.bar(df_year, y='count', x='vehicle.year', title='By vehicle year')
    )

    df_manufacturer = df.groupby(['vehicle.make', 'vehicle.model']).agg(count=('vehicle.make', 'count')).reset_index()
    st.plotly_chart(
        px.treemap(df_manufacturer,
                   path=['vehicle.make', 'vehicle.model'], values='count',
                   color='count',
                   title='Tree map of all rental cars (manufacturer->models)')
    )
    manufacturer_count = st.slider('Number of manufacturers to show (sorted descending)',
                                   min_value=5, max_value=df_manufacturer['vehicle.make'].nunique(), value=20)
    only_manufacturers = set(
        df_manufacturer.groupby('vehicle.make')
            .agg(count=('count', 'sum'))
            .sort_values('count', ascending=False)[:manufacturer_count]
            .reset_index()['vehicle.make'].values
    )

    st.plotly_chart(
        px.bar(df_manufacturer[df_manufacturer['vehicle.make'].isin(only_manufacturers)],
               y='count', x='vehicle.make', title='By manufacturer', color='vehicle.model')
            .update_layout(showlegend=False)
    )
    st.plotly_chart(
        px.box(df[df['vehicle.make'].isin(only_manufacturers)], x="vehicle.make", y="rate.daily",
               title='Rate per day by manufacturers')
    )

    df_model = df.groupby('vehicle.model').agg(count=('vehicle.model', 'count'),
                                               manufacturer=('vehicle.make', lambda x: x.values[0]))
    df_model = df_model.reset_index().sort_values('count', ascending=False)
    df_model['model'] = df_model['manufacturer'] + ': ' + df_model['vehicle.model']
    models_count = st.slider('Number of models to show (sorted descending)', 5, df_model.shape[0], 20, step=5)
    df_model = df_model[:models_count]
    st.write(df_model[['model', 'count']])
    st.plotly_chart(
        px.bar(df_model, y='count', x='model', title='By vehicle model', color='model')
    )


def draw_plotly_figures(figures):
    for fig in figures:
        st.plotly_chart(fig)


def investigate_dataset():
    """Shows different information about dataset"""
    if not os.path.exists(DATASET_NAME):
        st.markdown(f'''Original dataset is avalilable on [kaggle]({REFERENCE}).
            You have to download it and put {DATASET_NAME} to current dir''')
        return

    read_and_cache_csv = st.cache(pd.read_csv)
    df = read_and_cache_csv(DATASET_NAME)

    st.write('Sample of dataframe:')
    st.write(df.head(100))

    st.header('Rentals on map')
    draw_plotly_figures(draw_map_plotly(pd.DataFrame.copy(df)))

    st.header("Rentals by vehicle's characteristics")
    draw_vehicles_characteristics(df)

    corr = df.corr()

    st.plotly_chart(px.imshow(
        corr.values,
        labels=dict(color="Correlation"),
        x=corr.index.values,
        y=corr.columns.values,
        title='Correlation matrix of numeric features',
        color_continuous_scale='turbo',
    ))


def load_model(filename):
    with open(filename, 'rb') as model_file:
        return pickle.load(model_file)


def run_models():
    """Shows different information about dataset"""
    if not os.path.exists(DATASET_NAME):
        st.markdown(f'''Original dataset is avalilable on [kaggle]({REFERENCE}).
            You have to download it and put {DATASET_NAME} to current dir''')
        return

    # Download external dependencies.
    for resource in RESOURCES.keys():
        download_file(resource)

    # Once we have the models data - create
    read_and_cache_csv = st.cache(pd.read_csv)
    df = read_and_cache_csv(DATASET_NAME)
    data_df = df.fillna(df.mode().iloc[0])
    st.title('4 models were trained')

    test_size = 0.2
    st.write(f'''Dataset includes {df.shape[0]} samples. It was splitted by 
    train ({100*(1.0-test_size)}%) and test ({100*test_size}%) parts. MSE (MeanSquaredError) - loss function''')
    train_df, holdout_df = train_test_split(data_df, test_size=test_size, shuffle=True, random_state=42)

    X_holdout, y_holdout = holdout_df.drop(columns=['rate.daily']), holdout_df['rate.daily']

    cf = load_model(RESOURCES['transformer']['file'])
    X_test = cf.transform(X_holdout)
    st.write(f'Feature preprocessing extracts {X_test.shape[1]} features per sample')

    st.header(f'Results on test data ({X_test.shape[0]} samples):')
    linear_model = load_model(RESOURCES['linear']['file'])
    rf_model = load_model(RESOURCES['random_forest']['file'])
    catboost_model = load_model(RESOURCES['catboost']['file'])
    nn_model = keras.models.load_model(RESOURCES['nn']['file'])
    models = {
        'LinearRegression': linear_model,
        'RandomForest': rf_model,
        'CatBoost': catboost_model,
        'NeuralNetwork': nn_model,
    }
    test_results = []
    for name, model in models.items():
        prediction = model.predict(X_test.todense()) if name == 'NeuralNetwork' else model.predict(X_test)
        test_results.append({
            'Model': name,
            'MSE': mean_squared_error(prediction, y_holdout),
            'MAE': mean_absolute_error(prediction, y_holdout),
            'MAPE': my_mape(prediction.reshape(y_holdout.shape), y_holdout),
        })
    st.write(pd.DataFrame(test_results))

    for name, model in models.items():
        prediction = model.predict(X_test.todense()) if name == 'NeuralNetwork' else model.predict(X_test)
        holdout_df[f'{name}_Prediction'] = prediction
        holdout_df[f'{name}_Error'] = holdout_df[f'{name}_Prediction'] - holdout_df['rate.daily']

    result_with_predictions = ['rate.daily'] + [f'{name}_Prediction' for name in models]
    all_columns = list(holdout_df.columns)
    columns = st.multiselect('Select columns to show for test data prediction', all_columns, result_with_predictions)
    st.write(holdout_df[columns])

    sorted_by_rate = holdout_df.sort_values('rate.daily').reset_index()
    st.plotly_chart(
        px.line(sorted_by_rate, x=sorted_by_rate.index, y=result_with_predictions,
                title='Predictions (and ground truth)')
    )
    st.plotly_chart(
        px.line(sorted_by_rate, x=sorted_by_rate.index,
                y=[f'{name}_Error' for name in models],
                title='Errors of models (x - in ascending order of rate.daily)')
    )

    st.header('Try predict with models:')
    categorical_columns = list(df.select_dtypes(include=object).columns)
    params = {}
    for col in categorical_columns + ['vehicle.year', 'owner.id']:
        params[col] = st.selectbox(f'Select value for {col} column', df[col].value_counts().keys())
    params['rating'] = st.slider(f'Set rating of vehicle', min_value=1.0, max_value=5.0, value=5.0)
    params['renterTripsTaken'] = st.number_input(f'Set renterTripsTaken', min_value=0, max_value=500, value=30)
    params['reviewCount'] = st.number_input(f'Set renterTripsTaken', min_value=0, max_value=params['renterTripsTaken'],
                                            value=params['renterTripsTaken'])
    params['location.latitude'] = df[df['location.city'] == params['location.city']]['location.latitude'].mean()
    params['location.longitude'] = df[df['location.city'] == params['location.city']]['location.longitude'].mean()

    if st.button('Run models'):
        st.write('Features:')
        st.write(pd.DataFrame([params]))
        x = cf.transform(pd.DataFrame([params]))
        distances = np.array([cosine(x.toarray(), row.toarray()) for row in X_test])
        ind = np.argpartition(distances, -3)[-3:]
        st.write('3 most close items to this one from test data are')
        st.write(holdout_df.iloc[ind])

        predictions = pd.DataFrame([{
            f'{name}_Prediction': float(model.predict(x).reshape(1)[0])
            for name, model in models.items()}])
        st.write(predictions)


if __name__ == '__main__':
    main()
