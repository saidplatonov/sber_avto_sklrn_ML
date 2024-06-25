import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

def filter_columns(df):
    """
    get only usable columns
    :type df: DataFrame
    """
    df_new = df.copy()
    return df_new[['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_category',
               'device_os', 'device_brand', 'device_screen_resolution', 'device_browser', 'geo_country',
               'geo_city']]


def empty_data_standardization(df):
    df_new = df.copy()
    return df_new.replace(['(not set)', ''], pd.NA, inplace=True)

def device_brand_filling(df):
    sessions_cr_df = df.copy()

    # Заполняем Пустоты device_brand = 'Apple' если используемый браузер Safari
    sessions_cr_df.loc[(sessions_cr_df['device_brand'].isna()) &
                       (sessions_cr_df['device_browser'].isin(['Safari', 'Safari (in-app)'])
                        ), ['device_brand']] = 'Apple'

    # Заполняем Пустоты device_brand = 'Samsung' если используемый браузер Samsung Internet
    sessions_cr_df.loc[(sessions_cr_df['device_brand'].isna()) &
                       (sessions_cr_df['device_browser'].str.contains('Samsung')
                        ), ['device_brand']] = 'Samsung'

    # Заполняем device_brand = Apple если 'device_os' = ['Macintosh', 'iOS']
    sessions_cr_df.loc[(sessions_cr_df['device_brand'].isna()) &
                       (sessions_cr_df['device_os'].isin(['Macintosh', 'iOS'])
                        ), ['device_brand']] = 'Apple'

    return sessions_cr_df.fillna({'device_brand': 'empty'}, inplace=True)

def device_os_filling(df):
    sessions_cr_df = df.copy()

    # Заполняем 'device_os' == 'iOS' если 'device_category' = ['mobile', 'tablet'] и 'device_brand' = Apple
    sessions_cr_df.loc[(sessions_cr_df['device_os'].isna()) &
                       (sessions_cr_df['device_category'].isin(['mobile', 'tablet'])) &
                       (sessions_cr_df['device_brand'] == 'Apple'
                        ), ['device_os']] = 'iOS'

    # 'device_os' == 'Android' если 'device_category' = ['mobile', 'tablet']
    sessions_cr_df.loc[(sessions_cr_df['device_os'].isna()) &
                       (sessions_cr_df['device_category'].isin(['mobile', 'tablet'])) &
                       (sessions_cr_df['device_brand'] != 'Apple'
                        ), ['device_os']] = 'Android'

    # 'device_os' == 'Macintosh' если 'device_category' = desktop и 'device_brand' = Apple
    sessions_cr_df.loc[(sessions_cr_df['device_os'].isna()) &
                       (sessions_cr_df['device_category'] == 'desktop') &
                       (sessions_cr_df['device_brand'] == 'Apple'
                        ), ['device_os']] = 'Macintosh'

    # 'device_os' == 'Macintosh' если 'device_category' = desktop и 'device_browser' = Safari
    sessions_cr_df.loc[(sessions_cr_df['device_os'].isna()) &
                       (sessions_cr_df['device_category'] == 'desktop') &
                       (sessions_cr_df['device_browser'] == 'Safari'
                        ), ['device_os']] = 'Macintosh'

    # 'device_os' == 'Android' если 'device_browser' содержит Android и device_category = desktop
    sessions_cr_df.loc[(sessions_cr_df['device_os'].isna()) &
                       (sessions_cr_df['device_category'] == 'desktop') &
                       (sessions_cr_df['device_browser'].str.contains('Android')
                        ), ['device_os']] = 'Android'

    # 'device_os' == 'Windows' если 'device_browser' == ['Edge', 'Internet Explorer'] и device_category = desktop
    sessions_cr_df.loc[(sessions_cr_df['device_os'].isna()) &
                       (sessions_cr_df['device_category'] == 'desktop') &
                       (sessions_cr_df['device_browser'].isin(['Edge', 'Internet Explorer'])
                        ), ['device_os']] = 'Windows'

    return sessions_cr_df.fillna({'device_os': 'empty'}, inplace=True)


def get_data_by_cityname(df):


    return sessions_withnewdata_df


def main():
    print("Conversion Rate Prediction Pipeline")

    data_filters = Pipeline(steps=[
        ('columns_filter', FunctionTransformer(filter_columns)),
        ('empty_values_standardization', FunctionTransformer(empty_data_standardization)),
    ])

    data_generators = Pipeline(steps=[
        ('device_brand_generator', FunctionTransformer(device_brand_filling)),
        ('device_os_generator', FunctionTransformer(device_os_filling)),
        ('city_data_generator', FunctionTransformer( )),
    ])


    categorical_transformer = Pipeline(steps=[
        ('na_value_filling', SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='empty')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    col_transformers = ColumnTransformer(
        transformers=[('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64', 'int8'])),
                      ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
                      ])