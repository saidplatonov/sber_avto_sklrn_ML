import datetime
import os.path
import dill
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Metrics
from sklearn.metrics import roc_auc_score


def filter_columns(df):
    """
    get only usable columns
    :type df: DataFrame
    """
    df_new = df.copy()
    df_new = df_new[['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_category',
                     'device_os', 'device_brand', 'device_screen_resolution', 'device_browser', 'geo_country',
                     'geo_city']]
    print("filter_columns Done")
    return df_new



def empty_data_standardization(df):
    df_new = df.copy()
    df_new.replace(['(not set)', ''], np.nan, inplace=True)
    print("empty_data_standardization Done")
    return df_new


def change_rar_values(df):
    sessions_df_cleaned = df.copy()

    min_rarity = sessions_df_cleaned.shape[0] / 1000000  # 0.001

    column_to_update = ['utm_source', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                        'device_category', 'device_os', 'device_brand', 'device_browser',
                        'geo_country', 'geo_city']
    # Преоброзуем редкие значения в каждой категориальной колонке приравняв к 'other'
    for col in column_to_update:
        columns_val_counts = sessions_df_cleaned[col].value_counts()
        if columns_val_counts[columns_val_counts <= min_rarity].shape[0]:  # если есть редкие значения
            # назначим значение other для всех значения встречающихся min_rarity раз илли меньше.
            sessions_df_cleaned.loc[
                sessions_df_cleaned[col].isin(
                    columns_val_counts[columns_val_counts <= min_rarity].keys().to_list()), [col]] = 'other'
    print("change_rar_values Done")
    return sessions_df_cleaned


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

    sessions_cr_df.fillna({'device_brand': 'empty'}, inplace=True)
    print("device_brand_filling Done")
    return sessions_cr_df


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

    sessions_cr_df.fillna({'device_os': 'empty'}, inplace=True)
    print("device_os_filling Done")
    return sessions_cr_df


def get_is_organic(df):
    sessions_cr_df = df.copy()

    def set_organic(x, is_organic=0):
        if pd.notna(x) and x in ['organic', 'referral', '(none)']:
            is_organic = 1
        return is_organic

    sessions_cr_df['is_organic_visit'] = sessions_cr_df['utm_medium'].apply(set_organic)
    sessions_cr_df.drop(columns=['utm_medium'], inplace=True)
    print("get_is_organic Done")
    return sessions_cr_df


def add_screan_width_height(df):
    sessions_df_cleaned = df.copy()
    sessions_df_cleaned['device_screen_width'] = sessions_df_cleaned['device_screen_resolution'].apply(
        lambda w: int(w.lower().split('x')[0]))
    sessions_df_cleaned['device_screen_height'] = sessions_df_cleaned['device_screen_resolution'].apply(
        lambda h: int(h.lower().split('x')[1]))

    def get_boundaries(datacol):
        # функция определения границ выбросов
        minimum = datacol.mean() - 3 * datacol.std()
        maximum = datacol.mean() + 3 * datacol.std()
        boundaries = (minimum, maximum)
        return boundaries

    boundaries_w = get_boundaries(sessions_df_cleaned['device_screen_width'])
    boundaries_h = get_boundaries(sessions_df_cleaned['device_screen_height'])
    sessions_df_cleaned.loc[
        (sessions_df_cleaned.device_screen_width < boundaries_w[0]), ['device_screen_width']] = round(boundaries_w[0])
    sessions_df_cleaned.loc[
        (sessions_df_cleaned.device_screen_width > boundaries_w[1]), ['device_screen_width']] = round(boundaries_w[1])

    sessions_df_cleaned.loc[
        (sessions_df_cleaned.device_screen_height < boundaries_h[0]), ['device_screen_height']] = round(boundaries_h[0])
    sessions_df_cleaned.loc[
        (sessions_df_cleaned.device_screen_height > boundaries_h[1]), ['device_screen_height']] = round(boundaries_h[1])
    print("add_screan_width_height Done")
    return sessions_df_cleaned


def add_is_socialmedia_advert(df):
    sessions_df_new = df.copy()
    is_socmedia_advert = ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                          'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm')
    sessions_df_new['is_socialmedia_advert'] = sessions_df_new['utm_source'].apply(lambda u:
                                                                                   1 if u in is_socmedia_advert else 0)
    print("add_is_socialmedia_advert Done")
    return sessions_df_new


def add_display_megapixel(df):
    sessions_df_new = df.copy()
    sessions_df_new['device_display_megapixel'] = round(
        sessions_df_new['device_screen_width'] * sessions_df_new['device_screen_height'] / 1000000, 2)
    print("add_display_megapixel Done")
    return sessions_df_new


def add_orientation_vertical(df):
    sessions_df_new = df.copy()
    sessions_df_new['device_orientation_vertical'] = sessions_df_new.apply(
        lambda x: 1 if x['device_screen_width'] < x['device_screen_height'] else 0, axis=1)
    print("add_orientation_vertical Done")
    return sessions_df_new


def add_from_russia(df):
    sessions_df_new = df.copy()

    def set_from_russia(x, is_russia=0):
        if pd.notna(x) and x == 'Russia':
            is_russia = 1
        return is_russia

    sessions_df_new['from_russia'] = sessions_df_new['geo_country'].apply(set_from_russia)
    sessions_df_new.drop(columns=['geo_country'], inplace=True)
    print("add_from_russia Done")
    return sessions_df_new


def get_data_by_cityname(df):
    sessions_df_new = df.copy()

    # Так как услугу автоподписки целесообразна только для России, приравняем все города кроме Российских к 'other'
    sessions_df_new.loc[sessions_df_new['from_russia'] == 0, 'geo_city'] = 'other'

    # Откроем ранее созданный файл с данными о Российских городах
    full_cities_df = pd.read_csv("../cities_data/full_ru_cities_data.csv")

    def expand_rows(row):
        """
        Создаст новую строку для каждого альтернативного имени города.
        Сохранит остальные данные (Population, Timezone, Geo_lat, Geo_long, km_to_moscow) без изменений.
        Разделит значения в столбце Alternate Names по запятым и создаст новые строки в DataFrame для каждого альтернативного имени.
        Функция для расширения DataFrame

        :param row: DataFrame row
        :return: list with many rows content all alternate name in column Name and info about:
        (Population, Timezone, Geo_lat, Geo_long, km_to_moscow)
        """

        alternate_names = str(row['Alternate Names']).split(',')
        rows = []
        for name in alternate_names:
            new_row = row.copy()
            new_row['Name'] = name.strip()
            rows.append(new_row)
        return rows

    # Применение функции ко всем строкам датафрейма и объединение результата
    expanded_rows = full_cities_df.apply(lambda row: expand_rows(row), axis=1)

    # Использование explode для распаковки списков в строки DataFrame
    full_cities_df = pd.DataFrame([item for sublist in expanded_rows for item in sublist]).reset_index(drop=True)

    full_cities_df.drop_duplicates(subset=['Name'], keep=False, inplace=True)
    """
    Объединяем DataFrame sessions_df_new с full_cities_df по столбцу 'geo_city'
    Необходимо сделать датафрейм из full_cities_df где в Name будут все возможные вариации из Alternate Names    
    """
    sessions_withnewdata_df = pd.merge(sessions_df_new,
                                       full_cities_df[['Name', 'Population', 'Timezone', 'km_to_moscow']],
                                       left_on='geo_city', right_on='Name', how='left')

    sessions_withnewdata_df.rename(columns={
        'Population': 'city_population',
        'Timezone': 'city_timezone',
        'km_to_moscow': 'city_km_to_moscow'
    }, inplace=True)

    sessions_withnewdata_df.drop(columns='Name', inplace=True)

    # Заменяем пропущенные значения в случае отсутствия соответствия города в full_cities_df
    sessions_withnewdata_df.fillna({'city_population': round(full_cities_df['Population'].median()),
                                    'city_timezone': round(full_cities_df['Timezone'].max()),
                                    'city_km_to_moscow': round(full_cities_df['km_to_moscow'].median())
                                    }, inplace=True)

    # Преобразуем типы столбцов, если это необходимо
    sessions_withnewdata_df['city_population'] = sessions_withnewdata_df['city_population'].astype(int)

    # def get_boundaries(datacol):
    #     # определение границ выбросов
    #     minimum = datacol.mean() - 3 * datacol.std()
    #     maximum = datacol.mean() + 3 * datacol.std()
    #     boundaries = (minimum, maximum)
    #     return boundaries
    #
    # boundaries_tz = get_boundaries(sessions_withnewdata_df['city_timezone'])
    # boundaries_kmmsk = get_boundaries(sessions_withnewdata_df['city_km_to_moscow'])
    # # удаляем выбросы ['city_timezone']
    # sessions_withnewdata_df.loc[(sessions_withnewdata_df.city_timezone < boundaries_tz[0]
    #                              ), ['city_timezone']] = round(boundaries_tz[0])
    # sessions_withnewdata_df.loc[(sessions_withnewdata_df.city_timezone > boundaries_tz[1]
    #                              ), ['city_timezone']] = round(boundaries_tz[1])
    # # удаляем выбросы ['city_km_to_moscow']
    # sessions_withnewdata_df.loc[(sessions_withnewdata_df.city_km_to_moscow < boundaries_kmmsk[0]
    #                              ), ['city_km_to_moscow']] = round(boundaries_kmmsk[0])
    # sessions_withnewdata_df.loc[(sessions_withnewdata_df.city_km_to_moscow > boundaries_kmmsk[1]
    #                              ), ['city_km_to_moscow']] = round(boundaries_kmmsk[1])

    print("get_data_by_cityname Done, ", sessions_withnewdata_df.shape)
    return sessions_withnewdata_df


def main():
    print("Conversion Rate Prediction Pipeline")

    data_filters = Pipeline(steps=[
        ('columns_filter', FunctionTransformer(filter_columns)),
        # ('empty_values_standardization', FunctionTransformer(empty_data_standardization)),
        # ('delete_rar_values', FunctionTransformer(change_rar_values)),
    ])

    data_generators = Pipeline(steps=[
        ('device_brand_generator', FunctionTransformer(device_brand_filling)),
        ('device_os_generator', FunctionTransformer(device_os_filling)),
        ('organic_visitor_definition', FunctionTransformer(get_is_organic)),
        ('add_screen_width_height', FunctionTransformer(add_screan_width_height)),
        ('get_social_media_ad', FunctionTransformer(add_is_socialmedia_advert)),
        ('add_device_display_megapixel', FunctionTransformer(add_display_megapixel)),
        ('add_display_orientation', FunctionTransformer(add_orientation_vertical)),
        ('add_from_russia', FunctionTransformer(add_from_russia)),
        ('city_data_generator', FunctionTransformer(get_data_by_cityname)),
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('na_value_filling', SimpleImputer(strategy='constant', fill_value='empty')),  # missing_values=pd.NA,
        ('ohe_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    col_transformers = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64', 'int8'])),
            ('categorical', categorical_transformer, make_column_selector(dtype_include=[object, 'category']))
        ])

    preprocessor = Pipeline(steps=[
        ('filtering', data_filters),
        ('generator', data_generators),
        ('columns_transform', col_transformers)
    ])

    mymodel = SGDClassifier(random_state=42, loss='log_loss', penalty='l2', n_jobs=-1)

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', mymodel)
    ])

    def create_final_dataframes(path_test: str, path_train: str):
        sessions_df = pd.read_pickle("../data/ga_sessions.pkl")
        hits_df = pd.read_pickle("../data/ga_hits.pkl")

        # создание колонки о совершение одного из целевых действий
        target_action_types = (
        'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click', 'sub_custom_question_submit_click',
        'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click')
        hits_df['target_action'] = hits_df['event_action'].apply(lambda x: 1 if x in target_action_types else 0)

        # Создадим датафрейм содержащий session_id и CR (Conversion Rate)
        session_target_df = hits_df[['session_id', 'target_action']].groupby(['session_id'], as_index=False).max().rename(
            columns={'target_action': 'conversion_rate'})

        # Обьеденение 2 получившихся датафрейма
        sessions_cr_df = pd.merge(sessions_df, session_target_df, on='session_id')

        # Разделение датафрейма на обучающую и тестовую выборки
        sessions_cr_df, finaltest_df = train_test_split(sessions_cr_df, test_size=0.001, random_state=42,
                                                        stratify=sessions_cr_df['conversion_rate'])

        finaltest_df.to_pickle(path_test)
        sessions_cr_df.to_pickle(path_train)

        return print(f"DF with 99% data: {path_train} 1% data to test final model {path_test}")

    path_testfinaldf = "../tmp/data_to_test_api.pkl"
    path_trainfinaldf = "../tmp/data_to_train_api.pkl"

    if not os.path.isfile(path_testfinaldf) and not os.path.isfile(path_trainfinaldf):
        create_final_dataframes(path_testfinaldf, path_trainfinaldf)

    train_data = pd.read_pickle(path_trainfinaldf)
    train_target = train_data['conversion_rate']
    train_data.drop(columns=['conversion_rate'], inplace=True)
    print(train_data.shape, train_target.shape)


    # Использую только для выявления лучших гиперпараметров
    def hyper_param_tunning(pipe, train_data, train_target):
        # При необходимости можно прогнать pipe чтобы получить модель с оптимальными настройки Гиперпараметров
        # Эта функция использовалась для настройки гиперпараметров уже выбранной модели.

        # сетка для оптимизации Гипер Параметров выбранной модели
        param_grid = {
            'classifier__alpha': [0.0001, 0.001, 0.00001],
            # 'classifier__penalty': ['l2', 'l1', 'elasticnet'],
            'classifier__max_iter': [1000, 2000, 3000],
            # 'classifier__early_stopping': [True, False],
            'classifier__class_weight': [{0: 1, 1: 10}, {0: 1, 1: 2}, {0: 0.5, 1: 20}, {0: 0.51, 1: 17.22},
                                         {0: 0.1, 1: 0.34}]
        }

        rnd_search = RandomizedSearchCV(pipe, param_grid, n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)

        # Обучение модели с оптимизацией гиперпараметров
        rnd_search.fit(train_data, train_target)


        # Лучшие параметры
        print("Лучшие параметры: ", rnd_search.best_params_)

        # Лучший классификатор
        best_model = rnd_search.best_estimator_

        return best_model



    best_model = pipe.set_params(**{'classifier__max_iter': 3000,
                                  'classifier__class_weight': {0: 0.5, 1: 20},
                                  'classifier__alpha': 0.0001})

    best_model.fit(train_data, train_target)


    # Тестирование модели
    test_df = pd.read_pickle(path_testfinaldf)
    y = test_df['conversion_rate']
    X = test_df.drop(columns=['conversion_rate'])

    # Предсказания вероятностей для тестового набора
    y_pred_proba = best_model.predict_proba(X)[:, 1]

    # Оценка модели с использованием ROC-AUC
    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f'ROC-AUC Score: {roc_auc:.4f}')

    dump_data = {
        'model': best_model,
        'metadata': {
            'name': 'fitted model to predict conversion rate',
            'author': 'Said Platonov',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_model.named_steps["classifier"]).__name__,
            'ROC_AUC': roc_auc
        }
    }
    file_name = 'model_pipe.pkl'
    with open(file_name, 'wb') as file:
        dill.dump(dump_data, file)


if __name__ == '__main__':
    main()
