# sber_avto_sklrn_ML
Задача:
Задача: Научитесь предсказывать совершение целевого действия (ориентировочное значение ROC-AUC ~ 0.65) — факт совершения пользователем целевого действия.
Упакуйте получившуюся модель в сервис, который будет брать на вход все атрибуты, типа utm_, device_, geo_*, и отдавать на выход 0/1 (1 — если пользователь совершит любое целевое действие).

Project structure:<br>
![image](https://github.com/saidplatonov/sber_avto_sklrn_ML/assets/170549436/8e53bfde-607c-4f02-8768-7da10aa2dfeb)
<br>
./research_result.ipynb - Проведенные исследования данных и подбор моделей (с заметками и комментариями)<br>
./cities_data = Здесь распологают дополнительная информация о городах России, эти данные обработаны в JupyterNoutbook на выходе полчен DataFrame с данными которые мы можем использовать для генерации новых фитч.<br>
./cities_data/full_ru_cities_data.csv - Обработанные данные о городах россии для генерации новых фитч.<br>
./model/pipeline.py - Генератор model_pipeline.pkl модель которую мы в дальнейшем будем использовать в API<br>
./main.py - REST API сервис



