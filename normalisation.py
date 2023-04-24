'''
Скрипт машинного обучения. Анализирует базу собранную скриптом neiro.py
и пытается предсказать время жизни фалов в папке Загрузки
#https://www.bizkit.ru/2019/11/05/14921/
'''
import sqlite3
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


PATH_TO_BASE = "C:\\Users\\anvab\\OneDrive\\Документы\\Coding\\Python\\neiro\\base.db"
#PATH_TO_BASE = "/home/andrey/Dropbox/Coding/Python/neiro/base.db"

conn_sql = sqlite3.connect(PATH_TO_BASE)
df = pd.read_sql('SELECT rus_symbol_name, is_hidden, is_double, is_temp,'
                 'numbers_in_name,length_name, is_dir, ext, size, creation_time,'
                 'access_time, modification_time FROM source WHERE living = 0', conn_sql)

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

source_columns = df.columns
d = scaler.fit_transform(df)

normalise_df = pd.DataFrame(d, columns=source_columns)

# выводим информацию на экран
print(df.describe().transpose())
print(normalise_df)
#sns.pairplot(df[['length_name', 'numbers_in_name', 'access_time', 'modification_time']], diag_kind='kde')
# sns.distplot(df.modification_time) #распределение значений
plt.show()

# разделяем набор данных на обучающий набор и тестовый
train_dataset = normalise_df.sample(frac=0.7, random_state=0)
test_dataset = normalise_df.drop(train_dataset.index)

# отделяем параметры от меток (параметр, который будем прогнозировать)
x_train = train_dataset.drop(['modification_time'], axis=1).to_numpy()
y_train = train_dataset['modification_time'].to_numpy()
x_test = test_dataset.drop(['modification_time'], axis=1).to_numpy()
y_test = test_dataset['modification_time'].to_numpy()

# Строим простую полносвязную нейронную сеть
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())  # архитектура нашей модели

# компилируем сеть
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# обучаем
history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    validation_split=0.1,
                    verbose=2)


# Делаем прогноз. Возвращается копия предсказания в виде одномерного массива
pred = model.predict(x_test).flatten()


# TODO необходимо нормализовать вручную либо разобраться как денормализовать значения
# Возвращаем к прежнему размеру
##pred = pred * max_y + min_y
#y_test = y_test * max_y + min_y

# Средний модуль отклонения
err = np.mean(abs(pred - y_test))
print('Средний модуль отклонения: {}'.format(err))


# Предсказание vs правильный  ответ
for i in range(len(pred)):
    print("Сеть сказала: ", round(pred[i], 2), ", а верный ответ: ", round(
        y_test[i], 2), ", разница: ", round(pred[i] - y_test[i], 2))


# Считаем графики ошибки
plt.plot(history.history['mae'],
         label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'],
         label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()


# Разброс предсказаний может показать перекос, если есть
plt.scatter(y_test, pred)
plt.xlabel('Правильные значение')
plt.ylabel('Предсказания')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()


# Построение гистограммы ошибок
# Разность предсказанного и правильного ответа
error = pred - y_test
plt.hist(abs(error), bins=25)
plt.xlabel("Значение ошибки")
plt.ylabel("Количество")
plt.show()
