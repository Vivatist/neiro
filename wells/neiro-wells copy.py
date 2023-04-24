#https://www.bizkit.ru/2019/11/05/14921/

from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('wells/wells.csv')

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
d = scaler.fit_transform(df)
source_columns = df.columns
normalise_df = pd.DataFrame(d, columns=source_columns)

#выводим информацию на экран
print(df.describe().transpose())
print(normalise_df)
#sns.pairplot(df[['length_name', 'numbers_in_name', 'access_time', 'modification_time']], diag_kind='kde')
#sns.distplot(df.modification_time) #распределение значений
plt.show()

#разделяем набор данных на обучающий набор и тестовый
train_dataset = normalise_df.sample(frac=0.7)
test_dataset = normalise_df.drop(train_dataset.index)

#отделяем параметры от меток (параметр, который будем прогнозировать)
x_train = train_dataset.drop(['Qж после, куб.м/сут'], axis = 1).to_numpy()
y_train = train_dataset['Qж после, куб.м/сут'].to_numpy()
x_test = test_dataset.drop(['Qж после, куб.м/сут'], axis = 1).to_numpy()
y_test = test_dataset['Qж после, куб.м/сут'].to_numpy()

#Строим простую полносвязную нейронную сеть
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary()) # архитектура нашей модели
 
#компилируем сеть
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

#обучаем
history = model.fit(x_train, 
                    y_train, 
                    epochs=100, 
                    validation_split=0.1, 
                    verbose=2)


# Делаем прогноз. Возвращается копия предсказания в виде одномерного массива
pred = model.predict(x_test).flatten() 


#TODO необходимо нормализовать вручную либо разобраться как денормализовать значения
# Возвращаем к прежнему размеру
##pred = pred * max_y + min_y 
#y_test = y_test * max_y + min_y

# Средний модуль отклонения 
err = np.mean(abs(pred - y_test))
print('Средний модуль отклонения: {}'.format(err))


# Предсказание vs правильный  ответ
for i in range(len(pred)):
  #print("Сеть сказала: ", round(pred[i],2), ", а верный ответ: ", round(y_test[i],2), ", разница: ", round(pred[i] - y_test[i],2))
  print("Сеть сказала: ", round(pred[i],2), ", а верный ответ: ", round(y_test[i],2), ", разница: ", round(((pred[i] / y_test[i]) * 100) - 100,2),'%')

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



#Построение гистограммы ошибок
# Разность предсказанного и правильного ответа
error = pred - y_test
plt.hist(abs(error), bins = 25)
plt.xlabel("Значение ошибки")
plt.ylabel("Количество")
plt.show()
