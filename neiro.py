"""
Скрипт сбора статистики о файлах в папке Загрузки. Данные грузятся
в базу для дальнейшей нормализации и обработки нейросетью
"""
import os
import datetime
import pathlib
import sqlite3
import re


PATH = "C:\\Users\\anvab\\Downloads"

PATH_TO_BASE = "C:\\Users\\anvab\\OneDrive\\Документы\\Coding\\Python\\neiro\\base.db"

EXT = (
    ".torrent",
    ".exe",
    ".rpm",
    ".deb",
    ".arj",
    ".iso",
    ".zip",
    ".mp4",
    ".mp3",
    ".pdf",
    ".docx",
    ".doc",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".jpeg",
    ".jpg",
)


def deltahours(date_time: int):
    """возвращает прошедшее c date_time количество минут"""
    now = datetime.datetime.now()
    return (now - datetime.datetime.fromtimestamp(date_time)).total_seconds() // 60


def check_symbol(text):
    """Проверяет, есть ли в text русские символы"""
    alphabet = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    return not alphabet.isdisjoint(text.lower())


def check_numbers(text):
    """Проверяет, есть ли в text цифры"""
    alphabet = set("1234567890")
    return not alphabet.isdisjoint(text.lower())


class SourceCandidate:
    # TODO: заменить  на структуру
    """Класс файлового объекта. используется для
    анализа и хранения признаков файла"""
    # pylint: disable=too-many-instance-attributes
    # подавляем предупреждение pylint о большом количестве атрибутов класса

    def __init__(self, fullname):
        self.fullname = fullname

        self.name = os.path.splitext(os.path.basename(self.fullname))[0]

        self.rus_symbol_name = 1 if check_symbol(self.name) else 0

        self.is_hidden = 1 if re.match(r"^\..*", self.name) else 0

        self.is_temp = 1 if re.match(r"^\~.*", self.name) else 0

        self.is_double = 1 if re.match(r".*\([0-9]\)", self.name) else 0

        self.numbers_in_name = 1 if check_numbers(self.name) else 0

        self.length_name = len(self.name)

        self.is_dir = 1 if os.path.isdir(self.fullname) else 0

        extention = os.path.splitext(self.fullname)[1].lower()
        self.extention = EXT.index(extention) + 1 if extention in EXT else 0

        stat = os.stat(self.fullname)

        if self.is_dir == 1:
            total_size = 0
            # pylint: disable=unused-variable
            for root, dirs, files in os.walk(self.fullname):
                # pylint: enable=unused-variable
                for specific_file in files:
                    total_size += os.path.getsize(os.path.join(root, specific_file))
            self.size = total_size
        else:
            self.size = stat.st_size

        self.creation_time = deltahours(stat.st_ctime)

        self.access_time = deltahours(stat.st_atime)

        self.modification_time = deltahours(stat.st_mtime)

        self.deleted = 0


# pylint: enable=too-many-instance-attributes
# * конец описания класса ----------------------------------------------------


conn_sql = sqlite3.connect(PATH_TO_BASE)
cursor_sql = conn_sql.cursor()

# выводит в консоль содержимое SQL таблицы


def print_table(table_name):
    """печатает таблицу table_name"""
    print(f"Таблица {table_name}:")
    sql = f"SELECT * FROM {format(table_name)}"
    cursor_sql.execute(sql)
    print(cursor_sql.fetchall())


# добавляет запись в таблицу


def add_record(file_candidate, living):
    """добавляет в базу запись о файле"""
    sql = """INSERT INTO source
                          (living, full_name, name, rus_symbol_name, is_hidden,
                          is_double, is_temp, numbers_in_name,length_name, is_dir, 
                          ext, size, creation_time, access_time, modification_time)
                          VALUES
                          (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"""
    # pylint: disable=line-too-long
    cursor_sql.execute(
        sql,
        (
            living,
            file_candidate.fullname,
            file_candidate.name,
            file_candidate.rus_symbol_name,
            file_candidate.is_hidden,
            file_candidate.is_double,
            file_candidate.is_temp,
            file_candidate.numbers_in_name,
            file_candidate.length_name,
            file_candidate.is_dir,
            file_candidate.extention,
            file_candidate.size,
            file_candidate.creation_time,
            file_candidate.access_time,
            file_candidate.modification_time,
        ),
    )
    # pylint: enable=line-too-long
    conn_sql.commit()


print("Старт проверки")
file_name_list = []

# формируем список с файлами из папки
print(f"Формируем список файлов из папки {PATH} ")
for file in pathlib.Path(PATH).iterdir():
    full_file_name = os.path.join(PATH, file.name)
    file_name_list.append(full_file_name)

print(f"В папке {PATH} найдено {len(file_name_list)} файлов")

# Формируем список с файлами из базы
print("Формируем список с файлами зарегистрированными в базе")
info = cursor_sql.execute("SELECT [full_name] FROM source WHERE living=1")
base_name_list = info.fetchall()
# преобразуем кортеж в список
base_name_list = list(sum(base_name_list, ()))

print(f"В базе найдено {len(base_name_list)} живых файлов")

print("СИНХРОНИЗАЦИЯ:")

print(f"Обновляем даты в базе для всех файлов в папке {PATH} ")
for f in file_name_list:
    SQL_ADD_FILE = """UPDATE source SET creation_time = ?, access_time = ?, modification_time = ?
            WHERE full_name=? AND living=1"""
    sd = SourceCandidate(f)
    cursor_sql.execute(
        SQL_ADD_FILE,
        (
            sd.creation_time,
            sd.access_time,
            sd.modification_time,
            sd.fullname,
        ),
    )

# Добавляем в базу новые файлы
result = list(set(file_name_list) - set(base_name_list))
for r in result:
    sd = SourceCandidate(r)
    add_record(sd, 1)
    print(f" В базу добавлен файл {sd.fullname}")

# Помечаем в базе удаленные файлы
result = list(set(base_name_list) - set(file_name_list))
for r in result:
    SQL_MARK_AS_DELETED = """UPDATE source SET living=0 WHERE full_name=? AND living=1"""
    # sd = SourceCandidate(r)
    cursor_sql.execute(SQL_MARK_AS_DELETED, (r,))
    print(f" Файл {r} помечен как удаленный")

conn_sql.commit()  # применяем все изменения в базе
