import duckdb
import time
from functools import lru_cache
import pymorphy3

start_time = time.time()

# Инициализируем русский морфологический анализатор
morph = pymorphy3.MorphAnalyzer()

# 500_000 слов в кэше сэкономят нам часы вычислений.
@lru_cache(maxsize=500000)
def get_lemma(word):
    # [0] берет наиболее вероятный вариант. normal_form — это лемма.
    return morph.parse(word)[0].normal_form

def lemmatize_tokens_udf(tokens):
    """Лемматизация списка токенов"""
    if not tokens:
        return []
    return [get_lemma(token) for token in tokens]

# Подключаемся к DuckDB
conn = duckdb.connect()

# Указываем типы данных для максимальной скорости передачи между C++ и Python
conn.create_function(
    "lemmatize_tokens", 
    lemmatize_tokens_udf, 
    [duckdb.list_type(str)], 
    duckdb.list_type(str)
)

# Выполняем SQL-запрос
conn.execute("""
    CREATE OR REPLACE TABLE processed_data AS 
    
    -- Шаг 1: Очистка текста
    WITH cleaned AS (
        SELECT 
            project_short_name,
            LOWER(TRIM(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                COALESCE(question, ''),
                                '^/[a-zA-Z0-9_]+|\\s/[a-zA-Z0-9_]+', '' -- Удаление команд бота (/start, /help)
                            ),
                            '&[a-zA-Z0-9#]+;', ' ' -- HTML сущности (&nbsp;, &#123;)
                        ),
                        'https?://\\S+|www\\.\\S+', '' -- Ссылки
                    ),
                    '\\s+', ' ' -- Нормализация пробелов
                )
            )) AS question,
            
            LOWER(TRIM(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                COALESCE(answer, ''),
                                '^/[a-zA-Z0-9_]+|\\s/[a-zA-Z0-9_]+', ''
                            ),
                            '&[a-zA-Z0-9#]+;', ' '
                        ),
                        'https?://\\S+|www\\.\\S+', ''
                    ),
                    '\\s+', ' '
                )
            )) AS answer,
             
             LOWER(TRIM(COALESCE(project_short_name, ''),    
                )
            )) AS project 
        FROM read_parquet('chatbots_dataset.parquet')
    ),
    
    -- Шаг 2: Нативная токенизация DuckDB (С поддержкой русского языка)
    tokenized AS (
        SELECT 
            *,
            regexp_extract_all(question, '[а-яёa-z0-9_]+(?:-[а-яёa-z0-9_]+)*') AS question_tokens,
            regexp_extract_all(answer, '[а-яёa-z0-9_]+(?:-[а-яёa-z0-9_]+)*') AS answer_tokens,
        FROM cleaned
    )
    
    -- Шаг 3: Вызов Python UDF (Лемматизация)
    SELECT 
        project_short_name,
        question,
        answer,
        question_tokens,
        answer_tokens,
        lemmatize_tokens(question_tokens) AS question_lemmas,
        lemmatize_tokens(answer_tokens) AS answer_lemmas
    FROM tokenized
""")


## ПОИСК ПО СЛОВАРЮ 
print("Подготовка списка ключевых слов...")
# Загружаем CSV и превращаем столбец 'normalized_term' в один большой массив (LIST).

conn.execute("""
    CREATE OR REPLACE TEMP TABLE filter_words AS
    SELECT list(LOWER(TRIM(normalized_term))) as keyword_list
    FROM read_csv_auto(
    'illegal_terms_dictionary_edit.csv',
    delim=',',
    header=True,
    quote='"',
    escape='"',
    ignore_errors=True
)
    WHERE normalized_term IS NOT NULL AND TRIM(normalized_term) != ''
""")

print("Фильтрация датасета...")
# Создаем новую таблицу только с нужными строками 
# Ищем пересечения как в вопросах (question_lemmas), так и в ответах (answer_lemmas)
conn.execute("""
    CREATE OR REPLACE TABLE filtered_data AS 
    WITH target AS (
        SELECT keyword_list FROM filter_words LIMIT 1
    )
    SELECT pd.*
    FROM processed_data pd, target t
    WHERE 
        -- Условие: длина массива пересечений больше 0
        -- Ищем совпадения в вопросах
        len(list_intersect(pd.question_lemmas, t.keyword_list)) > 0
        
        -- ИЛИ ищем совпадения в ответах (закомментируйте строку ниже, если не нужно)
        OR len(list_intersect(pd.answer_lemmas, t.keyword_list)) > 0
""")

print(f"Фильтрация завершена за {time.time() - start_time:.2f} сек")

# Получаем статистику
original_count = conn.execute("SELECT COUNT(*) FROM processed_data").fetchone()[0]
filtered_count = conn.execute("SELECT COUNT(*) FROM filtered_data").fetchone()[0]

print(f"\n✅ Результаты фильтрации:")
print(f"Исходных строк: {original_count:,}")
print(f"Найдено строк с ключевыми словами: {filtered_count:,}")
print(f"Процент совпадений: {(filtered_count/original_count)*100:.2f}%")

print("\nСохранение отфильтрованных данных...")
conn.execute("""
    COPY filtered_data TO 'filtered_chatbots_dataset.parquet' (FORMAT PARQUET)
""")

