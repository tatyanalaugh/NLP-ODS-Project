import asyncio
import time
import os
import json
import csv
import re
import pandas as pd
import openai
from pathlib import Path
from tqdm.asyncio import tqdm
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv("/Users/vika/PyCharmMiscProject/.env")
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ──────────────────────────────────────────────
# НАСТРОЙКИ
# ──────────────────────────────────────────────
BASE_URL = "BASE_URL"
API_KEY  = "API_KEY"

MODEL_NAME      = "MODEL_NAME "
MAX_CONCURRENCY = 256
MAX_TEXT_LEN    = 1500


# ──────────────────────────────────────────────
# ЗАГРУЗКА СЛОВАРЯ
# ──────────────────────────────────────────────
def load_drug_terms(csv_path="illegal_terms_dictionary_edit.csv"):
    terms = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = (row.get("normalized_term") or "").strip()
            cat  = (row.get("category") or "").strip()
            if not term or cat == "crypto_payment":
                continue
            term = re.sub(r"\([^)]*\)", "", term)
            if "(" in term or ")" in term or len(term) > 25:
                continue
            term = term.strip(" .,:;").lower()
            if len(term) >= 2:
                terms.add(term)
    return ", ".join(sorted(terms))


DRUG_TERMS = load_drug_terms()
DRUG_TERMS_SHORT = (
    "клад, закладка, касание, ненаход, кладмен, стаф, ширево, камень, марка, соль, "
    "химия, мяу, пыха, бошка, шмаль, мутить, рисовать, духовка, баян, приход, кайф, "
    "ломки, спайс, мефедрон, метамфетамин, героин, кокс, гаш, трава, кислота, грибы, "
    "барыга, пушер, гонец, передоз, трип, эйфория"
)

print(f"Загружено терминов в DRUG_TERMS: {len(DRUG_TERMS.split(', '))}")
print(f"Терминов в DRUG_TERMS_SHORT: {len(DRUG_TERMS_SHORT.split(', '))}")

# Проверка на фигурные скобки (они ломают .format())
if '{' in DRUG_TERMS or '}' in DRUG_TERMS:
    print("ВНИМАНИЕ: DRUG_TERMS содержит фигурные скобки — экранируем")
    DRUG_TERMS = DRUG_TERMS.replace('{', '{{').replace('}', '}}')

# ──────────────────────────────────────────────
# ЗАГРУЗКА ДАТАСЕТА
# ──────────────────────────────────────────────
# Используем train.parquet для оценки
df_train = pd.read_parquet("train.parquet").reset_index(drop=True)

def row_to_input_json(row, idx):
    question = str(row["question"]) if pd.notna(row["question"]) else ""
    answer   = str(row["answer"])   if pd.notna(row["answer"])   else ""
    text = f"Вопрос: {question}\nОтвет: {answer}"
    return {
        "id":   f"{row['session_id']}___{idx}",  # составной ключ
        "text": text[:MAX_TEXT_LEN],
    }

input_jsons = [row_to_input_json(row, idx) for idx, row in df_train.iterrows()]

print(f"\nВсего записей в train: {len(input_jsons)}")
print(f"Уникальных session_id: {len({i['id'] for i in input_jsons})}")
print("\nПример:")
print(json.dumps(input_jsons[0], ensure_ascii=False, indent=2))

# Проверка уникальности ID — если сломается, значит session_id не уникален
assert len({i["id"] for i in input_jsons}) == len(input_jsons), (
    "session_id не уникален в train.parquet! "
    "Нужно использовать составной ключ (session_id + порядковый номер)."
)

# ──────────────────────────────────────────────
# ПРОМПТЫ
# ──────────────────────────────────────────────
SYSTEM_PROMPT = "Ты - помощник по классификации текста для задачи модерации на предмет упоминания наркотиков."

class DrugMentionClassification(BaseModel):
    id: str
    has_drug_mention: bool

prompt_b = """
РОЛЬ:
Ты — высокоточная система бинарной классификации (NLP-модель), предназначенная для модерации сообщений Telegram. Твоя цель — максимально точно (с приоритетом на высокий F1-score) определять наличие упоминаний наркотических и психоактивных веществ.

ЗАДАЧА:
Определи, содержит ли текст сообщения упоминания наркотиков или связанной с ними деятельности.

ФОРМАТ ВХОДА:
JSON с полями:
- id: идентификатор сообщения
- text: текст сообщения (единственный источник анализа)

ФОРМАТ ВЫХОДА:
Строго JSON:
{{
  "id": <id из входа>,
  "has_drug_mention": true | false
}}


ОПРЕДЕЛЕНИЕ КЛАССА true:
Ставь true, если выполнено ХОТЯ БЫ ОДНО из условий:

1. Прямое упоминание наркотиков:
   - каннабис, марихуана, гашиш, кокаин, героин, амфетамин, метамфетамин, экстази, LSD и т.д.
   - любые термины из словаря ниже (полное или частичное совпадение по корню)

2. Сленг, жаргон, эвфемизмы:
   - шишки, травка, соль (в наркотическом контексте), меф, спиды, колёса, бошки и т.д.
   - английский сленг: weed, coke, meth, molly, acid и т.д.

3. Намеренно искажённые слова:
   - замены символов: м@рuху@на, к0к@ин, мефедр0н
   - пробелы/разделители: "м е ф", "к о к с"
   - транслит: marikhuana, geroin, mefedron

4. Контекст действий:
   - покупка, продажа, обмен, доставка, закладки
   - употребление, хранение, производство
   - поиск: "где взять", "купить", "есть ли", "ищу"

5. Косвенные сигналы:
   - эмодзи: 💊 🌿 🍁 ❄️ 🔥 🚬 💉
   - сочетание нейтральных слов с подозрительным контекстом

6. Частично неоднозначные случаи:
   - если есть разумное подозрение на наркотический контекст → true

ОПРЕДЕЛЕНИЕ КЛАССА false:
Ставь false, если:

1. Упоминания отсутствуют полностью
2. Слова-омонимы используются в бытовом значении:
   - "соль", "сахар", "таблетки" без контекста наркотиков
3. Лекарства:
   - если это медицинский контекст без признаков злоупотребления
4. Явная ирония или метафоры:
   - "я подсел на кофе как на наркотик"
5. Общие разговоры без связи с наркотиками

ОГРАНИЧЕНИЯ:
- Используй ТОЛЬКО поле text
- НЕ добавляй объяснений
- НЕ добавляй новых полей
- НЕ изменяй структуру JSON
- Ответ ДОЛЖЕН быть валидным JSON

ПРИМЕР:

ВХОД:
{{
  "id": "12345",
  "text": "где купить меф?"
}}

ВЫХОД:
{{
  "id": "12345",
  "has_drug_mention": true
}}

СЛОВАРЬ ТЕРМИНОВ НАРКОТИЧЕСКОЙ ТЕМАТИКИ (используй как опорный список):
{DRUG_TERMS}

ВХОДНОЙ JSON:
{INPUT_JSON}
"""

prompt_c = """
РОЛЬ:
Ты — опытный помощник по классификации сообщений в чат-ботах Telegram для задачи модерации контента, связанного с покупкой, продажей, перепродажей, обменом наркотических средств.

ЗАДАЧА:
Определи, есть ли в сообщении упоминание наркотических средств и психогенных веществ.

КОНТЕКСТ:
На вход подаётся JSON с текстом сообщения бота. Нужно выполнить бинарную классификацию:
- true — если в тексте есть упоминание наркотиков, наркотических веществ, сленговых названий наркотиков, их покупки, продажи, употребления, хранения, изготовления или распространения
- false — если таких упоминаний нет
- Учитывай, что написание наркотиков может быть намеренно видоизменено посредством замены части символов в слове
- Анализируй смайлики, которые могут указывать на то, что в сообщении обсуждаются наркотики
- Обращай внимание как на русские, так и на английские названия наркотиков
- Учитывай, что названия наркотиков часто могут быть сленговыми

ОГРАНИЧЕНИЯ:
- Анализируй только текст из входного поля
- Не добавляй объяснений вне JSON
- Верни только JSON, строго по заданной структуре
- Если упоминание неоднозначное, но разумно связано с наркотиками, ставь true
- Учитывай, что названия наркотиков могут быть омонимичны словам из бытовой речи
- Не считай упоминания лекарств наркотиками, если нет явной связи
- Не присваивай true явно ироничному употреблению

ПРИМЕРЫ:

ПРИМЕР ВХОДА:
{{
  "id": "12345",
  "text": "где купить наркотики?"
}}

ПРИМЕР ВЫХОДА:
{{
  "id": "12345",
  "has_drug_mention": true
}}
ОПОРНЫЙ СЛОВАРЬ ТЕРМИНОВ (используй для распознавания сленга):
{DRUG_TERMS_SHORT}
Теперь обработай входной JSON и верни только выходной JSON по той же структуре.

ВХОДНОЙ JSON:
{INPUT_JSON}
"""

# ──────────────────────────────────────────────
# ПОСТРОИТЕЛИ СООБЩЕНИЙ
# ──────────────────────────────────────────────
def build_messages_b(item, use_dict=True):
    user_content = prompt_b.format(
        DRUG_TERMS=DRUG_TERMS if use_dict else "(словарь не используется)",
        INPUT_JSON=json.dumps(item, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_messages_c(item, use_dict=True):
    user_content = prompt_c.format(
        DRUG_TERMS_SHORT=DRUG_TERMS_SHORT if use_dict else "(словарь не используется)",
        INPUT_JSON=json.dumps(item, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


PROMPT_VARIANTS = {
    "prompt_b_no_dict":   lambda x: build_messages_b(x, use_dict=False),
    "prompt_b_with_dict": lambda x: build_messages_b(x, use_dict=True),
    "prompt_c_no_dict":   lambda x: build_messages_c(x, use_dict=False),
    "prompt_c_with_dict": lambda x: build_messages_c(x, use_dict=True),
}

# ──────────────────────────────────────────────
# АСИНХРОННЫЕ ЗАПРОСЫ
# ──────────────────────────────────────────────
async def send_one_request(client, model_name, messages):
    start = time.time()
    response = await client.chat.completions.parse(
        model=model_name,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        seed=42,
        response_format=DrugMentionClassification,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    end = time.time()
    return {
        "parsed": response.choices[0].message.parsed,
        "time":              end - start,
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


async def process_with_semaphore(client, model_name, messages):
    async with semaphore:
        return await send_one_request(client, model_name, messages)


def parse_response(raw_result, item_id: str):
    try:
        parsed = raw_result.get("parsed")
        if parsed is None:
            print(f"  ⚠️  Нет parsed для id={item_id}")
            return None
        return {
            "id":               parsed.id or item_id,
            "has_drug_mention": parsed.has_drug_mention,
        }
    except Exception as e:
        print(f"  ⚠️  Ошибка для id={item_id}: {e}")
        return None


# ──────────────────────────────────────────────
# ОЦЕНКА МЕТРИК
# ──────────────────────────────────────────────
def evaluate_results(results_file: str, df_truth: pd.DataFrame, label: str):
    if not Path(results_file).exists():
        print(f"[{label}] файл {results_file} не найден, пропускаю")
        return None

    with open(results_file) as f:
        preds = json.load(f)

    df_pred = pd.DataFrame(preds)
    if df_pred.empty:
        print(f"[{label}] файл пустой, пропускаю")
        return None

    df_pred = df_pred.drop_duplicates(subset="id", keep="last")
    df_pred["id"] = df_pred["id"].astype(str)

    df_truth_local = df_truth.copy()
    df_truth_local["composite_id"] = (
            df_truth_local["session_id"].astype(str) + "___" +
            df_truth_local.index.astype(str)
    )

    # Диагностика выравнивания
    expected_ids = set(df_truth_local["session_id"])
    actual_ids   = set(df_pred["id"])
    missing = expected_ids - actual_ids
    extra   = actual_ids   - expected_ids
    if missing or extra:
        print(f"[{label}] ВНИМАНИЕ: пропущено id из truth: {len(missing)}, лишних id в pred: {len(extra)}")
        if extra:
            print(f"  пример лишних id: {list(extra)[:3]}")
        if missing:
            print(f"  пример пропущенных id: {list(missing)[:3]}")

    df_merged = df_truth_local.merge(
        df_pred, left_on="composite_id", right_on="id", how="inner"
    )

    if len(df_merged) == 0:
        print(f"[{label}] нет совпадений по session_id, пропускаю.")
        return None

    if len(df_merged) != len(df_truth_local):
        print(f"[{label}] предупреждение: смержилось {len(df_merged)} из {len(df_truth_local)} строк")

    y_true = (df_merged["message_label"] == "illegal").astype(int)
    y_pred = df_merged["has_drug_mention"].astype(int)

    metrics = {
        "version":   label,
        "n":         len(df_merged),
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\n=== Промпт {label} (n={metrics['n']}) ===")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n  Матрица ошибок [строки=truth (legal, illegal), столбцы=pred]:")
    print(pd.DataFrame(
        cm,
        index=["truth_legal", "truth_illegal"],
        columns=["pred_legal", "pred_illegal"]
    ))

    print(f"\n  classification_report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["legal", "illegal"],
        zero_division=0
    ))

    return metrics


# ──────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ──────────────────────────────────────────────
async def run_variant(client, variant_name, items):
    """Классифицирует все items одним промптом, сохраняет результаты в JSON."""
    print(f"\n{'='*50}")
    print(f"Запускаем вариант: {variant_name}")
    print(f"Записей: {len(items)}")

    builder = PROMPT_VARIANTS[variant_name]
    tasks = [
        asyncio.create_task(
            process_with_semaphore(client, MODEL_NAME, builder(item))
        )
        for item in items
    ]

    start = time.time()
    raw_results = await tqdm.gather(*tasks, desc=variant_name)
    end = time.time()

    # Статистика времени
    valid = [r for r in raw_results if not isinstance(r, Exception)]
    print(f"Всего времени: {end - start:.2f} сек")
    print(f"Среднее время на запрос: {sum(x['time'] for x in valid) / len(valid):.2f} сек")
    print(f"Средний prompt_tokens: {sum(x['prompt_tokens'] for x in valid) / len(valid):.0f}")
    print(f"Средний completion_tokens: {sum(x['completion_tokens'] for x in valid) / len(valid):.0f}")

    # Парсим ответы
    parsed = []
    for item, raw in zip(items, raw_results):
        if isinstance(raw, Exception):
            print(f"  ⚠️  Ошибка запроса для id={item['id']}: {raw}")
            continue
        result = parse_response(raw, item["id"])
        if result:
            parsed.append(result)

    # Сохраняем результаты
    output_file = f"results_{variant_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(parsed)} результатов → {output_file}")

    return parsed


async def main():
    global semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    print(f"Модель: {MODEL_NAME}")
    print(f"Записей для классификации: {len(input_jsons)}")

    # Шаг 1 — классификация всеми вариантами промптов
    for variant_name in PROMPT_VARIANTS:
        await run_variant(client, variant_name, input_jsons)

    # Шаг 2 — считаем метрики
    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ МЕТРИКИ")
    print(f"{'='*50}")

    all_metrics = []
    for variant_name in PROMPT_VARIANTS:
        m = evaluate_results(
            f"results_{variant_name}.json",
            df_train,
            variant_name
        )
        if m is not None:
            all_metrics.append(m)

    if all_metrics:
        print("\n=== Сводная таблица ===")
        print(pd.DataFrame(all_metrics).set_index("version").round(4))


if __name__ == "__main__":
    asyncio.run(main())
