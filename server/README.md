### README.md

```markdown
# Flask-сервер для медицинских предсказаний

Этот проект представляет собой веб-приложение на Flask, которое использует модель CatBoost для предсказания потенциальных медицинских состояний на основе загружаемых данных пациентов. Результаты отображаются в HTML-интерфейсе.

---

## Структура проекта

```
server/
├── app.py               # Основное Flask-приложение
├── model.cbm            # Предобученная модель CatBoost
├── requirements.txt     # Зависимости проекта
├── templates/           # HTML-шаблоны для веб-интерфейса
│   ├── index.html       # Главная страница для загрузки файлов
│   ├── results.html     # Страница для отображения результатов
│   └── error.html       # Страница для отображения ошибок
└── README.md            # Документация проекта
```

---

## Как это работает

1. Пользователь загружает файл `.xlsx` с данными пациентов через веб-интерфейс.
2. Сервер обрабатывает файл:
   - Проверяет наличие всех необходимых столбцов.
   - Подготавливает данные для модели (упорядочивает и переименовывает столбцы).
3. Модель CatBoost выполняет предсказания на основе данных.
4. Результаты отображаются в браузере, включая:
   - `ID истории болезни`
   - Рекомендации на основе предсказания.

---

## Требования

- Python 3.8 или выше
- Библиотеки: Flask, pandas, catboost, openpyxl

---

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <адрес_репозитория>
   cd server
   ```

2. Настройте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Для Windows: venv\Scripts\activate
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Убедитесь, что модель CatBoost (`model.cbm`) находится в корневой директории проекта.

---

## Запуск проекта

1. Запустите Flask-сервер:
   ```bash
   python app.py
   ```

2. Откройте браузер и перейдите по адресу:
   ```
   http://127.0.0.1:5000/
   ```

3. Загрузите файл `.xlsx` с необходимыми столбцами:
   - `Пол`, `WBC`, `NE#`, `LY#`, `MO#`, `EO#`, `BA#`, `RBC`, `HGB`, `HCT`,
   - `MCV`, `MCH`, `MCHC`, `RDW`, `PLT`, `MPV`, `PCT`, `NE%`, `LY%`, `MO%`, `EO%`, `BA%`, `ID истории болезни`.

4. Просмотрите предсказания, отображаемые в браузере.

---

## О модели

- **Тип модели:** Предобученная CatBoost Classifier
- **Назначение:** Бинарная классификация для предсказания медицинских состояний.
  - `1`: Результаты могут иметь отклонения от нормы.
  - `0`: Результаты, скорее всего, в пределах нормы.
- **Входные признаки:** 22 числовых признака из данных анализов крови.
- **Выход:** Предсказание и читаемая рекомендация.

---

## Формат загружаемого файла

Файл `.xlsx`, загружаемый пользователем, должен содержать следующую структуру:

| Пол | WBC  | NE# | LY# | MO# | EO# | BA# | RBC  | HGB | HCT | ... | ID истории болезни                        |
|-----|------|-----|-----|-----|-----|-----|-------|-----|-----|-----|-------------------------------------------|
| 1   | 10.0 | 6.5 | 1.3 | 1.5 | 0.2 | 0.0 | 2.21  | 84.0| 26.3| ... | 0f59b100-2900-11ed-ab56-0050568844e6      |
| 0   | 9.1  | 6.0 | 1.1 | 1.3 | 0.2 | 0.0 | 1.91  | 74.0| 23.6| ... | 06a83aaa-8e27-11ec-ab52-0050568844e6      |

---

## Устранение проблем

1. **Отсутствие зависимостей:**
   Убедитесь, что все зависимости установлены:
   ```bash
   pip install -r requirements.txt
   ```

2. **Сервер не запускается:**
   Проверьте, что вы находитесь в правильной директории и виртуальное окружение активировано.

3. **Ошибки модели:**
   Убедитесь, что файл `model.cbm` находится в корневой директории проекта.

4. **Некорректные результаты:**
   Проверьте формат загружаемого файла и соответствие колонок.

---

## Возможные улучшения

- Добавить поддержку других форматов файлов (например, CSV).
- Улучшить обработку ошибок для некорректных данных.
- Реализовать более сложную систему рекомендаций.

---
