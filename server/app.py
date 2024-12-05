from flask import Flask, request, jsonify, render_template
from catboost import CatBoostClassifier
import pandas as pd
import os
import json

app = Flask(__name__)

# Загрузка обученной модели
model = CatBoostClassifier()
model.load_model("model.cbm")

# Ожидаемый порядок столбцов
EXPECTED_COLUMNS = [
    'Пол', 'WBC', 'NE#', 'LY#', 'MO#', 'EO#', 'BA#',
    'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW',
    'PLT', 'MPV', 'PCT', 'NE%', 'LY%', 'MO%', 'EO%', 'BA%'
]

# Главная страница
@app.route("/")
def home():
    return render_template("upload.html")

# Словарь для интерпретации диагнозов
DIAGNOSIS_MAPPING = {
    0: "Здоров",
    1: "Лейкоз"
}

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
        # Чтение файла
        df = pd.read_excel(file)
        
        # Проверка наличия всех колонок
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400
        
        # Сохраняем колонку ID истории болезни
        if "ID истории болезни" not in df.columns:
            return jsonify({"error": "Column 'ID истории болезни' not found"}), 400
        
        ids = df["ID истории болезни"].tolist()
        
        # Упорядочиваем колонки для модели
        df = df[EXPECTED_COLUMNS]
        
        # Переименовываем колонки для модели
        df.columns = [str(i) for i in range(len(EXPECTED_COLUMNS))]
        
        # Прогнозирование
        predictions = model.predict(df).tolist()
        
        # Формируем результаты
        results = [
            {
                "ID истории болезни": id_,
                "Рекомендация": (
                    "Результаты могут иметь отклонения от нормы, пожалуйста, обратитесь к врачу."
                    if pred == 1 else
                    "Скорее всего, ваши результаты находятся в пределах нормы, но для уточнения обратитесь к врачу."
                )
            }
            for id_, pred in zip(ids, predictions)
        ]
        

      # Передаем результаты в шаблон HTML
        return render_template("results.html", results=results)
    except Exception as e:
        return render_template("error.html", error=f"Error processing file: {str(e)}"), 500


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True)
