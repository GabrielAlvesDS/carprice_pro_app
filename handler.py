import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from carprice_pro.CarPrice_Pro import CarPrice_Pro

model = pickle.load(open('model/xgb_model.pkl', 'rb'))

# Criar uma instância da classe Flask
app = Flask(__name__)

# Definir as rotas da API
@app.route('/', methods=['GET', 'POST'])

def carprice_predict():

    try:
        data = request.get_json()

        # df para coletar os dados recebidos
        df = pd.DataFrame(data)

        # Processar os dados
        pipeline = CarPrice_Pro()
        df1 = pipeline.data_cleaning(df)
        df2 = pipeline.data_preparation(df1)
        predicted_price = pipeline.data_prediction(model, df2)

        return jsonify({'predicted_price': predicted_price.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

  
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host='0.0.0.0', port=port, debug=True )
