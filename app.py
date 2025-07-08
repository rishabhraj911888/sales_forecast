
from flask import Flask, render_template, request, send_file
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import os
import zipfile
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            graph_html = fig.to_html(full_html=False)

            return render_template('result.html', graph=graph_html)

    return render_template('index.html')

@app.route('/download-all')
def download_all():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            zf.write(filepath, arcname=filename)
    memory_file.seek(0)
    return send_file(memory_file, download_name='all_csvs.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
