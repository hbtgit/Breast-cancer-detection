FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app
COPY . /app

RUN pip install streamlit scikit-learn numpy pillow
RUN apt-get update && apt-get install -y unzip

COPY new_balanced_data.zip /app/
RUN unzip new_balanced_data.zip -d /app/new_balanced_data

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]