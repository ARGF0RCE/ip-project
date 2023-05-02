FROM python:3.8

WORKDIR /app

RUN apt update && apt install wget && wget https://storage.googleapis.com/ip-project-model/multi_input_multi_output_model.7z && apt install p7zip && p7zip -d multi_input_multi_output_model.7z
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
