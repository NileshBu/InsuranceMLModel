FROM python:3.7.3-stretch

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r python_requirements.txt

CMD ["python","app.py"]