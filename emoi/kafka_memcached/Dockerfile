FROM  python:3.10-slim

COPY requirements.txt /app/

COPY app/ /app/

ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR /app

# Upgrade to the newest available package version. 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python", "main.py"] # last command 

