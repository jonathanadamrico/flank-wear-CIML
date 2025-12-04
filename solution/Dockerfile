FROM python:3.10-slim
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install torch --index-url https://download.pytorch.org/whl/cpu


COPY lib/ lib/
COPY model/ model/
COPY main.py .
COPY run.sh .
COPY eval_features.csv .

RUN sed -i 's/\r$//' run.sh && chmod +x run.sh

CMD ["sh", "run.sh"]