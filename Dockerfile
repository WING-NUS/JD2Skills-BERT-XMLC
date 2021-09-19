FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /code
ADD pybert /code/pybert/
COPY __init__.py __init__.py
COPY run_bert.py run_bert.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD python run_bert.py --train --data_name job_dataset