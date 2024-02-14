FROM ubuntu:focal

RUN apt-get update

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get install -y build-essential

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;
    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN apt-get install -y wget

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh

RUN ~/miniconda3/bin/conda init bash

ENV PATH=~/miniconda3/bin:$PATH

WORKDIR /code

COPY environment.yml .
RUN /bin/bash -c "conda env create -f environment.yml"
COPY ./app /code/app

# CMD ["conda run", "-n", "mlops_fraud", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD /bin/bash -c "conda run -n mlops_fraud uvicorn app.main:app --host 0.0.0.0 --port 8000"