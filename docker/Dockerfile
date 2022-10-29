FROM pytorch/pytorch:latest 
RUN pip install fire 
RUN pip install rdkit-pypi==2022.3.1
RUN pip install gpytorch 
RUN pip install botorch
RUN pip install guacamol
RUN pip install wandb
RUN pip install selfies==2.0.0
RUN pip install pytorch_lightning
RUN pip install pandas
RUN pip install networkx
RUN pip install matplotlib
RUN apt update
RUN apt install -y build-essential
RUN apt install -y libxrender1 libxext6 software-properties-common apt-utils
RUN conda install -y pomegranate
RUN pip install --no-deps molsets==0.3.1
RUN pip install fcd-torch
RUN pip install free-mujoco-py
RUN pip install gym
RUN apt update
RUN apt install -y build-essential
RUN apt install -y libosmesa6-dev
RUN apt install -y libgl-dev
RUN pip install multiprocess
RUN pip install Box2D 
# wandb api key to log runs
# ENV WANDB_API_KEY=$YOUR_WANDB_API_KEY
