# Create Conda Env
```bash 
conda create -n Iot_forecast python=3.7
```

# Activate Conda Env.
```bash 
conda activate Iot_forecast 
```
# Installing FBprophet model 

To install fbprophet one must first install Pystan which is a library that helps in running Fbprophet with ease. To install Pystan just open you Command Prompt or Anaconda Prompt and then type:
```bash
pip install pystan
```

Wait for the installation to finish.

2. Once Pystan is successfully downloaded, the next step is to install Fbprophet either by pip or conda. Under the same Command Prompt just type:

```bash 

pip install fbprophet

```
or,
```bash 

conda install -c conda-forge fbprophet

```
