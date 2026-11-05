# Portfólio — Redes Neurais e Deep Learning

Repositório público das entregas da eletiva de Redes Neurais e Deep Learning do Insper.

https://sergioctf.github.io/redes-neurais-deep-learning

## Preparação

```bash
python -m venv env
```

No Windows:

```powershell
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Executar o exercício Data

Baixe o arquivo `train.csv` da competição Spaceship Titanic conforme as instruções em
`docs/exercises/data/data/README.md`. Depois, a partir da raiz do repositório, execute:

```bash
python docs/exercises/data/code/data_exercise.py
```

## Executar o exercício Perceptron

O exercício usa somente NumPy e Matplotlib para gerar os dados e as figuras. A partir da raiz do repositório, execute:

```bash
python docs/exercises/perceptron/code/perceptron_exercise.py
```

## Executar o exercício VAE

Instale a versão CPU de PyTorch separadamente para não tornar o build do site
dependente de uma biblioteca grande de treinamento. O MNIST é baixado pelo
`torchvision` na primeira execução para a pasta local `data/` (ignorada pelo Git).

```bash
python -m pip install -r docs/exercises/vae/code/requirements-vae.txt
python docs/exercises/vae/code/test_vae.py
python docs/exercises/vae/code/vae_exercise.py
```

O último comando treina os dois modelos por 20 épocas cada e gera as seis figuras,
dois históricos CSV e `results.json`.

## Visualizar o site

```bash
mkdocs serve
```
