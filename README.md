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

## Visualizar o site

```bash
mkdocs serve
```
