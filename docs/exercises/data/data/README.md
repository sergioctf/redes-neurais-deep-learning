# Dados do Spaceship Titanic

O exercício utiliza somente o arquivo `train.csv` da competição
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data).

Para obter o arquivo com a ferramenta oficial do Kaggle:

```bash
kaggle competitions download spaceship-titanic -f train.csv -p docs/exercises/data/data
```

Caso o download seja entregue como ZIP, extraia `train.csv` nesta pasta. O caminho final deve ser:

```text
docs/exercises/data/data/train.csv
```

O CSV não é versionado neste repositório. O script valida o cabeçalho, as 8.693 linhas e o SHA-256 esperado antes de iniciar a análise.

