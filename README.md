# Tech Challenge - Fase 2 | Classificação da qualidade de vinhos

## Sobre o projeto

Este projeto desenvolve um modelo de classificação para prever se um vinho é de alta qualidade a partir das suas características físico-químicas (acidez, teor alcoólico, sulfatos, densidade, entre outras).

A base utilizada é a Wine Quality Dataset, disponível no Kaggle, com 1.143 amostras de vinho tinto e uma nota de qualidade atribuída por especialistas. Conforme o enunciado, a nota foi transformada em duas classes: alta qualidade (nota maior ou igual a 7) e baixa/média qualidade (nota menor que 7).

O trabalho passa pelas etapas de análise exploratória, preparação dos dados, treinamento e comparação de modelos, avaliação e interpretação dos resultados.

## Estrutura do repositório

```
wine-quality-classification/
├── data/                  # base utilizada (WineQT.csv)
├── notebooks/             # notebook com a análise e a modelagem
├── src/                   # scripts auxiliares (config, dados, gráficos, modelagem e main.py)
├── tests/                 # testes das funções de preparação dos dados
├── results/               # gráficos e métricas dos modelos
├── requirements.txt       # bibliotecas utilizadas
└── README.md
```

O notebook `_Tech_Challenge_Fase_2` contém toda a análise passo a passo. A pasta `src` tem as mesmas etapas organizadas em funções, para rodar o projeto inteiro de uma vez e gerar os arquivos da pasta `results`.

## Como reproduzir a análise

1. Criar e ativar um ambiente virtual:

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (macOS/Linux)
```

2. Instalar as dependências:

```
pip install -r requirements.txt
```

3. Abrir o notebook `notebooks/_Tech_Challenge_Fase_2.ipynb` no Jupyter e executar as células de cima para baixo.

4. Para gerar os gráficos e métricas da pasta `results` sem abrir o notebook:

```
python -m src.main
```

5. Para rodar os testes:

```
python -m unittest discover tests
```

## Qualidade e tratamento de dados

A base não tem dados nulos e todas as colunas já estão no tipo numérico. A coluna Id foi removida por ser apenas o número da linha.

Foram encontradas 125 linhas duplicadas, que foram removidas para evitar que o mesmo vinho ficasse no treino e no teste. A base final tem 1.018 vinhos, dos quais 137 (13,5%) são de alta qualidade, ou seja, as classes estão desbalanceadas.

Os outliers foram identificados pelos boxplots (principalmente em açúcar residual e cloretos), mas foram mantidos porque são valores possíveis em vinhos reais.

A base foi dividida em 80% para treino e 20% para teste, com stratify para manter a proporção das classes. Para o KNN e a Regressão Logística foram testadas a padronização (StandardScaler) e a normalização (MinMaxScaler). O Random Forest não precisa de escala.

Foi criada uma variável nova, a proporção de SO₂ livre (SO₂ livre dividido pelo SO₂ total). Separados, o SO₂ livre e o total quase não se relacionam com a qualidade, mas a proporção tem correlação maior e melhorou o F1 dos três modelos na validação cruzada, por isso foi mantida.

## Modelos e resultados

Foram comparados três modelos: Regressão Logística, KNN e Random Forest. Os hiperparâmetros foram escolhidos com GridSearchCV usando o F1, e os modelos foram comparados com validação cruzada (K-Fold com 10 folds, o mesmo usado no GridSearch).

| Modelo | F1 validação cruzada | Acurácia | Precisão | Recall | F1 teste | AUC |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,45 | 0,90 | 0,65 | 0,48 | 0,55 | 0,89 |
| KNN | 0,51 | 0,89 | 0,60 | 0,44 | 0,51 | 0,88 |
| Random Forest | 0,48 | 0,91 | 0,79 | 0,41 | 0,54 | 0,91 |

O modelo escolhido foi o KNN (k = 7), por ter o maior F1 médio na validação cruzada. No teste ele ficou um pouco abaixo dos outros dois no F1, mas a diferença é de um ou dois vinhos, já que o teste tem apenas 27 vinhos de alta qualidade. Os três modelos deixam passar parte dos vinhos bons (recall entre 41% e 48%). A acurácia ficou perto de 88% em todos os modelos, um valor próximo ao de um modelo que sempre responde "baixa/média" (86,5%), por isso ela não foi usada para escolher o modelo.

As variáveis que mais influenciam a qualidade são o teor alcoólico, os sulfatos e a acidez volátil, seguidos da densidade e do ácido cítrico. Vinhos com mais álcool, sulfatos adequados e pouca acidez volátil têm muito mais chance de serem de alta qualidade.

## Entregáveis

- Apresentação executiva: `docs/apresentacao_executiva.pdf` (e a versão em PPTX)
- Vídeo: (adicionar o link após a gravação)
