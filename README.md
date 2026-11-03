## Sobre o projeto

A qualidade de um vinho normalmente é avaliada por especialistas, em uma análise sensorial que leva tempo, tem custo e depende da experiência de quem avalia. Este projeto verifica se as medidas físico-químicas coletadas em laboratório durante a produção conseguem antecipar essa avaliação.

O objetivo foi desenvolver um modelo de classificação capaz de prever se um vinho é de alta qualidade a partir das suas características físico-químicas. Conforme o enunciado, a nota dada pelos especialistas foi transformada em duas classes:

- Alta qualidade (1): nota maior ou igual a 7
- Baixa/média qualidade (0): nota menor que 7

## Base de dados

Foi utilizada a Wine Quality Dataset, disponível no Kaggle ([yasserh/wine-quality-dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)), arquivo `WineQT.csv`. A base tem 1.143 amostras de vinho tinto e as seguintes colunas:

| Coluna original | Nome no projeto | Descrição |
|---|---|---|
| fixed acidity | acidez_fixa | Acidez fixa (g/L) |
| volatile acidity | acidez_volatil | Acidez volátil, principalmente ácido acético (g/L) |
| citric acid | acido_citrico | Ácido cítrico (g/L) |
| residual sugar | acucar_residual | Açúcar residual (g/L) |
| chlorides | cloretos | Cloretos, ou seja, sal (g/L) |
| free sulfur dioxide | so2_livre | Dióxido de enxofre livre (mg/L) |
| total sulfur dioxide | so2_total | Dióxido de enxofre total (mg/L) |
| density | densidade | Densidade (g/cm³) |
| pH | ph | pH |
| sulphates | sulfatos | Sulfatos (g/L) |
| alcohol | alcool | Teor alcoólico (% vol.) |
| quality | qualidade | Nota dada pelos especialistas (0 a 10) |
| Id | removida | Número da linha, não descreve o vinho |

## Estrutura do repositório

```
tech-challenge-fase2/
├── data/
│   └── WineQT.csv                        # base utilizada
├── notebooks/
│   └── _Tech_Challenge_Fase_2.ipynb      # análise e modelagem completas, passo a passo
├── src/                                  # scripts auxiliares com as mesmas etapas do notebook
│   ├── config.py                         # caminhos, nomes das colunas e constantes
│   ├── dados.py                          # carga, limpeza, criação do alvo e da nova variável
│   ├── modelagem.py                      # treino/teste, modelos, GridSearch, K-Fold e métricas
│   ├── graficos.py                       # gráficos da análise e da avaliação
│   └── main.py                           # roda o projeto inteiro e gera a pasta results
├── tests/
│   └── test_dados.py                     # testes das funções de preparação dos dados
├── results/
│   ├── figures/                          # 19 gráficos (análise exploratória e modelos)
│   ├── comparacao_modelos.csv            # métricas dos 3 modelos
│   ├── classification_report.txt         # relatório de classificação de cada modelo
│   ├── metricas.json                     # resumo dos resultados e hiperparâmetros
│   └── ...                               # estatísticas, outliers, correlações e importâncias
├── docs/
│   ├── apresentacao_executiva.pptx       # storytelling da análise exploratória
│   └── video_executivo.mp4               # vídeo executivo (até 5 minutos)
├── requirements.txt                      # bibliotecas utilizadas
└── README.md
```

O notebook contém toda a análise, com o código e as conclusões de cada etapa. A pasta `src` tem as mesmas etapas organizadas em funções, para rodar o projeto inteiro de uma vez e gerar os arquivos da pasta `results`.

## Como reproduzir a análise

1. Clonar o repositório e entrar na pasta do projeto.

2. Criar e ativar um ambiente virtual:

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (macOS/Linux)
```

3. Instalar as dependências:

```
pip install -r requirements.txt
```

4. Abrir o notebook `notebooks/_Tech_Challenge_Fase_2.ipynb` no Jupyter ou no VS Code e executar as células de cima para baixo.

5. Para gerar os gráficos e as métricas da pasta `results` sem abrir o notebook (na pasta raiz do projeto):

```
python -m src.main
```

6. Para rodar os testes:

```
python -m unittest discover tests
```

A execução completa leva cerca de 2 minutos, por causa da busca de hiperparâmetros.

## Etapas do projeto

### 1. Compreensão do problema

Problema de aprendizado supervisionado de classificação binária. As variáveis independentes são as 11 medidas físico-químicas e a variável alvo é `alta_qualidade` (nota maior ou igual a 7).

### 2. Análise exploratória dos dados

- **Qualidade da base:** não há dados nulos e todas as colunas são numéricas. Foram encontradas e removidas 125 linhas duplicadas, para evitar que o mesmo vinho ficasse no treino e no teste. A base final tem 1.018 vinhos.
- **Balanceamento das classes:** as notas se concentram em 5 e 6, e apenas 13,5% dos vinhos (137) são de alta qualidade. As classes são desbalanceadas, por isso a acurácia sozinha não serve para avaliar os modelos.
- **Distribuição das variáveis:** densidade e pH têm distribuição próxima da normal. Açúcar residual, cloretos, sulfatos e SO₂ são assimétricos, com muitos valores baixos e poucos muito altos.
- **Outliers:** identificados pelos boxplots, principalmente em açúcar residual (9,3% dos vinhos), cloretos (7,0%) e sulfatos (4,0%). Não há valores impossíveis (pH entre 2,7 e 4,0, álcool entre 8,4% e 14,9%), então os outliers foram mantidos por serem vinhos reais.
- **Correlações entre as variáveis:** acidez fixa e pH (−0,69), acidez fixa e densidade (+0,68), acidez fixa e ácido cítrico (+0,67), SO₂ livre e SO₂ total (+0,66), ácido cítrico e acidez volátil (−0,54) e álcool e densidade (−0,50). Cada uma tem justificativa química, explicada no notebook.
- **Correlação com a alta qualidade:** teor alcoólico (+0,41), ácido cítrico (+0,24) e sulfatos (+0,21) favorecem; acidez volátil (−0,30) é o principal fator negativo.

Principais insights:

- A chance de um vinho ser de alta qualidade vai de 1% na faixa de menor teor alcoólico para 41% na faixa de maior teor alcoólico.
- Com acidez volátil baixa, 36% dos vinhos são de alta qualidade; com acidez volátil alta, apenas 2%.
- Sulfatos acima de 0,65 g/L triplicam a chance de alta qualidade.
- Vinhos com álcool de pelo menos 11%, acidez volátil abaixo de 0,47 g/L e sulfatos de pelo menos 0,65 g/L são 11% da base, mas 54% deles são de alta qualidade (4 vezes a média) e reúnem 42% de todos os vinhos de alta qualidade.

### 3. Pré-processamento

- Colunas renomeadas para português e coluna `Id` removida.
- Tipos verificados (todos numéricos) e duplicados removidos.
- Não há variáveis categóricas, então não foi necessário encoding.
- Criação de uma nova variável, a **proporção de SO₂ livre** (SO₂ livre dividido pelo SO₂ total). Ela tem correlação de 0,13 com a alta qualidade, maior que o SO₂ livre e o SO₂ total separados, e melhorou o F1 dos três modelos na validação cruzada, por isso foi mantida.
- Divisão em 80% treino e 20% teste com `stratify` (mesma proporção de vinhos de alta qualidade nos dois conjuntos) e `random_state=42`.
- Padronização (`StandardScaler`) e normalização (`MinMaxScaler`) testadas para os modelos sensíveis à escala, dentro de um `Pipeline`, para que a escala seja calculada só com os dados de treino.

### 4. Modelos

Foram comparados três modelos de classificação com abordagens diferentes:

- **Regressão Logística:** modelo linear.
- **KNN:** classifica pela maioria dos vizinhos mais próximos.
- **Random Forest:** conjunto de árvores de decisão.

Os hiperparâmetros foram escolhidos com `GridSearchCV` usando o F1 como métrica, e os modelos foram comparados com validação cruzada K-Fold de 10 folds no conjunto de treino. Para o KNN, também foi feita a curva de erro e F1 para k de 1 a 20, que confirmou o k = 7 escolhido pelo GridSearch.

### 5. Avaliação dos modelos

Resultados no conjunto de teste (204 vinhos, sendo 27 de alta qualidade), que não foi usado no treinamento:

| Modelo | F1 validação cruzada | Acurácia | Precisão | Recall | F1 teste | AUC |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,45 | 0,90 | 0,65 | 0,48 | 0,55 | 0,89 |
| **KNN (escolhido)** | **0,51** | 0,89 | 0,60 | 0,44 | 0,51 | 0,88 |
| Random Forest | 0,48 | 0,91 | 0,79 | 0,41 | 0,54 | 0,91 |

- O modelo escolhido foi o **KNN**, por ter o maior F1 médio na validação cruzada. Esse critério foi definido antes de olhar o resultado no teste.
- Os três modelos têm acurácia perto de 88% na validação cruzada, valor próximo ao de um modelo que sempre responde "baixa/média" (86,5%). Por isso a escolha foi feita pelo F1.
- No teste, o KNN apontou 20 vinhos como de alta qualidade e 12 realmente eram (precisão de 60%, contra 13% se a escolha fosse ao acaso). Ele encontrou 12 dos 27 vinhos de alta qualidade (recall de 44%).
- No teste, o KNN ficou um pouco abaixo dos outros dois no F1. A diferença corresponde a um ou dois vinhos, já que o teste tem apenas 27 vinhos de alta qualidade.
- O Random Forest mostrou overfitting (F1 de 1,00 no treino e 0,54 no teste). No KNN, o F1 de 1,00 no treino é consequência do `weights="distance"` e não indica overfitting sozinho.

Também foram usados a matriz de confusão, o `classification_report` e a curva ROC de cada modelo (ver notebook e pasta `results/figures`).

### 6. Interpretação dos resultados

A importância das variáveis no Random Forest confirma a análise exploratória: **teor alcoólico** (16,8%), **sulfatos** (12,1%) e **acidez volátil** (11,5%) são as variáveis mais importantes, seguidas de ácido cítrico e densidade. A variável criada (proporção de SO₂ livre) aparece em 6º lugar.

Implicações para o processo de produção:

- **Controlar a acidez volátil**, com cuidado na higiene, na temperatura e no contato com oxigênio. É o defeito que mais tira vinhos da faixa de alta qualidade.
- **Garantir a maturação da uva e uma fermentação completa**, que elevam o teor alcoólico. Vinhos com menos de 10% de álcool quase nunca chegam a alta qualidade (1,9%).
- **Dosar os sulfitos com precisão:** sulfatos adequados protegem o vinho, mas o excesso de SO₂ total aparece ligado a notas menores.
- **Usar o modelo como triagem de laboratório**, para indicar os lotes com mais chance de serem de alta qualidade e priorizar a avaliação dos especialistas. O modelo apoia, mas não substitui, a análise sensorial.

## Limitações

- A base é pequena (137 vinhos de alta qualidade) e vem de uma única origem (vinho tinto português), então os resultados precisam ser validados antes de serem aplicados a outros vinhos.
- As relações encontradas são correlações, não causas. As recomendações devem ser testadas na produção.
- O modelo ainda deixa passar mais da metade dos vinhos de alta qualidade. Ter mais exemplos de vinhos de alta qualidade é a forma mais segura de melhorar esse resultado.

## Entregáveis

- **Código:** este repositório (notebook, scripts da pasta `src`, testes e resultados).
- **Apresentação executiva:** [`docs/apresentacao_executiva.pdf`](docs/apresentacao_executiva.pdf), com o storytelling da análise exploratória. Também disponível em [PowerPoint](docs/apresentacao_executiva.pptx).
- **Vídeo executivo:** [`docs/video_executivo.mp4`](docs/video_executivo.mp4), com até 5 minutos.

## Tecnologias

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn e Jupyter.
