"""Configurações usadas em todo o projeto (caminhos, colunas e constantes)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "data"
PASTA_RESULTADOS = RAIZ / "results"
PASTA_FIGURAS = PASTA_RESULTADOS / "figures"
ARQUIVO_DADOS = PASTA_DADOS / "WineQT.csv"

RANDOM_STATE = 42
TAMANHO_TESTE = 0.2
N_FOLDS = 10
NOTA_MINIMA_ALTA = 7  # nota >= 7 é considerada alta qualidade

# nomes das colunas em português
NOMES_COLUNAS = {
    "fixed acidity": "acidez_fixa",
    "volatile acidity": "acidez_volatil",
    "citric acid": "acido_citrico",
    "residual sugar": "acucar_residual",
    "chlorides": "cloretos",
    "free sulfur dioxide": "so2_livre",
    "total sulfur dioxide": "so2_total",
    "density": "densidade",
    "pH": "ph",
    "sulphates": "sulfatos",
    "alcohol": "alcool",
    "quality": "qualidade",
}

ALVO_ORIGINAL = "qualidade"
ALVO = "alta_qualidade"
VARIAVEIS_ORIGINAIS = ["acidez_fixa", "acidez_volatil", "acido_citrico", "acucar_residual",
                       "cloretos", "so2_livre", "so2_total", "densidade", "ph", "sulfatos", "alcool"]
# feature criada na etapa de feature engineering (ver notebook)
NOVAS_FEATURES = ["proporcao_so2_livre"]
FEATURES = VARIAVEIS_ORIGINAIS + NOVAS_FEATURES

# nomes bonitos para os gráficos
ROTULOS = {
    "acidez_fixa": "Acidez fixa",
    "acidez_volatil": "Acidez volátil",
    "acido_citrico": "Ácido cítrico",
    "acucar_residual": "Açúcar residual",
    "cloretos": "Cloretos",
    "so2_livre": "SO₂ livre",
    "so2_total": "SO₂ total",
    "densidade": "Densidade",
    "ph": "pH",
    "sulfatos": "Sulfatos",
    "alcool": "Teor alcoólico",
    "proporcao_so2_livre": "Proporção SO₂ livre",
    "qualidade": "Qualidade (nota)",
    "alta_qualidade": "Alta qualidade",
}
NOMES_CLASSES = ["Baixa/Média (<7)", "Alta (≥7)"]

COR_ALTA = "#2a78d6"   # azul
COR_BAIXA = "#eb6834"  # laranja
CORES_MODELOS = ["#2a78d6", "#eb6834", "#1baf7a"]

# cores da apresentação executiva (usadas nos gráficos que viraram slides)
COR_VINHO = "#7b2240"   # alta qualidade / fatores a favor
COR_CONTRA = "#c24d2c"  # fatores contra
COR_CINZA = "#b8afb2"   # baixa/média qualidade / acaso


def rotulo(coluna):
    """Devolve o nome em português da coluna para usar nos gráficos."""
    return ROTULOS.get(coluna, coluna)
