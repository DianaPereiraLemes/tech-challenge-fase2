"""Funções para carregar e preparar a base de vinhos."""
import pandas as pd

from src.config import (ALVO, ALVO_ORIGINAL, ARQUIVO_DADOS, NOMES_COLUNAS, NOTA_MINIMA_ALTA,
                        VARIAVEIS_ORIGINAIS)


def carregar_dados(caminho=ARQUIVO_DADOS):
    """Lê o csv da base."""
    return pd.read_csv(caminho)


def renomear_colunas(df):
    """Tira a coluna Id (é só o número da linha) e coloca os nomes em português."""
    return df.drop(columns=["Id"], errors="ignore").rename(columns=NOMES_COLUNAS)


def resumo_qualidade(df):
    """Retorna um resumo rápido: tamanho, nulos, duplicados e tipos."""
    return {
        "linhas": df.shape[0],
        "colunas": df.shape[1],
        "nulos": int(df.isnull().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "tipos": df.dtypes.astype(str).value_counts().to_dict(),
    }


def remover_duplicados(df):
    """Remove linhas repetidas para o mesmo vinho não cair no treino e no teste."""
    return df.drop_duplicates().reset_index(drop=True)


def criar_alvo(df, nota_minima=NOTA_MINIMA_ALTA):
    """Cria a coluna alta_qualidade: 1 se a nota for >= 7, 0 caso contrário."""
    df = df.copy()
    df[ALVO] = (df[ALVO_ORIGINAL] >= nota_minima).astype(int)
    return df


def criar_features(df):
    """Cria a proporção de SO₂ livre (parte do SO₂ total que está livre para proteger o vinho)."""
    df = df.copy()
    df["proporcao_so2_livre"] = df["so2_livre"] / df["so2_total"]
    return df


def preparar_base(caminho=ARQUIVO_DADOS):
    """Carrega, renomeia, tira duplicados, cria o alvo e a feature nova."""
    df = renomear_colunas(carregar_dados(caminho))
    df = remover_duplicados(df)
    df = criar_alvo(df)
    return criar_features(df)


def contar_outliers(df):
    """Conta quantos pontos ficam fora dos bigodes do boxplot (1,5 x IQR) em cada variável."""
    linhas = []
    for col in VARIAVEIS_ORIGINAIS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        fora = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        linhas.append({"variavel": col, "n_outliers": int(fora.sum()),
                       "pct_outliers": round(100 * fora.mean(), 2),
                       "minimo": df[col].min(), "maximo": df[col].max()})
    return pd.DataFrame(linhas).sort_values("n_outliers", ascending=False)


def taxa_por_faixa(df, coluna, faixas=5):
    """% de vinhos de alta qualidade em cada faixa (quintil) da variável."""
    grupos = pd.qcut(df[coluna], q=faixas, duplicates="drop")
    return df.groupby(grupos, observed=True)[ALVO].mean()
