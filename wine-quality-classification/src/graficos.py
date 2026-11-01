"""Gráficos da EDA e da avaliação dos modelos.

Cada função gera um gráfico, salva em results/figures e devolve o caminho do png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

from src.config import (ALVO, ALVO_ORIGINAL, COR_ALTA, COR_BAIXA, CORES_MODELOS, NOMES_CLASSES,
                        PASTA_FIGURAS, VARIAVEIS_ORIGINAIS, rotulo)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlelocation"] = "left"

CORES_CLASSES = {NOMES_CLASSES[0]: COR_BAIXA, NOMES_CLASSES[1]: COR_ALTA}


def salvar(fig, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_FIGURAS / f"{nome}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def com_classe(df):
    """Copia o df com uma coluna 'Classe' escrita por extenso (para as legendas)."""
    df = df.copy()
    df["Classe"] = df[ALVO].map({0: NOMES_CLASSES[0], 1: NOMES_CLASSES[1]})
    return df


# ---------------------------- EDA ----------------------------

def distribuicao_qualidade(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    notas = df[ALVO_ORIGINAL].value_counts().sort_index()
    cores = [COR_ALTA if n >= 7 else COR_BAIXA for n in notas.index]
    barras = ax1.bar(notas.index.astype(str), notas.values, color=cores)
    ax1.bar_label(barras, padding=3)
    ax1.set(title="Vinhos por nota", xlabel="Nota", ylabel="Quantidade")

    classes = df[ALVO].value_counts().sort_index()
    barras = ax2.bar(NOMES_CLASSES, classes.values, color=[COR_BAIXA, COR_ALTA])
    ax2.bar_label(barras, labels=[f"{v} ({v / classes.sum():.1%})" for v in classes.values], padding=3)
    ax2.set(title="Balanceamento das classes", ylabel="Quantidade")
    ax2.set_ylim(0, classes.max() * 1.15)

    fig.tight_layout()
    return salvar(fig, "01_distribuicao_qualidade")


def histogramas(df):
    eixos = df[VARIAVEIS_ORIGINAIS].rename(columns=rotulo).hist(bins=30, figsize=(15, 10),
                                                    color=COR_ALTA, edgecolor="white")
    fig = eixos.flat[0].figure
    fig.suptitle("Histogramas das variáveis", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return salvar(fig, "02_histogramas")


def distribuicao_por_classe(df):
    dados = com_classe(df)
    fig, eixos = plt.subplots(3, 4, figsize=(15, 10))
    for i, col in enumerate(VARIAVEIS_ORIGINAIS):
        ax = eixos.flat[i]
        sns.histplot(data=dados, x=col, hue="Classe", stat="density", common_norm=False,
                     element="step", palette=CORES_CLASSES, ax=ax, legend=(i == 0))
        ax.set(title=rotulo(col), xlabel="", ylabel="")
    eixos.flat[-1].axis("off")
    fig.suptitle("Distribuição de cada variável por classe", x=0.01, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    return salvar(fig, "03_distribuicao_por_classe")


def mapa_calor(df):
    colunas = VARIAVEIS_ORIGINAIS + [ALVO_ORIGINAL, ALVO]
    corr = df[colunas].corr().rename(index=rotulo, columns=rotulo)
    mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)  # mostra só a metade de baixo
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mascara, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Mapa de calor das correlações")
    return salvar(fig, "04_mapa_calor_correlacoes")


def dispersao_alcool_acidez(df):
    dados = com_classe(df)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=dados, x="alcool", y="acidez_volatil", hue="Classe", hue_order=NOMES_CLASSES,
                    palette=CORES_CLASSES, alpha=0.7, s=25, ax=ax)
    ax.set(title="Teor alcoólico x acidez volátil", xlabel="Teor alcoólico (%)",
           ylabel="Acidez volátil (g/L)")
    return salvar(fig, "05_dispersao_alcool_acidez")


def boxplots_por_classe(df):
    dados = com_classe(df)
    fig, eixos = plt.subplots(3, 4, figsize=(15, 10))
    for i, col in enumerate(VARIAVEIS_ORIGINAIS):
        ax = eixos.flat[i]
        sns.boxplot(data=dados, x="Classe", y=col, hue="Classe", order=NOMES_CLASSES,
                    palette=CORES_CLASSES, legend=False, fliersize=2, ax=ax)
        ax.set(title=rotulo(col), xlabel="", ylabel="")
    eixos.flat[-1].axis("off")
    fig.suptitle("Boxplots por classe (os pontos soltos são outliers)", x=0.01, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    return salvar(fig, "06_boxplots_por_classe")


# ---------------------------- Modelos ----------------------------

def erro_por_k(ks, erros, f1s):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ks, erros, "-o", color=COR_BAIXA, label="Erro médio")
    ax.plot(ks, f1s, "-o", color=COR_ALTA, label="F1 médio")
    melhor = ks[int(np.argmax(f1s))]
    ax.axvline(melhor, color="gray", linestyle="--", linewidth=1)
    ax.set(title=f"KNN: erro e F1 por valor de k (maior F1 em k={melhor})", xlabel="k", ylabel="")
    ax.set_xticks(ks)
    ax.legend()
    return salvar(fig, "07_knn_erro_por_k")


def comparacao_kfold(scores):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordem = scores.groupby("modelo")["f1"].mean().sort_values(ascending=False).index
    sns.boxplot(data=scores, x="modelo", y="f1", hue="modelo", order=ordem,
                palette=CORES_MODELOS[:len(ordem)], legend=False, ax=ax)
    sns.stripplot(data=scores, x="modelo", y="f1", order=ordem, color="black", size=3, ax=ax)
    ax.set(title="F1 em cada fold da validação cruzada (10 folds)", xlabel="", ylabel="F1")
    return salvar(fig, "08_comparacao_kfold")


def matrizes_confusao(modelos, x_teste, y_teste):
    fig, eixos = plt.subplots(1, len(modelos), figsize=(4.5 * len(modelos), 4))
    for ax, (nome, modelo) in zip(eixos, modelos.items()):
        ConfusionMatrixDisplay.from_estimator(modelo, x_teste, y_teste,
                                              display_labels=["Baixa/Média", "Alta"],
                                              cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(nome)
        ax.grid(False)
    fig.tight_layout()
    return salvar(fig, "09_matrizes_confusao")


def curvas_roc(modelos, x_teste, y_teste):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for (nome, modelo), cor in zip(modelos.items(), CORES_MODELOS):
        prob = modelo.predict_proba(x_teste)[:, 1]
        fpr, tpr, _ = roc_curve(y_teste, prob)
        ax.plot(fpr, tpr, color=cor, linewidth=2, label=f"{nome} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(title="Curva ROC no teste", xlabel="Taxa de falsos positivos",
           ylabel="Taxa de verdadeiros positivos")
    ax.legend(loc="lower right")
    return salvar(fig, "10_curvas_roc")


def treino_vs_teste(tabela):
    dados = (tabela[["F1 treino", "F1 teste"]].astype(float).reset_index()
             .melt(id_vars="modelo", var_name="Conjunto", value_name="valor"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=dados, x="modelo", y="valor", hue="Conjunto",
                palette=["#9bbbe6", COR_ALTA], ax=ax)
    for barras in ax.containers:
        ax.bar_label(barras, fmt="%.2f", padding=2, fontsize=9)
    ax.set(title="F1 no treino x F1 no teste", xlabel="", ylabel="F1", ylim=(0, 1.1))
    return salvar(fig, "11_treino_vs_teste")


def importancia_variaveis(importancias, nome_modelo):
    serie = importancias.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    barras = ax.barh([rotulo(c) for c in serie.index], serie.values, color=COR_ALTA)
    ax.bar_label(barras, labels=[f"{v:.1%}" for v in serie], padding=3, fontsize=9)
    ax.set(title=f"Importância das variáveis ({nome_modelo})", xlabel="Importância (Gini)")
    ax.set_xlim(0, serie.max() * 1.2)
    return salvar(fig, "12_importancia_variaveis")

