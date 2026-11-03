"""Roda o projeto inteiro e salva os gráficos e as métricas em results/.

Na raiz do projeto:
    python -m src.main
"""
import json
import warnings

from src import graficos
from src.config import ALVO, FEATURES, PASTA_RESULTADOS, VARIAVEIS_ORIGINAIS
from src.dados import (carregar_dados, contar_outliers, preparar_base, renomear_colunas,
                       resumo_qualidade, taxa_por_faixa)
from src.modelagem import (arvore_explicativa, buscar_hiperparametros, definir_modelos,
                           erro_knn_por_k, importancia_gini, relatorio, sem_vazamento,
                           separar_treino_teste, tabela_resultados, validacao_cruzada)

warnings.filterwarnings("ignore")


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    # 1. dados
    resumo_base = resumo_qualidade(renomear_colunas(carregar_dados()))
    df = preparar_base()
    print("Base original:", resumo_base)
    print(f"Sem duplicados: {len(df)} vinhos, {df[ALVO].mean():.1%} de alta qualidade")

    # 2. EDA
    df[VARIAVEIS_ORIGINAIS].describe().T.to_csv(PASTA_RESULTADOS / "estatisticas_descritivas.csv")
    contar_outliers(df).to_csv(PASTA_RESULTADOS / "outliers_boxplot.csv", index=False)
    corr_alvo = df[FEATURES + [ALVO]].corr()[ALVO].drop(ALVO).sort_values()
    corr_alvo.to_csv(PASTA_RESULTADOS / "correlacao_com_alvo.csv")

    graficos.vinhos_por_nota(df)
    graficos.balanceamento_classes(df)
    graficos.histogramas(df)
    graficos.distribuicao_por_classe(df)
    graficos.boxplots_por_classe(df)
    graficos.mapa_calor(df)
    graficos.correlacao_com_alvo(df)
    graficos.pairplot_principais(df)
    graficos.dispersao_alcool_acidez(df)
    taxas = {col: taxa_por_faixa(df, col) for col in ["alcool", "acidez_volatil", "sulfatos"]}
    graficos.taxas_por_faixa(taxas, df[ALVO].mean())

    # 3. treino e teste
    x_treino, x_teste, y_treino, y_teste = separar_treino_teste(df)
    print(f"Treino: {len(x_treino)} | Teste: {len(x_teste)} | sem vazamento: {sem_vazamento(x_treino, x_teste)}")

    # 4. modelos
    modelos = {}
    parametros = {}
    for nome, (pipeline, grade) in definir_modelos().items():
        busca = buscar_hiperparametros(pipeline, grade, x_treino, y_treino)
        modelos[nome] = busca.best_estimator_
        parametros[nome] = {k: str(v) for k, v in busca.best_params_.items()}
        print(f"{nome}: F1 no GridSearch = {busca.best_score_:.3f} | {busca.best_params_}")

    # curva de erro e F1 por k usando a mesma configuração do KNN escolhido no GridSearch
    knn = modelos["KNN"]
    ks, erros, f1s = erro_knn_por_k(x_treino, y_treino, knn.named_steps["escala"],
                                    knn.named_steps["modelo"].metric, knn.named_steps["modelo"].weights)
    graficos.erro_por_k(ks, erros, f1s)

    cv = validacao_cruzada(modelos, x_treino, y_treino)
    cv.to_csv(PASTA_RESULTADOS / "validacao_cruzada_folds.csv", index=False)
    graficos.comparacao_kfold(cv)

    # 5. avaliação
    tabela = tabela_resultados(modelos, x_treino, y_treino, x_teste, y_teste, cv)
    tabela.round(4).to_csv(PASTA_RESULTADOS / "comparacao_modelos.csv")
    escolhido = tabela["F1 CV (média)"].astype(float).idxmax()
    print(tabela.round(3).to_string())
    print(f"\nModelo escolhido (maior F1 médio na validação cruzada): {escolhido}")
    print(relatorio(modelos[escolhido], x_teste, y_teste))

    graficos.matrizes_confusao(modelos, x_teste, y_teste)
    graficos.curvas_roc(modelos, x_teste, y_teste)
    graficos.treino_vs_teste(tabela)
    graficos.precisao_vs_acaso(modelos[escolhido], x_teste, y_teste, escolhido)
    with open(PASTA_RESULTADOS / "classification_report.txt", "w", encoding="utf-8") as f:
        for nome, modelo in modelos.items():
            f.write(f"{nome}\n{relatorio(modelo, x_teste, y_teste)}\n")

    # 6. interpretação
    importancias = importancia_gini(modelos["Random Forest"])
    importancias.to_csv(PASTA_RESULTADOS / "importancia_variaveis.csv")
    graficos.importancia_variaveis(importancias, "Random Forest")

    resumo = {
        "linhas_originais": resumo_base["linhas"],
        "duplicados_removidos": resumo_base["duplicados"],
        "linhas_finais": len(df),
        "pct_alta_qualidade": round(float(df[ALVO].mean()), 4),
        "treino": len(x_treino),
        "teste": len(x_teste),
        "modelo_escolhido": escolhido,
        "melhores_parametros": parametros,
        "metricas": tabela.astype(float).round(4).to_dict(orient="index"),
        "importancia_rf": importancias.round(4).to_dict(),
    }
    with open(PASTA_RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
