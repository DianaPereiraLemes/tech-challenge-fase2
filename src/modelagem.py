"""Separação treino/teste, modelos, validação cruzada e métricas."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import ALVO, FEATURES, N_FOLDS, RANDOM_STATE, TAMANHO_TESTE

# como as classes são desbalanceadas, usamos o F1 em vez da acurácia
f1_scorer = make_scorer(f1_score)


def separar_treino_teste(df):
    """80% treino e 20% teste. O stratify mantém a mesma proporção de vinhos bons nos dois."""
    x = df[FEATURES]
    y = df[ALVO]
    return train_test_split(x, y, test_size=TAMANHO_TESTE, stratify=y, random_state=RANDOM_STATE)


def definir_modelos():
    """Retorna os 3 modelos (dentro de um Pipeline) e os hiperparâmetros que o GridSearch vai testar.

    KNN e Regressão Logística precisam dos dados na mesma escala, então testamos
    StandardScaler e MinMaxScaler. O Random Forest não precisa de escala.
    """
    escalas = [StandardScaler(), MinMaxScaler()]

    logistica = Pipeline([("escala", StandardScaler()),
                          ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    grade_logistica = {"escala": escalas, "modelo__C": [0.01, 0.1, 1, 10, 100]}

    knn = Pipeline([("escala", StandardScaler()), ("modelo", KNeighborsClassifier())])
    grade_knn = {"escala": escalas,
                 "modelo__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
                 "modelo__weights": ["uniform", "distance"],
                 "modelo__metric": ["euclidean", "manhattan", "cosine"]}

    rf = Pipeline([("modelo", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    grade_rf = {"modelo__n_estimators": [100, 300],
                "modelo__criterion": ["gini", "entropy"],
                "modelo__max_depth": [None, 5, 10],
                "modelo__min_samples_leaf": [1, 3, 5]}

    return {
        "Regressão Logística": (logistica, grade_logistica),
        "KNN": (knn, grade_knn),
        "Random Forest": (rf, grade_rf),
    }


def buscar_hiperparametros(pipeline, grade, x_treino, y_treino):
    """GridSearchCV escolhendo pelo F1, com o mesmo K-Fold de 10 folds usado na comparação."""
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    busca = GridSearchCV(pipeline, param_grid=grade, scoring=f1_scorer, cv=kfold, n_jobs=-1)
    busca.fit(x_treino, y_treino)
    return busca


def validacao_cruzada(modelos, x_treino, y_treino):
    """K-Fold com 10 folds no treino. Devolve o F1 e a acurácia de cada fold."""
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    linhas = []
    for nome, modelo in modelos.items():
        f1 = cross_val_score(modelo, x_treino, y_treino, cv=kfold, scoring=f1_scorer)
        acc = cross_val_score(modelo, x_treino, y_treino, cv=kfold, scoring="accuracy")
        for fold in range(N_FOLDS):
            linhas.append({"modelo": nome, "fold": fold + 1, "f1": f1[fold], "acuracia": acc[fold]})
    return pd.DataFrame(linhas)


def erro_knn_por_k(x_treino, y_treino, escala, metrica, pesos, ks=range(1, 21)):
    """Testa k de 1 a 20 no KNN (com a mesma escala, métrica e pesos do modelo final).

    Calcula o erro médio e o F1 médio na validação cruzada, só com o treino.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    erros, f1s = [], []
    for k in ks:
        knn = Pipeline([("escala", escala),
                        ("modelo", KNeighborsClassifier(n_neighbors=k, metric=metrica, weights=pesos))])
        erros.append(1 - cross_val_score(knn, x_treino, y_treino, cv=kfold, scoring="accuracy").mean())
        f1s.append(cross_val_score(knn, x_treino, y_treino, cv=kfold, scoring=f1_scorer).mean())
    return list(ks), erros, f1s


def avaliar(modelo, x, y):
    pred = modelo.predict(x)
    prob = modelo.predict_proba(x)[:, 1]
    return {
        "Acurácia": accuracy_score(y, pred),
        "Precisão": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "AUC": roc_auc_score(y, prob),
    }


def tabela_resultados(modelos, x_treino, y_treino, x_teste, y_teste, cv):
    """Junta numa tabela: F1 da validação cruzada, métricas no teste e F1 no treino."""
    resultados = {}
    for nome, modelo in modelos.items():
        teste = avaliar(modelo, x_teste, y_teste)
        cv_modelo = cv[cv["modelo"] == nome]
        resultados[nome] = {
            "F1 CV (média)": cv_modelo["f1"].mean(),
            "F1 CV (desvio)": cv_modelo["f1"].std(),
            "Acurácia CV (média)": cv_modelo["acuracia"].mean(),
            **teste,
            "F1 treino": f1_score(y_treino, modelo.predict(x_treino)),
            "F1 teste": teste["F1"],
        }
    tabela = pd.DataFrame(resultados).T
    tabela.index.name = "modelo"
    return tabela


def relatorio(modelo, x_teste, y_teste):
    return classification_report(y_teste, modelo.predict(x_teste),
                                 target_names=["Baixa/Média", "Alta"], digits=3)


def importancia_gini(modelo):
    """Importância das variáveis do Random Forest (feature_importances_)."""
    rf = modelo.named_steps["modelo"]
    return pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)


def arvore_explicativa(x_treino, y_treino, profundidade=3):
    """Árvore pequena só para desenhar as regras. Não entra na comparação dos modelos."""
    arvore = DecisionTreeClassifier(max_depth=profundidade, random_state=RANDOM_STATE)
    return arvore.fit(x_treino, y_treino)


def sem_vazamento(x_treino, x_teste):
    """Confere se algum vinho do teste é igual a um vinho do treino."""
    treino = set(map(tuple, np.round(x_treino.values, 6)))
    return not any(tuple(linha) in treino for linha in np.round(x_teste.values, 6))
