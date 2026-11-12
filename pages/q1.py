
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.polynomial import Polynomial
from itertools import combinations


def interpolar_lagrange(x, y):

    P = Polynomial([0.0])

    for i in range(len(x)):

        Li = Polynomial([1.0])

        for j in range(len(x)):

            if i != j:
                fator = Polynomial([-x[j], 1.0]) / (x[i] - x[j])
                Li *= fator

        P += y[i] * Li

    return P


def calcular_rmse(x, y):

    erros = []

    if len(x) < 2:
        return None

    for i in range(len(x)):

        x_teste = np.delete(x, i)
        y_teste = np.delete(y, i)

        P = interpolar_lagrange(x_teste, y_teste)

        previsao = P(x[i])

        erros.append((y[i] - previsao) ** 2)

    return np.sqrt(np.mean(erros))



def pagina_questao_1():

    st.title("Questão 1 - Interpolação de Lagrange")

    st.write(
        "Selecione diretamente na tabela os pontos que deseja "
        "utilizar na interpolação."
    )

    # Tabela inicial
    dados_iniciais = pd.DataFrame({
        "Usar": [True, True, True, False, False, False, False],
        "x": [0, 0.5, 1, 1.5, 2, 2.5, 3],
        "y": [1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727]
    })

    # Editor da tabela
    dados = st.data_editor(
        dados_iniciais,
        num_rows="fixed",
        use_container_width=True,
        key="tabela_pontos",
        column_config={
            "Usar": st.column_config.CheckboxColumn(
                "Selecionar",
                help="Marque os pontos que deseja utilizar.",
                default=False
            ),
            "x": st.column_config.NumberColumn("X"),
            "y": st.column_config.NumberColumn("Y")
        },
        disabled=["x", "y"]
    )

    # Identificar pontos selecionados
    selecionados = dados[dados["Usar"] == True]

    quantidade = len(selecionados)

    st.write(f"**Pontos selecionados: {quantidade}/7**")

    # Validação da quantidade
    if quantidade == 0:
        st.info("Selecione pelo menos um ponto na tabela.")
        return

    # Extrair coordenadas
    x = selecionados["x"].to_numpy(dtype=float)
    y = selecionados["y"].to_numpy(dtype=float)

    x_todos = dados["x"].to_numpy(dtype=float)
    y_todos = dados["y"].to_numpy(dtype=float)

    # Validar valores
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        st.error("Os pontos selecionados devem possuir valores válidos.")
        return

    if len(np.unique(x)) != len(x):
        st.error("Os valores de X devem ser únicos.")
        return

    # Organizar pontos por X
    ordem = np.argsort(x)
    x = x[ordem]
    y = y[ordem]

    # Construção do polinômio
    P = interpolar_lagrange(x, y)

    st.subheader("Polinômio interpolador")

    st.latex(str(P))

    # Identificação das raízes
    raizes = P.roots()

    raizes_reais = [
        r.real for r in raizes
        if abs(r.imag) < 1e-8
    ]

    st.subheader("Raízes encontradas")

    if raizes_reais:

        for i, raiz in enumerate(raizes_reais):
            st.write(f"Raiz {i+1}: x = {raiz:.8f}")

    else:
        st.info("Nenhuma raiz real encontrada.")

    
    # Gráfico
    st.subheader("Gráfico do polinômio")

    # Definir intervalo incluindo todas as raízes
    x_min = min(x)
    x_max = max(x)

    if raizes_reais:
        x_min = min(x_min, min(raizes_reais))
        x_max = max(x_max, max(raizes_reais))

    amplitude = x_max - x_min

    if amplitude == 0:
        amplitude = 2

    margem = amplitude * 0.15

    x_min -= margem
    x_max += margem

    x_grafico = np.linspace(x_min, x_max, 1000)
    y_grafico = P(x_grafico)

    fig = go.Figure()

    # Todos os pontos da tabela
    fig.add_trace(go.Scatter(
        x=x_todos,
        y=y_todos,
        mode="markers",
        name="Pontos disponíveis",
        marker=dict(color="gray", size=8)
    ))

    # Pontos selecionados
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Pontos selecionados",
        marker=dict(color="orange", size=12, symbol="diamond")
    ))

    # Polinômio
    fig.add_trace(go.Scatter(
        x=x_grafico,
        y=y_grafico,
        mode="lines",
        name="Polinômio interpolador",
        line=dict(color="blue", width=2)
    ))

    # Eixo X
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black"
    )

    # Raízes
    if raizes_reais:

        fig.add_trace(go.Scatter(
            x=raizes_reais,
            y=[0] * len(raizes_reais),
            mode="markers+text",
            name="Raízes reais",
            text=[f"{r:.5f}" for r in raizes_reais],
            textposition="top center",
            marker=dict(
                color="red",
                size=13,
                symbol="x"
            )
        ))

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(range=[x_min, x_max])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabela dos pontos utilizados
    st.subheader("Pontos utilizados")

    tabela_selecionados = pd.DataFrame({
        "x": x,
        "y": y
    })

    st.dataframe(
        tabela_selecionados,
        use_container_width=True
    )
