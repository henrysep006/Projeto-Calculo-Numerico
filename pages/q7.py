import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from scipy.integrate import trapezoid, simpson
     

    
def pagina_questao_7():
   
    
    # ============================================================
    # CONFIGURACAO DA PAGINA
    # ============================================================
    
    st.set_page_config(
        page_title="Simulador de Velocidade e Integracao Numerica",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .stMetric {
            background-color: #1e222b;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #ff4b4b;
        }
        .stButton button {
            width: 100%;
            border-radius: 6px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    
    # ============================================================
    # METODOS DE INTEGRACAO NUMERICA
    # ============================================================
    
    def validar_dados(x, y):
        """Valida os dados usados pelos metodos de integracao."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    
        if len(x) != len(y):
            raise ValueError("X e Y devem possuir a mesma quantidade de valores.")
    
        if len(x) < 2:
            raise ValueError("Informe pelo menos dois pontos.")
    
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Os dados devem conter apenas valores finitos.")
    
        if not np.all(np.diff(x) > 0):
            raise ValueError("Os valores de tempo devem estar em ordem crescente.")
    
        return x, y
    
    
    def verificar_espacamento_uniforme(x):
        """Retorna h e verifica se os pontos sao igualmente espacados."""
        diferencas = np.diff(x)
        h = diferencas[0]
    
        if not np.allclose(diferencas, h, rtol=1e-7, atol=1e-9):
            raise ValueError("Este metodo exige pontos igualmente espacados.")
    
        return h
    
    
    def regra_trapezios(x, y):
        """
        Regra dos trapezios composta.
        Aceita pontos com espacamento uniforme ou nao uniforme.
        """
        x, y = validar_dados(x, y)
    
        areas = []
        integral = 0.0
    
        for i in range(len(x) - 1):
            h_i = x[i + 1] - x[i]
            area = h_i * (y[i] + y[i + 1]) / 2
            areas.append(area)
            integral += area
    
        return integral, areas
    
    
    def regra_simpson_13(x, y):
        """Simpson 1/3; para n impar, combina Simpson 1/3 e 3/8."""
        x, y = validar_dados(x, y)
        n = len(x) - 1
        if n < 1:
            raise ValueError("Informe pelo menos dois pontos.")
        h = verificar_espacamento_uniforme(x)
    
        integral = 0.0
        etapas = []
    
        if n % 2 == 0:
            limite = n
            descricao = "Simpson 1/3 composto."
        elif n >= 3:
            limite = n - 3
            descricao = (
                "Aproximacao combinada: Simpson 1/3 na parte inicial "
                "e Simpson 3/8 nos tres ultimos subintervalos."
            )
        else:
            limite = 0
            descricao = (
                "Aproximacao pelo trapezio, pois existe apenas "
                "um subintervalo e nao e possivel aplicar Simpson."
            )
    
        for i in range(0, limite, 2):
            area = (h / 3) * (y[i] + 4*y[i+1] + y[i+2])
            integral += area
            etapas.append({
                "Metodo": "Simpson 1/3",
                "Pontos": f"{i}, {i+1}, {i+2}",
                "Contribuicao": area
            })
    
        if n % 2 == 1 and n >= 3:
            i = n - 3
            area = (3*h / 8) * (y[i] + 3*y[i+1] + 3*y[i+2] + y[i+3])
            integral += area
            etapas.append({
                "Metodo": "Simpson 3/8 complementar",
                "Pontos": f"{i}, {i+1}, {i+2}, {i+3}",
                "Contribuicao": area
            })
        elif n == 1:
            area = h * (y[0] + y[1]) / 2
            integral += area
            etapas.append({
                "Metodo": "Trapezio complementar",
                "Pontos": "0, 1",
                "Contribuicao": area
            })
    
        return integral, etapas, descricao
    
    
    # Pesos das formulas fechadas de Newton-Cotes.
    # Grau 1: h/2 * [1, 1]
    # Grau 2: h/3 * [1, 4, 1]
    # Grau 3: 3h/8 * [1, 3, 3, 1]
    # Grau 4: 2h/45 * [7, 32, 12, 32, 7]
    
    PESOS_NEWTON_COTES = {
        1: (np.array([1, 1], dtype=float), 1 / 2),
        2: (np.array([1, 4, 1], dtype=float), 1 / 3),
        3: (np.array([1, 3, 3, 1], dtype=float), 3 / 8),
        4: (np.array([7, 32, 12, 32, 7], dtype=float), 2 / 45)
    }
    
    
    def newton_cotes_composta(x, y, grau):
        """
        Newton-Cotes fechada composta, graus 1 a 4.
        Se houver subintervalos restantes, usa Newton-Cotes de grau menor
        para integrar o trecho final, identificando-o como aproximacao.
        """
        x, y = validar_dados(x, y)
        if grau not in PESOS_NEWTON_COTES:
            raise ValueError("O grau deve ser 1, 2, 3 ou 4.")
    
        n = len(x) - 1
        if n < grau:
            raise ValueError(
                f"Newton-Cotes de grau {grau} precisa de pelo menos {grau + 1} pontos."
            )
    
        h = verificar_espacamento_uniforme(x)
        pesos, fator = PESOS_NEWTON_COTES[grau]
        integral = 0.0
        etapas = []
        inicio = 0
    
        quantidade_paineis = n // grau
        restante = n % grau
    
        for _ in range(quantidade_paineis):
            fim = inicio + grau + 1
            area = fator * h * np.dot(pesos, y[inicio:fim])
            integral += area
            etapas.append({
                "Grau utilizado": grau,
                "Pontos": f"{inicio} ate {fim - 1}",
                "Contribuicao": area,
                "Tipo": "Metodo solicitado"
            })
            inicio += grau
    
        if restante:
            pesos_restantes, fator_restante = PESOS_NEWTON_COTES[restante]
            fim = inicio + restante + 1
            area = fator_restante * h * np.dot(pesos_restantes, y[inicio:fim])
            integral += area
            etapas.append({
                "Grau utilizado": restante,
                "Pontos": f"{inicio} ate {fim - 1}",
                "Contribuicao": area,
                "Tipo": f"Aproximacao complementar (grau {restante})"
            })
            descricao = (
                f"Aproximacao mista: grau {grau} nos paineis completos "
                f"e grau {restante} no trecho final."
            )
        else:
            descricao = f"Newton-Cotes composto de grau {grau}."
    
        return integral, etapas, descricao
    
    
    # ============================================================
    # GRAFICOS GEOMETRICOS DOS METODOS
    # ============================================================
    
    def figura_trapezios(x, y):
        """
        Monta o grafico com os trapezios da regra dos trapezios.
        Cada trapezio liga (x_i, 0) -> (x_i, y_i) -> (x_i+1, y_i+1) -> (x_i+1, 0).
        """
        x, y = validar_dados(x, y)
    
        xs_poligonos, ys_poligonos = [], []
        for i in range(len(x) - 1):
            xs_poligonos += [x[i], x[i], x[i + 1], x[i + 1], x[i], None]
            ys_poligonos += [0.0, y[i], y[i + 1], 0.0, 0.0, None]
    
        fig = go.Figure()
    
        # Todos os trapezios em um unico trace (None separa os poligonos)
        fig.add_trace(go.Scatter(
            x=xs_poligonos,
            y=ys_poligonos,
            mode="lines",
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.30)",
            line=dict(color="#636EFA", width=1.5),
            name="Trapezios",
            hoverinfo="skip"
        ))
    
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=8, color="#FFFFFF", line=dict(color="#636EFA", width=2)),
            name="Pontos medidos"
        ))
    
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Tempo (s)",
            yaxis_title="Velocidade (m/s)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        )
    
        return fig
    
    
    def _interpolacao_lagrange(xn, yn, pontos=60):
        """Polinomio interpolador de Lagrange que passa por (xn, yn)."""
        xn = np.asarray(xn, dtype=float)
        yn = np.asarray(yn, dtype=float)
    
        xx = np.linspace(xn[0], xn[-1], pontos)
        yy = np.zeros_like(xx)
    
        for j in range(len(xn)):
            termo = np.full_like(xx, yn[j])
            for m in range(len(xn)):
                if m != j:
                    termo *= (xx - xn[m]) / (xn[j] - xn[m])
            yy += termo
    
        return xx, yy
    
    
    def figura_simpson_13(x, y):
        """
        Monta o grafico com as parabolas da regra de Simpson 1/3.
        Cada parabola passa por 3 pontos consecutivos (2 subintervalos).
        Segue a mesma logica de regra_simpson_13: se o numero de
        subintervalos for impar, os 3 ultimos usam Simpson 3/8 (cubica).
        """
        x, y = validar_dados(x, y)
        n = len(x) - 1
        verificar_espacamento_uniforme(x)
    
        paineis = []  # cada painel: (indices dos pontos, tipo)
    
        if n % 2 == 0:
            for i in range(0, n, 2):
                paineis.append(([i, i + 1, i + 2], "parabola"))
        elif n >= 3:
            for i in range(0, n - 3, 2):
                paineis.append(([i, i + 1, i + 2], "parabola"))
            paineis.append(([n - 3, n - 2, n - 1, n], "cubica"))
        else:
            paineis.append(([0, 1], "reta"))
    
        estilos = {
            "parabola": ("Parabolas (Simpson 1/3)", ["#00CC96", "#AB63FA"]),
            "cubica": ("Cubica (Simpson 3/8 complementar)", ["#FFA15A"]),
            "reta": ("Reta (trapezio complementar)", ["#FFA15A"])
        }
    
        fig = go.Figure()
        legendas_exibidas = set()
        contador_parabolas = 0
    
        for indices, tipo in paineis:
            nome, cores = estilos[tipo]
    
            if tipo == "parabola":
                cor = cores[contador_parabolas % len(cores)]
                contador_parabolas += 1
            else:
                cor = cores[0]
    
            xx, yy = _interpolacao_lagrange(x[indices], y[indices])
    
            # fecha o poligono ate o eixo x para preencher a area
            xs_poligono = np.concatenate([xx, [xx[-1], xx[0]]])
            ys_poligono = np.concatenate([yy, [0.0, 0.0]])
    
            fig.add_trace(go.Scatter(
                x=xs_poligono,
                y=ys_poligono,
                mode="lines",
                fill="toself",
                fillcolor=_cor_com_alpha(cor, 0.28),
                line=dict(color=cor, width=2),
                name=nome,
                legendgroup=tipo,
                showlegend=tipo not in legendas_exibidas,
                hoverinfo="skip"
            ))
            legendas_exibidas.add(tipo)
    
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=8, color="#FFFFFF", line=dict(color="#00CC96", width=2)),
            name="Pontos medidos"
        ))
    
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Tempo (s)",
            yaxis_title="Velocidade (m/s)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        )
    
        return fig
    
    
    def _cor_com_alpha(hex_cor, alpha):
        """Converte '#RRGGBB' em 'rgba(r, g, b, alpha)'."""
        hex_cor = hex_cor.lstrip("#")
        r, g, b = (int(hex_cor[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    
    
    # ============================================================
    # ESTADO DA SIMULACAO
    # ============================================================
    
    valores_iniciais = {
        "time_step": 1.0,
        "max_time": 20,
        "sim_finished": False,
        "current_time": 0.0,
        "current_velocity": 0.0,
        "history_t": [0.0],
        "history_v": [0.0]
    }
    
    for chave, valor in valores_iniciais.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor
    
    
    # ============================================================
    # SIDEBAR - CONTROLES
    # ============================================================
    
    st.sidebar.title("🏎️ Painel de Controle")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Parametros da pista")
    
    max_t_input = st.sidebar.slider(
        "Duracao desejada do percurso (s)",
        min_value=10,
        max_value=60,
        value=20,
        step=5
    )
    
    dt_input = st.sidebar.select_slider(
        "Passo de tempo Δt (s)",
        options=[0.5, 1.0, 2.0],
        value=1.0
    )
    
    accel_power = st.sidebar.slider(
        "Potencia de aceleracao (m/s²)",
        1.0, 10.0, 3.0
    )
    
    brake_power = st.sidebar.slider(
        "Potencia de frenagem (m/s²)",
        1.0, 15.0, 5.0
    )
    
    
    def reset_simulation():
        st.session_state.time_step = dt_input
        st.session_state.max_time = max_t_input
        st.session_state.sim_finished = False
        st.session_state.current_time = 0.0
        st.session_state.current_velocity = 0.0
        st.session_state.history_t = [0.0]
        st.session_state.history_v = [0.0]
    
    
    st.sidebar.button(
        "🔄 Reiniciar percurso",
        on_click=reset_simulation
    )
    
    
    # ============================================================
    # CABECALHO
    # ============================================================
    
    st.title("🏎️ Telemetria e Integracao Numerica de Velocidade")
    
    st.markdown("""
    Controle a aceleracao do veiculo em intervalos discretos de tempo.
    Ao finalizar o percurso, a distancia sera calculada usando:
    
    - Regra dos trapezios;
    - Regra de Simpson 1/3;
    - Newton-Cotes fechado dos graus 1, 2, 3 e 4.
    
    A distancia percorrida e representada pela integral da velocidade:
    """)
    
    st.latex(r"s = \int_{t_0}^{t_f} v(t)\,dt")
    
    st.markdown("---")
    
    
    # ============================================================
    # SIMULACAO DO VEICULO
    # ============================================================
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("🎮 Controles do veiculo")
    
        c1, c2 = st.columns(2)
    
        c1.metric(
            "Tempo decorrido",
            f"{st.session_state.current_time:.1f} s",
            f"Meta: {max_t_input} s"
        )
    
        c2.metric(
            "Velocidade atual",
            f"{st.session_state.current_velocity:.1f} m/s",
            f"{st.session_state.current_velocity * 3.6:.1f} km/h"
        )
    
        progresso = min(
            st.session_state.current_time / max_t_input,
            1.0
        )
        st.progress(progresso)
    
        if not st.session_state.sim_finished:
            st.markdown("### Comando para o proximo intervalo")
    
            c_accel, c_keep, c_brake = st.columns(3)
    
            acelerar = c_accel.button("🚀 Acelerar")
            manter = c_keep.button("➡️ Manter")
            frear = c_brake.button("🛑 Frear")
    
            acao = None
    
            if acelerar:
                acao = "acelerar"
            elif manter:
                acao = "manter"
            elif frear:
                acao = "frear"
    
            if acao is not None:
                dt = st.session_state.time_step
                velocidade = st.session_state.current_velocity
    
                if acao == "acelerar":
                    nova_velocidade = velocidade + accel_power * dt
                elif acao == "frear":
                    nova_velocidade = max(0.0, velocidade - brake_power * dt)
                else:
                    nova_velocidade = velocidade
    
                st.session_state.current_velocity = nova_velocidade
                st.session_state.current_time += dt
    
                st.session_state.history_t.append(
                    st.session_state.current_time
                )
                st.session_state.history_v.append(nova_velocidade)
    
                if st.session_state.current_time >= max_t_input:
                    st.session_state.sim_finished = True
    
                st.rerun()
    
        else:
            st.success("🏁 Percurso finalizado! Consulte os resultados abaixo.")
    
    
    # ============================================================
    # GRAFICO DE VELOCIDADE POR TEMPO
    # ============================================================
    
    with col_right:
        st.subheader("📈 Grafico de velocidade x tempo")
    
        t_arr = np.array(st.session_state.history_t, dtype=float)
        v_arr = np.array(st.session_state.history_v, dtype=float)
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=t_arr,
            y=v_arr,
            mode="lines+markers",
            name="Velocidade v(t)",
            line=dict(color="#00CC96", width=3),
            marker=dict(size=8, color="#00FFB2")
        ))
    
        fig.add_trace(go.Scatter(
            x=t_arr,
            y=v_arr,
            fill="tozeroy",
            fillcolor="rgba(0, 204, 150, 0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Area sob a curva"
        ))
    
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Tempo (s)",
            yaxis_title="Velocidade (m/s)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified"
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("---")
    
    
    # ============================================================
    # INTEGRACAO NUMERICA
    # ============================================================
    
    if len(st.session_state.history_t) > 1:
    
        st.header("🧮 Calculo da distancia por integracao numerica")
    
        t_data = np.array(st.session_state.history_t, dtype=float)
        v_data = np.array(st.session_state.history_v, dtype=float)
    
        n = len(t_data) - 1
    
        st.write(
            f"Quantidade de pontos: **{len(t_data)}**  |  "
            f"Subintervalos: **{n}**  |  "
            f"Passo de tempo: **{st.session_state.time_step} s**"
        )
    
        # --------------------------------------------------------
        # Calculo dos metodos
        # --------------------------------------------------------
    
        resultado_trapezios, etapas_trapezios = regra_trapezios(
            t_data, v_data
        )
    
        resultado_simpson, etapas_simpson, descricao_simpson = regra_simpson_13(
            t_data, v_data
        )
    
        resultados_nc = {}
    
        for grau in range(1, 5):
            try:
                resultado, etapas, descricao = newton_cotes_composta(
                    t_data, v_data, grau
                )
                resultados_nc[grau] = {
                    "resultado": resultado,
                    "etapas": etapas,
                    "descricao": descricao,
                    "erro": None
                }
            except ValueError as erro:
                resultados_nc[grau] = {
                    "resultado": None,
                    "etapas": None,
                    "descricao": None,
                    "erro": str(erro)
                }
    
        # --------------------------------------------------------
        # Cards de resultados
        # --------------------------------------------------------
    
        st.subheader("Resultados")
    
        c1, c2, c3 = st.columns(3)
    
        c1.metric(
            "Regra dos trapezios",
            f"{resultado_trapezios:.4f} m"
        )
    
        if resultado_simpson is not None:
            c2.metric(
                "Simpson 1/3",
                f"{resultado_simpson:.4f} m"
            )
        else:
            c2.metric("Simpson 1/3", f"{resultado_simpson:.4f} m")
    
        if resultados_nc[3]["resultado"] is not None:
            c3.metric(
                "Newton-Cotes grau 3",
                f"{resultados_nc[3]['resultado']:.4f} m"
            )
        else:
            c3.metric("Newton-Cotes grau 3", "Indisponivel")
    
        # Tabela comparativa
        comparacao = [{
            "Metodo": "Trapezios",
            "Resultado (m)": resultado_trapezios,
            "Situacao": "Calculado"
        }]
    
        comparacao.append({
            "Metodo": "Simpson 1/3",
            "Resultado (m)": resultado_simpson,
            "Situacao": descricao_simpson
        })
    
        for grau in range(1, 5):
            info = resultados_nc[grau]
    
            comparacao.append({
                "Metodo": f"Newton-Cotes grau {grau}",
                "Resultado (m)": info["resultado"],
                "Situacao": "Calculado" if info["resultado"] is not None else info["erro"]
            })
    
        st.dataframe(
            pd.DataFrame(comparacao),
            use_container_width=True,
            hide_index=True
        )
    
        st.caption(
            "Newton-Cotes grau 1 equivale a trapezios; "
            "Newton-Cotes grau 2 equivale a Simpson 1/3. "
            "Os resultados repetidos sao esperados."
        )
    
        # --------------------------------------------------------
        # Graficos geometricos: trapezios e parabolas
        # --------------------------------------------------------
    
        st.subheader("📐 Visualizacao geometrica dos metodos")
    
        col_g_trap, col_g_simp = st.columns(2)
    
        with col_g_trap:
            st.markdown("**Regra dos trapezios**")
            st.plotly_chart(
                figura_trapezios(t_data, v_data),
                use_container_width=True,
                key="grafico_trapezios"
            )
            st.caption(
                f"Cada trecho entre dois pontos e aproximado por uma reta, "
                f"formando {n} trapezio(s). Area total: "
                f"{resultado_trapezios:.4f} m."
            )
    
        with col_g_simp:
            st.markdown("**Regra de Simpson 1/3**")
            st.plotly_chart(
                figura_simpson_13(t_data, v_data),
                use_container_width=True,
                key="grafico_simpson"
            )
            st.caption(
                f"Cada parabola passa por 3 pontos consecutivos "
                f"(2 subintervalos). Area total: {resultado_simpson:.4f} m. "
                f"{descricao_simpson}"
            )
    
        # --------------------------------------------------------
        # Explicacao dos metodos e etapas
        # --------------------------------------------------------
    
        st.subheader("Detalhamento dos metodos")
    
        tab_trap, tab_simp, tab_nc = st.tabs([
            "Regra dos trapezios",
            "Simpson 1/3",
            "Newton-Cotes graus 1 a 4"
        ])
    
        with tab_trap:
            st.markdown("""
            A regra dos trapezios aproxima cada trecho da curva por uma reta.
            A area total e a soma das areas dos trapezios:
            """)
            st.latex(
                r"A_T = \sum_{i=0}^{n-1}"
                r"\frac{(t_{i+1}-t_i)(v_i+v_{i+1})}{2}"
            )
    
            tabela_trap = []
    
            for i, area in enumerate(etapas_trapezios):
                h_i = t_data[i + 1] - t_data[i]
    
                tabela_trap.append({
                    "Intervalo": f"{i + 1}",
                    "Inicio": t_data[i],
                    "Fim": t_data[i + 1],
                    "Area": area
                })
    
            st.dataframe(
                pd.DataFrame(tabela_trap),
                use_container_width=True,
                hide_index=True
            )
    
            st.success(
                f"Distancia aproximada pelos trapezios: "
                f"{resultado_trapezios:.6f} m"
            )
    
        with tab_simp:
            st.markdown("""
            Simpson 1/3 aproxima a funcao por uma parabola a cada dois
            subintervalos, utilizando tres pontos consecutivos.
            """)
            st.latex(
                r"A_S = \frac{h}{3}"
                r"[v_0+4v_1+2v_2+\cdots+4v_{n-1}+v_n]"
            )
    
            st.info(descricao_simpson)
            st.success(
                f"Distancia aproximada por Simpson: {resultado_simpson:.6f} m"
            )
            st.dataframe(
                pd.DataFrame(etapas_simpson),
                use_container_width=True,
                hide_index=True
            )
    
        with tab_nc:
            st.markdown("""
            As formulas fechadas de Newton-Cotes aproximam a funcao por
            polinomios interpoladores. Cada painel de grau `m` utiliza
            `m + 1` pontos igualmente espacados.
            """)
    
            st.markdown("**Formulas utilizadas:**")
    
            st.latex(
                r"NC_1 = \frac{h}{2}(f_0+f_1)"
            )
            st.latex(
                r"NC_2 = \frac{h}{3}(f_0+4f_1+f_2)"
            )
            st.latex(
                r"NC_3 = \frac{3h}{8}(f_0+3f_1+3f_2+f_3)"
            )
            st.latex(
                r"NC_4 = \frac{2h}{45}"
                r"(7f_0+32f_1+12f_2+32f_3+7f_4)"
            )
    
            for grau in range(1, 5):
                info = resultados_nc[grau]
    
                st.markdown(f"**Newton-Cotes de grau {grau}**")
    
                if info["resultado"] is None:
                    st.warning(info["erro"])
                else:
                    st.info(info["descricao"])
                    st.success(f"Resultado: {info['resultado']:.6f} m")
                    st.dataframe(
                        pd.DataFrame(info["etapas"]),
                        use_container_width=True,
                        hide_index=True
                    )
    
        # --------------------------------------------------------
        # Historico da telemetria
        # --------------------------------------------------------
    
        with st.expander("Ver tabela completa da telemetria"):
            df_telemetria = pd.DataFrame({
                "Tempo (s)": t_data,
                "Velocidade (m/s)": v_data,
                "Velocidade (km/h)": v_data * 3.6
            })
    
            st.dataframe(
                df_telemetria,
                use_container_width=True,
                hide_index=True
            )
    
    else:
        st.info(
            "Use os botoes do painel para acelerar, manter ou frear "
            "e comecar a registrar a telemetria."
        )