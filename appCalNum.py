import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.polynomial import Polynomial

# ============================================================
# 1. CONFIGURAÇÃO GERAL (DEVE SER A PRIMEIRA COISA)
# ============================================================
st.set_page_config(
    page_title="Projeto - Cálculo Numérico",
    page_icon="📊",
    layout="wide"
)

# Importações dos módulos locais (certifique-se de que os arquivos existem na pasta 'questoes')
try:
    from pages.q1 import pagina_questao_1
    from pages.q7 import pagina_questao_7
except ImportError:
    # Funções de fallback caso os arquivos ainda não estejam configurados
    def pagina_questao_1():
        st.title("Questão 1")
        st.write("Conteúdo da Questão 1 em desenvolvimento.")
        
    def pagina_questao_7():
        st.title("Questão 7")
        st.write("Conteúdo da Questão 7 em desenvolvimento.")

# ============================================================
# 2. ESTADO DA SESSÃO E NAVEGAÇÃO NA BARRA LATERAL
# ============================================================
if "questoes" not in st.session_state:
    st.session_state.questoes = ["Questão 1", "Questão 7"]

if "pagina_atual" not in st.session_state:
    st.session_state.pagina_atual = "Questão 1"

with st.sidebar:
    st.title("📚 Navegação")
    
    # Menu de rádio vinculado ao session_state
    pagina = st.radio(
        "Selecione uma questão:",
        st.session_state.questoes,
        key="pagina_atual"
    )

# ============================================================
# 3. ROTEAMENTO / EXIBIÇÃO DAS PÁGINAS
# ============================================================
if pagina == "Questão 1":
    pagina_questao_1()
elif pagina == "Questão 7":
    pagina_questao_7()