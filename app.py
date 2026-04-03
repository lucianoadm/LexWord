import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
import os
import numpy as np
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Configuração da Página
st.set_page_config(page_title="LexWord - Sentimento & Léxico", layout="wide")

# Estilização
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {
        background-color: #1e40af !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white !important;
    }
    .stMetric label { color: #e0e7ff !important; }
    .stMetric .metric-value { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ====================== CARREGAMENTO STOP WORDS (CSV) ======================
@st.cache_data
def carregar_stopwords():
    caminho = os.path.join("data", "stopwords.csv")
    try:
        df_stop = pd.read_csv(caminho)
        # Aceita coluna 'stopword', 'palavra' ou 'termo'
        coluna = next((col for col in ['stopword', 'palavra', 'termo'] if col in df_stop.columns), None)
        if coluna:
            stopwords = set(df_stop[coluna].astype(str).str.lower().str.strip())
            return stopwords
        else:
            st.warning("⚠️ Arquivo stopwords.csv encontrado, mas sem coluna válida.")
            return set()
    except FileNotFoundError:
        st.warning("⚠️ Arquivo 'data/stopwords.csv' não encontrado. Usando lista básica.")
        return {
            "o", "a", "os", "as", "um", "uma", "de", "da", "do", "das", "dos", "em", "no", "na",
            "por", "para", "com", "que", "e", "ou", "mas", "se", "como", "é", "era", "foi"
        }
    except Exception:
        return set()

STOP_WORDS = carregar_stopwords()

# ====================== CARREGAMENTO DO BANCO LÉXICO ======================
@st.cache_data
def carregar_banco_lexico():
    caminho = "data"
    try:
        pos = pd.read_csv(os.path.join(caminho, "positivo.csv"))
        neg = pd.read_csv(os.path.join(caminho, "negativo.csv"))
        neu = pd.read_csv(os.path.join(caminho, "neutro.csv"))
        pos["categoria"] = "Positivo"
        neg["categoria"] = "Negativo"
        neu["categoria"] = "Neutro"
        df = pd.concat([pos, neg, neu], ignore_index=True)

        df["termo"] = df["termo"].astype(str).str.lower().str.strip()
        df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
        df = df.dropna(subset=["peso"]).reset_index(drop=True)
        df["peso_abs"] = df["peso"].abs()
        return df
    except FileNotFoundError:
        st.error("⚠️ Erro: Arquivos CSV não encontrados na pasta 'data/'.")
        return pd.DataFrame(columns=["termo", "peso", "categoria", "peso_abs"])

df_ref = carregar_banco_lexico()

# ====================== INTERFACE ======================
st.title("📊 LexWord: Dashboard Analítico")

st.sidebar.title("⚙️ Configurações")
st.sidebar.info(f"Banco Léxico: {len(df_ref)} palavras")
st.sidebar.info(f"Stop Words: {len(STOP_WORDS)} carregadas")

# Controle de Stop Words
usar_stopwords = st.sidebar.checkbox("Filtrar Stop Words", value=True)
stopwords_custom = st.sidebar.text_area(
    "Stop Words adicionais (separadas por vírgula):",
    value="",
    help="Ex: então, porém, muito, apenas, realmente"
)

# --- Análise de Frase ---
st.subheader("🔍 Analisador de Sentimento em Tempo Real")
frase_input = st.text_input(
    "Digite uma frase ou parágrafo:",
    placeholder="Ex: O amor traz paz e alegria, mas a dor causa sofrimento."
)

encontradas = pd.DataFrame(columns=["termo", "peso", "categoria", "peso_abs"])
score = None
sentimento = "—"

if frase_input and not df_ref.empty:
    texto_limpo = frase_input.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(";", "")
    palavras = texto_limpo.split()

    if usar_stopwords:
        stopwords = STOP_WORDS.copy()
        if stopwords_custom.strip():
            stopwords.update([w.strip().lower() for w in stopwords_custom.split(",") if w.strip()])
        palavras = [p for p in palavras if p not in stopwords]

    encontradas = df_ref[df_ref["termo"].isin(palavras)].copy()

    if not encontradas.empty:
        score = float(encontradas["peso"].mean())
        sentimento = "Positivo" if score > 0.1 else ("Negativo" if score < -0.1 else "Neutro")

        c1, c2, c3 = st.columns(3)
        c1.metric("Score da Frase", f"{score:.2f}")
        c2.write(f"**Sentimento Detectado:** {sentimento}")
        c3.write(f"**Termos Identificados:** {len(encontradas)}")
        st.dataframe(encontradas[["termo", "peso", "categoria"]], width="stretch")
    else:
        st.info("Nenhum termo léxico encontrado após filtragem.")

st.divider()

# ====================== BASE ATIVA + GRÁFICOS + INFERÊNCIA ======================
if frase_input and not encontradas.empty:
    df_ativo = encontradas.copy()
    contexto = "Base filtrada pela frase"
else:
    df_ativo = df_ref.copy()
    contexto = "Base completa"

pesos = pd.to_numeric(df_ativo.get("peso", pd.Series(dtype=float)), errors="coerce").dropna()
if pesos.empty:
    pesos = pd.Series([0.0])

# Métricas
st.subheader("📈 Estatísticas do Banco de Referência")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Média", f"{float(pesos.mean()):.2f}")
col_b.metric("Mediana", f"{float(pesos.median()):.2f}")
col_c.metric("Desvio Padrão", f"{float(pesos.std(ddof=1)) if len(pesos) > 1 else 0.0:.2f}")
col_d.metric("Assimetria", f"{float(stats.skew(pesos)) if len(pesos) > 2 else 0.0:.2f}")

# Gráficos (mantidos iguais)
g1, g2 = st.columns(2)
with g1:
    fig_hist = px.histogram(
        df_ativo,
        x="peso",
        color="categoria" if "categoria" in df_ativo.columns else None,
        opacity=0.7,
        barmode="overlay",
        title=f"Distribuição de Pesos — {contexto}"
    )
    # Ajuste: use_container_width=True -> width="stretch"
    st.plotly_chart(fig_hist, width="stretch")

with g2:
    fig_box = px.box(
        df_ativo,
        x="categoria" if "categoria" in df_ativo.columns else None,
        y="peso",
        color="categoria" if "categoria" in df_ativo.columns else None,
        title=f"Quartis por Categoria — {contexto}"
    )
    # Ajuste: use_container_width=True -> width="stretch"
    st.plotly_chart(fig_box, width="stretch")

if "peso_abs" not in df_ativo.columns:
    df_ativo["peso_abs"] = pd.to_numeric(df_ativo["peso"], errors="coerce").abs()

# ====================== GRÁFICO "INTENSIDADE" (MELHOR VISUALIZAÇÃO) ======================
# Substitui o scatter (que polui com muitos termos) por barra horizontal ordenada por intensidade (peso_abs).
# Mantém a mesma intenção: mostrar "força" do termo no léxico.
df_int = df_ativo.copy()
df_int["peso_abs"] = pd.to_numeric(df_int["peso_abs"], errors="coerce")
df_int = df_int.dropna(subset=["peso_abs"])

# Opcional: limitar visualização aos termos mais intensos (sem afetar estatística/inferência)
top_n = min(40, len(df_int))  # mantém boa legibilidade quando o léxico é grande
df_int = df_int.sort_values("peso_abs", ascending=True).tail(top_n)

fig_intensidade = px.bar(
    df_int,
    x="peso_abs",
    y="termo",
    orientation="h",
    color="categoria" if "categoria" in df_int.columns else None,
    title=f"Intensidade Léxica por Termo (Top {top_n}) — {contexto}",
    hover_data={"peso": True, "peso_abs": True, "termo": True}
)
# Ajuste: use_container_width=True -> width="stretch"
st.plotly_chart(fig_intensidade, width="stretch")

# --- INFERÊNCIA (SUPER RELATÓRIO APRIMORADO) ---
with st.expander("📄 Relatório de Inferência Estatística"):
    # ==========================================================
    # ANÁLISE MAIS APROFUNDADA (SUPER INFERÊNCIA)
    # ==========================================================
    n = int(len(pesos))
    media = float(pesos.mean()) if n else 0.0
    mediana = float(pesos.median()) if n else 0.0
    dp = float(pesos.std(ddof=1)) if n > 1 else 0.0
    var = float(pesos.var(ddof=1)) if n > 1 else 0.0
    minimo = float(pesos.min()) if n else 0.0
    maximo = float(pesos.max()) if n else 0.0

    st.write(f"**Amostra:** n = `{n}` | min = `{minimo:.3f}` | máx = `{maximo:.3f}`")
    st.write(f"**Centro e dispersão:** média = `{media:.3f}` | mediana = `{mediana:.3f}` | DP = `{dp:.3f}` | variância = `{var:.3f}`")

    # 1) Normalidade
    if n >= 4:
        sample_n = min(5000, n)
        shapiro_stat, p_val = stats.shapiro(pesos.sample(sample_n, random_state=42))
    else:
        shapiro_stat, p_val = (np.nan, 1.0)
    st.write(f"**Normalidade (Shapiro-Wilk):** p = `{p_val:.4e}`")
    if p_val < 0.05:
        st.write("❌ **Interpretação:** Evidência forte de **não-normalidade**. Comum em léxicos emocionais com polarização extrema.")
    else:
        st.write("✅ **Interpretação:** Distribuição aproximadamente normal.")

    # 2) Teste t vs 0
    if n >= 2:
        t_stat, t_p = stats.ttest_1samp(pesos, 0.0)
    else:
        t_stat, t_p = (0.0, 1.0)
    st.write(f"**Teste t (média vs 0):** t = `{t_stat:.3f}` | p = `{t_p:.4e}`")
    if t_p < 0.05:
        direcao = "positivo" if media > 0 else "negativo" if media < 0 else "neutro"
        st.write(f"✅ **Conclusão:** Viés estatisticamente significativo **{direcao}** no léxico.")
    else:
        st.write("ℹ️ **Conclusão:** Sem evidência de viés médio diferente de zero.")

    # 3) Intervalo de Confiança
    if n >= 2 and dp > 0:
        sem = float(stats.sem(pesos))
        ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=media, scale=sem)
        st.write(f"**IC 95% da média:** [`{ci_low:.3f}`, `{ci_high:.3f}`]")
    else:
        st.write("**IC 95% da média:** indisponível.")

    # 4) Tamanho de efeito
    cohen_d = (media / dp) if dp > 0 else 0.0
    st.write(f"**Tamanho de efeito (Cohen’s d):** `{cohen_d:.2f}` (0.2=pequeno | 0.5=médio | 0.8=grande)")

    # 5) Assimetria e Curtose
    if n >= 4:
        skew = float(stats.skew(pesos))
        kurt = float(stats.kurtosis(pesos, fisher=True))
    else:
        skew, kurt = 0.0, 0.0
    st.write(f"**Forma da distribuição:** skew = `{skew:.2f}` | kurtose = `{kurt:.2f}`")

    # 6) Robustez
    mad = float(np.median(np.abs(pesos - mediana))) if n else 0.0
    st.write(f"**Robustez (MAD):** `{mad:.3f}`")

    # 7) Outliers
    if n >= 4:
        q1, q3 = np.percentile(pesos, [25, 75])
        iqr = float(q3 - q1)
        lim_inf = float(q1 - 1.5 * iqr)
        lim_sup = float(q3 + 1.5 * iqr)
        outliers = int(((pesos < lim_inf) | (pesos > lim_sup)).sum())
        st.write(f"**Outliers (1.5×IQR):** `{outliers}` | IQR = `{iqr:.3f}`")
    else:
        st.write("**Outliers:** indisponível.")

    # 8) Balanço Emocional
    pos_ratio = float((pesos > 0).mean()) if n else 0.0
    neg_ratio = float((pesos < 0).mean()) if n else 0.0
    neu_ratio = float((pesos == 0).mean()) if n else 0.0
    st.write(f"**Balanço Emocional:** Positivos = `{pos_ratio:.1%}` | Negativos = `{neg_ratio:.1%}` | Neutros = `{neu_ratio:.1%}`")

    # 9) Kruskal-Wallis
    try:
        grupos = [g["peso"].values for _, g in df_ativo.groupby("categoria")]
        if len(grupos) >= 2 and all(len(g) > 1 for g in grupos):
            kw_stat, kw_p = stats.kruskal(*grupos)
            st.write(f"**Kruskal-Wallis (diferença entre categorias):** H = `{kw_stat:.3f}` | p = `{kw_p:.4e}`")
    except:
        pass

    # 10) Resumo por Categoria
    try:
        resumo_cat = df_ativo.groupby("categoria")["peso"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        resumo_cat.columns = ["categoria", "n", "media", "mediana", "dp", "min", "max"]
        st.write("**Resumo Estatístico por Categoria:**")
        st.dataframe(resumo_cat, width="stretch")
    except:
        pass

    # 11) Termos Extremos
    try:
        top_pos = df_ativo.sort_values("peso", ascending=False).head(8)[["termo", "peso", "categoria"]]
        top_neg = df_ativo.sort_values("peso", ascending=True).head(8)[["termo", "peso", "categoria"]]
        st.write("**Extremos do Léxico (Polarização):**")
        cpos, cneg = st.columns(2)
        with cpos:
            st.write("**Top 8 Positivos:**")
            st.dataframe(top_pos, width="stretch")
        with cneg:
            st.write("**Top 8 Negativos:**")
            st.dataframe(top_neg, width="stretch")
    except:
        pass

    # --- PDF ---
    def gerar_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        y = 800
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "LexWord — Relatório de Análise Léxica (Inferência Avançada)")
        y -= 24
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        y -= 18
        c.drawString(50, y, f"Amostra: n={n} | min={minimo:.3f} | max={maximo:.3f}")
        y -= 14
        c.drawString(50, y, f"Média={media:.3f} | Mediana={mediana:.3f} | DP={dp:.3f} | Var={var:.3f}")
        y -= 14
        c.drawString(50, y, f"Normalidade (Shapiro): p={p_val:.4e}")
        y -= 14
        c.drawString(50, y, f"Teste t (média vs 0): t={t_stat:.3f} | p={t_p:.4e}")
        y -= 14
        if n >= 2 and dp > 0:
            sem = float(stats.sem(pesos))
            ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=media, scale=sem)
            c.drawString(50, y, f"IC 95% da média: [{ci_low:.3f}, {ci_high:.3f}]")
            y -= 14
        c.drawString(50, y, f"Cohen’s d={cohen_d:.2f} | Skew={skew:.2f} | Kurtose={kurt:.2f} | MAD={mad:.3f}")
        y -= 18
        c.drawString(50, y, f"Proporções: +={pos_ratio:.1%} | -={neg_ratio:.1%} | 0={neu_ratio:.1%}")
        y -= 18
        if score is not None:
            c.drawString(50, y, f"Score da frase: {score:.3f} | Sentimento: {sentimento} | Termos identificados: {len(encontradas)}")
            y -= 18
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, 60, "© 2026 Luciano Paiva — Criador do LexWord. Todos os direitos reservados.")
        c.drawString(50, 46, "Uso autorizado apenas para fins de análise e pesquisa. Reprodução/redistribuição requer permissão do autor.")
        c.drawString(50, 32, "Contato do autor: lucianopaiva35@hotmail.com")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    st.download_button(
        "📥 Gerar e baixar PDF da análise",
        data=gerar_pdf(),
        file_name="LexWord_Analise_Avancada.pdf",
        mime="application/pdf"
    )

# --- RODAPÉ ---
st.caption("LexWord v1.0 | Análise Léxica Avançada | © 2026 Luciano Paiva (Criador) — Todos os direitos reservados.")