📊 LexWord — Análise Léxica e Sentimento com Inferência Estatística
LexWord é uma aplicação analítica desenvolvida em Python + Streamlit para análise léxica, detecção de sentimento e inferência estatística avançada, baseada em bancos léxicos ponderados (positivo, negativo e neutro).
O sistema foi projetado para uso acadêmico, científico e exploratório, permitindo desde análises rápidas de frases até relatórios estatísticos completos com exportação em PDF.

🔗 Este módulo faz parte do ecossistema LexOS.

🚀 Principais Funcionalidades

✅ Análise de sentimento em tempo real a partir de frases ou parágrafos
✅ Banco léxico customizável (positivo / negativo / neutro)
✅ Filtro inteligente de Stop Words
✅ Métricas estatísticas completas

Média, mediana, desvio padrão, variância
Assimetria, curtose, MAD
Detecção de outliers (IQR)


✅ Inferência estatística avançada

Teste de normalidade (Shapiro-Wilk)
Teste t (média vs zero)
Intervalo de confiança (95%)
Tamanho de efeito (Cohen’s d)
Kruskal-Wallis entre categorias


✅ Visualizações interativas (Plotly)

Histogramas
Boxplots
Intensidade léxica por termo


✅ Geração automática de relatório em PDF
✅ Interface web responsiva via Streamlit


🧠 Conceito de Funcionamento

O texto é tokenizado e normalizado
Stop Words são removidas (padrão + customizadas)
Termos são cruzados com o banco léxico
O sentimento é inferido pela média ponderada dos pesos
Estatísticas descritivas e inferenciais são aplicadas
Resultados são apresentados visualmente e em relatório

🗂️ Estrutura do Projeto
LexWord/
│
├── app.py                     # Aplicação principal (Streamlit)
├── data/
│   ├── positivo.csv           # Banco léxico positivo
│   ├── negativo.csv           # Banco léxico negativo
│   ├── neutro.csv             # Banco léxico neutro
│   └── stopwords.csv          # Stop Words customizadas
│
├── README.md                  # Documentação do projeto
└── requirements.txt           # Dependências


📦 Dependências
Principais bibliotecas utilizadas:

streamlit
pandas
numpy
plotly
scipy
reportlab

Exemplo de requirements.txt

streamlit>=1.32
pandas
numpy
plotly
scipy
reportlab

▶️ Como Executar Localmente

Clone o repositório:

git clone https://github.com/seu-usuario/lexword.git
cd lexword
``

Crie e ative um ambiente virtual (opcional, recomendado):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Instale as dependências:
pip install -r requirements.txt

Execute a aplicação:
streamlit run app.py

Acesse no navegador:
http://localhost:8501

🧪 Exemplo de Uso
Digite uma frase como:

“O amor traz paz e alegria, mas a dor causa sofrimento.”

O LexWord irá:

Identificar termos léxicos
Calcular o score médio
Classificar o sentimento
Atualizar gráficos e estatísticas
Permitir gerar um PDF analítico completo


📄 Relatório em PDF
O relatório gerado inclui:

Estatísticas descritivas
Testes de hipóteses
Intervalos de confiança
Medidas de efeito
Proporção emocional
Resultados da frase analisada (se aplicável)

Ideal para documentação, pesquisa e apresentações.

⚠️ Observações Importantes

O LexWord não substitui análise humana
Os resultados dependem diretamente da qualidade do banco léxico
Léxicos emocionais tendem a não seguir normalidade — isso é esperado e tratado


📌 Possíveis Extensões Futuras

🔹 Análise temporal de textos
🔹 Comparação entre múltiplas frases/documentos
🔹 Upload de arquivos .txt / .pdf
🔹 Dashboards comparativos
🔹 API REST para integração externa


👤 Autor
Luciano Paiva
Criador do LexWord
📧 Contato: lucianopaiva35@hotmail.com

📜 Licença
© 2026 Luciano Paiva.
Uso autorizado para fins acadêmicos, analíticos e de pesquisa.
Reprodução, redistribuição ou uso comercial requer autorização do autor.

