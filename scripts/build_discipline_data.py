#!/usr/bin/env python3
"""
build_discipline_data.py
========================
Constrói os dados curados de disciplinas de Engenharia de Sistemas da UFMG.

Etapas:
  1. Baixa a planilha de dificuldade e salva como CSV local.
  2. Extrai as disciplinas do PDF "Mapa de Oferta por Percurso Curricular" (2026/1).
  3. Enriquece com período, pré-requisitos, área, ciclo, ementa e dificuldade.
  4. Salva o JSON final em data/disciplinas_engsis.json.

Uso:
  python scripts/build_discipline_data.py [--pdf CAMINHO_PDF]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import pdfplumber

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIFICULDADE = DATA_DIR / "dificuldade_engsis.csv"
JSON_DISCIPLINAS = DATA_DIR / "disciplinas_engsis.json"

# URL pública da planilha de dificuldade (Google Sheets → CSV)
URL_DIFICULDADE = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vT7KtZA2fWbfXp1K5Ek8EcAZl6Ae6kzSw5qDryR5_Jqu89aKJvF15o6YOJhvYLre-"
    "7st3tmw5zRsv9H/pub?output=csv&gid=0"
)

# Caminho padrão do PDF de oferta
DEFAULT_PDF = Path("/var/home/andrerezende/Downloads/FIA/arquivo.pdf")

# Caminho do CSV de horários da grade
SCHEDULE_CSV = Path("/var/home/andrerezende/Downloads/grade_horarios_geesufmg.csv")

# Mapeamento de códigos placeholder do CSV de horários → códigos reais
PLACEHOLDER_MAP: dict[str, str] = {
    "ELEXXS": "ELE630",
    "DCCXXA": "DCC217",
    "ELEXXA": "ELE631",
    "ELEXXG": "ELE632",
    "ESAZZZ": "ESA019",
    "ELTXXA": "ELT136",
    "EEEEXXB": "EEE048",
    "EMTXXX": "EMT122",
    "EEEXXC": "EEE050",
    "EEEXXA": "EEE049",
    "ELEXXC": "ELE633",
    "DCCXXX": "DCC218",
    "ELEXXD": "ELE634",
    "ELEXXE": "ELE635",
    "ELEXXH": "EEE051",
    "EEEXXP": "EEE052",
    "EEEXXQ": "EEE019",
    "EEEXXR": "EEE020",
}

# ---------------------------------------------------------------------------
# Dados do PPC – grade curricular completa
# ---------------------------------------------------------------------------
# Formato: (codigo, nome, carga_horaria, periodo, [pre_requisitos])
PPC_GRADE: list[tuple[str, str, int, int, list[str]]] = [
    # --- Período 1 (Ciclo Básico Ciências) ---
    ("MAT001", "CALCULO DIFERENCIAL E INTEGRAL I", 90, 1, []),
    ("MAT038", "GEOMETRIA ANALITICA E ALGEBRA LINEAR", 60, 1, []),
    ("QUI628", "QUÍMICA GERAL E", 60, 1, []),
    ("ELE630", "INTRODUÇÃO A ENGENHARIA DE SISTEMAS", 30, 1, []),
    ("DCC217", "MATEMÁTICA DISCRETA PARA ENGENHARIA", 60, 1, []),
    # --- Período 2 ---
    ("MAT039", "CÁLCULO DIFERENCIAL E INTEGRAL II", 60, 2, ["MAT001"]),
    ("FIS065", "FUNDAMENTOS DE MECANICA", 60, 2, ["MAT001"]),
    ("DCC203", "PROGRAMAÇÃO E DESENVOLVIMENTO DE SOFTWARE I", 60, 2, []),
    ("EMT122", "INTRODUÇÃO À CIÊNCIA DOS MATERIAIS", 30, 2, ["QUI628"]),
    # --- Período 3 ---
    ("MAT002", "CALCULO DIFERENCIAL E INTEGRAL III", 60, 3, ["MAT039"]),
    ("FIS069", "FUNDAMENTOS DE ELETROMAGNETISMO", 60, 3, ["MAT039", "FIS065"]),
    ("DCC204", "PROGRAMAÇÃO E DESENVOLVIMENTO DE SOFTWARE II", 60, 3, ["DCC203"]),
    ("DCC205", "ESTRUTURAS DE DADOS", 60, 3, ["DCC203"]),
    ("DCC218", "INTRODUÇÃO A SISTEMAS COMPUTACIONAIS", 60, 3, ["DCC203"]),
    # --- Período 4 ---
    ("MAT015", "EQUAÇÕES DIFERENCIAIS A", 60, 4, ["MAT039"]),
    ("FIS086", "FUNDAMENTOS DE OSCILAÇÕES, ONDAS E ÓPTICA", 60, 4, ["FIS065", "MAT039"]),
    ("FIS152", "FUNDAMENTOS DE MECÂNICA DOS FLUIDOS E TERMODINÂMICA", 30, 4, ["FIS065"]),
    ("ELE064", "ANALISE DE CIRCUITOS ELETRICOS I", 30, 4, ["FIS069"]),
    ("ELT124", "SISTEMAS DIGITAIS", 60, 4, ["DCC217"]),
    ("EST773", "FUNDAMENTOS DE ESTATÍSTICA E CIÊNCIA DE DADOS", 60, 4, ["MAT001"]),
    # --- Período 5 ---
    ("ELE065", "ANALISE DE CIRCUITOS ELETRICOS II", 30, 5, ["ELE064"]),
    ("ELE028", "LABORATORIO DE CIRCUITOS ELETRICOS I", 30, 5, ["ELE064"]),
    ("ELT029", "LABORATORIO DE SISTEMAS DIGITAIS", 30, 5, ["ELT124"]),
    ("ELT084", "DISPOSITIVOS E CIRCUITOS ELETRONICOS BASICOS", 60, 5, ["ELE064"]),
    ("ELT123", "ARQUITETURA E ORGANIZAÇÃO DE COMPUTADORES", 60, 5, ["ELT124"]),
    ("ELE631", "ANÁLISE, PROJETO E PROGRAMAÇÃO ORIENTADOS A OBJETOS", 60, 5, ["DCC204"]),
    ("ESA019", "CIÊNCIAS DO AMBIENTE", 30, 5, []),
    # --- Período 6 ---
    ("ELT136", "FUNDAMENTOS DE SISTEMAS DINÂMICOS E CONTROLE", 60, 6, ["MAT015"]),
    ("ELT080", "LABORATORIO DE CIRCUITOS ELETRONICOS E PROJETOS", 30, 6, ["ELT084"]),
    ("EEE050", "MODELAGEM E SIMULAÇÃO MULTIFÍSICA", 60, 6, ["MAT015"]),
    ("ELE082", "PESQUISA OPERACIONAL", 60, 6, ["MAT038"]),
    ("ELE077", "OTIMIZACAO NAO LINEAR", 30, 6, ["MAT038"]),
    # --- Período 7 ---
    ("EEE046", "SISTEMAS A EVENTOS DISCRETOS", 60, 7, ["DCC217"]),
    ("EEE017", "CONFIABILIDADE DE SISTEMAS", 60, 7, []),
    ("ELE088", "TEORIA DA DECISAO", 30, 7, ["EST773"]),
    ("ELE632", "PROCESSOS E MÉTODOS EM ENGENHARIA DE SISTEMAS", 60, 7, ["ELE630"]),
    ("ELE633", "LABORATORIO DE SISTEMAS I", 60, 7, ["ELE632"]),
    # --- Período 8 ---
    ("EEE048", "FUNDAMENTOS DE INTELIGÊNCIA ARTIFICIAL", 60, 8, ["EST773"]),
    ("EEE049", "APRENDIZADO DE MÁQUINA", 60, 8, ["EST773"]),
    ("ELE634", "LABORATORIO DE SISTEMAS II", 60, 8, ["ELE633"]),
    ("EEE051", "LABORATORIO DE GERENCIAMENTO DE SISTEMAS", 60, 8, ["ELE632"]),
    # --- Período 9 ---
    ("ELE635", "LABORATORIO DE SISTEMAS III", 60, 9, ["ELE634"]),
    # --- Período 10 ---
    ("EEE018", "TRABALHO DE CONCLUSAO DE CURSO I", 90, 10, []),
    ("EEE052", "TRABALHO DE CONCLUSAO DE CURSO I", 90, 10, []),
    # --- Período 11 ---
    ("EEE019", "TRABALHO DE CONCLUSAO DE CURSO II", 90, 11, []),
    ("EEE053", "TRABALHO DE CONCLUSAO DE CURSO II", 90, 11, []),
    ("EEE020", "ESTAGIO SUPERVISIONADO", 165, 11, []),
    ("EEE054", "ESTAGIO SUPERVISIONADO", 165, 11, []),
]

# Índice rápido: código → dados do PPC
PPC_INDEX: dict[str, dict] = {}
for _cod, _nome, _ch, _per, _prereqs in PPC_GRADE:
    PPC_INDEX[_cod] = {
        "nome": _nome,
        "carga_horaria": _ch,
        "periodo_sugerido": _per,
        "pre_requisitos": _prereqs,
    }

# ---------------------------------------------------------------------------
# Ementas (descrições curtas em português)
# ---------------------------------------------------------------------------
# Ementas extraídas do site GEES UFMG (geesufmg.com/curso/grade-curricular)
EMENTAS: dict[str, str] = {
    # Período 1
    "DCC203": "Introdução ao funcionamento de um computador e ao desenvolvimento de programas; Desenvolvimento de programas em uma linguagem de alto nível; Tipos de dados simples, apontadores, variáveis compostas homogêneas e heterogêneas; Entrada e saída; Estruturas de controle e repetição; Funções e ferramentas de modularização.",
    "MAT001": "Funções de R em R; Derivadas; Integrais; Aplicações.",
    "MAT038": "Álgebra vetorial; Retas e planos; Matrizes, sistemas lineares e determinantes; Espaço vetorial Rn; Autovalores e autovetores de matrizes; Diagonalização de matrizes simétricas.",
    "QUI628": "Reações em solução aquosa e estequiometria; Termoquímica, cinética química, equilíbrio químico; Fundamentos de eletroquímica; Ligação iônica, ligação covalente, interações intermoleculares; Química aplicada a engenharia e geologia.",
    "ELE630": "Integração do estudante à UFMG, à Escola de Engenharia e ao curso de Engenharia de Sistemas; Normas acadêmicas; Normas de segurança em laboratórios; Prevenção e combate a incêndios e desastres; Contextualização da Engenharia e da Engenharia de Sistemas; Aplicações da Engenharia e da Engenharia de Sistemas.",
    # Período 2
    "DCC204": "Programação estruturada e linguagem de programação modular; Metodologias de desenvolvimento de software; Compreensão, corretude e depuração de programas; Resolução de problemas de forma modular e eficiente.",
    "FIS065": "Cinemática e dinâmica da partícula; Sistemas de partículas; Cinemática e dinâmica de rotação; Leis de Conservação da Energia e dos Momentos Linear e Angular.",
    "MAT039": "Coordenadas polares; Cônicas; Séries; Série e fórmula de Taylor; Diferenciabilidade de funções de várias variáveis.",
    "ELT124": "Sistemas de numeração; Álgebra Booleana; Portas lógicas; Circuitos combinacionais: análise, síntese e técnicas de minimização; Circuitos sequenciais síncronos e assíncronos; Análise, síntese e técnicas de minimização de circuitos sequenciais; Famílias de circuitos lógicos; Dispositivos lógicos programáveis; Gate arrays; Análise e projeto de sistemas digitais; Fundamentos de Linguagem de Descrição de Hardware e sua aplicação no projeto de Sistemas Digitais.",
    "DCC217": "Estudo de fundamentos de lógica; Técnicas de prova, indução matemática; Teoria de conjuntos, análise combinatória; Funções, recursão, relações em conjuntos e teoria dos grafos.",
    # Período 3
    "DCC205": "Análise de algoritmos; Abstração de dados; Introdução às técnicas de análise de algoritmos; Estruturas de dados estáticas e dinâmicas na memória principal e secundária; Estruturas de dados para realização eficiente de operações sobre dados.",
    "FIS069": "Carga elétrica, campo elétrico e a lei de Gauss; Potencial elétrico, capacitores e dielétricos; Corrente e resistência elétricas; Campo magnético e lei de Ampère; Lei de Faraday e indutância; Materiais magnéticos; Equações de Maxwell.",
    "EST773": "Introdução à Estatística e Ciência de Dados; Visualização de dados: tipos de variáveis, gráficos e tabelas, medidas de posição e variabilidade; Fundamentos de probabilidade e modelos probabilísticos; Tomadas de decisão com base em evidências: estimação pontual e intervalar, conceitos de testes de hipóteses; Prática Computacional.",
    "MAT040": "Equações Diferenciais Ordinárias de 1a e 2a Ordens; Soluções de Equações Diferenciais em Séries de Potências; Sistemas de Equações Diferenciais Lineares; Transformada de Laplace; Séries de Fourier; Equações Diferenciais Parciais.",
    "ELE064": "Fontes de tensão e corrente dependentes e independentes; Leis fundamentais de circuitos; Circuitos resistivos; Métodos de análise de circuitos; Teoremas de rede; Circuitos com amplificador operacional ideal.",
    "ELT029": "Aulas práticas envolvendo circuitos combinacionais, circuitos sequenciais síncronos e assíncronos e dispositivos lógicos programáveis; Gate Arrays; Análise e projeto de sistemas digitais.",
    # Período 4
    "ELE631": "Gerenciamento da Complexidade; Introdução a Análise e Projeto Orientados a Objetos; Engenharia de Requisitos; Modelagem Orientada a Objetos; Implementação dos Conceitos em uma Linguagem OO; Introdução aos Padrões de Projeto OO; Introdução à Modelagem Estrutural, Comportamental e de Arquitetura.",
    "FIS086": "Oscilações Mecânicas e Eletromagnéticas; Ondas Mecânicas. Som; Ondas Eletromagnéticas. Óptica.",
    "MAT002": "Integral dupla e tripla; Vetores aleatórios: distribuições marginais e condicionais; Momentos condicionais; Correlações parciais; Independência estocástica; Distribuições multivariadas; Transformação de variáveis aleatórias n-dimensionais; Função geratriz de momentos e Função característica.",
    "ELE065": "Indutância, capacitância e indutância mútua; Circuitos de primeira ordem (RC, RL e outros circuitos); Circuitos de segunda ordem (RLC série, RLC paralelo e outros circuitos); Excitação senoidal e fasores; Análise em Regime permanente senoidal; Potência em regime permanente senoidal.",
    "ELE632": "Visão holística do sistema socioeconômico atual e seus principais processos; Abordagens de ciclo de vida para sistemas; Processos de ciclo de vida para sistemas segundo normas e referências técnicas; Métodos relacionados a estes processos; Aplicação do conteúdo visto junto à comunidade externa.",
    "ESA019": "Meio Ambiente e Engenharia; Causas e efeitos da poluição hídrica, atmosférica e do solo; Processos e equipamentos usados na prevenção e no controle da poluição; Noções de Legislação Ambiental; Sistema de Gestão Ambiental e Certificação; Produção Mais Limpa (P+L); Engenharia e Sustentabilidade.",
    # Período 5
    "EEE048": "Inteligência artificial e agentes inteligentes; Arquiteturas de agentes inteligentes; Representação do conhecimento; Algoritmos e Heurísticas de busca; Algoritmo A*; Redes de restrições e satisfação de restrições; Proposições e inferência; Raciocínio com incertezas e Raciocínio baseado em casos; Planejamento com incertezas; Redes de decisão e Processos de decisão markovianos; Teoria de jogos; Aspectos sociais e éticos da inteligência artificial.",
    "ELT136": "Sistemas de controle fundamentais e tecnologia de sistemas de controle; Sensores, atuadores, modelagem física de sistemas; Técnicas operacionais para sistemas lineares em tempo contínuo; Transformadas de Laplace; Resposta via funções de transferência; Estabilidade; Especificações de desempenho; Projeto de controladores via funções de transferência; Resposta em frequência; Não linearidades simples.",
    "ELE082": "Programação linear e suas aplicações; Método simplex; Análise de sensibilidade e dualidade; Otimização em redes; Programação dinâmica; Otimização combinatória e heurísticas; Modelagem.",
    "ELE028": "Organização e Segurança em laboratórios; Medição de grandezas elétricas; Experimentos básicos com elementos de circuitos (resistivos, fontes dependentes, capacitores, indutores); Circuitos em regime transitório e em regime permanente senoidal; Circuitos com Amplificadores Operacionais.",
    "EMT122": "Ligações químicas, tipos de materiais e suas características básicas; Estrutura cristalina e amorfa; Defeitos cristalinos; Diagramas de fases; Propriedades Mecânicas; Materiais Cerâmicos e Poliméricos.",
    # Período 6
    "DCC011": "Memória auxiliar; organização física e lógica; Métodos de acesso; Estruturas de arquivos; Manipulação de bancos de dados; Linguagens e pacotes; Recuperação de informação.",
    "FIS152": "Temperatura e dilatação; Modelo cinético do gás ideal; Calor e a primeira lei da termodinâmica; Entropia e a segunda lei da termodinâmica; Estática e dinâmica de fluidos; Equação de Bernoulli.",
    "ELT084": "Introdução à Eletrônica; Circuitos Eletrônicos com Amplificadores Operacionais ideais e reais; Junção PN; Diodos Retificadores e Zener: característica, circuitos com diodos e aplicações; Transistores (FET e BJT): características, modelos, polarização, análises; Funcionamento dos transistores como chaves; Conversores A/D e D/A: conceitos básicos; Representação gráfica de circuitos eletrônicos e simulação.",
    "ELT123": "Arquitetura de microprocessadores: unidade de controle, memória, entrada e saída; Programação de microprocessadores: instruções, modos de endereçamento, Assembly e C; Dispositivos periféricos, interrupção, acesso direto à memória; Barramentos-padrão; Ferramentas para análise, desenvolvimento e depuração; Microprocessadores comerciais; Projetos de aplicações com microprocessadores e interfaces de E/S; Multiprocessamento.",
    "ELE077": "Formulação de problemas de otimização; Propriedades geométricas dos espaços de busca; Condições de otimalidade; Métodos determinísticos para otimização irrestrita; Métodos sem derivadas; Métodos para otimização restrita.",
    # Período 7
    "ELE633": "Ciclo de vida de sistemas: definição, requisitos, projeto, implementação, verificação, integração e validação; Engenharia Baseada em Modelos; SysML; Desenho universal e sua aplicação em projetos de engenharia; Desenvolvimento de um projeto de engenharia com a comunidade externa (extensão).",
    "EEE049": "Princípios Estatísticos do Aprendizado de Máquina; Análise Exploratória e Métodos de Visualização de Dados; Modelos de Regressão Linear e Logística; Métodos de Correlação e Relações Causais; Partição de Dados e Clustering; Métodos de amostragem; Classificadores e Regressores baseados em Árvores.",
    "EEE050": "Modelagem de Sistemas Físicos usando ODEs e PDEs; Exemplos de aplicações: Transferência de Calor, Eletromagnetismo, Análise Acústica, Mecânica; Introdução a métodos numéricos para a simulação de ODEs e PDEs (Diferenças finitas e Elementos Finitos).",
    "ELT080": "Utilização dos equipamentos e instrumentos de Laboratório; Análise e Projetos com Amplificadores Operacionais Reais; Análise e Projeto de circuitos com Diodos retificadores e Zener; Análise e Projeto de circuitos com Transistores bipolares e MOSFETs.",
    "ELE088": "Otimização multiobjetivo e conjuntos de Pareto; Modelagem de preferências; Modelagem do risco e decisão sob incerteza; Jogos e decisão minimax; Decisão bayesiana; Sistemas de suporte à decisão.",
    "EEE046": "Fundamentos de Sistemas a Eventos Discretos; Modelagem e análise utilizando autômatos e redes de Petri; Síntese e implementação de controladores; Abordagem temporizada.",
    # Período 8
    "DCC218": "Arquiteturas de redes de computadores; Aplicações e infraestrutura de redes; Conceitos de segurança; Processos, threads, sincronização, gerenciamento de memória, sistemas de arquivos; Tempo, nomes, replicação e consistência em Sistemas Distribuídos.",
    "ELE634": "Modelos caixa-preta; Otimização e problemas inversos; Síntese de sistemas por otimização; Desenvolvimento de um projeto de engenharia com a comunidade externa (extensão).",
    "EEE017": "Conceitos Básicos de Estatística e Probabilidade; Engenharia de Confiabilidade: Confiabilidade, Manutenabilidade e Disponibilidade; Engenharia de Manutenção; FMEA e FTA; Manutenção Centrada na Confiabilidade; Estudo de caso de um processo industrial / comercial.",
    # Período 9
    "EEE051": "Ferramentas e técnicas para o gerenciamento de sistemas; Processos de gestão: projetos, operações, aquisição, qualidade, riscos, etc.; Engenharia concorrente; Análise de questões étnico-raciais e de direitos humanos; Aplicação do conteúdo visto junto à comunidade externa.",
    "ELE635": "Desenvolvimento de um projeto de engenharia envolvendo sistemas discretos com a comunidade externa (extensão).",
    # Período 10
    "EEE052": "Metodologias científico-tecnológicas e ferramentas de apoio; Ferramentas de apoio à escrita científica e à pesquisa e gerenciamento bibliográfico; Elaboração das etapas iniciais de um projeto completo de engenharia; Concepção, avaliação de alternativas, seleção e especificação da solução; Elaboração de documentação preliminar com discussão sobre humanidades e aspectos técnicos.",
    # --- Disciplinas com códigos duplicados (TCC, Estágio) ---
    "EEE018": "Metodologias científico-tecnológicas e ferramentas de apoio; Ferramentas de apoio à escrita científica e à pesquisa e gerenciamento bibliográfico; Elaboração das etapas iniciais de um projeto completo de engenharia; Concepção, avaliação de alternativas, seleção e especificação da solução; Elaboração de documentação preliminar com discussão sobre humanidades e aspectos técnicos.",
    "EEE019": "Redação da monografia final; Análise dos resultados obtidos; Preparação e realização da defesa oral do Trabalho de Conclusão de Curso perante banca examinadora.",
    "EEE053": "Redação da monografia final; Análise dos resultados obtidos; Preparação e realização da defesa oral do Trabalho de Conclusão de Curso perante banca examinadora.",
    "EEE020": "Estágio supervisionado em empresa ou instituição pública ou privada, com supervisão docente e elaboração de relatório técnico final.",
    "EEE054": "Estágio supervisionado em empresa ou instituição pública ou privada, com supervisão docente e elaboração de relatório técnico final.",
    # --- Disciplinas do PPC com código alternativo (EDO) ---
    "MAT015": "Equações Diferenciais Ordinárias de 1a e 2a Ordens; Soluções de Equações Diferenciais em Séries de Potências; Sistemas de Equações Diferenciais Lineares; Transformada de Laplace; Séries de Fourier; Equações Diferenciais Parciais.",
    "MAT016": "Equações Diferenciais Ordinárias de 1a e 2a Ordens; Soluções de Equações Diferenciais em Séries de Potências; Sistemas de Equações Diferenciais Lineares; Transformada de Laplace; Séries de Fourier; Equações Diferenciais Parciais.",
}

# ---------------------------------------------------------------------------
# Classificação por área de conhecimento (prefixo do código)
# ---------------------------------------------------------------------------
AREA_MAP: list[tuple[list[str], str]] = [
    (["DCC"], "Computação"),
    (["MAT"], "Matemática"),
    (["FIS"], "Física"),
    (["QUI"], "Química"),
    (["ELE", "EEE", "ELT"], "Engenharia Elétrica e Sistemas"),
    (["EMT", "EMC"], "Materiais"),
    (["EST"], "Estatística"),
    (["ESA"], "Meio Ambiente"),
    (["DCP", "DIT"], "Ciências Sociais e Direito"),
    (["ECN"], "Economia"),
    (["EMA"], "Engenharia Mecânica/Aeroespacial"),
    (["EPD"], "Engenharia de Produção"),
    (["ENU"], "Engenharia Nuclear"),
    (["ENG"], "Engenharia (Tópicos Especiais)"),
]


def classificar_area(codigo: str) -> str:
    """Retorna a área de conhecimento com base no prefixo do código."""
    prefixo = re.match(r"[A-Z]+", codigo)
    if not prefixo:
        return "Outros"
    pref = prefixo.group()
    for prefixos, area in AREA_MAP:
        if pref in prefixos:
            return area
    return "Outros"


def classificar_ciclo(periodo: int | None) -> str:
    """Retorna o ciclo curricular com base no período sugerido."""
    if periodo is None:
        return "Optativa"
    if periodo <= 2:
        return "Básico Ciências"
    if periodo <= 4:
        return "Básico Engenharia"
    if periodo <= 6:
        return "Profissional Engenharia"
    if periodo <= 9:
        return "Profissional Engenharia de Sistemas"
    return "Integrador"


def gerar_ementa_generica(nome: str) -> str:
    """Gera uma ementa genérica para disciplinas sem ementa cadastrada."""
    nome_limpo = re.sub(r"\s+", " ", nome).strip()
    nome_lower = nome_limpo.lower()
    if "tópicos" in nome_lower or "topicos" in nome_lower:
        return f"Estudo de tópicos avançados em {nome_lower}."
    if "laboratório" in nome_lower or "laboratorio" in nome_lower:
        return f"Atividades práticas de laboratório relacionadas a {nome_lower}."
    if "seminário" in nome_lower or "seminario" in nome_lower:
        return f"Apresentação e discussão de temas atuais em {nome_lower}."
    if "projeto" in nome_lower:
        return f"Desenvolvimento de projeto prático em {nome_lower}."
    return f"Estudo dos fundamentos e aplicações de {nome_lower}."


# ---------------------------------------------------------------------------
# Fuzzy matching para cruzar nomes com a planilha de dificuldade
# ---------------------------------------------------------------------------
def normalizar_nome(nome: str) -> str:
    """Normaliza nome para comparação fuzzy."""
    texto = nome.upper().strip()
    # Remove acentos comuns
    for de, para in [("Á", "A"), ("É", "E"), ("Í", "I"), ("Ó", "O"), ("Ú", "U"),
                     ("Â", "A"), ("Ê", "E"), ("Ô", "O"), ("Ã", "A"), ("Õ", "O"),
                     ("Ç", "C"), ("À", "A")]:
        texto = texto.replace(de, para)
    texto = re.sub(r"[^A-Z0-9 ]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def buscar_dificuldade(nome_disc: str, df_dif: pd.DataFrame) -> dict[str, float | None]:
    """Busca valores de dificuldade por nome (fuzzy matching)."""
    resultado = {"dificuldade": None, "trabalhosa": None, "dificuldade_geral": None}

    if df_dif.empty:
        return resultado

    # Identifica colunas de nome e métricas
    col_nome = None
    for c in df_dif.columns:
        if "disciplina" in c.lower() or "nome" in c.lower() or "materia" in c.lower():
            col_nome = c
            break
    if col_nome is None:
        col_nome = df_dif.columns[0]

    # Mapeia colunas de métricas
    col_map = {}
    for c in df_dif.columns:
        cl = c.lower()
        if "geral" in cl:
            col_map["dificuldade_geral"] = c
        elif "trabalhosa" in cl or "trabalho" in cl:
            col_map["trabalhosa"] = c
        elif "dificuldade" in cl or "dificil" in cl:
            col_map["dificuldade"] = c

    nome_norm = normalizar_nome(nome_disc)
    melhor_ratio = 0.0
    melhor_idx = None

    for idx, row in df_dif.iterrows():
        nome_plan = normalizar_nome(str(row[col_nome]))
        ratio = SequenceMatcher(None, nome_norm, nome_plan).ratio()
        if ratio > melhor_ratio:
            melhor_ratio = ratio
            melhor_idx = idx

    # Aceita match com similaridade >= 0.7
    if melhor_ratio >= 0.7 and melhor_idx is not None:
        row = df_dif.loc[melhor_idx]
        for chave, coluna in col_map.items():
            val = row.get(coluna)
            if pd.notna(val):
                # Converte vírgula decimal para ponto
                if isinstance(val, str):
                    val = val.replace(",", ".")
                try:
                    resultado[chave] = round(float(val), 2)
                except (ValueError, TypeError):
                    pass

    return resultado


# ---------------------------------------------------------------------------
# Extração de disciplinas do PDF
# ---------------------------------------------------------------------------
def extrair_disciplinas_pdf(pdf_path: Path) -> list[dict]:
    """Extrai disciplinas de Engenharia de Sistemas do PDF de oferta."""
    disciplinas = []
    codigos_vistos = set()

    with pdfplumber.open(pdf_path) as pdf:
        all_rows = []
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    if row and len(row) >= 4:
                        all_rows.append(row)

    for row in all_rows:
        # Pula cabeçalhos
        if row[0] and "Código" in str(row[0]):
            continue

        codigo = str(row[0] or "").strip()
        nome = re.sub(r"\s+", " ", str(row[1] or "")).strip()
        ch_raw = str(row[2] or "").strip()
        cursos = str(row[3] or "").strip()

        # Filtra por Engenharia de Sistemas
        if "ENGENHARIA DE SISTEMAS" not in cursos.upper():
            continue

        # Valida código
        if not re.match(r"[A-Z]{2,4}\d{3,4}", codigo):
            continue

        # Evita duplicatas
        if codigo in codigos_vistos:
            continue
        codigos_vistos.add(codigo)

        # Carga horária
        try:
            ch = int(re.sub(r"[^\d]", "", ch_raw))
        except ValueError:
            ch = 0

        # Natureza/tipo (coluna 6, se existir)
        natureza = "OB"
        if len(row) > 6 and row[6]:
            tipo_raw = str(row[6]).strip().upper()
            if tipo_raw in ("OB", "OP", "OE", "AC", "EL"):
                natureza = tipo_raw

        # Horário e professor (últimas colunas, se existirem)
        horario = str(row[8]).strip() if len(row) > 8 and row[8] else None
        professor = str(row[9]).strip() if len(row) > 9 and row[9] else None

        disciplinas.append({
            "codigo": codigo,
            "nome": nome,
            "carga_horaria": ch,
            "natureza": natureza,
            "horario": horario,
            "professor": professor,
        })

    return disciplinas


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def load_schedule(csv_path: Path) -> dict[str, list[dict]]:
    """Carrega o CSV de horários e retorna dicionário codigo -> lista de slots."""
    if not csv_path.exists():
        print(f"  ⚠ CSV de horários não encontrado: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    schedule: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        codigo = str(row.get("Codigo", "")).strip()
        dia = str(row.get("Dia", "")).strip()
        turno = str(row.get("Turno", "")).strip()
        if not codigo or codigo in ("LIVRE",) or codigo.startswith("OPT"):
            continue
        if codigo.startswith("NÚCLEO"):
            continue
        # Resolve placeholder → código real
        codigo = PLACEHOLDER_MAP.get(codigo, codigo)
        slot = {"dia": dia, "turno": turno}
        if codigo not in schedule:
            schedule[codigo] = []
        # Evita duplicatas
        if slot not in schedule[codigo]:
            schedule[codigo].append(slot)
    return schedule

def main(pdf_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Baixar planilha de dificuldade ---
    print("[1/5] Baixando planilha de dificuldade...")
    try:
        df_dif = pd.read_csv(URL_DIFICULDADE)
        df_dif.to_csv(CSV_DIFICULDADE, index=False)
        print(f"  ✓ Salvo em {CSV_DIFICULDADE} ({len(df_dif)} linhas)")
    except Exception as e:
        print(f"  ⚠ Erro ao baixar planilha: {e}")
        print("  → Tentando carregar cópia local...")
        if CSV_DIFICULDADE.exists():
            df_dif = pd.read_csv(CSV_DIFICULDADE)
        else:
            df_dif = pd.DataFrame()

    # --- 2. Extrair disciplinas do PDF ---
    print(f"\n[2/5] Extraindo disciplinas do PDF: {pdf_path}")
    if not pdf_path.exists():
        print(f"  ✗ PDF não encontrado: {pdf_path}")
        sys.exit(1)

    disciplinas_pdf = extrair_disciplinas_pdf(pdf_path)
    print(f"  ✓ {len(disciplinas_pdf)} disciplinas extraídas do PDF")

    # --- 3. Carregar horários ---
    print(f"\n[3/5] Carregando horários da grade: {SCHEDULE_CSV}")
    schedule = load_schedule(SCHEDULE_CSV)
    codes_with_schedule = sum(1 for slots in schedule.values() if slots)
    print(f"  ✓ {len(schedule)} códigos com horários ({codes_with_schedule} com slots)")

    # --- 4. Enriquecer com dados do PPC e dificuldade ---
    print("\n[4/5] Enriquecendo com dados do PPC, ementas, dificuldade e horários...")
    resultado_final = []

    for disc in disciplinas_pdf:
        codigo = disc["codigo"]

        # Dados do PPC
        ppc = PPC_INDEX.get(codigo, {})
        periodo = ppc.get("periodo_sugerido")
        prereqs = ppc.get("pre_requisitos", [])

        # Área de conhecimento e ciclo
        area = classificar_area(codigo)
        ciclo = classificar_ciclo(periodo)

        # Ementa
        ementa = EMENTAS.get(codigo, gerar_ementa_generica(disc["nome"]))

        # Dificuldade (fuzzy match pelo nome)
        dif_vals = buscar_dificuldade(disc["nome"], df_dif)

        registro = {
            "codigo": codigo,
            "nome": disc["nome"],
            "carga_horaria": disc["carga_horaria"],
            "natureza": disc["natureza"],
            "periodo_sugerido": periodo,
            "pre_requisitos": prereqs,
            "area_conhecimento": area,
            "ciclo": ciclo,
            "ementa": ementa,
            "dificuldade": dif_vals["dificuldade"],
            "trabalhosa": dif_vals["trabalhosa"],
            "dificuldade_geral": dif_vals["dificuldade_geral"],
            "horarios": schedule.get(codigo, []),
        }
        resultado_final.append(registro)

    # Ordena: obrigatórias primeiro (por período), depois optativas (por código)
    resultado_final.sort(key=lambda d: (
        d["periodo_sugerido"] if d["periodo_sugerido"] is not None else 999,
        d["codigo"],
    ))

    # --- 5. Salvar JSON ---
    print(f"\n[5/5] Salvando JSON em {JSON_DISCIPLINAS}...")
    with open(JSON_DISCIPLINAS, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, ensure_ascii=False, indent=2)

    # Estatísticas finais
    obrigatorias = [d for d in resultado_final if d["periodo_sugerido"] is not None]
    optativas = [d for d in resultado_final if d["periodo_sugerido"] is None]
    com_dif = [d for d in resultado_final if d["dificuldade"] is not None]

    print(f"\n{'='*60}")
    print(f"  Total de disciplinas:      {len(resultado_final)}")
    print(f"  Obrigatórias (com período): {len(obrigatorias)}")
    print(f"  Optativas (sem período):    {len(optativas)}")
    print(f"  Com dados de dificuldade:   {len(com_dif)}")
    print(f"  Com dados de horários:      {sum(1 for d in resultado_final if d['horarios'])}")
    print(f"{'='*60}")

    # Remove .gitkeep antigos nos subdiretórios de data
    for subdir in ("raw", "processed", "synthetic_pairs"):
        gitkeep = DATA_DIR / subdir / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
            print(f"  🗑 Removido {gitkeep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrói dados curados de disciplinas")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF,
                        help="Caminho do PDF de oferta")
    args = parser.parse_args()
    main(args.pdf)
