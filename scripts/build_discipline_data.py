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
EMENTAS: dict[str, str] = {
    "MAT001": "Funções reais de uma variável, limites, continuidade, derivadas e integrais. Aplicações do cálculo diferencial e integral.",
    "MAT038": "Vetores, matrizes, sistemas lineares, autovalores e autovetores. Transformações lineares e espaços vetoriais.",
    "QUI628": "Estrutura atômica, ligações químicas, termodinâmica e equilíbrio químico. Fundamentos para engenharia.",
    "ELE630": "Visão geral da engenharia de sistemas: conceitos, aplicações e metodologias. Introdução à abordagem sistêmica.",
    "DCC217": "Lógica proposicional, teoria de conjuntos, relações, grafos e combinatória. Base matemática para computação.",
    "MAT039": "Técnicas de integração, integrais impróprias, sequências e séries numéricas. Séries de potências e Taylor.",
    "FIS065": "Cinemática e dinâmica da partícula, leis de Newton, trabalho, energia, momento e rotação de corpos rígidos.",
    "DCC203": "Introdução à programação: variáveis, estruturas de controle, funções e tipos de dados. Prática em linguagem de alto nível.",
    "EMT122": "Estrutura cristalina, propriedades mecânicas, diagrama de fases e materiais de engenharia.",
    "MAT002": "Funções de várias variáveis, derivadas parciais, integrais duplas e triplas. Curvas e superfícies no espaço.",
    "FIS069": "Campo elétrico, potencial, corrente, campo magnético, indução eletromagnética e equações de Maxwell.",
    "DCC204": "Modularização, recursão, ponteiros, alocação dinâmica e manipulação de arquivos. Projetos de média complexidade.",
    "DCC205": "Listas, pilhas, filas, árvores, grafos e tabelas hash. Análise de complexidade de algoritmos.",
    "DCC218": "Representação de dados, organização de computadores, linguagem de montagem e interface hardware-software.",
    "MAT015": "Equações diferenciais ordinárias de primeira e segunda ordem, transformada de Laplace e sistemas lineares.",
    "FIS086": "Oscilações harmônicas, ondas mecânicas e eletromagnéticas, óptica geométrica e física.",
    "FIS152": "Estática e dinâmica dos fluidos, termodinâmica: leis, ciclos e entropia.",
    "ELE064": "Análise de circuitos resistivos, leis de Kirchhoff, teoremas de circuitos e métodos de análise nodal/malhas.",
    "ELT124": "Álgebra booleana, portas lógicas, circuitos combinacionais e sequenciais. Máquinas de estados finitos.",
    "EST773": "Probabilidade, distribuições, inferência estatística, regressão e introdução à ciência de dados.",
    "ELE065": "Circuitos em regime transitório e permanente senoidal, fasores, potência e frequência complexa.",
    "ELE028": "Montagem e medição de circuitos elétricos em laboratório. Uso de instrumentos de medida.",
    "ELT029": "Projeto e implementação de circuitos digitais em laboratório. Uso de FPGA e ferramentas de simulação.",
    "ELT084": "Diodos, transistores BJT e MOSFET, amplificadores e circuitos de polarização.",
    "ELT123": "Processadores, hierarquia de memória, pipeline, entrada/saída e organização de sistemas computacionais.",
    "ELE631": "Paradigma orientado a objetos: classes, herança, polimorfismo, padrões de projeto e UML.",
    "ESA019": "Ecologia, poluição, recursos naturais e desenvolvimento sustentável. Responsabilidade ambiental na engenharia.",
    "ELT136": "Modelagem de sistemas dinâmicos, espaço de estados, estabilidade, resposta em frequência e controle PID.",
    "ELT080": "Projeto e montagem de circuitos eletrônicos analógicos e digitais integrados.",
    "EEE050": "Modelagem computacional de sistemas multifísicos, simulação numérica e validação de modelos.",
    "ELE082": "Programação linear, simplex, problemas de transporte, designação e programação inteira.",
    "ELE077": "Métodos de otimização sem restrições e com restrições: gradiente, Newton e multiplicadores de Lagrange.",
    "EEE046": "Autômatos, redes de Petri, linguagens formais e modelagem de sistemas a eventos discretos.",
    "EEE017": "Análise de confiabilidade, taxa de falhas, disponibilidade, manutenção e árvore de falhas.",
    "ELE088": "Teoria da utilidade, critérios de decisão sob incerteza e análise multicritério.",
    "ELE632": "Ciclo de vida de sistemas, requisitos, arquitetura de sistemas, verificação e validação.",
    "ELE633": "Projeto prático integrador de engenharia de sistemas: definição de requisitos e arquitetura.",
    "EEE048": "Busca, representação do conhecimento, lógica, planejamento e fundamentos de IA simbólica e conexionista.",
    "EEE049": "Aprendizado supervisionado, não supervisionado e por reforço. Redes neurais e avaliação de modelos.",
    "ELE634": "Projeto integrador: implementação, testes e integração de subsistemas.",
    "EEE051": "Gestão de projetos de engenharia, indicadores de desempenho, qualidade e gerenciamento de riscos.",
    "ELE635": "Projeto integrador final: validação, documentação e apresentação de sistema completo.",
    "EEE018": "Elaboração do projeto de TCC: revisão bibliográfica, metodologia e desenvolvimento inicial.",
    "EEE052": "Elaboração do projeto de TCC: revisão bibliográfica, metodologia e desenvolvimento inicial.",
    "EEE019": "Conclusão do TCC: resultados, análise, redação final e defesa do trabalho.",
    "EEE053": "Conclusão do TCC: resultados, análise, redação final e defesa do trabalho.",
    "EEE020": "Estágio em empresa ou instituição, com acompanhamento docente e relatório final.",
    "EEE054": "Estágio em empresa ou instituição, com acompanhamento docente e relatório final.",
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
    nome_lower = nome.lower()
    # Tenta inferir algo razoável pelo nome
    if "tópicos" in nome_lower or "topicos" in nome_lower:
        return f"Estudo de tópicos avançados em {nome.title().split(' ')[-1].lower()}."
    if "laboratório" in nome_lower or "laboratorio" in nome_lower:
        return f"Atividades práticas de laboratório relacionadas a {nome.title().lower()}."
    if "seminário" in nome_lower or "seminario" in nome_lower:
        return f"Apresentação e discussão de temas atuais em {nome.title().lower()}."
    if "projeto" in nome_lower:
        return f"Desenvolvimento de projeto prático em {nome.title().lower()}."
    return f"Estudo dos fundamentos e aplicações de {nome.title().lower()}."


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
        nome = str(row[1] or "").strip()
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
def main(pdf_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Baixar planilha de dificuldade ---
    print("[1/4] Baixando planilha de dificuldade...")
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
    print(f"\n[2/4] Extraindo disciplinas do PDF: {pdf_path}")
    if not pdf_path.exists():
        print(f"  ✗ PDF não encontrado: {pdf_path}")
        sys.exit(1)

    disciplinas_pdf = extrair_disciplinas_pdf(pdf_path)
    print(f"  ✓ {len(disciplinas_pdf)} disciplinas extraídas do PDF")

    # --- 3. Enriquecer com dados do PPC e dificuldade ---
    print("\n[3/4] Enriquecendo com dados do PPC, ementas e dificuldade...")
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
        }
        resultado_final.append(registro)

    # Ordena: obrigatórias primeiro (por período), depois optativas (por código)
    resultado_final.sort(key=lambda d: (
        d["periodo_sugerido"] if d["periodo_sugerido"] is not None else 999,
        d["codigo"],
    ))

    # --- 4. Salvar JSON ---
    print(f"\n[4/4] Salvando JSON em {JSON_DISCIPLINAS}...")
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
