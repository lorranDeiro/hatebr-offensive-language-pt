# %% [markdown]
# # 1. Configuração Inicial e Imports
# 
# Nesta célula, são realizadas as configurações iniciais do projeto, incluindo:
# - Importação das bibliotecas necessárias para análise de dados, processamento de linguagem natural (PLN) e visualização.
# - Configuração de avisos (`warnings`) para evitar mensagens desnecessárias durante a execução.
# - Download de recursos do NLTK, como stopwords e modelos de tokenização, que são essenciais para o pré-processamento do texto.
# 
# **Por que está estruturado assim?**
# - A consolidação dos imports e downloads em uma única célula facilita a organização e a reutilização do código.
# - O uso de `try-except` para os downloads garante que o código não falhe caso os recursos já estejam disponíveis.

# %%
# =============================================================================
# 1. CONFIGURAÇÃO INICIAL E IMPORTS
# =============================================================================

import re
import string
import warnings
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk import FreqDist, bigrams, trigrams, pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# Downloads necessários do NLTK (executar apenas uma vez)
recursos_nltk = [
    'floresta', 'mac_morpho', 'punkt', 'stopwords',
    'punkt_tab', 'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]
for recurso in recursos_nltk:
    try:
        nltk.download(recurso, quiet=True)
    except:
        pass


# %% [markdown]
# # 2. Carregamento e Análise Inicial dos Dados
# 
# Esta célula contém a função `carregar_e_analisar_corpus`, que realiza as seguintes tarefas:
# 1. Carrega o arquivo CSV contendo os dados do corpus.
# 2. Exibe estatísticas básicas, como o número total de comentários e a distribuição das classes (ofensivo e não-ofensivo).
# 
# **Por que está estruturado assim?**
# - A análise inicial ajuda a entender a estrutura e o equilíbrio das classes no conjunto de dados.
# - A exibição de estatísticas básicas é útil para verificar se os dados foram carregados corretamente e identificar possíveis problemas, como classes desbalanceadas.

# %%
# =============================================================================
# 2. CARREGAMENTO E ANÁLISE INICIAL DOS DADOS
# =============================================================================

def carregar_e_analisar_corpus(caminho_csv):
    """Carrega o corpus e exibe estatísticas básicas."""
    df = pd.read_csv(caminho_csv)

    print("=" * 70)
    print("ANÁLISE INICIAL DO CORPUS HateBR")
    print("=" * 70)
    print(f"\nTotal de comentários: {len(df):,}")
    print(f"Colunas: {df.columns.tolist()}")

    print("\nDISTRIBUIÇÃO DE CLASSES:")
    dist = df['label_final'].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  Classe {label}: {count:,} ({pct:.1f}%)")

    return df

# %% [markdown]
# # 3. Pré-Processamento
# 
# A função `preprocessar_corpus` realiza o pré-processamento dos textos, que é uma etapa essencial para a análise de linguagem natural. As principais tarefas incluem:
# 1. Tokenização: Divide os textos em palavras ou tokens.
# 2. Limpeza: Remove stopwords, pontuações, números e tokens indesejados.
# 3. Estatísticas: Calcula o número médio de tokens por comentário e o número de tokens únicos.
# 
# **Por que está estruturado assim?**
# - A limpeza dos dados reduz o ruído e melhora a qualidade das análises subsequentes.
# - A tokenização é necessária para aplicar técnicas de PLN, como análise de frequência e n-gramas.
# - As estatísticas ajudam a entender a complexidade do corpus e verificar a eficácia do pré-processamento.

# %%
# =============================================================================
# 3. PRÉ-PROCESSAMENTO
# =============================================================================

def preprocessar_corpus(df):
    """Aplica tokenização e limpeza no corpus."""
    print("\n" + "=" * 70)
    print("PRÉ-PROCESSAMENTO")
    print("=" * 70)

    # Configuração de stopwords e pontuação
    stopwords_pt = set(stopwords.words('portuguese'))
    pontuacao = set(string.punctuation)
    tokens_indesejados = {'>', '<', '..', '...', '....', '``', "''", '@'}

    def limpar_tokens(tokens):
        """Remove stopwords, pontuação e tokens indesejados."""
        return [
            t for t in tokens
            if t not in stopwords_pt
            and t not in pontuacao
            and t not in tokens_indesejados
            and len(t) > 1
            and not t.isdigit()
        ]

    # Tokenização e limpeza
    df['tokens'] = df['comentario'].apply(word_tokenize)
    df['cleaned_tokens'] = df['tokens'].apply(limpar_tokens)
    df['num_tokens'] = df['cleaned_tokens'].apply(len)

    # Estatísticas
    tokens_unicos = len(set([t for tokens in df['cleaned_tokens'] for t in tokens]))
    media_tokens = df['num_tokens'].mean()

    print(f"\nTokens únicos: {tokens_unicos:,}")
    print(f"Média de tokens por comentário: {media_tokens:.2f}")

    return df

# %% [markdown]
# # 4. Análise Léxica e N-Gramas
# 
# A função `analisar_frequencias` realiza a análise léxica do corpus, incluindo:
# 1. Cálculo das frequências de palavras (unigramas) para cada classe (ofensivo e não-ofensivo).
# 2. Identificação dos bigramas mais frequentes (pares de palavras consecutivas).
# 
# **Por que está estruturado assim?**
# - A análise de frequências ajuda a identificar padrões linguísticos característicos de cada classe.
# - A separação por classe permite entender as diferenças entre os textos ofensivos e não-ofensivos.
# - A análise de bigramas captura expressões compostas que podem ser relevantes para a classificação.

# %%
# =============================================================================
# 4. ANÁLISE LÉXICA E N-GRAMAS
# =============================================================================

def analisar_frequencias(df):
    """Analisa frequências de tokens e n-gramas por classe."""
    print("\n" + "=" * 70)
    print("ANÁLISE LÉXICA POR CLASSE")
    print("=" * 70)

    # Separar tokens por classe
    tokens_nao_of = [t for tokens in df[df['label_final'] == 0]['cleaned_tokens'] for t in tokens]
    tokens_of = [t for tokens in df[df['label_final'] == 1]['cleaned_tokens'] for t in tokens]

    # Frequências de unigramas
    fdist_nao_of = FreqDist(tokens_nao_of)
    fdist_of = FreqDist(tokens_of)

    print("\nPalavras mais frequentes - NÃO OFENSIVO (Top 15):")
    for palavra, freq in fdist_nao_of.most_common(15):
        print(f"  {palavra}: {freq}")

    print("\nPalavras mais frequentes - OFENSIVO (Top 15):")
    for palavra, freq in fdist_of.most_common(15):
        print(f"  {palavra}: {freq}")

    # Bigramas
    print("\n" + "-" * 70)
    print("ANÁLISE DE BIGRAMAS")
    print("-" * 70)

    bigramas_nao_of = [bg for tokens in df[df['label_final'] == 0]['cleaned_tokens']
                    for bg in bigrams(tokens)]
    bigramas_of = [bg for tokens in df[df['label_final'] == 1]['cleaned_tokens']
                for bg in bigrams(tokens)]

    fdist_bg_nao_of = FreqDist(bigramas_nao_of)
    fdist_bg_of = FreqDist(bigramas_of)

    print("\nBigramas - NÃO OFENSIVO (Top 10):")
    for bigrama, freq in fdist_bg_nao_of.most_common(10):
        print(f"  {' '.join(bigrama)}: {freq}")

    print("\nBigramas - OFENSIVO (Top 10):")
    for bigrama, freq in fdist_bg_of.most_common(10):
        print(f"  {' '.join(bigrama)}: {freq}")

    # Trigramas
    print("\n" + "-" * 70)
    print("ANÁLISE DE TRIGRAMAS")
    print("-" * 70)

    trigramas_nao_of = [tg for tokens in df[df['label_final'] == 0]['cleaned_tokens']
                        for tg in trigrams(tokens)]
    trigramas_of = [tg for tokens in df[df['label_final'] == 1]['cleaned_tokens']
                    for tg in trigrams(tokens)]

    fdist_tg_nao_of = FreqDist(trigramas_nao_of)
    fdist_tg_of = FreqDist(trigramas_of)

    print("\nTrigramas - NÃO OFENSIVO (Top 10):")
    for trigrama, freq in fdist_tg_nao_of.most_common(10):
        print(f"  {' '.join(trigrama)}: {freq}")

    print("\nTrigramas - OFENSIVO (Top 10):")
    for trigrama, freq in fdist_tg_of.most_common(10):
        print(f"  {' '.join(trigrama)}: {freq}")


    return {
        'fdist_nao_of': fdist_nao_of,
        'fdist_of': fdist_of,
        'fdist_bg_nao_of': fdist_bg_nao_of,
        'fdist_bg_of': fdist_bg_of
    }


# %%
# =============================================================================
# 4.1. ANÁLISE DE CLASSES GRAMATICAIS (POS TAG)
# =============================================================================
from nltk import pos_tag

def analisar_pos_tagging(df):
    """Analisa classes gramaticais e seu papel na detecção de ofensas."""
    print("\n" + "="*70)
    print("ANÁLISE DE CLASSES GRAMATICAIS (POS TAGGING)")
    print("="*70)

    # POS tagging por classe
    pos_of = []
    pos_nao_of = []

    # Iterar sobre as linhas dos DataFrames filtrados para acessar 'tokens'
    # Isso é mais seguro do que o slicing direto em alguns casos de visualizações/cópias
    for index, row in df[df['label_final'] == 1].head(1000).iterrows():  # Amostra
        pos_of.extend([tag for word, tag in pos_tag(row['tokens'])])

    for index, row in df[df['label_final'] == 0].head(1000).iterrows():
        pos_nao_of.extend([tag for word, tag in pos_tag(row['tokens'])])

    fdist_pos_of = FreqDist(pos_of)
    fdist_pos_nao_of = FreqDist(pos_nao_of)

    print("\nClasses gramaticais mais frequentes - OFENSIVO:")
    for tag, freq in fdist_pos_of.most_common(10):
        print(f"  {tag}: {freq}")

    print("\nClasses gramaticais mais frequentes - NÃO-OFENSIVO:")
    for tag, freq in fdist_pos_nao_of.most_common(10):
        print(f"  {tag}: {freq}")

    # Análise: substantivos vs adjetivos vs verbos
    print("\n📊 ANÁLISE COMPARATIVA:")
    for pos_type, descricao in [('NN', 'Substantivos'), ('JJ', 'Adjetivos'), ('VB', 'Verbos')]:
        # Garante que as chaves existem antes de somar
        freq_of = sum(freq for tag, freq in fdist_pos_of.items() if tag.startswith(pos_type)) if fdist_pos_of else 0
        freq_nao_of = sum(freq for tag, freq in fdist_pos_nao_of.items() if tag.startswith(pos_type)) if fdist_pos_nao_of else 0

        ratio = freq_of / freq_nao_of if freq_nao_of > 0 else (freq_of if freq_of > 0 else 0) # Handle division by zero and cases where one is zero
        print(f"  {descricao}: Ratio Ofensivo/Não-Ofensivo = {ratio:.2f}")

# %%
# =============================================================================
# 4.2. ANÁLISE DE ENTIDADES NOMEADAS (NER)
# =============================================================================

import spacy

def analisar_entidades(df):
    """Analisa entidades nomeadas e seu papel na classificação."""
    print("\n" + "="*70)
    print("ANÁLISE DE ENTIDADES NOMEADAS (NER)")
    print("="*70)

    nlp = spacy.load('pt_core_news_sm')  # Modelo português

    entidades_of = defaultdict(list)
    entidades_nao_of = defaultdict(list)

    # Amostra para análise (NER é computacionalmente caro)
    amostra_of = df[df['label_final'] == 1]['comentario'].sample(min(500, len(df)), random_state=42)
    amostra_nao_of = df[df['label_final'] == 0]['comentario'].sample(min(500, len(df)), random_state=42)

    for texto in amostra_of:
        doc = nlp(texto)
        for ent in doc.ents:
            entidades_of[ent.label_].append(ent.text)

    for texto in amostra_nao_of:
        doc = nlp(texto)
        for ent in doc.ents:
            entidades_nao_of[ent.label_].append(ent.text)

    print("\n🏷️  TIPOS DE ENTIDADES - OFENSIVO:")
    for label, entities in sorted(entidades_of.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"  {label}: {len(entities)} ocorrências")
        print(f"    Exemplos: {', '.join(list(set(entities))[:5])}")

    print("\n🏷️  TIPOS DE ENTIDADES - NÃO-OFENSIVO:")
    for label, entities in sorted(entidades_nao_of.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"  {label}: {len(entities)} ocorrências")
        print(f"    Exemplos: {', '.join(list(set(entities))[:5])}")

    # Conclusão para o relatório
    print("\n📝 INSIGHTS:")
    print("  - Menções a PER (pessoas) podem indicar ataques pessoais")
    print("  - ORG (organizações) aparecem em contextos políticos")
    print("  - LOC (locais) mais neutros em comentários não-ofensivos")

# %%
# =============================================================================
# 4.3. ANÁLISE DE SENTIMENTO
# =============================================================================
from textblob import TextBlob

def analisar_sentimento(df):
    """Analisa polaridade de sentimento e sua relação com ofensividade."""
    print("\n" + "="*70)
    print("ANÁLISE DE SENTIMENTO")
    print("="*70)

    def calcular_sentimento(texto):
        blob = TextBlob(texto)
        return blob.sentiment.polarity  # -1 (negativo) a +1 (positivo)

    df['sentimento'] = df['comentario'].apply(calcular_sentimento)

    sent_of = df[df['label_final'] == 1]['sentimento']
    sent_nao_of = df[df['label_final'] == 0]['sentimento']

    print(f"\n📊 POLARIDADE MÉDIA:")
    print(f"  Ofensivo:     {sent_of.mean():.3f} (DP: {sent_of.std():.3f})")
    print(f"  Não-Ofensivo: {sent_nao_of.mean():.3f} (DP: {sent_nao_of.std():.3f})")

    # Distribuição
    print(f"\n📈 DISTRIBUIÇÃO:")
    print(f"  Ofensivo - Negativo: {(sent_of < -0.1).sum()} ({(sent_of < -0.1).mean()*100:.1f}%)")
    print(f"  Ofensivo - Neutro:   {((sent_of >= -0.1) & (sent_of <= 0.1)).sum()}")
    print(f"  Ofensivo - Positivo: {(sent_of > 0.1).sum()}")

    return df

# %%
# =============================================================================
# 4.4. VISUALIZAÇÃO DA ANÁLISE LINGUÍSTICA
# =============================================================================

def visualizar_analise_linguistica(df, frequencias):
    """Cria visualizações da análise linguística."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Top palavras por classe
    top_of = frequencias['fdist_of'].most_common(15)
    top_nao_of = frequencias['fdist_nao_of'].most_common(15)

    palavras_of = [p for p, _ in top_of]
    freqs_of = [f for _, f in top_of]
    axes[0, 0].barh(palavras_of, freqs_of, color='red', alpha=0.7)
    axes[0, 0].set_title('Top 15 Palavras - OFENSIVO', fontweight='bold')
    axes[0, 0].invert_yaxis()

    palavras_nao_of = [p for p, _ in top_nao_of]
    freqs_nao_of = [f for _, f in top_nao_of]
    axes[0, 1].barh(palavras_nao_of, freqs_nao_of, color='green', alpha=0.7)
    axes[0, 1].set_title('Top 15 Palavras - NÃO-OFENSIVO', fontweight='bold')
    axes[0, 1].invert_yaxis()

    # 2. Distribuição de tokens por classe
    df.boxplot(column='num_tokens', by='label_final', ax=axes[1, 0])
    axes[1, 0].set_title('Distribuição do Comprimento dos Textos')
    axes[1, 0].set_xlabel('Classe')
    axes[1, 0].set_ylabel('Número de Tokens')

    # 3. Word Cloud comparativo
    from wordcloud import WordCloud
    tokens_of_text = ' '.join([t for tokens in df[df['label_final']==1]['cleaned_tokens'] for t in tokens])
    wc_of = WordCloud(width=400, height=300, background_color='white').generate(tokens_of_text)
    axes[1, 1].imshow(wc_of, interpolation='bilinear')
    axes[1, 1].set_title('Word Cloud - OFENSIVO')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('analise_linguistica.png', dpi=300)
    plt.show()

# %% [markdown]
# # 5. Identificação de Vocabulário Distintivo
# 
# A função `criar_vocabulario_distintivo` identifica palavras e n-gramas que são distintivos para cada classe. As principais etapas incluem:
# 1. Cálculo das frequências relativas de palavras em cada classe.
# 2. Identificação de palavras com alta razão de frequência entre as classes.
# 3. Identificação de bigramas e trigramas distintivos.
# 
# **Por que está estruturado assim?**
# - O vocabulário distintivo é usado para criar regras no classificador.
# - A análise de frequências relativas ajuda a identificar palavras que são mais comuns em uma classe do que na outra.
# - A inclusão de n-gramas permite capturar expressões compostas que podem ser indicativas de ofensividade.

# %%
# =============================================================================
# 5. IDENTIFICAÇÃO DE VOCABULÁRIO DISTINTIVO
# =============================================================================

def criar_vocabulario_distintivo(df, min_freq=3, min_ratio=3, max_ratio=1000):
    """
    Identifica palavras e n-gramas distintivos para cada classe.

    Args:
        df: DataFrame com colunas 'cleaned_tokens' e 'label_final'
        min_freq: Frequência mínima para considerar uma palavra
        min_ratio: Ratio mínimo para considerar distintivo
        max_ratio: Ratio máximo para evitar outliers
    """
    print("\n" + "=" * 70)
    print("CRIANDO VOCABULÁRIO DISTINTIVO")
    print("=" * 70)

    # Separar tokens por classe
    tokens_nao_of = [t for tokens in df[df['label_final'] == 0]['cleaned_tokens'] for t in tokens]
    tokens_of = [t for tokens in df[df['label_final'] == 1]['cleaned_tokens'] for t in tokens]

    # Calcular frequências
    fdist_nao_of = FreqDist(tokens_nao_of)
    fdist_of = FreqDist(tokens_of)

    total_nao_of = sum(fdist_nao_of.values())
    total_of = sum(fdist_of.values())

    # Identificar palavras distintivas
    palavras_ofensivas = set()
    palavras_contextuais = set()
    palavras_positivas = set()

    todas_palavras = set(fdist_of.keys()) | set(fdist_nao_of.keys())

    for palavra in todas_palavras:
        freq_of = fdist_of.get(palavra, 0)
        freq_nao_of = fdist_nao_of.get(palavra, 0)

        if freq_of + freq_nao_of < min_freq:
            continue

        freq_rel_of = freq_of / total_of if total_of > 0 else 0
        freq_rel_nao_of = freq_nao_of / total_nao_of if total_nao_of > 0 else 0

        # Classificar baseado no ratio
        if freq_rel_nao_of == 0 and freq_of > min_freq:
            palavras_ofensivas.add(palavra)
        elif freq_rel_of == 0 and freq_nao_of > min_freq:
            palavras_positivas.add(palavra)
        elif freq_rel_of > 0 and freq_rel_nao_of > 0:
            ratio_of = freq_rel_of / freq_rel_nao_of
            ratio_nao_of = freq_rel_nao_of / freq_rel_of

            if min_ratio < ratio_of < max_ratio:
                if ratio_of > 5:
                    palavras_ofensivas.add(palavra)
                else:
                    palavras_contextuais.add(palavra)
            elif min_ratio < ratio_nao_of < max_ratio:
                palavras_positivas.add(palavra)

    # Identificar bigramas distintivos
    bigramas_of = list(bigrams([t for tokens in df[df['label_final'] == 1]['cleaned_tokens']
                                 for t in tokens]))
    bigramas_nao_of = list(bigrams([t for tokens in df[df['label_final'] == 0]['cleaned_tokens']
                                     for t in tokens]))

    fdist_bg_of = FreqDist(bigramas_of)
    fdist_bg_nao_of = FreqDist(bigramas_nao_of)

    bigramas_ofensivos = set()
    for bigrama, freq in fdist_bg_of.most_common(100):
        freq_nao_of = fdist_bg_nao_of.get(bigrama, 0)
        if freq > 5 and freq > freq_nao_of * 2:
            bigramas_ofensivos.add(bigrama)

    # Trigramas distintivos
    trigramas_of = list(trigrams([t for tokens in df[df['label_final'] == 1]['cleaned_tokens']
                                   for t in tokens]))
    fdist_tg_of = FreqDist(trigramas_of)

    trigramas_ofensivos = {trigrama for trigrama, freq in fdist_tg_of.most_common(50) if freq > 3}

    print(f"\nPalavras fortemente ofensivas: {len(palavras_ofensivas)}")
    print(f"Palavras contextuais: {len(palavras_contextuais)}")
    print(f"Palavras positivas: {len(palavras_positivas)}")
    print(f"Bigramas ofensivos: {len(bigramas_ofensivos)}")
    print(f"Trigramas ofensivos: {len(trigramas_ofensivos)}")

    return {
        'palavras_ofensivas': palavras_ofensivas,
        'palavras_contextuais': palavras_contextuais,
        'palavras_positivas': palavras_positivas,
        'bigramas_ofensivos': bigramas_ofensivos,
        'trigramas_ofensivos': trigramas_ofensivos
    }


# %% [markdown]
# # 6. Classificador Baseado em Regras
# 
# A classe `ClassificadorRegras` implementa um sistema de classificação baseado em regras linguísticas. As principais funcionalidades incluem:
# 1. Extração de features linguísticas, como palavras ofensivas, bigramas e trigramas.
# 2. Aplicação de regras para calcular scores de ofensividade e não-ofensividade.
# 3. Classificação dos textos com base nos scores calculados.
# 
# **Por que está estruturado assim?**
# - O uso de regras linguísticas permite interpretar as decisões do classificador.
# - A combinação de diferentes tipos de features (palavras, n-gramas, padrões) melhora a precisão do modelo.
# - A implementação modular facilita a adição de novas regras ou ajustes nas existentes.

# %%
# =============================================================================
# 6. CLASSIFICADOR BASEADO EM REGRAS
# =============================================================================

class ClassificadorRegras:
    """Sistema de classificação baseado em regras linguísticas."""

    def __init__(self, vocabularios):
        """Inicializa com vocabulários pré-calculados."""
        self.palavras_ofensivas = vocabularios['palavras_ofensivas']
        self.palavras_contextuais = vocabularios['palavras_contextuais']
        self.palavras_positivas = vocabularios['palavras_positivas']
        self.bigramas_ofensivos = vocabularios['bigramas_ofensivos']
        self.trigramas_ofensivos = vocabularios['trigramas_ofensivos']
        self.nlp = spacy.load('pt_core_news_sm')

        # Intensificadores e negações
        self.intensificadores = {
            'muito', 'demais', 'total', 'completo', 'puro', 'tão',
            'super', 'mega', 'completamente', 'totalmente', 'extremamente'
        }
        self.negacoes = {'não', 'nem', 'nunca', 'jamais', 'nada', 'ninguém'}

        # Padrões regex
        self.padroes = {
            'xingamento': re.compile(r'\b\w*[*#@]+\w*\b'),
            'gritaria': re.compile(r'\b[A-Z]{4,}\b'),
            'repeticao': re.compile(r'(.)\1{3,}'),
            'emoji_raiva': re.compile(r'[😡🤬😤💢🖕]'),
            'emoji_positivo': re.compile(r'[😊😃😄😁💕❤️👍🙏💖😘]')
        }

        # Pesos das regras
        self.pesos = {
            'palavra_ofensiva': 3.0,
            'palavra_contextual': 1.5,
            'palavra_positiva': 2.0,
            'bigrama_ofensivo': 2.5,
            'trigrama_ofensivo': 2.0,
            'intensificador': 1.5,
            'negacao_positivo': 2.0,
            'xingamento': 2.0,
            'emoji_raiva': 1.5,
            'emoji_positivo': 1.5
        }

    def extrair_features(self, texto):
        """Extrai features linguísticas do texto."""
        texto_lower = texto.lower()
        tokens = word_tokenize(texto_lower)
        tokens_limpos = [t for t in tokens if t not in string.punctuation and len(t) > 1]

        features = {
            'num_tokens': len(tokens_limpos),
            'palavras_ofensivas': [t for t in tokens_limpos if t in self.palavras_ofensivas],
            'palavras_contextuais': [t for t in tokens_limpos if t in self.palavras_contextuais],
            'palavras_positivas': [t for t in tokens_limpos if t in self.palavras_positivas],
            'bigramas_ofensivos': [bg for bg in bigrams(tokens_limpos) if bg in self.bigramas_ofensivos],
            'trigramas_ofensivos': [tg for tg in trigrams(tokens_limpos) if tg in self.trigramas_ofensivos],
            'tem_intensificador': any(t in self.intensificadores for t in tokens_limpos),
            'tem_negacao': any(t in self.negacoes for t in tokens_limpos),
            'tem_xingamento': bool(self.padroes['xingamento'].search(texto)),
            'num_emoji_raiva': len(self.padroes['emoji_raiva'].findall(texto)),
            'num_emoji_positivo': len(self.padroes['emoji_positivo'].findall(texto)),
            'sentimento': TextBlob(texto).sentiment.polarity
        }

        # Combinações especiais
        features['intensificador_com_ofensa'] = (
            features['tem_intensificador'] and features['palavras_ofensivas']
        )
        features['negacao_com_positivo'] = (
            features['tem_negacao'] and features['palavras_positivas']
        )

        return features

    def tem_mencao_pessoa(self, texto):
        """Detecta menções a pessoas (possível ataque pessoal)."""
        doc = self.nlp(texto)
        return any(ent.label_ == 'PER' for ent in doc.ents)

    def calcular_score(self, features, texto):
        """Calcula score de ofensividade baseado nas features."""
        score_of = 0
        score_nao_of = 0
        regras = []

        # Regras para OFENSIVO
        if features['palavras_ofensivas']:
            n = len(features['palavras_ofensivas'])
            score_of += n * self.pesos['palavra_ofensiva']
            regras.append(f"R1: {n} palavra(s) ofensiva(s)")

        if features['palavras_contextuais']:
            n = len(features['palavras_contextuais'])
            score_of += n * self.pesos['palavra_contextual']
            regras.append(f"R2: {n} palavra(s) contextual(is)")

        if features['bigramas_ofensivos']:
            n = len(features['bigramas_ofensivos'])
            score_of += n * self.pesos['bigrama_ofensivo']
            regras.append(f"R3: {n} bigrama(s) ofensivo(s)")

        if features['trigramas_ofensivos']:
            n = len(features['trigramas_ofensivos'])
            score_of += n * self.pesos['trigrama_ofensivo']
            regras.append(f"R4: {n} trigrama(s) ofensivo(s)")

        if features['intensificador_com_ofensa']:
            score_of += self.pesos['intensificador']
            regras.append("R5: Intensificador + ofensa")

        if features['negacao_com_positivo']:
            score_of += self.pesos['negacao_positivo']
            regras.append("R6: Negação + positivo")

        if features['tem_xingamento']:
            score_of += self.pesos['xingamento']
            regras.append("R7: Xingamento censurado")

        if features['num_emoji_raiva'] > 0:
            score_of += features['num_emoji_raiva'] * self.pesos['emoji_raiva']
            regras.append(f"R8: {features['num_emoji_raiva']} emoji(s) de raiva")

        # Regras para NÃO-OFENSIVO
        if features['palavras_positivas']:
            n = len(features['palavras_positivas'])
            score_nao_of += n * self.pesos['palavra_positiva']
            regras.append(f"R9: {n} palavra(s) positiva(s)")

        if features['num_emoji_positivo'] > 0:
            score_nao_of += features['num_emoji_positivo'] * self.pesos['emoji_positivo']
            regras.append(f"R10: {features['num_emoji_positivo']} emoji(s) positivo(s)")

        if features['num_tokens'] > 20 and not features['palavras_ofensivas']:
            score_nao_of += 1.0
            regras.append("R11: Texto longo sem ofensas")

        if self.tem_mencao_pessoa(texto) and features['palavras_ofensivas']:
            score_of += 2.0
            regras.append("R12: Menção a pessoa + ofensa (ataque pessoal)")

        if features['sentimento'] < -0.3:
            score_of += 1.5
            regras.append("R13: Sentimento fortemente negativo")

        return score_of, score_nao_of, regras

    def classificar(self, texto):
        """
        Classifica um texto como ofensivo (1) ou não-ofensivo (0).

        Returns:
            tuple: (classe, confiança, regras_ativadas)
        """
        features = self.extrair_features(texto)
        score_of, score_nao_of, regras = self.calcular_score(features, texto)

        threshold = 0.5

        if score_of > score_nao_of + threshold:
            classe = 1
            total = score_of + score_nao_of
            confianca = score_of / total if total > 0 else 0.5
        else:
            classe = 0
            total = score_of + score_nao_of
            confianca = score_nao_of / total if total > 0 else 0.5

        if not regras:
            classe = 0
            confianca = 0.5
            regras = ["DEFAULT: Sem regras ativadas"]

        return classe, confianca, regras

# %% [markdown]
# # 7. Avaliação do Classificador
# 
# A função `avaliar_classificador` avalia o desempenho do classificador em termos de métricas como acurácia, precisão, recall e F1-score. As principais etapas incluem:
# 1. Aplicação do classificador a cada comentário no corpus.
# 2. Cálculo das métricas de desempenho.
# 3. Análise de erros, incluindo exemplos de falsos positivos e falsos negativos.
# 
# **Por que está estruturado assim?**
# - A avaliação quantitativa é essencial para medir a eficácia do classificador.
# - A análise de erros ajuda a identificar áreas de melhoria no modelo.
# - A exibição de exemplos concretos facilita a interpretação dos resultados.

# %%
# =============================================================================
# 7. AVALIAÇÃO DO CLASSIFICADOR
# =============================================================================

def avaliar_classificador(df, classificador):
    """Avalia o classificador e exibe métricas detalhadas."""
    print("\n" + "=" * 70)
    print("AVALIAÇÃO DO CLASSIFICADOR")
    print("=" * 70)

    # Aplicar classificador
    resultados = []
    for idx, row in df.iterrows():
        classe, confianca, regras = classificador.classificar(row['comentario'])
        resultados.append({
            'previsao': classe,
            'confianca': confianca,
            'regras': regras
        })

        if (idx + 1) % 500 == 0:
            print(f"  Processados: {idx + 1}/{len(df)}")

    # Adicionar resultados ao DataFrame
    df['previsao'] = [r['previsao'] for r in resultados]
    df['confianca'] = [r['confianca'] for r in resultados]
    df['regras_ativadas'] = [r['regras'] for r in resultados]

    # Calcular métricas
    y_true = df['label_final'].values
    y_pred = df['previsao'].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Exibir resultados
    print("\n📊 MÉTRICAS:")
    print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precisão:  {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\n📈 MATRIZ DE CONFUSÃO:")
    print(f"              Pred_0  Pred_1")
    print(f"  Real_0:     {cm[0,0]:6}  {cm[0,1]:6}")
    print(f"  Real_1:     {cm[1,0]:6}  {cm[1,1]:6}")

    print("\n📋 RELATÓRIO DETALHADO:")
    print(classification_report(y_true, y_pred,
                                target_names=['Não-Ofensivo', 'Ofensivo']))

    # Análise de erros
    df['tipo_erro'] = 'Correto'
    df.loc[(df['label_final'] == 0) & (df['previsao'] == 1), 'tipo_erro'] = 'FP'
    df.loc[(df['label_final'] == 1) & (df['previsao'] == 0), 'tipo_erro'] = 'FN'

    print(f"\n🔍 ANÁLISE DE ERROS:")
    print(f"  Falsos Positivos (FP): {(df['tipo_erro'] == 'FP').sum()}")
    print(f"  Falsos Negativos (FN): {(df['tipo_erro'] == 'FN').sum()}")

    # Exemplos de erros
    print("\n⚠️ EXEMPLOS DE FALSOS NEGATIVOS:")
    fn_df = df[df['tipo_erro'] == 'FN']
    for idx, row in fn_df.head(3).iterrows():
        print(f"\n  '{row['comentario'][:80]}...'")
        print(f"  Regras: {row['regras_ativadas']}")

    print("\n⚠️ EXEMPLOS DE FALSOS POSITIVOS:")
    fp_df = df[df['tipo_erro'] == 'FP']
    for idx, row in fp_df.head(3).iterrows():
        print(f"\n  '{row['comentario'][:80]}...'")
        print(f"  Regras: {row['regras_ativadas']}")

    return df

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# =============================================================================
# 7.2. ANÁLISE DE IMPORTÂNCIA DAS REGRAS
# =============================================================================

def analise_de_importancia_das_regras(df, vocabularios):
    """Testa a importância individual de cada tipo de regra."""
    print("\n" + "="*70)
    print("ANÁLISE DE IMPORTÂNCIA DAS REGRAS")
    print("="*70)

    resultados = {}

    # Baseline: todas as regras
    clf_completo = ClassificadorRegras(vocabularios)
    y_pred_completo = [clf_completo.classificar(t)[0] for t in df['comentario']]
    f1_completo = f1_score(df['label_final'], y_pred_completo)
    resultados['Completo'] = f1_completo

    # Teste removendo cada tipo de regra
    regras_para_testar = [
        ('sem_unigramas', {'palavras_ofensivas': set()}),
        ('sem_bigramas', {'bigramas_ofensivos': set()}),
        ('sem_trigramas', {'trigramas_ofensivos': set()}),
        ('sem_intensificadores', {'intensificadores': set()}),
        ('sem_negacoes', {'negacoes': set()}),  # Exemplo adicional
        ('sem_emojis', {'emoji_raiva': set(), 'emoji_positivo': set()})  # Exemplo adicional
    ]

    for nome, vocabulario_modificado in regras_para_testar:
        vocab_teste = vocabularios.copy()
        vocab_teste.update(vocabulario_modificado)
        clf_teste = ClassificadorRegras(vocab_teste)
        y_pred_teste = [clf_teste.classificar(t)[0] for t in df['comentario']]
        f1_teste = f1_score(df['label_final'], y_pred_teste)
        resultados[nome] = f1_teste
        delta = f1_completo - f1_teste
        print(f"  {nome}: F1={f1_teste:.4f} (Δ={delta:+.4f})")

    return resultados

# %%
# =============================================================================
# 7.3. ANÁLISE QUALITATIVA DE ERROS
# =============================================================================

def analise_qualitativa_erros(df):
    """Categoriza e analisa qualitativamente os erros."""
    print("\n" + "="*70)
    print("ANÁLISE QUALITATIVA DE ERROS")
    print("="*70)

    # Falsos Negativos (FN) - Ofensivos não detectados
    fn_df = df[(df['label_final'] == 1) & (df['previsao'] == 0)]

    print(f"\n❌ FALSOS NEGATIVOS: {len(fn_df)}")
    print("\nCATEGORIAS DE ERROS:")

    # Categorizar erros manualmente (amostra)
    categorias_fn = {
        'ironia_sarcasmo': [],
        'ofensa_implícita': [],
        'vocabulário_fora_léxico': [],
        'contexto_necessário': []
    }

    # Exemplo de categorização (fazer manualmente para ~20 exemplos)
    print("  • Ironia/Sarcasmo não detectado")
    print("  • Ofensa implícita (sem palavrões diretos)")
    print("  • Vocabulário novo/gírias não no léxico")
    print("  • Necessita contexto externo para entender ofensa")

    # Falsos Positivos (FP) - Não-ofensivos classificados como ofensivos
    fp_df = df[(df['label_final'] == 0) & (df['previsao'] == 1)]

    print(f"\n✅ FALSOS POSITIVOS: {len(fp_df)}")
    print("\nCATEGORIAS DE ERROS:")
    print("  • Palavras ambíguas (ex: 'pena' = compaixão)")
    print("  • Citações de ofensas (mencionando, não praticando)")
    print("  • Discussão sobre o tema ofensividade")
    print("  • Falsos triggers (emojis, padrões)")

# %%
# =============================================================================
# 8. PIPELINE PRINCIPAL
# =============================================================================

def executar_pipeline(caminho_csv):
    """Executa o pipeline completo de análise e classificação."""

    print("\n" + "=" * 70)
    print("✅ PIPELINE Iniciado COM SUCESSO")
    print("=" * 70)


    # 1. Carregar dados
    print("\n" + "=" * 70)
    print("Carregando Dados")
    df = carregar_e_analisar_corpus(caminho_csv)
    print("\n" + "=" * 70)

    # 2. Pré-processar
    print("\n" + "=" * 70)
    print("Pré-processando Os dados")
    df = preprocessar_corpus(df) # df agora tem 'tokens' e 'cleaned_tokens'
    print("\n" + "=" * 70)

    # 3. Analisar frequências (usa 'cleaned_tokens')
    print("\n" + "=" * 70)
    print("Analisando Frequências")
    frequencias = analisar_frequencias(df)
    print("\n" + "=" * 70)

    # 4. Criar vocabulário distintivo (usa 'cleaned_tokens')
    print("\n" + "=" * 70)
    print("Criando Vocabulário Distintivo")
    vocabularios = criar_vocabulario_distintivo(df)
    print("\n" + "=" * 70)

    print("\n"+"Visualizando Análise Linguística")
    visualizar_analise_linguistica(df, frequencias)
    print("\n" + "=" * 70)

    # 4.1. Analisar Pós Tagging (precisa de 'tokens') - REMOVIDO DO PIPELINE
    # print("\n"+"Analisando POS Tagging")
    # analisar_pos_tagging(df) # Passa o df que já tem 'tokens'
    # print("\n" + "=" * 70)

    # 4.2. Analisar entidades nomeadas (precisa de 'comentario') - REMOVIDO DO PIPELINE
    # print("\n"+"Analisando Entidades Nomeadas")
    # analisar_entidades(df) # Passa o df que tem 'comentario'
    # print("\n" + "=" * 70)

    # 4.3. Analisar sentimento (precisa de 'comentario') - REMOVIDO DO PIPELINE
    # print("\n"+"Analisando Sentimento")
    # df = analisar_sentimento(df) # df agora tem 'sentimento'
    # print("\n" + "=" * 70)

    # 5. Criar e treinar classificador (usa vocabularios e espera texto original/features)
    print("\n"+"Criando e Treinando Classificador Baseado em Regras")
    classificador = ClassificadorRegras(vocabularios)
    print("\n" + "=" * 70)

    # 6. Avaliar (usa o classificador e o df com 'comentario')
    print("\n"+"Avaliando Classificador Baseado em Regras")
    # Passamos o df original para avaliação, pois a avaliação do classificador de regras
    # não depende das colunas de análise linguística que foram removidas do pipeline.
    df_resultado = avaliar_classificador(df, classificador)
    print("\n" + "=" * 70)

    # 7. Análise de importância das regras (usa o df com 'comentario' e 'label_final')
    print("\n"+"Analisando Importância das Regras")
    analise_de_importancia_das_regras(df_resultado, vocabularios) # usa o df_resultado que tem previsoes e labels
    print("\n" + "=" * 70)

    print("\n"+"Analisando Qualitativamente os Erros")
    analise_qualitativa_erros(df_resultado) # usa o df_resultado que tem tipo_erro


    print("\n" + "=" * 70)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO")
    print("=" * 70)


    # Retorna o df_resultado que contém as colunas originais, processamento básico,
    # previsões do classificador de regras e tipo de erro.
    return df_resultado, classificador, df # Return the DataFrame after basic processing

# %%
# =============================================================================
# FUNÇÕES DE EXTRAÇÃO DE FEATURES LINGUÍSTICAS
# =============================================================================

def extract_pos_features(tokens):
    """
    Extrai features de POS tagging (contagem de tags) para um texto.
    Retorna um dicionário com contagens de tags.
    """
    # Realiza POS tagging nos tokens
    pos_tags = pos_tag(tokens)

    # Conta a frequência de cada tag
    tag_counts = Counter(tag for word, tag in pos_tags)

    # Podemos selecionar as tags mais relevantes ou usar todas
    # Para simplificar, vamos retornar as contagens de algumas tags comuns
    selected_tags = ['NN', 'JJ', 'VB', 'RB', 'DT', 'PRP', 'IN', 'NNS', 'JJS', 'VBP']
    features = {f'pos_count_{tag}': tag_counts.get(tag, 0) for tag in selected_tags}

    # Adicionar contagem total de tags (excluindo pontuação)
    total_tags = sum(count for tag, count in tag_counts.items() if tag not in string.punctuation)
    features['pos_total_count'] = total_tags

    return features

def extract_ner_features(texto, nlp):
    """
    Extrai features de NER (contagem de tipos de entidades) para um texto.
    Retorna um dicionário com contagens de tipos de entidades.
    """
    doc = nlp(texto)
    entity_counts = Counter(ent.label_ for ent in doc.ents)

    # Selecionar alguns tipos de entidades comuns ou usar todos
    selected_entities = ['PER', 'ORG', 'LOC', 'MISC']
    features = {f'ner_count_{ent_type}': entity_counts.get(ent_type, 0) for ent_type in selected_entities}

    # Adicionar contagem total de entidades
    features['ner_total_count'] = len(doc.ents)

    return features

def extract_sentiment_features(texto):
    """
    Extrai features de sentimento (polaridade e subjetividade) para um texto.
    Retorna um dicionário com os scores.
    """
    blob = TextBlob(texto)
    features = {
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity # Subjetividade também pode ser útil
    }
    return features

print("Funções de extração de features linguísticas criadas.")

# %% [markdown]
# # Função de Análise do Naive Bayes
# 

# %%
# =============================================================================
# FUNÇÃO DE ANÁLISE DO NAIVE BAYES
# =============================================================================

def analisar_naive_bayes(modelo, X_train, X_test, y_train, y_test,
                         tfidf_vectorizer, df_test):
    """
    Analisa o desempenho do classificador Naive Bayes com métricas detalhadas.

    Parâmetros:
    -----------
    modelo : MultinomialNB
        Modelo Naive Bayes treinado
    X_train, X_test : sparse matrix
        Features de treino e teste (TF-IDF)
    y_train, y_test : array
        Labels de treino e teste
    tfidf_vectorizer : TfidfVectorizer
        Vetorizador TF-IDF treinado
    df_test : DataFrame
        DataFrame com os dados de teste (para análise qualitativa)

    Retorna:
    --------
    dict : Dicionário com métricas e resultados da análise
    """

    print("\n" + "=" * 70)
    print("ANÁLISE DETALHADA - NAIVE BAYES")
    print("=" * 70)

    # 1. PREDIÇÕES
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)

    # 2. MÉTRICAS PRINCIPAIS
    print("\n📊 MÉTRICAS DE DESEMPENHO:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precisão:  {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 3. MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 MATRIZ DE CONFUSÃO:")
    print(f"                  Predito 0  Predito 1")
    print(f"  Real 0 (Não-Of):    {cm[0,0]:6}    {cm[0,1]:6}")
    print(f"  Real 1 (Ofensivo):  {cm[1,0]:6}    {cm[1,1]:6}")

    # Cálculo de VP, VN, FP, FN
    vn, fp, fn, vp = cm.ravel()
    print(f"\n  Verdadeiros Negativos (VN): {vn}")
    print(f"  Falsos Positivos (FP):      {fp}")
    print(f"  Falsos Negativos (FN):      {fn}")
    print(f"  Verdadeiros Positivos (VP): {vp}")

    # 4. RELATÓRIO DETALHADO
    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred,
                                target_names=['Não-Ofensivo', 'Ofensivo'],
                                digits=4))

    # 5. ANÁLISE DE ERROS
    print("\n🔍 ANÁLISE DE ERROS:")

    # Criar DataFrame temporário com predições
    df_analise = df_test.copy()
    df_analise['predicao'] = y_pred
    df_analise['confianca'] = y_pred_proba.max(axis=1)
    df_analise['label_real'] = y_test.values

    # Identificar tipos de erro
    df_analise['tipo'] = 'Correto'
    df_analise.loc[(df_analise['label_real'] == 0) &
                   (df_analise['predicao'] == 1), 'tipo'] = 'FP'
    df_analise.loc[(df_analise['label_real'] == 1) &
                   (df_analise['predicao'] == 0), 'tipo'] = 'FN'

    num_fp = (df_analise['tipo'] == 'FP').sum()
    num_fn = (df_analise['tipo'] == 'FN').sum()

    print(f"  Falsos Positivos (FP): {num_fp} ({num_fp/len(df_analise)*100:.2f}%)")
    print(f"  Falsos Negativos (FN): {num_fn} ({num_fn/len(df_analise)*100:.2f}%)")

    # 6. EXEMPLOS DE FALSOS NEGATIVOS (deveria ser ofensivo, mas predito como não-ofensivo)
    print("\n⚠️  EXEMPLOS DE FALSOS NEGATIVOS (FN):")
    print("    (Texto ofensivo classificado como não-ofensivo)")
    fn_samples = df_analise[df_analise['tipo'] == 'FN'].head(3)

    for idx, (i, row) in enumerate(fn_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    Confiança da predição: {row['confianca']:.3f}")

    # 7. EXEMPLOS DE FALSOS POSITIVOS (deveria ser não-ofensivo, mas predito como ofensivo)
    print("\n⚠️  EXEMPLOS DE FALSOS POSITIVOS (FP):")
    print("    (Texto não-ofensivo classificado como ofensivo)")
    fp_samples = df_analise[df_analise['tipo'] == 'FP'].head(3)

    for idx, (i, row) in enumerate(fp_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    Confiança da predição: {row['confianca']:.3f}")

    # 8. FEATURES MAIS IMPORTANTES
    print("\n🎯 FEATURES MAIS DISCRIMINATIVAS:")

    # Obter as features (palavras) mais importantes para cada classe
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Log-probabilidades das features para cada classe
    # Classe 0 (Não-Ofensivo)
    top_nao_ofensivo = np.argsort(modelo.feature_log_prob_[0])[-10:]
    print("\n  Top 10 features para NÃO-OFENSIVO:")
    for feature_idx in reversed(top_nao_ofensivo):
        print(f"    - {feature_names[feature_idx]}: {modelo.feature_log_prob_[0][feature_idx]:.4f}")

    # Classe 1 (Ofensivo)
    top_ofensivo = np.argsort(modelo.feature_log_prob_[1])[-10:]
    print("\n  Top 10 features para OFENSIVO:")
    for feature_idx in reversed(top_ofensivo):
        print(f"    - {feature_names[feature_idx]}: {modelo.feature_log_prob_[1][feature_idx]:.4f}")

    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 70)

    # Retornar dicionário com resultados
    resultados = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'df_analise': df_analise,
        'num_fp': num_fp,
        'num_fn': num_fn
    }

    return resultados

# %% [markdown]
# # **Função de Análise do SVM**

# %%
# =============================================================================
# FUNÇÃO DE ANÁLISE DO SVM
# =============================================================================

def analisar_svm(modelo, X_train, X_test, y_train, y_test,
                 tfidf_vectorizer, df_test):
    """
    Analisa o desempenho do classificador SVM com métricas detalhadas.

    Parâmetros:
    -----------
    modelo : SVC
        Modelo SVM treinado (kernel='linear')
    X_train, X_test : sparse matrix
        Features de treino e teste (TF-IDF)
    y_train, y_test : array
        Labels de treino e teste
    tfidf_vectorizer : TfidfVectorizer
        Vetorizador TF-IDF treinado
    df_test : DataFrame
        DataFrame com os dados de teste (para análise qualitativa)

    Retorna:
    --------
    dict : Dicionário com métricas e resultados da análise
    """

    print("\n" + "=" * 70)
    print("ANÁLISE DETALHADA - SVM (LINEAR KERNEL)")
    print("=" * 70)

    # 1. PREDIÇÕES
    y_pred = modelo.predict(X_test)

    # Para SVM linear, usamos decision_function para obter "confiança"
    # Valores mais distantes de 0 indicam maior confiança
    decision_scores = modelo.decision_function(X_test)

    # 2. MÉTRICAS PRINCIPAIS
    print("\n📊 MÉTRICAS DE DESEMPENHO:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precisão:  {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 3. MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 MATRIZ DE CONFUSÃO:")
    print(f"                  Predito 0  Predito 1")
    print(f"  Real 0 (Não-Of):    {cm[0,0]:6}    {cm[0,1]:6}")
    print(f"  Real 1 (Ofensivo):  {cm[1,0]:6}    {cm[1,1]:6}")

    # Cálculo de VP, VN, FP, FN
    vn, fp, fn, vp = cm.ravel()
    print(f"\n  Verdadeiros Negativos (VN): {vn}")
    print(f"  Falsos Positivos (FP):      {fp}")
    print(f"  Falsos Negativos (FN):      {fn}")
    print(f"  Verdadeiros Positivos (VP): {vp}")

    # 4. RELATÓRIO DETALHADO
    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred,
                                target_names=['Não-Ofensivo', 'Ofensivo'],
                                digits=4))

    # 5. ANÁLISE DE ERROS
    print("\n🔍 ANÁLISE DE ERROS:")

    # Criar DataFrame temporário com predições
    df_analise = df_test.copy()
    df_analise['predicao'] = y_pred
    df_analise['decision_score'] = decision_scores
    df_analise['confianca'] = np.abs(decision_scores)  # Distância do hiperplano
    df_analise['label_real'] = y_test.values

    # Identificar tipos de erro
    df_analise['tipo'] = 'Correto'
    df_analise.loc[(df_analise['label_real'] == 0) &
                   (df_analise['predicao'] == 1), 'tipo'] = 'FP'
    df_analise.loc[(df_analise['label_real'] == 1) &
                   (df_analise['predicao'] == 0), 'tipo'] = 'FN'

    num_fp = (df_analise['tipo'] == 'FP').sum()
    num_fn = (df_analise['tipo'] == 'FN').sum()

    print(f"  Falsos Positivos (FP): {num_fp} ({num_fp/len(df_analise)*100:.2f}%)")
    print(f"  Falsos Negativos (FN): {num_fn} ({num_fn/len(df_analise)*100:.2f}%)")

    # 6. EXEMPLOS DE FALSOS NEGATIVOS
    print("\n⚠️  EXEMPLOS DE FALSOS NEGATIVOS (FN):")
    print("    (Texto ofensivo classificado como não-ofensivo)")
    fn_samples = df_analise[df_analise['tipo'] == 'FN'].nsmallest(3, 'confianca')

    for idx, (i, row) in enumerate(fn_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    Decision Score: {row['decision_score']:.3f}")
        print(f"    Confiança: {row['confianca']:.3f}")

    # 7. EXEMPLOS DE FALSOS POSITIVOS
    print("\n⚠️  EXEMPLOS DE FALSOS POSITIVOS (FP):")
    print("    (Texto não-ofensivo classificado como ofensivo)")
    fp_samples = df_analise[df_analise['tipo'] == 'FP'].nsmallest(3, 'confianca')

    for idx, (i, row) in enumerate(fp_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    Decision Score: {row['decision_score']:.3f}")
        print(f"    Confiança: {row['confianca']:.3f}")

    # 8. ANÁLISE DE COEFICIENTES (FEATURES MAIS IMPORTANTES)
    print("\n🎯 FEATURES MAIS DISCRIMINATIVAS (COEFICIENTES SVM):")
    print("\nNota: Coeficientes positivos indicam correlação com classe OFENSIVO")
    print("      Coeficientes negativos indicam correlação com classe NÃO-OFENSIVO")

    try:
        # Obter coeficientes do modelo SVM linear
        # IMPORTANTE: Converter de matriz esparsa para array denso
        if hasattr(modelo.coef_, 'toarray'):
            # Se for matriz esparsa, converter para array denso
            coeficientes = modelo.coef_.toarray()[0]
        else:
            # Se já for array denso
            coeficientes = modelo.coef_[0]

        # Obter nomes das features
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Converter para array numpy se necessário
        coeficientes = np.array(coeficientes).flatten()

        print(f"\n  Total de features analisadas: {len(coeficientes)}")

        # Top features para classe OFENSIVO (coeficientes positivos maiores)
        indices_positivos = np.argsort(coeficientes)[-15:]
        print("\n  Top 15 features para OFENSIVO (coeficientes > 0):")
        for rank, idx in enumerate(reversed(indices_positivos), 1):
            feature = feature_names[idx]
            coef = coeficientes[idx]
            print(f"    {rank:2d}. {feature:30s}: {coef:+.4f}")

        # Top features para classe NÃO-OFENSIVO (coeficientes negativos menores)
        indices_negativos = np.argsort(coeficientes)[:15]
        print("\n  Top 15 features para NÃO-OFENSIVO (coeficientes < 0):")
        for rank, idx in enumerate(indices_negativos, 1):
            feature = feature_names[idx]
            coef = coeficientes[idx]
            print(f"    {rank:2d}. {feature:30s}: {coef:+.4f}")

        # Estatísticas sobre os coeficientes
        print("\n📊 ESTATÍSTICAS DOS COEFICIENTES:")
        print(f"  Total de features:        {len(coeficientes)}")
        print(f"  Coeficientes positivos:   {(coeficientes > 0).sum()}")
        print(f"  Coeficientes negativos:   {(coeficientes < 0).sum()}")
        print(f"  Coeficientes zero:        {(coeficientes == 0).sum()}")
        print(f"  Coeficiente máximo:       {coeficientes.max():+.4f}")
        print(f"  Coeficiente mínimo:       {coeficientes.min():+.4f}")
        print(f"  Coeficiente médio:        {coeficientes.mean():+.4f}")
        print(f"  Desvio padrão:            {coeficientes.std():.4f}")

    except Exception as e:
        print(f"\n⚠️  Erro ao analisar coeficientes: {str(e)}")
        print(f"    Tipo de coef_: {type(modelo.coef_)}")
        print(f"    Shape: {modelo.coef_.shape}")
        print("    Continuando com o restante da análise...")

    # 9. ESTATÍSTICAS DOS DECISION SCORES
    print("\n📉 DISTRIBUIÇÃO DOS DECISION SCORES:")
    print(f"  Mínimo:  {decision_scores.min():.3f}")
    print(f"  Máximo:  {decision_scores.max():.3f}")
    print(f"  Média:   {decision_scores.mean():.3f}")
    print(f"  Mediana: {np.median(decision_scores):.3f}")

    # Separar por classe
    scores_corretos = df_analise[df_analise['tipo'] == 'Correto']['decision_score']
    scores_erros = df_analise[df_analise['tipo'].isin(['FP', 'FN'])]['decision_score']

    print(f"\n  Confiança média (predições corretas): {np.abs(scores_corretos).mean():.3f}")
    print(f"  Confiança média (predições erradas):  {np.abs(scores_erros).mean():.3f}")

    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 70)

    # Retornar dicionário com resultados
    resultados = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'decision_scores': decision_scores,
        'coeficientes': coeficientes,
        'feature_names': feature_names,
        'df_analise': df_analise,
        'num_fp': num_fp,
        'num_fn': num_fn
    }

    return resultados

# %%
# =============================================================================
# FUNÇÃO DE ANÁLISE DA REGRESSÃO LOGÍSTICA
# =============================================================================

def analisar_regressao_logistica(modelo, X_train, X_test, y_train, y_test,
                                  tfidf_vectorizer, df_test):
    """
    Analisa o desempenho do classificador de Regressão Logística com métricas detalhadas.

    A Regressão Logística combina características de Naive Bayes (probabilidades)
    e SVM (coeficientes), permitindo análise rica e interpretável.

    Parâmetros:
    -----------
    modelo : LogisticRegression
        Modelo de Regressão Logística treinado
    X_train, X_test : sparse matrix
        Features de treino e teste (TF-IDF)
    y_train, y_test : array
        Labels de treino e teste
    tfidf_vectorizer : TfidfVectorizer
        Vetorizador TF-IDF treinado
    df_test : DataFrame
        DataFrame com os dados de teste (para análise qualitativa)

    Retorna:
    --------
    dict : Dicionário com métricas e resultados da análise
    """

    print("\n" + "=" * 70)
    print("ANÁLISE DETALHADA - REGRESSÃO LOGÍSTICA")
    print("=" * 70)

    # 1. PREDIÇÕES E PROBABILIDADES
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)

    # Probabilidades para cada classe
    proba_classe_0 = y_pred_proba[:, 0]  # P(Não-Ofensivo)
    proba_classe_1 = y_pred_proba[:, 1]  # P(Ofensivo)

    # 2. MÉTRICAS PRINCIPAIS
    print("\n📊 MÉTRICAS DE DESEMPENHO:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precisão:  {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 3. MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 MATRIZ DE CONFUSÃO:")
    print(f"                  Predito 0  Predito 1")
    print(f"  Real 0 (Não-Of):    {cm[0,0]:6}    {cm[0,1]:6}")
    print(f"  Real 1 (Ofensivo):  {cm[1,0]:6}    {cm[1,1]:6}")

    # Cálculo de VP, VN, FP, FN
    vn, fp, fn, vp = cm.ravel()
    print(f"\n  Verdadeiros Negativos (VN): {vn}")
    print(f"  Falsos Positivos (FP):      {fp}")
    print(f"  Falsos Negativos (FN):      {fn}")
    print(f"  Verdadeiros Positivos (VP): {vp}")

    # 4. RELATÓRIO DETALHADO
    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred,
                                target_names=['Não-Ofensivo', 'Ofensivo'],
                                digits=4))

    # 5. ANÁLISE DE PROBABILIDADES
    print("\n🎲 ANÁLISE DAS PROBABILIDADES PREDITAS:")

    # Estatísticas das probabilidades
    print(f"\n  Probabilidades da classe OFENSIVO (P(y=1)):")
    print(f"    Mínima:  {proba_classe_1.min():.4f}")
    print(f"    Máxima:  {proba_classe_1.max():.4f}")
    print(f"    Média:   {proba_classe_1.mean():.4f}")
    print(f"    Mediana: {np.median(proba_classe_1):.4f}")

    # Distribuição de confiança
    alta_confianca = np.sum((proba_classe_1 > 0.7) | (proba_classe_1 < 0.3))
    media_confianca = np.sum((proba_classe_1 >= 0.3) & (proba_classe_1 <= 0.7))

    print(f"\n  Distribuição de Confiança:")
    print(f"    Alta confiança (p<0.3 ou p>0.7):   {alta_confianca} ({alta_confianca/len(y_test)*100:.1f}%)")
    print(f"    Média confiança (0.3≤p≤0.7):       {media_confianca} ({media_confianca/len(y_test)*100:.1f}%)")

    # 6. ANÁLISE DE ERROS
    print("\n🔍 ANÁLISE DE ERROS:")

    # Criar DataFrame temporário com predições
    df_analise = df_test.copy()
    df_analise['predicao'] = y_pred
    df_analise['prob_ofensivo'] = proba_classe_1
    df_analise['prob_nao_ofensivo'] = proba_classe_0
    df_analise['confianca'] = np.maximum(proba_classe_0, proba_classe_1)
    df_analise['label_real'] = y_test.values

    # Identificar tipos de erro
    df_analise['tipo'] = 'Correto'
    df_analise.loc[(df_analise['label_real'] == 0) &
                   (df_analise['predicao'] == 1), 'tipo'] = 'FP'
    df_analise.loc[(df_analise['label_real'] == 1) &
                   (df_analise['predicao'] == 0), 'tipo'] = 'FN'

    num_fp = (df_analise['tipo'] == 'FP').sum()
    num_fn = (df_analise['tipo'] == 'FN').sum()

    print(f"  Falsos Positivos (FP): {num_fp} ({num_fp/len(df_analise)*100:.2f}%)")
    print(f"  Falsos Negativos (FN): {num_fn} ({num_fn/len(df_analise)*100:.2f}%)")

    # Confiança média por tipo
    conf_corretos = df_analise[df_analise['tipo'] == 'Correto']['confianca'].mean()
    conf_erros = df_analise[df_analise['tipo'].isin(['FP', 'FN'])]['confianca'].mean()

    print(f"\n  Confiança média (predições corretas): {conf_corretos:.4f}")
    print(f"  Confiança média (predições erradas):  {conf_erros:.4f}")

    # 7. EXEMPLOS DE FALSOS NEGATIVOS (casos mais incertos)
    print("\n⚠️  EXEMPLOS DE FALSOS NEGATIVOS (FN):")
    print("    (Texto ofensivo classificado como não-ofensivo)")
    fn_samples = df_analise[df_analise['tipo'] == 'FN'].nlargest(3, 'prob_nao_ofensivo')

    for idx, (i, row) in enumerate(fn_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    P(Não-Ofensivo): {row['prob_nao_ofensivo']:.3f}")
        print(f"    P(Ofensivo):     {row['prob_ofensivo']:.3f}")

    # 8. EXEMPLOS DE FALSOS POSITIVOS (casos mais incertos)
    print("\n⚠️  EXEMPLOS DE FALSOS POSITIVOS (FP):")
    print("    (Texto não-ofensivo classificado como ofensivo)")
    fp_samples = df_analise[df_analise['tipo'] == 'FP'].nlargest(3, 'prob_ofensivo')

    for idx, (i, row) in enumerate(fp_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    P(Não-Ofensivo): {row['prob_nao_ofensivo']:.3f}")
        print(f"    P(Ofensivo):     {row['prob_ofensivo']:.3f}")

    # 9. ANÁLISE DE COEFICIENTES (FEATURES MAIS IMPORTANTES)
    print("\n🎯 FEATURES MAIS DISCRIMINATIVAS (COEFICIENTES):")
    print("\nNota: Regressão Logística usa P(y=1) = σ(W·x + b)")
    print("      Coeficientes > 0 → aumentam P(Ofensivo)")
    print("      Coeficientes < 0 → aumentam P(Não-Ofensivo)")

    try:
        # Extrair coeficientes (garantir conversão correta)
        if hasattr(modelo.coef_, 'toarray'):
            coeficientes = modelo.coef_.toarray()[0]
        else:
            coeficientes = modelo.coef_[0]

        coeficientes = np.array(coeficientes).flatten()
        feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

        # Intercepto (bias)
        intercepto = modelo.intercept_[0]
        print(f"\n  Intercepto (b): {intercepto:.4f}")
        print(f"  Total de features: {len(coeficientes)}")

        # Top features para OFENSIVO (coeficientes positivos)
        indices_positivos = np.argsort(coeficientes)[-15:]
        print("\n  Top 15 features que aumentam P(Ofensivo):")
        for rank, idx in enumerate(reversed(indices_positivos), 1):
            print(f"    {rank:2d}. {feature_names[idx]:30s}: {coeficientes[idx]:+.4f}")

        # Top features para NÃO-OFENSIVO (coeficientes negativos)
        indices_negativos = np.argsort(coeficientes)[:15]
        print("\n  Top 15 features que aumentam P(Não-Ofensivo):")
        for rank, idx in enumerate(indices_negativos, 1):
            print(f"    {rank:2d}. {feature_names[idx]:30s}: {coeficientes[idx]:+.4f}")

        # Estatísticas dos coeficientes
        print("\n📊 ESTATÍSTICAS DOS COEFICIENTES:")
        print(f"  Coeficientes positivos:   {(coeficientes > 0).sum():,}")
        print(f"  Coeficientes negativos:   {(coeficientes < 0).sum():,}")
        print(f"  Coeficiente máximo:       {coeficientes.max():+.4f}")
        print(f"  Coeficiente mínimo:       {coeficientes.min():+.4f}")
        print(f"  Desvio padrão:            {coeficientes.std():.4f}")

    except Exception as e:
        print(f"\n⚠️  Erro ao analisar coeficientes: {str(e)}")

    # 10. ANÁLISE DE CASOS LIMITE (próximos do threshold 0.5)
    print("\n🔬 ANÁLISE DE CASOS LIMITE (probabilidade próxima de 0.5):")

    casos_limite = df_analise[(df_analise['prob_ofensivo'] > 0.45) &
                               (df_analise['prob_ofensivo'] < 0.55)]

    print(f"  Total de casos limite: {len(casos_limite)} ({len(casos_limite)/len(df_analise)*100:.1f}%)")

    if len(casos_limite) > 0:
        acertos_limite = casos_limite[casos_limite['tipo'] == 'Correto']
        erros_limite = casos_limite[casos_limite['tipo'] != 'Correto']

        print(f"  Acertos em casos limite: {len(acertos_limite)} ({len(acertos_limite)/len(casos_limite)*100:.1f}%)")
        print(f"  Erros em casos limite:   {len(erros_limite)} ({len(erros_limite)/len(casos_limite)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 70)

    # Retornar dicionário com resultados
    resultados = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'coeficientes': coeficientes if 'coeficientes' in locals() else None,
        'feature_names': feature_names if 'feature_names' in locals() else None,
        'intercepto': intercepto if 'intercepto' in locals() else None,
        'df_analise': df_analise,
        'num_fp': num_fp,
        'num_fn': num_fn,
        'confianca_corretos': conf_corretos,
        'confianca_erros': conf_erros
    }

    return resultados

# %% [markdown]
# # **Análise da CNN**

# %% [markdown]
# Analisa o desempenho da CNN com métricas detalhadas e visualizações.
#     
#     Parâmetros:
#     -----------
#     modelo : Sequential
#         Modelo CNN treinado (Keras/TensorFlow)
#     history : History
#         Objeto retornado pelo model.fit() contendo histórico de treinamento
#     X_train, X_test : array
#         Features de treino e teste (sequências padronizadas)
#     y_train, y_test : array
#         Labels de treino e teste
#     df_test : DataFrame
#         DataFrame com os dados de teste (para análise qualitativa)
#     Retorna:
#     --------
#     dict : Dicionário com métricas e resultados da análise

# %%
# =============================================================================
# FUNÇÃO DE ANÁLISE DA CNN
# =============================================================================

def analisar_cnn(modelo, history, X_train, X_test, y_train, y_test, df_test):

    print("\n" + "=" * 70)
    print("ANÁLISE DETALHADA - CNN (CONVOLUTIONAL NEURAL NETWORK)")
    print("=" * 70)

    # 1. PREDIÇÕES
    print("\n[1/8] Gerando predições...")
    y_pred_proba = modelo.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype("int32").flatten()

    # 2. MÉTRICAS PRINCIPAIS
    print("\n📊 MÉTRICAS DE DESEMPENHO:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precisão:  {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 3. MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 MATRIZ DE CONFUSÃO:")
    print(f"                  Predito 0  Predito 1")
    print(f"  Real 0 (Não-Of):    {cm[0,0]:6}    {cm[0,1]:6}")
    print(f"  Real 1 (Ofensivo):  {cm[1,0]:6}    {cm[1,1]:6}")

    vn, fp, fn, vp = cm.ravel()
    print(f"\n  Verdadeiros Negativos (VN): {vn}")
    print(f"  Falsos Positivos (FP):      {fp}")
    print(f"  Falsos Negativos (FN):      {fn}")
    print(f"  Verdadeiros Positivos (VP): {vp}")

    # 4. RELATÓRIO DETALHADO
    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred,
                                target_names=['Não-Ofensivo', 'Ofensivo'],
                                digits=4))

    # 5. ANÁLISE DO HISTÓRICO DE TREINAMENTO
    print("\n📚 ANÁLISE DO HISTÓRICO DE TREINAMENTO:")

    # Extrair dados do histórico
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    num_epochs = len(train_loss)

    print(f"\n  Número de épocas treinadas: {num_epochs}")
    print(f"\n  Época final:")
    print(f"    Loss (treino):      {train_loss[-1]:.4f}")
    print(f"    Loss (validação):   {val_loss[-1]:.4f}")
    print(f"    Accuracy (treino):  {train_acc[-1]:.4f}")
    print(f"    Accuracy (validação): {val_acc[-1]:.4f}")

    # Melhor época
    best_epoch = np.argmin(val_loss) + 1
    print(f"\n  Melhor época (menor val_loss): {best_epoch}")
    print(f"    Loss (validação):   {val_loss[best_epoch-1]:.4f}")
    print(f"    Accuracy (validação): {val_acc[best_epoch-1]:.4f}")

    # 6. ANÁLISE DE OVERFITTING
    print("\n🔍 ANÁLISE DE OVERFITTING/UNDERFITTING:")

    # Diferença entre treino e validação
    diff_loss = train_loss[-1] - val_loss[-1]
    diff_acc = train_acc[-1] - val_acc[-1]

    print(f"  Diferença Loss (treino - val):     {diff_loss:+.4f}")
    print(f"  Diferença Accuracy (treino - val): {diff_acc:+.4f}")

    if diff_acc > 0.05:
        print(f"\n  ⚠️  OVERFITTING DETECTADO!")
        print(f"      O modelo está memorizando os dados de treino.")
        print(f"      Sugestões: usar Dropout, Early Stopping, ou mais dados.")
    elif val_acc[-1] < 0.7:
        print(f"\n  ⚠️  POSSÍVEL UNDERFITTING!")
        print(f"      O modelo não está aprendendo bem.")
        print(f"      Sugestões: treinar mais épocas, aumentar complexidade.")
    else:
        print(f"\n  ✓ Modelo bem ajustado (generaliza bem).")

    # 7. ANÁLISE DE PROBABILIDADES
    print("\n🎲 ANÁLISE DAS PROBABILIDADES PREDITAS:")

    proba_classe_1 = y_pred_proba.flatten()

    print(f"\n  Probabilidades da classe OFENSIVO:")
    print(f"    Mínima:  {proba_classe_1.min():.4f}")
    print(f"    Máxima:  {proba_classe_1.max():.4f}")
    print(f"    Média:   {proba_classe_1.mean():.4f}")
    print(f"    Mediana: {np.median(proba_classe_1):.4f}")

    # Distribuição de confiança
    alta_confianca = np.sum((proba_classe_1 > 0.7) | (proba_classe_1 < 0.3))
    media_confianca = np.sum((proba_classe_1 >= 0.3) & (proba_classe_1 <= 0.7))

    print(f"\n  Distribuição de Confiança:")
    print(f"    Alta (p<0.3 ou p>0.7):   {alta_confianca} ({alta_confianca/len(y_test)*100:.1f}%)")
    print(f"    Média (0.3≤p≤0.7):       {media_confianca} ({media_confianca/len(y_test)*100:.1f}%)")

    # 8. ANÁLISE DE ERROS
    print("\n🔍 ANÁLISE DE ERROS:")

    # Criar DataFrame de análise
    df_analise = df_test.copy()
    df_analise['predicao'] = y_pred
    df_analise['prob_ofensivo'] = proba_classe_1
    df_analise['prob_nao_ofensivo'] = 1 - proba_classe_1
    df_analise['confianca'] = np.maximum(proba_classe_1, 1 - proba_classe_1)
    df_analise['label_real'] = y_test

    # Identificar tipos de erro
    df_analise['tipo'] = 'Correto'
    df_analise.loc[(df_analise['label_real'] == 0) &
                   (df_analise['predicao'] == 1), 'tipo'] = 'FP'
    df_analise.loc[(df_analise['label_real'] == 1) &
                   (df_analise['predicao'] == 0), 'tipo'] = 'FN'

    num_fp = (df_analise['tipo'] == 'FP').sum()
    num_fn = (df_analise['tipo'] == 'FN').sum()

    print(f"  Falsos Positivos (FP): {num_fp} ({num_fp/len(df_analise)*100:.2f}%)")
    print(f"  Falsos Negativos (FN): {num_fn} ({num_fn/len(df_analise)*100:.2f}%)")

    # Confiança média
    conf_corretos = df_analise[df_analise['tipo'] == 'Correto']['confianca'].mean()
    conf_erros = df_analise[df_analise['tipo'].isin(['FP', 'FN'])]['confianca'].mean()

    print(f"\n  Confiança média (predições corretas): {conf_corretos:.4f}")
    print(f"  Confiança média (predições erradas):  {conf_erros:.4f}")

    # 9. EXEMPLOS DE FALSOS NEGATIVOS
    print("\n⚠️  EXEMPLOS DE FALSOS NEGATIVOS (FN):")
    print("    (Texto ofensivo classificado como não-ofensivo)")

    fn_samples = df_analise[df_analise['tipo'] == 'FN'].nsmallest(3, 'prob_ofensivo')

    for idx, (i, row) in enumerate(fn_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    P(Não-Ofensivo): {row['prob_nao_ofensivo']:.3f}")
        print(f"    P(Ofensivo):     {row['prob_ofensivo']:.3f}")

    # 10. EXEMPLOS DE FALSOS POSITIVOS
    print("\n⚠️  EXEMPLOS DE FALSOS POSITIVOS (FP):")
    print("    (Texto não-ofensivo classificado como ofensivo)")

    fp_samples = df_analise[df_analise['tipo'] == 'FP'].nlargest(3, 'prob_ofensivo')

    for idx, (i, row) in enumerate(fp_samples.iterrows(), 1):
        texto = row.get('comentario', row.get('cleaned_text', 'N/A'))
        print(f"\n  Exemplo {idx}:")
        print(f"    Texto: '{texto[:100]}...'")
        print(f"    P(Não-Ofensivo): {row['prob_nao_ofensivo']:.3f}")
        print(f"    P(Ofensivo):     {row['prob_ofensivo']:.3f}")

    # 11. INFORMAÇÕES DO MODELO
    print("\n🏗️  ARQUITETURA DO MODELO:")
    print(f"  Total de parâmetros: {modelo.count_params():,}")

    trainable_params = sum([np.prod(w.shape) for w in modelo.trainable_weights])
    print(f"  Parâmetros treináveis: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 70)

    # Retornar resultados
    resultados = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba.flatten(),
        'history': history.history,
        'df_analise': df_analise,
        'num_fp': num_fp,
        'num_fn': num_fn,
        'best_epoch': best_epoch,
        'confianca_corretos': conf_corretos,
        'confianca_erros': conf_erros
    }

    return resultados

# %% [markdown]
# # **Visualização para CNN**

# %% [markdown]
#     Cria visualizações específicas do treinamento da CNN.
#     
#     Parâmetros:
#     -----------
#     history : History
#         Objeto de histórico do treinamento
#     resultados_cnn : dict
#         Resultados da análise da CNN

# %%
# =============================================================================
# FUNÇÃO DE VISUALIZAÇÃO ESPECÍFICA PARA CNN
# =============================================================================

def visualizar_treinamento_cnn(history, resultados_cnn):

    print("\n" + "=" * 70)
    print("VISUALIZAÇÕES DO TREINAMENTO DA CNN")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 10))

    # Extrair dados do histórico
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    # =========================================================================
    # 1. CURVA DE LOSS
    # =========================================================================
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_loss, 'b-o', label='Loss (Treino)', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-s', label='Loss (Validação)', linewidth=2)
    ax1.set_xlabel('Épocas', fontweight='bold')
    ax1.set_ylabel('Loss (Binary Crossentropy)', fontweight='bold')
    ax1.set_title('Evolução da Loss Durante o Treinamento',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Marcar melhor época
    best_epoch = resultados_cnn['best_epoch']
    ax1.axvline(x=best_epoch, color='green', linestyle='--',
                label=f'Melhor Época ({best_epoch})', alpha=0.5)

    # =========================================================================
    # 2. CURVA DE ACCURACY
    # =========================================================================
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, train_acc, 'b-o', label='Accuracy (Treino)', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-s', label='Accuracy (Validação)', linewidth=2)
    ax2.set_xlabel('Épocas', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Evolução da Accuracy Durante o Treinamento',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)

    # =========================================================================
    # 3. GAP ENTRE TREINO E VALIDAÇÃO (Overfitting)
    # =========================================================================
    ax3 = plt.subplot(2, 3, 3)

    gap_loss = [train - val for train, val in zip(train_loss, val_loss)]
    gap_acc = [train - val for train, val in zip(train_acc, val_acc)]

    ax3.plot(epochs, gap_loss, 'purple', label='Gap Loss', linewidth=2, marker='o')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.fill_between(epochs, gap_loss, 0, alpha=0.3, color='purple')
    ax3.set_xlabel('Épocas', fontweight='bold')
    ax3.set_ylabel('Diferença (Treino - Validação)', fontweight='bold')
    ax3.set_title('Análise de Overfitting (Gap)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # =========================================================================
    # 4. MATRIZ DE CONFUSÃO
    # =========================================================================
    ax4 = plt.subplot(2, 3, 4)
    sns.heatmap(resultados_cnn['confusion_matrix'], annot=True, fmt='d',
                cmap='YlOrRd', cbar=False, ax=ax4,
                xticklabels=['Não-Of', 'Ofensivo'],
                yticklabels=['Não-Of', 'Ofensivo'])
    ax4.set_title('Matriz de Confusão - CNN', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Real', fontweight='bold')
    ax4.set_xlabel('Predito', fontweight='bold')

    # =========================================================================
    # 5. DISTRIBUIÇÃO DE PROBABILIDADES
    # =========================================================================
    ax5 = plt.subplot(2, 3, 5)

    df_analise = resultados_cnn['df_analise']
    proba_real0 = df_analise[df_analise['label_real'] == 0]['prob_ofensivo']
    proba_real1 = df_analise[df_analise['label_real'] == 1]['prob_ofensivo']

    ax5.hist(proba_real0, bins=30, alpha=0.6, label='Real: Não-Ofensivo',
             color='#3498db', edgecolor='black')
    ax5.hist(proba_real1, bins=30, alpha=0.6, label='Real: Ofensivo',
             color='#e74c3c', edgecolor='black')
    ax5.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                label='Threshold (0.5)')
    ax5.set_xlabel('P(Ofensivo)', fontweight='bold')
    ax5.set_ylabel('Frequência', fontweight='bold')
    ax5.set_title('Distribuição de Probabilidades Preditas',
                  fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # =========================================================================
    # 6. MÉTRICAS POR ÉPOCA
    # =========================================================================
    ax6 = plt.subplot(2, 3, 6)

    # Tabela de métricas finais
    metricas_texto = f"""
    MÉTRICAS FINAIS:

    Acurácia:  {resultados_cnn['accuracy']:.4f}
    Precisão:  {resultados_cnn['precision']:.4f}
    Recall:    {resultados_cnn['recall']:.4f}
    F1-Score:  {resultados_cnn['f1_score']:.4f}

    ERROS:
    FP: {resultados_cnn['num_fp']}
    FN: {resultados_cnn['num_fn']}

    MELHOR ÉPOCA: {best_epoch}
    """

    ax6.text(0.1, 0.5, metricas_texto, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.axis('off')

    plt.tight_layout()
    plt.show()

    print("\n✓ Visualizações do treinamento geradas!")

# %% [markdown]
# # Função para analisar e comparar:
# **Modelos SVM x Naivy Bais x Regressão Linear**

# %%
# =============================================================================
# FUNÇÃO DE VISUALIZAÇÃO COMPARATIVA DOS MODELOS
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def visualizar_comparacao_modelos(resultados_nb, resultados_svm, resultados_lr,
                                   y_test, save_plots=False):
    """
    Cria visualizações comparativas entre os três modelos de classificação.

    Parâmetros:
    -----------
    resultados_nb : dict
        Resultados do Naive Bayes
    resultados_svm : dict
        Resultados do SVM
    resultados_lr : dict
        Resultados da Regressão Logística
    y_test : array
        Labels reais do conjunto de teste
    save_plots : bool
        Se True, salva os gráficos em arquivos
    """

    print("\n" + "=" * 70)
    print("GERANDO VISUALIZAÇÕES COMPARATIVAS")
    print("=" * 70)

    # Configurar estilo dos gráficos
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10

    # Criar figura com subplots
    fig = plt.figure(figsize=(18, 12))

    # =========================================================================
    # 1. COMPARAÇÃO DE MÉTRICAS (Gráfico de Barras)
    # =========================================================================
    ax1 = plt.subplot(2, 3, 1)

    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    nb_valores = [resultados_nb['accuracy'], resultados_nb['precision'],
                  resultados_nb['recall'], resultados_nb['f1_score']]
    svm_valores = [resultados_svm['accuracy'], resultados_svm['precision'],
                   resultados_svm['recall'], resultados_svm['f1_score']]
    lr_valores = [resultados_lr['accuracy'], resultados_lr['precision'],
                  resultados_lr['recall'], resultados_lr['f1_score']]

    x = np.arange(len(metricas))
    width = 0.25

    bars1 = ax1.bar(x - width, nb_valores, width, label='Naive Bayes',
                    color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, svm_valores, width, label='SVM Linear',
                    color='#e74c3c', alpha=0.8)
    bars3 = ax1.bar(x + width, lr_valores, width, label='Reg. Logística',
                    color='#2ecc71', alpha=0.8)

    ax1.set_xlabel('Métricas', fontweight='bold')
    ax1.set_ylabel('Valor', fontweight='bold')
    ax1.set_title('Comparação de Métricas de Desempenho', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metricas, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.75, 0.90])
    ax1.grid(axis='y', alpha=0.3)

    # Adicionar valores nas barras
    def adicionar_valores(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    adicionar_valores(bars1)
    adicionar_valores(bars2)
    adicionar_valores(bars3)

    # =========================================================================
    # 2. MATRIZES DE CONFUSÃO (3 subplots)
    # =========================================================================

    # Naive Bayes
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(resultados_nb['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Não-Of', 'Ofensivo'],
                yticklabels=['Não-Of', 'Ofensivo'])
    ax2.set_title('Matriz de Confusão - Naive Bayes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Real', fontweight='bold')
    ax2.set_xlabel('Predito', fontweight='bold')

    # SVM
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(resultados_svm['confusion_matrix'], annot=True, fmt='d',
                cmap='Reds', cbar=False, ax=ax3,
                xticklabels=['Não-Of', 'Ofensivo'],
                yticklabels=['Não-Of', 'Ofensivo'])
    ax3.set_title('Matriz de Confusão - SVM Linear', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Real', fontweight='bold')
    ax3.set_xlabel('Predito', fontweight='bold')

    # Regressão Logística
    ax4 = plt.subplot(2, 3, 4)
    sns.heatmap(resultados_lr['confusion_matrix'], annot=True, fmt='d',
                cmap='Greens', cbar=False, ax=ax4,
                xticklabels=['Não-Of', 'Ofensivo'],
                yticklabels=['Não-Of', 'Ofensivo'])
    ax4.set_title('Matriz de Confusão - Reg. Logística', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Real', fontweight='bold')
    ax4.set_xlabel('Predito', fontweight='bold')

    # =========================================================================
    # 3. COMPARAÇÃO DE ERROS (FP vs FN)
    # =========================================================================
    ax5 = plt.subplot(2, 3, 5)

    modelos = ['Naive\nBayes', 'SVM\nLinear', 'Reg.\nLogística']
    fp_valores = [resultados_nb['num_fp'], resultados_svm['num_fp'],
                  resultados_lr['num_fp']]
    fn_valores = [resultados_nb['num_fn'], resultados_svm['num_fn'],
                  resultados_lr['num_fn']]

    x = np.arange(len(modelos))
    width = 0.35

    bars_fp = ax5.bar(x - width/2, fp_valores, width, label='Falsos Positivos',
                      color='#e67e22', alpha=0.8)
    bars_fn = ax5.bar(x + width/2, fn_valores, width, label='Falsos Negativos',
                      color='#9b59b6', alpha=0.8)

    ax5.set_xlabel('Modelos', fontweight='bold')
    ax5.set_ylabel('Quantidade de Erros', fontweight='bold')
    ax5.set_title('Comparação de Tipos de Erros', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(modelos)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # Adicionar valores
    for bars in [bars_fp, bars_fn]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 4. CURVAS ROC
    # =========================================================================
    ax6 = plt.subplot(2, 3, 6)

    # Calcular curvas ROC para cada modelo
    # Naive Bayes
    fpr_nb, tpr_nb, _ = roc_curve(y_test, resultados_nb['y_pred_proba'][:, 1])
    roc_auc_nb = auc(fpr_nb, tpr_nb)

    # SVM (usar decision_function)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, resultados_svm['decision_scores'])
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    # Regressão Logística
    fpr_lr, tpr_lr, _ = roc_curve(y_test, resultados_lr['y_pred_proba'][:, 1])
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    # Plotar curvas
    ax6.plot(fpr_nb, tpr_nb, color='#3498db', lw=2,
             label=f'Naive Bayes (AUC = {roc_auc_nb:.3f})')
    ax6.plot(fpr_svm, tpr_svm, color='#e74c3c', lw=2,
             label=f'SVM Linear (AUC = {roc_auc_svm:.3f})')
    ax6.plot(fpr_lr, tpr_lr, color='#2ecc71', lw=2,
             label=f'Reg. Logística (AUC = {roc_auc_lr:.3f})')

    # Linha diagonal (classificador aleatório)
    ax6.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Classificador Aleatório')

    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
    ax6.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
    ax6.set_title('Curvas ROC - Comparação', fontsize=12, fontweight='bold')
    ax6.legend(loc="lower right")
    ax6.grid(alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico salvo: comparacao_modelos.png")

    plt.show()

    print("\n✓ Visualizações geradas com sucesso!")


# =============================================================================
# FUNÇÃO DE ANÁLISE DETALHADA DE PROBABILIDADES
# =============================================================================

def visualizar_distribuicao_probabilidades(resultados_nb, resultados_lr, y_test):
    """
    Visualiza a distribuição de probabilidades preditas pelos modelos.

    Parâmetros:
    -----------
    resultados_nb : dict
        Resultados do Naive Bayes
    resultados_lr : dict
        Resultados da Regressão Logística
    y_test : array
        Labels reais do conjunto de teste
    """

    print("\n" + "=" * 70)
    print("ANÁLISE DE DISTRIBUIÇÃO DE PROBABILIDADES")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # =========================================================================
    # 1. DISTRIBUIÇÃO - NAIVE BAYES
    # =========================================================================
    ax1 = axes[0, 0]

    proba_nb_classe1 = resultados_nb['y_pred_proba'][:, 1]

    # Separar por classe real
    proba_nb_real0 = proba_nb_classe1[y_test == 0]
    proba_nb_real1 = proba_nb_classe1[y_test == 1]

    ax1.hist(proba_nb_real0, bins=30, alpha=0.6, label='Real: Não-Ofensivo',
             color='#3498db', edgecolor='black')
    ax1.hist(proba_nb_real1, bins=30, alpha=0.6, label='Real: Ofensivo',
             color='#e74c3c', edgecolor='black')
    ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                label='Threshold (0.5)')
    ax1.set_xlabel('P(Ofensivo)', fontweight='bold')
    ax1.set_ylabel('Frequência', fontweight='bold')
    ax1.set_title('Distribuição de Probabilidades - Naive Bayes',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # =========================================================================
    # 2. DISTRIBUIÇÃO - REGRESSÃO LOGÍSTICA
    # =========================================================================
    ax2 = axes[0, 1]

    proba_lr_classe1 = resultados_lr['y_pred_proba'][:, 1]

    proba_lr_real0 = proba_lr_classe1[y_test == 0]
    proba_lr_real1 = proba_lr_classe1[y_test == 1]

    ax2.hist(proba_lr_real0, bins=30, alpha=0.6, label='Real: Não-Ofensivo',
             color='#3498db', edgecolor='black')
    ax2.hist(proba_lr_real1, bins=30, alpha=0.6, label='Real: Ofensivo',
             color='#e74c3c', edgecolor='black')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                label='Threshold (0.5)')
    ax2.set_xlabel('P(Ofensivo)', fontweight='bold')
    ax2.set_ylabel('Frequência', fontweight='bold')
    ax2.set_title('Distribuição de Probabilidades - Regressão Logística',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # =========================================================================
    # 3. BOX PLOT - COMPARAÇÃO DE CONFIANÇA
    # =========================================================================
    ax3 = axes[1, 0]

    # Preparar dados
    conf_nb = np.maximum(resultados_nb['y_pred_proba'][:, 0],
                         resultados_nb['y_pred_proba'][:, 1])
    conf_lr = np.maximum(resultados_lr['y_pred_proba'][:, 0],
                         resultados_lr['y_pred_proba'][:, 1])

    data_confianca = [conf_nb, conf_lr]
    labels_conf = ['Naive Bayes', 'Reg. Logística']

    bp = ax3.boxplot(data_confianca, labels=labels_conf, patch_artist=True,
                     notch=True, showmeans=True)

    # Colorir as caixas
    colors = ['#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_ylabel('Confiança da Predição', fontweight='bold')
    ax3.set_title('Distribuição de Confiança das Predições',
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # =========================================================================
    # 4. ANÁLISE DE CALIBRAÇÃO
    # =========================================================================
    ax4 = axes[1, 1]

    # Criar bins de probabilidade
    bins = np.linspace(0, 1, 11)

    # Naive Bayes
    bin_indices_nb = np.digitize(proba_nb_classe1, bins) - 1
    bin_sums_nb = np.bincount(bin_indices_nb, weights=y_test,
                               minlength=len(bins)-1)
    bin_counts_nb = np.bincount(bin_indices_nb, minlength=len(bins)-1)
    bin_accuracies_nb = np.divide(bin_sums_nb, bin_counts_nb,
                                   where=bin_counts_nb > 0)

    # Regressão Logística
    bin_indices_lr = np.digitize(proba_lr_classe1, bins) - 1
    bin_sums_lr = np.bincount(bin_indices_lr, weights=y_test,
                              minlength=len(bins)-1)
    bin_counts_lr = np.bincount(bin_indices_lr, minlength=len(bins)-1)
    bin_accuracies_lr = np.divide(bin_sums_lr, bin_counts_lr,
                                   where=bin_counts_lr > 0)

    # Plotar
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax4.plot(bin_centers, bin_accuracies_nb, 'o-', label='Naive Bayes',
             color='#3498db', linewidth=2, markersize=8)
    ax4.plot(bin_centers, bin_accuracies_lr, 's-', label='Reg. Logística',
             color='#2ecc71', linewidth=2, markersize=8)
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado', alpha=0.5)

    ax4.set_xlabel('Probabilidade Predita', fontweight='bold')
    ax4.set_ylabel('Fração de Positivos', fontweight='bold')
    ax4.set_title('Curva de Calibração', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

    print("\n✓ Análise de probabilidades concluída!")


# =============================================================================
# FUNÇÃO DE TABELA RESUMO COM PANDAS (ALTERNATIVA MAIS SIMPLES)
# =============================================================================

def criar_tabela_resumo_pandas(resultados_nb, resultados_svm, resultados_lr):
    """
    Cria uma tabela resumo usando pandas com estilo.
    """

    print("\n" + "=" * 70)
    print("TABELA RESUMO COMPARATIVA")
    print("=" * 70)

    # Extrair valores da matriz de confusão
    cm_nb = resultados_nb['confusion_matrix']
    cm_svm = resultados_svm['confusion_matrix']
    cm_lr = resultados_lr['confusion_matrix']

    # Criar DataFrame
    dados = {
        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score',
                    'VP', 'VN', 'FP', 'FN'],
        'Naive Bayes': [
            f"{resultados_nb['accuracy']:.4f}",
            f"{resultados_nb['precision']:.4f}",
            f"{resultados_nb['recall']:.4f}",
            f"{resultados_nb['f1_score']:.4f}",
            cm_nb[1,1], cm_nb[0,0], cm_nb[0,1], cm_nb[1,0]
        ],
        'SVM Linear': [
            f"{resultados_svm['accuracy']:.4f}",
            f"{resultados_svm['precision']:.4f}",
            f"{resultados_svm['recall']:.4f}",
            f"{resultados_svm['f1_score']:.4f}",
            cm_svm[1,1], cm_svm[0,0], cm_svm[0,1], cm_svm[1,0]
        ],
        'Reg. Logística': [
            f"{resultados_lr['accuracy']:.4f}",
            f"{resultados_lr['precision']:.4f}",
            f"{resultados_lr['recall']:.4f}",
            f"{resultados_lr['f1_score']:.4f}",
            cm_lr[1,1], cm_lr[0,0], cm_lr[0,1], cm_lr[1,0]
        ]
    }

    df_resumo = pd.DataFrame(dados)

    # Exibir tabela formatada
    print("\n")
    print(df_resumo.to_string(index=False))
    print("\n")

    # Criar visualização da tabela
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    # Converter para lista de listas para a tabela matplotlib
    cell_text = []
    for row in range(len(df_resumo)):
        cell_text.append([df_resumo.iloc[row, 0],
                         df_resumo.iloc[row, 1],
                         df_resumo.iloc[row, 2],
                         df_resumo.iloc[row, 3]])

    # Criar tabela
    tabela = ax.table(cellText=cell_text,
                     colLabels=['Métrica', 'Naive Bayes', 'SVM Linear', 'Reg. Logística'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2])

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1, 2)

    # Estilizar
    for (i, j), cell in tabela.get_celld().items():
        if i == 0:  # Cabeçalho
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        elif i <= 4:  # Métricas
            if j == 0:
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#e8f5e9')
        else:  # Matriz de confusão
            if j == 0:
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#fff3e0')

    plt.title('Resumo Comparativo dos Modelos', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    print("✓ Tabela resumo criada!")

    return df_resumo

# %%
# =============================================================================
# EXECUÇÃO DO PIPELINE E EXTRAÇÃO DE FEATURES LINGUÍSTICAS
# =============================================================================
import spacy
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_sm"], check=True)
    nlp = spacy.load('pt_core_news_sm')


if __name__ == "__main__":
    # Ajustar o caminho conforme necessário
    #CAMINHO_CSV = 'D:/Lorran/UC/3° Ano/PLN/projeto/HateBR.csv'  # ou 'D:/Lorran/UC/3º Ano/PLN/projeto/HateBR.csv'
    CAMINHO_CSV = '/content/sample_data/HateBR.csv'

    # Executar o pipeline principal para carregar, pré-processar e obter vocabulários
    # O pipeline agora retorna (df_resultado_regras, classificador_regras, df_processado_basico)
    print("\n" + "=" * 70)
    print("EXECUÇÃO DO PIPELINE PRINCIPAL (Etapas Iniciais)")
    print("=" * 70)
    df_resultado_regras, classificador_regras, df = executar_pipeline(CAMINHO_CSV)

    # =============================================================================
    # REALIZAR ANÁLISES LINGUÍSTICAS E EXTRAIR FEATURES (FORA DO PIPELINE)
    # =============================================================================
    print("\n" + "=" * 70)
    print("REALIZANDO ANÁLISES LINGUÍSTICAS E EXTRAINDO FEATURES")
    print("=" * 70)

    # Re-executar as análises linguísticas no df_processado_basico
    # Analisar Pós Tagging (precisa de 'tokens')
    print("\n"+"Analisando POS Tagging")
    analisar_pos_tagging(df) # O retorno é impresso, não modifica o df

    # Analisar entidades nomeadas (precisa de 'comentario')
    print("\n"+"Analisando Entidades Nomeadas")
    analisar_entidades(df) # O retorno é impresso, não modifica o df

    # Analisar sentimento (precisa de 'comentario')
    print("\n"+"Analisando Sentimento")
    # Esta função retorna o df modificado com a coluna 'sentimento'
    df_processado_com_sentimento = analisar_sentimento(df.copy()) # .copy() para evitar SettingWithCopyWarning


    # =============================================================================
    # APLICAR FUNÇÕES DE EXTRAÇÃO DE FEATURES LINGUÍSTICAS
    # =============================================================================

    print("\n" + "=" * 70)
    print("APLICANDO FUNÇÕES DE EXTRAÇÃO DE FEATURES LINGUÍSTICAS")
    print("=" * 70)

    # Aplicar a extração de features de POS tagging usando o DataFrame processado
    # Como a função extract_pos_features espera tokens, usamos a coluna 'tokens' do df_processado_basico
    pos_features_list = df['tokens'].apply(extract_pos_features).tolist()
    pos_features_df = pd.DataFrame(pos_features_list)

    # Aplicar a extração de features de NER usando o DataFrame processado
    # A função extract_ner_features espera o texto original e o objeto nlp
    ner_features_list = df['comentario'].apply(lambda x: extract_ner_features(x, nlp)).tolist()
    ner_features_df = pd.DataFrame(ner_features_list)

    # Aplicar a extração de features de sentimento usando o DataFrame processado
    # A função extract_sentiment_features espera o texto original
    sentiment_features_list = df['comentario'].apply(extract_sentiment_features).tolist()
    sentiment_features_df = pd.DataFrame(sentiment_features_list)

    # Concatenar as novas features ao DataFrame original processado
    # Começamos com o df_processado_basico e adicionamos as novas features
    # Removemos colunas que não serão usadas nos modelos de ML baseados em features combinadas
    df_with_linguistic_features = pd.concat([
        df.drop(columns=[
            'tokens', 'cleaned_tokens', 'num_tokens', # Colunas de processamento que não são features diretas
            'previsao', 'confianca', 'regras_ativadas', 'tipo_erro', # Colunas do classificador de regras
            'cleaned_text', 'sentimento' # 'sentimento' será substituído pela feature extraída
        ], errors='ignore'),
        pos_features_df,
        ner_features_df,
        sentiment_features_df
    ], axis=1)

    print("\nNovas features linguísticas adicionadas ao DataFrame.")
    print(f"Shape do DataFrame com features linguísticas: {df_with_linguistic_features.shape}")
    display(df_with_linguistic_features.head())

    # Salvar resultados
    df_with_linguistic_features.to_csv('df_com_features_linguisticas.csv', index=False)

# %% [markdown]
# META 2: Desenvolvimento de um sistema de classificação de texto, com recurso a métodos
# de AC

# %% [markdown]
# ## Criar funções de extração de features linguísticas
# 
# ### Subtask:
# Desenvolver funções que extraiam features numéricas ou categóricas a partir dos resultados da análise de POS tagging, NER e sentimento para cada comentário.

# %%
# =============================================================================
# APLICAR FUNÇÕES DE EXTRAÇÃO DE FEATURES LINGUÍSTICAS
# =============================================================================

print("\n" + "=" * 70)
print("APLICANDO FUNÇÕES DE EXTRAÇÃO DE FEATURES LINGUÍSTICAS")
print("=" * 70)

# Certifique-se de que o modelo spaCy para português está carregado
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    print("Modelo spaCy 'pt_core_news_sm' não encontrado. Baixando...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_sm"], check=True)
    nlp = spacy.load('pt_core_news_sm')


# Aplicar a extração de features de POS tagging usando o DataFrame processado
# Como a função extract_pos_features espera tokens, usamos a coluna 'tokens' do df_processed
pos_features_list = df['tokens'].apply(extract_pos_features).tolist()
pos_features_df = pd.DataFrame(pos_features_list)

# Aplicar a extração de features de NER usando o DataFrame processado
# A função extract_ner_features espera o texto original e o objeto nlp
ner_features_list = df['comentario'].apply(lambda x: extract_ner_features(x, nlp)).tolist()
ner_features_df = pd.DataFrame(ner_features_list)

# Aplicar a extração de features de sentimento usando o DataFrame processado
# A função extract_sentiment_features espera o texto original
sentiment_features_list =df['comentario'].apply(extract_sentiment_features).tolist()
sentiment_features_df = pd.DataFrame(sentiment_features_list)

# Concatenar as novas features ao DataFrame original processado
# Mantemos as colunas essenciais do DataFrame processado e adicionamos as novas features
# Removendo colunas que não são mais necessárias ou serão substituídas/combinadas
df_with_linguistic_features = pd.concat([df.drop(columns=['tokens', 'cleaned_tokens', 'num_tokens',
                                                         'previsao', 'confianca', 'regras_ativadas',
                                                         'tipo_erro', 'cleaned_text', 'sentimento'], errors='ignore'), # Use errors='ignore' caso a coluna não exista
                                         pos_features_df,
                                         ner_features_df,
                                         sentiment_features_df], axis=1)

print("\nNovas features linguísticas adicionadas ao DataFrame.")
print(f"Shape do DataFrame com features linguísticas: {df_with_linguistic_features.shape}")
display(df_with_linguistic_features.head())

# %%
# =============================================================================
# COMBINAR FEATURES LINGUÍSTICAS COM TF-IDF
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

print("\n" + "=" * 70)
print("COMBINANDO FEATURES LINGUÍSTICAS COM TF-IDF")
print("=" * 70)

# 1. Gerar features TF-IDF a partir do texto limpo
# Reutilizamos o código anterior de vetorização
if 'cleaned_text' not in df.columns:
    df['cleaned_text'] = df['cleaned_tokens'].apply(lambda x: ' '.join(x))


tfidf = TfidfVectorizer(max_features=5000) # Manter o mesmo limite de features
X_tfidf = tfidf.fit_transform(df['cleaned_text']) # Usar o df_processado_basico que tem 'cleaned_text'

print(f"\nShape da matriz TF-IDF: {X_tfidf.shape}")

# 2. Converter as features linguísticas (DataFrame) para formato esparso, se necessário,
# ou garantir que estão em um formato compatível para concatenação.
# As features linguísticas são numéricas, podemos convertê-las para um array numpy
# e depois para uma matriz esparsa para compatibilidade com X_tfidf.
# Certifique-se de que df_with_linguistic_features contém as colunas corretas antes de dropar
cols_to_drop = [
    'id', 'comentario', 'anotator1', 'anotator2', 'anotator3', 'label_final',
    'links_post', 'account_post' # Remover colunas não-feature
]
# Drop columns only if they exist in the DataFrame
cols_to_drop_existing = [col for col in cols_to_drop if col in df_with_linguistic_features.columns]


X_linguistic = df_with_linguistic_features.drop(columns=cols_to_drop_existing).values # Converter para array NumPy

# Converter o array NumPy para formato esparso (CSR matrix)
X_linguistic_sparse = sp.csr_matrix(X_linguistic)

print(f"Shape da matriz de features linguísticas: {X_linguistic_sparse.shape}")

# 3. Combinar as matrizes de features
# Concatenação horizontal (axis=1)
X_combined = sp.hstack([X_tfidf, X_linguistic_sparse])

# Definir o target variable (y) - usando o label_final do df_with_linguistic_features
y_combined = df_with_linguistic_features['label_final']

print(f"Shape do vetor de features combinado: {X_combined.shape}")
print(f"Shape do vetor target (y): {y_combined.shape}")

print("\nFeatures combinadas criadas: X_combined e y_combined.")

# %% [markdown]
# Primeiro modelo: Nayve Bayes

# %%
# =============================================================================
# IMPLEMENTAÇÃO DO CLASSIFICADOR NAIVE BAYES
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("\n" + "=" * 70)
print("IMPLEMENTANDO CLASSIFICADOR NAIVE BAYES")
print("=" * 70)

# 1. Preparar os dados (Vetorização com TF-IDF)
# Usaremos os comentários originais para o TF-IDF, pois a limpeza já foi feita
# para a análise linguística, mas para o TF-IDF precisamos do texto para contexto.
# No entanto, para consistência com o pipeline que já utiliza tokens limpos
# e para evitar re-tokenização/limpeza aqui, vamos unir os 'cleaned_tokens'
# de volta em strings para o TF-IDF.

df['cleaned_text'] = df['cleaned_tokens'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer(max_features=5000) # Limitando para um número razoável de features
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label_final']

print(f"\nShape da matriz TF-IDF: {X.shape}")

# 2. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Shape do conjunto de treino: {X_train.shape}, {y_train.shape}")
print(f"Shape do conjunto de teste: {X_test.shape}, {y_test.shape}")

# 3. Treinar o modelo Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

print("\nModelo Naive Bayes treinado.")

# 4. Avaliar o desempenho do modelo
y_pred_nb = model_nb.predict(X_test)

print("\n📊 MÉTRICAS (Naive Bayes):")
print(f"  Acurácia:  {accuracy_score(y_test, y_pred_nb):.4f}")
print("\n📈 MATRIZ DE CONFUSÃO (Naive Bayes):")
print(confusion_matrix(y_test, y_pred_nb))
print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO (Naive Bayes):")
print(classification_report(y_test, y_pred_nb, target_names=['Não-Ofensivo', 'Ofensivo']))

print("\n" + "=" * 70)
print("CLASSIFICADOR NAIVE BAYES IMPLEMENTADO E AVALIADO")
print("=" * 70)


# %%
# Preparar DataFrame de teste para análise qualitativa
df_test = df.iloc[y_test.index].reset_index(drop=True)

# Chamar a função de análise
resultados = analisar_naive_bayes(
        modelo=model_nb,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tfidf_vectorizer=tfidf,
        df_test=df_test
    )

# %% [markdown]
# Segundo Modelo: SVM

# %%
# =============================================================================
# IMPLEMENTAÇÃO DO CLASSIFICADOR SVM
# =============================================================================

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("\n" + "=" * 70)
print("IMPLEMENTANDO CLASSIFICADOR SVM")
print("=" * 70)

# 1. Treinar o modelo SVM
# Usaremos um kernel linear, que geralmente funciona bem para classificação de texto esparsa
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

print("\nModelo SVM treinado.")

# 2. Avaliar o desempenho do modelo
y_pred_svm = model_svm.predict(X_test)

print("\n📊 MÉTRICAS (SVM):")
print(f"  Acurácia:  {accuracy_score(y_test, y_pred_svm):.4f}")
print("\n📈 MATRIZ DE CONFUSÃO (SVM):")
print(confusion_matrix(y_test, y_pred_svm))
print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO (SVM):")
print(classification_report(y_test, y_pred_svm, target_names=['Não-Ofensivo', 'Ofensivo']))

print("\n" + "=" * 70)
print("CLASSIFICADOR SVM IMPLEMENTADO E AVALIADO")
print("=" * 70)

# %%
print("\n Analisando resultados do SVM...")
resultados_svm = analisar_svm(
        modelo=model_svm,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tfidf_vectorizer=tfidf,
        df_test=df_test
    )

# %% [markdown]
# Terceiro modelo: Regressão Linear

# %%
# =============================================================================
# IMPLEMENTAÇÃO DO CLASSIFICADOR REGRESSÃO LOGÍSTICA
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("\n" + "=" * 70)
print("IMPLEMENTANDO CLASSIFICADOR REGRESSÃO LOGÍSTICA")
print("=" * 70)

# 1. Treinar o modelo Regressão Logística
# Usaremos um solver adequado para dados esparsos e de alta dimensão
model_lr = LogisticRegression(solver='liblinear', random_state=42)
model_lr.fit(X_train, y_train)

print("\nModelo Regressão Logística treinado.")

# 2. Avaliar o desempenho do modelo
y_pred_lr = model_lr.predict(X_test)

print("\n📊 MÉTRICAS (Regressão Logística):")
print(f"  Acurácia:  {accuracy_score(y_test, y_pred_lr):.4f}")
print("\n📈 MATRIZ DE CONFUSÃO (Regressão Logística):")
print(confusion_matrix(y_test, y_pred_lr))
print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO (Regressão Logística):")
print(classification_report(y_test, y_pred_lr, target_names=['Não-Ofensivo', 'Ofensivo']))

print("\n" + "=" * 70)
print("CLASSIFICADOR REGRESSÃO LOGÍSTICA IMPLEMENTADO E AVALIADO")
print("=" * 70)

# %%
print("\n Analisando resultados do CLASSIFICADOR REGRESSÃO LOGÍSTICA...")
resultados_lr = analisar_regressao_logistica(
        modelo=model_lr,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        tfidf_vectorizer=tfidf,
        df_test=df_test
    )

# %% [markdown]
# # Comparação Dos 3 Modelos ja implementados

# %%
print("COMPARAÇÃO DOS TRÊS MODELOS")
print("=" * 70)

print("\n📊 TABELA COMPARATIVA DE MÉTRICAS:")
print(f"\n{'Métrica':<15} {'Naive Bayes':>15} {'SVM Linear':>15} {'Reg. Logística':>15}")
print("-" * 65)
print(f"{'Acurácia':<15} {resultados['accuracy']:>14.4f} {resultados_svm['accuracy']:>14.4f} {resultados_lr['accuracy']:>14.4f}")
print(f"{'Precisão':<15} {resultados['precision']:>14.4f} {resultados_svm['precision']:>14.4f} {resultados_lr['precision']:>14.4f}")
print(f"{'Recall':<15} {resultados['recall']:>14.4f} {resultados_svm['recall']:>14.4f} {resultados_lr['recall']:>14.4f}")
print(f"{'F1-Score':<15} {resultados['f1_score']:>14.4f} {resultados_svm['f1_score']:>14.4f} {resultados_lr['f1_score']:>14.4f}")
print("-" * 65)

# Determinar melhor modelo
f1_scores = {
    'Naive Bayes': resultados['f1_score'],
    'SVM Linear': resultados_svm['f1_score'],
    'Regressão Logística': resultados_lr['f1_score']
}

melhor_modelo = max(f1_scores, key=f1_scores.get)
print(f"\n🏆 MELHOR MODELO: {melhor_modelo}")
print(f"   F1-Score: {f1_scores[melhor_modelo]:.4f}")

# Ranking
print("\n📈 RANKING POR F1-SCORE:")
for rank, (modelo, score) in enumerate(sorted(f1_scores.items(),
                                                key=lambda x: x[1],
                                                reverse=True), 1):
    print(f"   {rank}º. {modelo:20s}: {score:.4f}")


print("\n" + "=" * 70)
print("ETAPA 7: VISUALIZAÇÕES COMPARATIVAS")
print("=" * 70)

# Gráficos comparativos principais
visualizar_comparacao_modelos(
    resultados_nb=resultados,
    resultados_svm=resultados_svm,
    resultados_lr=resultados_lr,
    y_test=y_test,
    save_plots=True  # Salvar em arquivo
)

# Análise detalhada de probabilidades
visualizar_distribuicao_probabilidades(
    resultados_nb=resultados,
    resultados_lr=resultados_lr,
    y_test=y_test
)


df_resumo = criar_tabela_resumo_pandas(
    resultados_nb=resultados,
    resultados_svm=resultados_svm,
    resultados_lr=resultados_lr
)

# %% [markdown]
# ## Preparação dos dados para cnn
# 
# ### Subtask:
# Tokenizar os textos, converter palavras em sequências numéricas (indexação do vocabulário), e aplicar padding para que todas as sequências tenham o mesmo comprimento.
# 

# %% [markdown]
# **Reasoning**:
# The subtask requires tokenizing the text, converting tokens to sequences, and padding these sequences for consistent length, which can be done using Keras preprocessing utilities.
# 
# 

# %%
# =============================================================================
# 1. PREPARAÇÃO DOS DADOS PARA CNN
# =============================================================================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

print("\n" + "=" * 70)
print("PREPARAÇÃO DOS DADOS PARA CNN")
print("=" * 70)

# Usar a coluna 'cleaned_tokens' que já está no DataFrame 'df'
texts = df['cleaned_tokens'].apply(lambda x: ' '.join(x)).tolist()
labels = df['label_final'].values

# 1. Tokenização e Criação do Vocabulário
# Vamos usar o Tokenizer do Keras para criar um vocabulário e indexar as palavras.
# O número máximo de palavras a serem consideradas (vocabulário) pode ser ajustado.
max_words = 10000 # Definir o tamanho máximo do vocabulário
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>") # Out-of-vocabulary token
tokenizer.fit_on_texts(texts)

# Converter textos em sequências de inteiros
sequences = tokenizer.texts_to_sequences(texts)

# Visualizar algumas informações
word_index = tokenizer.word_index
print(f"\nEncontrados {len(word_index)} tokens únicos.")
print(f"Tamanho do vocabulário (considerando max_words): {min(len(word_index), max_words)}")
print(f"Exemplo de sequência: {sequences[0]}")

# 2. Padding das Sequências
# Definir o comprimento máximo das sequências.
# Podemos usar a média + desvio padrão ou um valor fixo baseado na análise exploratória
# Vamos usar um valor fixo por enquanto, que pode ser ajustado.
maxlen = 100 # Definir o comprimento máximo das sequências

padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

print(f"\nShape das sequências acolchoadas: {padded_sequences.shape}")
print(f"Exemplo de sequência acolchoada: {padded_sequences[0]}")

# 3. Dividir os dados em conjuntos de treino e teste
# Reutilizar a divisão feita para os modelos anteriores para consistência
# X será as sequências acolchoadas, y serão os labels
X_cnn = padded_sequences
y_cnn = labels

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y_cnn, test_size=0.25, random_state=42, stratify=y_cnn
)

print(f"\nShape do conjunto de treino para CNN: {X_train_cnn.shape}, {y_train_cnn.shape}")
print(f"Shape do conjunto de teste para CNN: {X_test_cnn.shape}, {y_test_cnn.shape}")

print("\n" + "=" * 70)
print("PREPARAÇÃO DOS DADOS PARA CNN CONCLUÍDA")
print("=" * 70)

# %%
# =============================================================================
# 2. CRIAÇÃO DA CAMADA DE EMBEDDINGS DE PALAVRAS
# =============================================================================

from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

print("\n" + "=" * 70)
print("CRIAÇÃO DA CAMADA DE EMBEDDINGS")
print("=" * 70)

# Definir as dimensões do embedding
# Este é um hiperparâmetro que pode ser ajustado. Valores comuns são 50, 100, 200, 300.
embedding_dim = 100

# O tamanho do vocabulário é max_words + 1 (para incluir o token OOV e o padding)
vocab_size = max_words + 1

# Criar a camada de Embedding
# input_dim: tamanho do vocabulário
# output_dim: dimensão dos embeddings
# input_length: comprimento máximo das sequências (maxlen definido anteriormente)
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=maxlen,
                            name="embedding_layer")

print(f"\nCamada de Embedding criada com input_dim={vocab_size}, output_dim={embedding_dim}, input_length={maxlen}")

# Podemos visualizar a forma da saída desta camada se a adicionarmos a um modelo Sequential simples
# Isso é apenas para demonstração
model_test_embedding = Sequential([embedding_layer])
model_test_embedding.build(input_shape=(None, maxlen)) # None para batch size
print("\nShape de saída da camada de Embedding (exemplo):")
print(model_test_embedding.output_shape)

print("\n" + "=" * 70)
print("CAMADA DE EMBEDDINGS CRIADA")
print("=" * 70)

# %%
# =============================================================================
# 3. DEFINIÇÃO DA ARQUITETURA DA CNN
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Embedding

print("\n" + "=" * 70)
print("DEFINIÇÃO DA ARQUITETURA DA CNN")
print("=" + "=" * 70)

# A arquitetura da CNN pode ser ajustada (número de filtros, tamanho do kernel, etc.)
filters = 128 # Número de filtros convolucionais
kernel_size = 5 # Tamanho do kernel da convolução (ex: 3, 4, 5 palavras)

model_cnn = Sequential()

# Adicionar a camada de Embedding criada anteriormente
model_cnn.add(embedding_layer) # Reutiliza a camada de embedding definida no passo 2

# Camada Convolucional 1D
# Aplica filtros sobre a sequência de embeddings para extrair features locais
model_cnn.add(Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu')) # Função de ativação ReLU

# Camada de Pooling Global
# Reduz cada mapa de features para um único valor, capturando as features mais importantes
model_cnn.add(GlobalMaxPooling1D())

# Camadas Densas (Fully Connected) para Classificação
model_cnn.add(Dense(64, activation='relu')) # Camada oculta com 64 neurônios e ativação ReLU
model_cnn.add(Dropout(0.5)) # Camada de Dropout para regularização (ajuda a evitar overfitting)

# Camada de Saída
# Uma única unidade com ativação sigmoide para classificação binária (ofensivo/não-ofensivo)
model_cnn.add(Dense(1, activation='sigmoid'))

# Visualizar a arquitetura do modelo
print("\nArquitetura do Modelo CNN:")
model_cnn.summary()

print("\n" + "=" * 70)
print("ARQUITETURA DA CNN DEFINIDA")
print("=" + "=" * 70)

# %%
# =============================================================================
# 4. COMPILAÇÃO E TREINAMENTO DA CNN
# =============================================================================

from tensorflow.keras.optimizers import Adam

print("\n" + "=" * 70)
print("COMPILAÇÃO E TREINAMENTO DA CNN")
print("=" + "=" * 70)

# 1. Compilar o Modelo
# Usaremos o otimizador Adam, função de perda binary_crossentropy (para classificação binária)
# e acompanharemos a acurácia.
model_cnn.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

print("\nModelo CNN compilado.")

# 2. Treinar o Modelo
# Definir o número de épocas e o tamanho do batch.
# Esses são hiperparâmetros que podem ser ajustados.
epochs = 10  # Número de vezes que o modelo verá todo o conjunto de treino
batch_size = 32 # Número de amostras por atualização de gradiente

print(f"\nIniciando treinamento por {epochs} épocas com batch size {batch_size}...")

# Usamos os dados de treino e teste preparados para a CNN (X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn)
history = model_cnn.fit(X_train_cnn, y_train_cnn,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_cnn, y_test_cnn),
                        verbose=1) # verbose=1 mostra o progresso do treinamento

print("\nTreinamento concluído.")

# O objeto 'history' contém informações sobre o treinamento (perda e métricas por época)
# Você pode usar history.history para acessar esses dados e plotá-los posteriormente.

print("\n" + "=" * 70)
print("COMPILAÇÃO E TREINAMENTO DA CNN CONCLUÍDOS")
print("=" + "=" * 70)

# %%
# =============================================================================
# 5. AVALIAÇÃO DA CNN
# =============================================================================

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("\n" + "=" * 70)
print("AVALIAÇÃO DA CNN")
print("=" + "=" * 70)

# 1. Fazer previsões no conjunto de teste
# A saída da camada final é uma probabilidade (devido à ativação sigmoide).
# Precisamos converter essas probabilidades em classes binárias (0 ou 1).
y_pred_prob_cnn = model_cnn.predict(X_test_cnn)
y_pred_cnn = (y_pred_prob_cnn > 0.5).astype("int32") # Converter probabilidades para classes (0 ou 1)

# 2. Avaliar o desempenho do modelo
accuracy_cnn = accuracy_score(y_test_cnn, y_pred_cnn)
cm_cnn = confusion_matrix(y_test_cnn, y_pred_cnn)
report_cnn = classification_report(y_test_cnn, y_pred_cnn, target_names=['Não-Ofensivo', 'Ofensivo'])

# Exibir resultados
print("\n📊 MÉTRICAS (CNN):")
print(f"  Acurácia:  {accuracy_cnn:.4f}")

print("\n📈 MATRIZ DE CONFUSÃO (CNN):")
print(cm_cnn)

print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO (CNN):")
print(report_cnn)

print("\n" + "=" * 70)
print("AVALIAÇÃO DA CNN CONCLUÍDA")
print("=" + "=" * 70)

# %%
print("\n[6.2] Analisando resultados da CNN...")
resultados_cnn = analisar_cnn(
    modelo=model_cnn,
    history=history,
    X_train=X_train_cnn,
    X_test=X_test_cnn,
    y_train=y_train_cnn,
    y_test=y_test_cnn,
    df_test=df_test
)

# Visualizações específicas da CNN
visualizar_treinamento_cnn(history, resultados_cnn)

# =========================================================================
# 7. COMPARAÇÃO DE TODOS OS MODELOS (incluindo CNN)
# =========================================================================
print("\n" + "=" * 70)
print("ETAPA 7: COMPARAÇÃO DE TODOS OS MODELOS")
print("=" * 70)

print("\n📊 TABELA COMPARATIVA COMPLETA:")
print(f"\n{'Métrica':<15} {'Naive Bayes':>13} {'SVM':>13} {'Reg. Log.':>13} {'CNN':>13}")
print("-" * 70)
print(f"{'Acurácia':<15} {resultados['accuracy']:>12.4f} {resultados_svm['accuracy']:>12.4f} {resultados_lr['accuracy']:>12.4f} {resultados_cnn['accuracy']:>12.4f}")
print(f"{'Precisão':<15} {resultados['precision']:>12.4f} {resultados_svm['precision']:>12.4f} {resultados_lr['precision']:>12.4f} {resultados_cnn['precision']:>12.4f}")
print(f"{'Recall':<15} {resultados['recall']:>12.4f} {resultados_svm['recall']:>12.4f} {resultados_lr['recall']:>12.4f} {resultados_cnn['recall']:>12.4f}")
print(f"{'F1-Score':<15} {resultados['f1_score']:>12.4f} {resultados_svm['f1_score']:>12.4f} {resultados_lr['f1_score']:>12.4f} {resultados_cnn['f1_score']:>12.4f}")
print("-" * 70)

# Determinar melhor modelo
f1_scores = {
    'Naive Bayes': resultados['f1_score'],
    'SVM Linear': resultados_svm['f1_score'],
    'Regressão Logística': resultados_lr['f1_score'],
    'CNN': resultados_cnn['f1_score']
}

melhor_modelo = max(f1_scores, key=f1_scores.get)
print(f"\n🏆 MELHOR MODELO: {melhor_modelo}")
print(f"   F1-Score: {f1_scores[melhor_modelo]:.4f}")


