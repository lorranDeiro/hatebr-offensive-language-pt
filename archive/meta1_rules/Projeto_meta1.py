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
    
    for tokens in df[df['label_final'] == 1]['tokens'][:1000]:  # Amostra
        pos_of.extend([tag for word, tag in pos_tag(tokens)])
    
    for tokens in df[df['label_final'] == 0]['tokens'][:1000]:
        pos_nao_of.extend([tag for word, tag in pos_tag(tokens)])
    
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
        freq_of = sum(freq for tag, freq in fdist_pos_of.items() if tag.startswith(pos_type))
        freq_nao_of = sum(freq for tag, freq in fdist_pos_nao_of.items() if tag.startswith(pos_type))
        ratio = freq_of / freq_nao_of if freq_nao_of > 0 else 0
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
    df = preprocessar_corpus(df)
    print("\n" + "=" * 70)
    
    # 3. Analisar frequências
    print("\n" + "=" * 70)
    print("Analisando Frequências")
    frequencias = analisar_frequencias(df)
    print("\n" + "=" * 70)
    
    # 4. Criar vocabulário distintivo
    print("\n" + "=" * 70)
    print("Criando Vocabulário Distintivo")
    vocabularios = criar_vocabulario_distintivo(df)
    print("\n" + "=" * 70)

    print("\n"+"Visualizando Análise Linguística")
    visualizar_analise_linguistica(df, frequencias)
    print("\n" + "=" * 70)

    # 4.1. Analisar Pós Tagging
    print("\n"+"Analisando POS Tagging")
    analisar_pos_tagging(df)
    print("\n" + "=" * 70)

    # 4.2. Analisar entidades nomeadas
    print("\n"+"Analisando Entidades Nomeadas")
    analisar_entidades(df)
    print("\n" + "=" * 70)

    # 4.3. Analisar sentimento
    print("\n"+"Analisando Sentimento")
    df = analisar_sentimento(df)
    print("\n" + "=" * 70)

    # 5. Criar e treinar classificador
    print("\n"+"Criando e Treinando Classificador")
    classificador = ClassificadorRegras(vocabularios)
    print("\n" + "=" * 70)

    # 6. Avaliar
    print("\n"+"Avaliando Classificador")
    df_resultado = avaliar_classificador(df, classificador)
    print("\n" + "=" * 70)

    # 7. Análise de importância das regras
    print("\n"+"Analisando Importância das Regras")
    analise_de_importancia_das_regras(df, vocabularios)
    print("\n" + "=" * 70)

    print("\n"+"Analisando Qualitativamente os Erros")
    analise_qualitativa_erros(df_resultado) 
    

    print("\n" + "=" * 70)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO")
    print("=" * 70)

    
    return df_resultado, classificador

# %%
# =============================================================================
# EXECUÇÃO
# =============================================================================
import spacy
try:
    spacy.load('pt_core_news_sm')
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_sm"], check=True)


if __name__ == "__main__":
    # Ajustar o caminho conforme necessário
    #CAMINHO_CSV = 'D:/Lorran/UC/3° Ano/PLN/projeto/HateBR.csv'  # ou 'D:/Lorran/UC/3º Ano/PLN/projeto/HateBR.csv'
    CAMINHO_CSV = 'D:/Lorran/UC/3° Ano/PLN/projeto/Meta1_final/HateBR.csv'
    
    df_final, classificador = executar_pipeline(CAMINHO_CSV)
    
    # Salvar resultados (opcional)
    # df_final.to_csv('resultados_classificacao.csv', index=False)


