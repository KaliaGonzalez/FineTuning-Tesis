"""
Módulo de métricas para evaluación de modelos de generación de texto
Implementa: BLEU, ROUGE, METEOR, BERTScore
"""

import re
from collections import Counter
from typing import List, Tuple
import numpy as np


# ==================== BLEU ====================
def calcular_bleu(referencia: str, hipotesis: str, n_gramas: int = 4) -> dict:
    """
    Calcula BLEU score (0-100)
    Mide n-grama precision entre referencia e hipótesis
    """
    ref_tokens = referencia.lower().split()
    hyp_tokens = hipotesis.lower().split()

    bleu_scores = []

    for n in range(1, n_gramas + 1):
        # Calcular n-gramas
        ref_ngrams = [
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        ]
        hyp_ngrams = [
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        ]

        # Contar coincidencias
        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)

        matches = sum((hyp_counts & ref_counts).values())
        total = max(1, len(hyp_ngrams))

        precision = matches / total if total > 0 else 0
        bleu_scores.append(precision)

    # Aplicar brevity penalty
    brevity_penalty = (
        1.0
        if len(hyp_tokens) >= len(ref_tokens)
        else np.exp(1 - len(ref_tokens) / len(hyp_tokens))
    )
    bleu = brevity_penalty * np.exp(
        np.mean(np.log([s if s > 0 else 1e-16 for s in bleu_scores]))
    )

    return {
        "BLEU": round(bleu * 100, 2),
        "BLEU_1": round(bleu_scores[0] * 100, 2),
        "BLEU_2": round(bleu_scores[1] * 100, 2) if len(bleu_scores) > 1 else 0,
        "BLEU_3": round(bleu_scores[2] * 100, 2) if len(bleu_scores) > 2 else 0,
        "BLEU_4": round(bleu_scores[3] * 100, 2) if len(bleu_scores) > 3 else 0,
    }


# ==================== ROUGE ====================
def calcular_rouge(referencia: str, hipotesis: str) -> dict:
    """
    Calcula ROUGE scores (0-100)
    ROUGE-1: Unigrama overlap
    ROUGE-2: Bigrama overlap
    ROUGE-L: Longest common subsequence
    """
    ref_tokens = referencia.lower().split()
    hyp_tokens = hipotesis.lower().split()

    # ROUGE-1
    ref_unigrams = set(ref_tokens)
    hyp_unigrams = set(hyp_tokens)
    rouge1_recall = (
        len(ref_unigrams & hyp_unigrams) / len(ref_unigrams) if ref_unigrams else 0
    )
    rouge1_precision = (
        len(ref_unigrams & hyp_unigrams) / len(hyp_unigrams) if hyp_unigrams else 0
    )
    rouge1_f = (
        2 * (rouge1_precision * rouge1_recall) / (rouge1_precision + rouge1_recall)
        if (rouge1_precision + rouge1_recall) > 0
        else 0
    )

    # ROUGE-2
    ref_bigrams = [tuple(ref_tokens[i : i + 2]) for i in range(len(ref_tokens) - 1)]
    hyp_bigrams = [tuple(hyp_tokens[i : i + 2]) for i in range(len(hyp_tokens) - 1)]
    ref_bigram_set = set(ref_bigrams)
    hyp_bigram_set = set(hyp_bigrams)
    rouge2_recall = (
        len(ref_bigram_set & hyp_bigram_set) / len(ref_bigram_set)
        if ref_bigram_set
        else 0
    )
    rouge2_precision = (
        len(ref_bigram_set & hyp_bigram_set) / len(hyp_bigram_set)
        if hyp_bigram_set
        else 0
    )
    rouge2_f = (
        2 * (rouge2_precision * rouge2_recall) / (rouge2_precision + rouge2_recall)
        if (rouge2_precision + rouge2_recall) > 0
        else 0
    )

    # ROUGE-L (Longest Common Subsequence)
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = lcs_length(ref_tokens, hyp_tokens)
    rougeL_recall = lcs / len(ref_tokens) if ref_tokens else 0
    rougeL_precision = lcs / len(hyp_tokens) if hyp_tokens else 0
    rougeL_f = (
        2 * (rougeL_precision * rougeL_recall) / (rougeL_precision + rougeL_recall)
        if (rougeL_precision + rougeL_recall) > 0
        else 0
    )

    return {
        "ROUGE_1": round(rouge1_f * 100, 2),
        "ROUGE_2": round(rouge2_f * 100, 2),
        "ROUGE_L": round(rougeL_f * 100, 2),
    }


# ==================== METEOR ====================
def calcular_meteor(referencia: str, hipotesis: str) -> dict:
    """
    Calcula METEOR score (0-100)
    Mide coincidencia de palabras exactas, stem y synonyms
    """
    ref_tokens = referencia.lower().split()
    hyp_tokens = hipotesis.lower().split()

    # Coincidencias exactas
    exact_matches = sum(1 for token in hyp_tokens if token in ref_tokens)

    # Coincidencias por prefijo (stemming simple)
    stem_matches = 0
    for hyp_token in hyp_tokens:
        for ref_token in ref_tokens:
            if hyp_token.startswith(ref_token[:3]) or ref_token.startswith(
                hyp_token[:3]
            ):
                stem_matches += 1
                break

    total_matches = (
        exact_matches + stem_matches * 0.5
    )  # Pesar menos las coincidencias de stem

    precision = total_matches / len(hyp_tokens) if hyp_tokens else 0
    recall = total_matches / len(ref_tokens) if ref_tokens else 0

    meteor = 0
    if precision + recall > 0:
        f_score = (precision * recall) / (0.9 * precision + 0.1 * recall)
        meteor = f_score

    return {"METEOR": round(meteor * 100, 2)}


# ==================== BERT SCORE (Simplificado) ====================
def calcular_bertscore_simple(referencia: str, hipotesis: str) -> dict:
    """
    BERTScore simplificado basado en similitud de tokens
    (Para BERTScore completo necesitas transformers y bert-score library)
    """
    ref_tokens = set(referencia.lower().split())
    hyp_tokens = set(hipotesis.lower().split())

    # Similitud simple: Jaccard
    intersection = len(ref_tokens & hyp_tokens)
    union = len(ref_tokens | hyp_tokens)

    jaccard = intersection / union if union > 0 else 0

    return {
        "SemanticSim": round(jaccard * 100, 2),
    }


# ==================== FUNCIÓN GENERAL ====================
def calcular_todas_metricas(referencia: str, hipotesis: str) -> dict:
    """
    Calcula TODAS las métricas y retorna un diccionario consolidado
    """
    metricas = {}

    # BLEU
    metricas.update(calcular_bleu(referencia, hipotesis))

    # ROUGE
    metricas.update(calcular_rouge(referencia, hipotesis))

    # METEOR
    metricas.update(calcular_meteor(referencia, hipotesis))

    # BERTScore simple
    metricas.update(calcular_bertscore_simple(referencia, hipotesis))

    return metricas
