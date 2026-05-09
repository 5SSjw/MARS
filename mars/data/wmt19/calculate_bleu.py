#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import argparse

import math
from collections import Counter

def simple_bleu(hypotheses, references):
    """
    Very basic BLEU implementation for fallback.
    Assumes tokenized input or does basic whitespace tokenization.
    Only computes BLEU-4.
    """
    def n_grams(text, n):
        words = text.split()
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

    precisions = []
    for n in range(1, 5):
        match_count = 0
        total_count = 0
        for h, r in zip(hypotheses, references):
            h_ngrams = Counter(n_grams(h, n))
            r_ngrams = Counter(n_grams(r[0], n)) # r is list of refs
            match_count += sum((h_ngrams & r_ngrams).values())
            total_count += sum(h_ngrams.values())
        
        if total_count == 0:
            precisions.append(0)
        else:
            precisions.append(match_count / total_count)

    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    
    # Brevity Penalty
    c = sum(len(h.split()) for h in hypotheses)
    r = sum(len(ref[0].split()) for ref in references) # using first ref length
    if c == 0: return 0.0
    bp = 1.0 if c > r else math.exp(1 - r / c)

    return geo_mean * bp * 100

def simple_chrf(hypotheses, references, beta=2):
    """
    Basic chrF implementation.
    Adjusted to match sacrebleu's handling where possible, but simplified.
    """
    def get_char_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text)-n+1)]

    total_f = 0
    for h, r in zip(hypotheses, references):
        # r is list of refs, take first
        ref = r[0]
        
        # Calculate for 1 to 6-grams
        precisions = []
        recalls = []
        
        for n in range(1, 7):
            h_ngrams = Counter(get_char_ngrams(h, n))
            r_ngrams = Counter(get_char_ngrams(ref, n))
            
            match = sum((h_ngrams & r_ngrams).values())
            h_total = sum(h_ngrams.values())
            r_total = sum(r_ngrams.values())
            
            p = match / h_total if h_total > 0 else 0
            r_score = match / r_total if r_total > 0 else 0
            
            precisions.append(p)
            recalls.append(r_score)
        
        avg_p = sum(precisions) / 6
        avg_r = sum(recalls) / 6
        
        if avg_p + avg_r == 0:
            f = 0
        else:
            f = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
        
        total_f += f
        
    return (total_f / len(hypotheses)) * 100

def install_sacrebleu():
    try:
        import sacrebleu
        return True
    except ImportError:
        print("sacrebleu not found. Installing...", flush=True)
        ret = os.system("pip install sacrebleu")
        if ret != 0:
            print("Failed to install sacrebleu.")
            return False
        return True

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_bleu(reference_file, generated_file, no_sacrebleu=False):
    use_sacrebleu = False
    if not no_sacrebleu:
        use_sacrebleu = install_sacrebleu()
    
    if not use_sacrebleu:
        print("Using simple internal BLEU implementation as fallback.")

    if use_sacrebleu:
        import sacrebleu

    print(f"Loading references from {reference_file}...")
    ref_data = load_jsonl(reference_file)
    references = {}
    for item in ref_data:
        # Assuming 'reference' is a list of strings
        if 'reference' in item and len(item['reference']) > 0:
            references[item['question_id']] = item['reference'][0]

    print(f"Loading generations from {generated_file}...")
    gen_data = load_jsonl(generated_file)
    
    hyps = []
    refs = []
    
    matched_count = 0
    for item in gen_data:
        qid = item.get('question_id')
        if qid is None:
            continue
            
        if qid not in references:
            continue
            
        # Extract generation
        # Expected format: "choices": [{"turns": ["..."]}]
        gen_text = ""
        if 'choices' in item and len(item['choices']) > 0:
            turns = item['choices'][0].get('turns', [])
            if turns:
                gen_text = turns[0]
        
        if not gen_text:
            print(f"Warning: Empty generation for qid {qid}")
            
        hyps.append(gen_text)
        refs.append(references[qid])
        matched_count += 1

    print(f"Matched {matched_count} examples.")
    
    if not hyps:
        print("No valid examples to evaluate.")
        return

    # Calculate BLEU
    if use_sacrebleu:
        import sacrebleu
        # SacreBLEU expects list of hypotheses and list of references (where each ref is a list of references)
        # Since we have 1 reference per example, we wrap it: [ [ref1, ref2, ...] ]
        # Check if tokenizer is needed (e.g. 'zh') but target is EN so default (13a) is OK.
        bleu_score = sacrebleu.corpus_bleu(hyps, [refs])
        print(f"BLEU: {bleu_score.score:.2f}")
        print(f"Signature: {bleu_score}")

        # Calculate chrF
        try:
            chrf_score = sacrebleu.corpus_chrf(hyps, [refs])
            print(f"chrF: {chrf_score.score:.2f}")
            print(f"Signature: {chrf_score}")
        except Exception as e:
            print(f"Error calculating chrF with sacrebleu: {e}")
            score = simple_chrf(hyps, [[r] for r in refs])
            print(f"chrF (Simple Fallback): {score:.2f}")
    else:
        # Fallback
        score = simple_bleu(hyps, [[r] for r in refs])
        print(f"BLEU (Simple Fallback): {score:.2f}")
        
        chrf = simple_chrf(hyps, [[r] for r in refs])
        print(f"chrF (Simple Fallback): {chrf:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Calculate BLEU for EAGLE generations.")
    default_ref = "/work/xinyu/EAGLE/eagle/data_2/wmt19/question.jsonl"
    parser.add_argument("--reference", default=default_ref, help=f"Path to reference (ground truth) jsonl file (default: {default_ref})")
    parser.add_argument("--generated", required=True, help="Path to generated (model output) jsonl file")
    parser.add_argument("--no-sacrebleu", action="store_true", help="Do not use sacrebleu even if installed")
    args = parser.parse_args()

    # Pass the flag to calculate_bleu (need to update function signature)
    calculate_bleu(args.reference, args.generated, no_sacrebleu=args.no_sacrebleu)

if __name__ == "__main__":
    main()
