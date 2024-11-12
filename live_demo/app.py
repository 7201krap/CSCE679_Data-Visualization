from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from scipy.stats import kurtosis
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import hmean
from scipy.stats import gmean
from evaluate import load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import pickle
import evaluate
import torch
import time
import nltk
import numpy as np
import torch.nn.functional as F

from flask import Flask, send_from_directory, request, jsonify

"""
    Score Evaluation 
"""
nltk.download("wordnet")
bertscore = load("bertscore")

t5_tokenizer_small = AutoTokenizer.from_pretrained("t5-small")
t5_model_small = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
print("Loaded in t5_small")

t5_tokenizer_base = AutoTokenizer.from_pretrained("t5-base")
t5_model_base = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
print("Loaded in t5_base")


# jhpark: verified that this is a correct way to use this
def calculate_rouge(true_sentence, predicted_sentence):
    # jhpark: rouge1/rouge2 (e.g. rouge1, rouge2): n-gram based scoring.
    # jhpark: rougeL: Longest common subsequence based scoring.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(true_sentence, predicted_sentence)
    return scores


# jhpark: verified that this is a correct way to use this
def calculate_bleu(true_tokens, predicted_tokens):
    """
    * reference for smoothing: A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU, Boxing Chen and Collin Cherry (2014)
    * method1: Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
    * https://www.nltk.org/_modules/nltk/translate/bleu_score.html for more details
    """
    bleu_score = sentence_bleu(
        true_tokens, predicted_tokens, smoothing_function=SmoothingFunction().method1
    )
    return bleu_score


def calculate_probs_from_logits(logits):
    return F.softmax(logits, dim=-1)


def generate_output_with_probs(model, tokenizer, example, top_k=5):
    '''
    - High Kurtosis: An indicator that the data has heavy tails or outliers.
    - Low Kurtosis:  An indicator that the data has thin tails and lacks outliers.
    '''
    inputs = tokenizer.encode(example, return_tensors="pt")
    output_ids = model.generate(inputs, max_new_tokens=np.inf, return_dict_in_generate=True, output_scores=True, return_legacy_cache=False)
    output_tokens = output_ids.sequences[0]
    output_probs = []
    output_kurtosis = []
    decoded_output = []

    # Retrieve probabilities for each token
    for i in range(1, len(output_tokens) - 1):  # Skip the first token and the last token
        # Fetch the logits from output_ids.scores, aligning with output_tokens[1:]
        probs = calculate_probs_from_logits(output_ids.scores[i - 1])[0]  # i-1 to align with `scores` index
        token_id = output_tokens[i]
        token_prob = probs[token_id].item()

        # Compute kurtosis on top probabilities
        top_probs, _ = torch.topk(probs, 1000)
        kurt = kurtosis(top_probs)

        output_probs.append(token_prob)
        output_kurtosis.append(kurt)
        decoded_output.append(tokenizer.decode([token_id], skip_special_tokens=True))
    
    return decoded_output, output_probs, output_kurtosis


"""
    End Score Evaluation
"""

"""
    Web App
"""

app = Flask(__name__, static_url_path="")


# Route to serve index.html
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


"""
    
    Request
        text: str
        model: str
        language: str

    Response
        scores: dict[score_name: str, score: num] # Not implemented yet
        tokens: array[str]
        uq:     array[num]
"""


# Route to handle GET request to /api/analyze
@app.route("/api/analyze", methods=["GET"])
def analyze():
    # Retrieve query parameters from the URL
    usr_text = request.args.get('text')
    usr_model = request.args.get('model')
    usr_language = request.args.get('language')

    input = f"Translate English to {usr_language}: {usr_text}"
    model = None
    tokenizer = None

    if usr_model == "small":
        model = t5_model_small
        tokenizer = t5_tokenizer_small
    elif usr_model == "base":
        model = t5_model_base
        tokenizer = t5_tokenizer_base

    response = {"scores": {}, "tokens": [], "uq": []}

    response["tokens"], response["uq"], token_kurtosis = generate_output_with_probs(model, tokenizer, input)
    response["scores"]["Confidence-G-T"] = gmean(response["uq"])
    response["scores"]["Confidence-A-T"] = np.average(response["uq"])

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)

"""
    End Web App
"""
