"""
Heuristic entity extraction: methods + datasets mentioned in a paper abstract
(novelty item 5, extraction step).

DELIBERATELY curated-lexicon + regex, not LLM-based open extraction -- see
hetero_graph.py's module docstring for the full scoping rationale (determinism, zero
API cost, reproducibility). Two independent extraction strategies, unioned:

  1. Lexicon match: word-boundary substring match against curated lists of well-known
     method/model names and dataset names, covering the CS/NLP/CV/biomedical domains
     this project's corpus spans (arXiv, ACL, PubMed). Recall-limited to whatever's in
     the lists below; precision is high because these are unambiguous proper nouns in
     this context.

  2. Acronym-definition pattern: "Some Multi Word Name (ACRONYM)" -> ACRONYM. Catches
     methods/datasets NOT in the curated lists, at the cost of also catching acronyms
     for things that are neither (e.g. "Root Mean Squared Error (RMSE)" -- a metric,
     not a method or dataset). Accepted as noise in the "method" bucket rather than
     trying to disambiguate metric-vs-method acronyms with more regex, which would not
     meaningfully improve precision at this scope.

Nothing in this module executes against real project data by itself -- `extract_entities`
is a pure function, called from hetero_graph.py's build step.
"""
from __future__ import annotations

import re
from typing import Dict, Set

METHOD_LEXICON: Set[str] = {
    # architectures / model families
    "Transformer", "BERT", "RoBERTa", "DistilBERT", "ALBERT", "XLNet", "ELMo", "GPT",
    "GPT-2", "GPT-3", "GPT-4", "T5", "BART", "ELECTRA", "DeBERTa", "ERNIE",
    "ResNet", "VGG", "AlexNet", "Inception", "DenseNet", "EfficientNet", "MobileNet",
    "U-Net", "YOLO", "Faster R-CNN", "Mask R-CNN", "ViT", "Swin Transformer",
    "LSTM", "BiLSTM", "GRU", "RNN", "CNN", "Seq2Seq", "Encoder-Decoder",
    # graph learning
    "GCN", "GAT", "GraphSAGE", "R-GCN", "Graph Neural Network", "Graph Convolutional Network",
    # generative
    "GAN", "VAE", "Autoencoder", "Diffusion Model", "Stable Diffusion", "DDPM",
    # embeddings
    "Word2Vec", "GloVe", "FastText",
    # attention / transformer internals
    "Attention Mechanism", "Self-Attention", "Multi-Head Attention", "Cross-Attention",
    # RL
    "Reinforcement Learning", "Q-Learning", "PPO", "DQN", "A3C", "Actor-Critic",
    # classical ML
    "Random Forest", "XGBoost", "LightGBM", "SVM", "k-NN", "Naive Bayes",
    "Logistic Regression", "k-means", "PCA", "t-SNE", "UMAP",
    # optimization
    "Adam", "SGD", "AdamW",
    # self-/contrastive/transfer learning
    "Contrastive Learning", "SimCLR", "MoCo", "BYOL", "CLIP",
    "Knowledge Distillation", "Transfer Learning", "Meta-Learning",
    "Few-Shot Learning", "Zero-Shot Learning", "Self-Supervised Learning",
    "Active Learning", "Federated Learning", "Adversarial Training",
    # regularization / training tricks
    "Data Augmentation", "Batch Normalization", "Dropout", "Layer Normalization",
    "Residual Connection",
    # LLM-era fine-tuning / alignment
    "LoRA", "QLoRA", "PEFT", "RLHF", "DPO", "Chain-of-Thought", "In-Context Learning",
    "Prompt Tuning", "Retrieval-Augmented Generation", "RAG",
    "Beam Search", "Greedy Decoding",
}

DATASET_LEXICON: Set[str] = {
    # vision
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "Fashion-MNIST", "COCO", "PASCAL VOC",
    "CelebA", "LSUN", "STL-10", "SVHN", "Omniglot", "ADE20K", "Cityscapes", "KITTI",
    "ShapeNet", "ModelNet", "Visual Genome", "VQA",
    # NLP / QA
    "SQuAD", "GLUE", "SuperGLUE", "WMT", "Penn Treebank", "WikiText", "BookCorpus",
    "Common Crawl", "C4", "The Pile", "OpenWebText", "Gigaword",
    "CoNLL", "CoNLL-2003", "OntoNotes", "TREC", "MS MARCO", "Natural Questions",
    "HotpotQA", "TriviaQA", "RACE", "SWAG", "HellaSwag", "WinoGrande", "ARC", "MMLU",
    "BIG-bench", "XSum", "CNN/DailyMail", "Multi30k", "IWSLT", "Europarl",
    "FewRel", "TACRED", "SemEval", "ACE2005", "ScienceQA",
    "PIQA", "CommonsenseQA", "OpenBookQA", "BoolQ", "COPA", "MultiRC", "ReCoRD",
    "RTE", "WiC", "WSC",
    # speech / multimodal
    "LibriSpeech", "LAION",
    # RL / control
    "Atari", "MuJoCo", "OpenAI Gym",
    # knowledge bases
    "DBpedia", "Freebase", "WordNet", "FrameNet", "PropBank",
    "Universal Dependencies",
    # biomedical (PubMed domain coverage)
    "MIMIC-III", "MIMIC-IV", "i2b2", "n2c2", "PubMedQA", "BioASQ", "BC5CDR",
    "UMLS", "SNOMED CT",
    # general / tabular
    "UCI", "Reddit", "Twitter", "Yelp", "IMDB", "AG News",
}

_ACRONYM_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z\-]{2,}(?:\s+[A-Za-z][a-zA-Z\-]{2,}){1,6})\s*\(([A-Z][A-Za-z0-9\-]{1,10})\)"
)

_DATASET_CONTEXT_WORDS = ("dataset", "corpus", "benchmark", "test set", "training set", "test suite")


def _lexicon_hits(text: str, lexicon: Set[str]) -> Set[str]:
    hits = set()
    for term in lexicon:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            hits.add(term)
    return hits


def extract_entities(text: str) -> Dict[str, Set[str]]:
    """Returns {"methods": {...}, "datasets": {...}} extracted from `text` (typically
    an abstract). Deterministic -- same input always gives the same output, no LLM
    sampling involved."""
    if not text:
        return {"methods": set(), "datasets": set()}

    methods = _lexicon_hits(text, METHOD_LEXICON)
    datasets = _lexicon_hits(text, DATASET_LEXICON)

    for m in _ACRONYM_PATTERN.finditer(text):
        acronym = m.group(2)
        if acronym in methods or acronym in datasets:
            continue
        window = text[max(0, m.start() - 40):m.end() + 40].lower()
        if any(w in window for w in _DATASET_CONTEXT_WORDS):
            datasets.add(acronym)
        else:
            methods.add(acronym)

    return {"methods": methods, "datasets": datasets}


_LLM_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract every machine learning METHOD/MODEL name and every DATASET/BENCHMARK "
    "name mentioned in the abstract below. Return strict JSON: "
    '{{"methods": [...], "datasets": [...]}}. Use the names exactly as they appear '
    "(prefer the acronym if both a full name and acronym are given). If none, return "
    "empty lists.\n\nAbstract:\n{abstract}\n"
)


def build_llm_extraction_prompt(abstract: str) -> str:
    """Optional upgrade path, NOT auto-run (needs an LLM call, costs API credit) --
    same pattern as rationale_revision.py's build_regeneration_prompt. Wire this into
    the same OpenRouter call pattern as ../notebooks/06_critic_generator_agent.ipynb
    to actually execute it, and UNION its output with extract_entities()'s heuristic
    hits (not replace them) to raise recall without giving up the free, deterministic
    baseline pass."""
    return _LLM_EXTRACTION_PROMPT_TEMPLATE.format(abstract=abstract.strip())
