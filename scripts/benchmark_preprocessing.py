"""Benchmark full-train TF-IDF preprocessing and record resource use."""

import os
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_scotus_dataset
from src.preprocess import PreprocessingConfig, build_tfidf_vectorizer, fit_on_train
from src.utils import save_json


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()


def fit_and_measure(config: PreprocessingConfig):
    dataset = load_scotus_dataset()
    train_texts = dataset["train"]["text"]
    vectorizer = build_tfidf_vectorizer(config)
    started = time.perf_counter()
    with MemorySampler() as sampler:
        fit_on_train(vectorizer, train_texts)
    elapsed = time.perf_counter() - started
    return vectorizer, elapsed, sampler.peak_rss


def main():
    config = PreprocessingConfig(tfidf_ngram_range=(1, 2))
    vectorizer, elapsed, peak_rss = fit_and_measure(config)
    peak_gb = peak_rss / (1024 ** 3)
    fallback = None

    if elapsed > 600 or peak_gb > 4:
        fallback = "unigram"
        config = PreprocessingConfig(tfidf_ngram_range=(1, 1))
        vectorizer, elapsed, peak_rss = fit_and_measure(config)

    result = {
        "seed": 42,
        "split": "train",
        "documents": 5000,
        "requested_ngram_range": [1, 2],
        "selected_ngram_range": list(config.tfidf_ngram_range),
        "max_features": config.tfidf_max_features,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "fit_wall_time_seconds": round(elapsed, 3),
        "peak_process_rss_bytes": int(peak_rss),
        "peak_process_rss_gb": round(peak_rss / (1024 ** 3), 3),
        "fallback": fallback,
        "thresholds": {
            "max_wall_time_seconds": 600,
            "max_peak_process_rss_gb": 4,
        },
    }
    save_json(result, "results/preprocessing_benchmark.json")
    print(result)


if __name__ == "__main__":
    main()
