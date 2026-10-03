"""
Script 04: Train and Evaluate Legal-BERT (nlpaueb/legal-bert-base-uncased).
Resumable, memory-safe fine-tuning with automatic hardware detection.
Can be executed in --smoke_test mode on CPU, or on GPU (local / Colab / Kaggle).
Saves model predictions, probabilities, and runtime hardware info to results/.
"""

import os
import sys
import time
import argparse
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_scotus_dataset, SCDB_LABEL_NAMES, SCDB_LABEL_MAP
from src.utils import set_seed, detect_hardware, save_json


class ScotusDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def train_transformer(args):
    set_seed(args.seed)
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Hardware Detection
    hardware_info = detect_hardware()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Executing on device: {device}")
    print(f"Hardware Info: {hardware_info['gpu_name']} | RAM: {hardware_info['total_ram_gb']} GB")

    # 2. Load Dataset
    print("Loading LexGLUE SCOTUS dataset...")
    ds = load_scotus_dataset()

    if args.smoke_test:
        print(f"SMOKE TEST MODE: subsampling to {args.smoke_train_size} train, {args.smoke_eval_size} val/test.")
        train_texts = ds["train"]["text"][:args.smoke_train_size]
        train_labels = ds["train"]["label"][:args.smoke_train_size]
        val_texts = ds["validation"]["text"][:args.smoke_eval_size]
        val_labels = ds["validation"]["label"][:args.smoke_eval_size]
        test_texts = ds["test"]["text"][:args.smoke_eval_size]
        test_labels = ds["test"]["label"][:args.smoke_eval_size]
    else:
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        val_texts = ds["validation"]["text"]
        val_labels = ds["validation"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]

    # 3. Load Tokenizer & Model
    print(f"Loading tokenizer and model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Failed loading {args.model_name}: {e}. Falling back to bert-base-uncased.")
        args.model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    id2label = {i: name for i, name in enumerate(SCDB_LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(SCDB_LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=13,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 4. Data Loaders
    train_dataset = ScotusDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = ScotusDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    test_dataset = ScotusDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # 6. Checkpoint Resume check
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    start_epoch = 0
    best_val_macro_f1 = -1.0

    if os.path.exists(checkpoint_path) and args.resume:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_macro_f1 = checkpoint.get("best_val_macro_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}, previous best val macro F1: {best_val_macro_f1:.4f}")

    # 7. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    use_amp = (device.type == "cuda")
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_res = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_res['accuracy']:.4f} | Val Macro F1: {val_res['macro_f1']:.4f}")

        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            print(f"--> New best validation Macro F1: {best_val_macro_f1:.4f}. Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
            }, checkpoint_path)

    # 8. Load Best Checkpoint for Final Test Evaluation
    if os.path.exists(checkpoint_path):
        print("Loading best model checkpoint for test evaluation...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_res = evaluate_model(model, test_loader, device)
    val_res = evaluate_model(model, val_loader, device)
    elapsed_time = time.time() - start_time

    print(f"\nFinal Test Results:")
    print(f"  Test Accuracy : {test_res['accuracy']:.4f}")
    print(f"  Test Macro F1 : {test_res['macro_f1']:.4f}")
    print(f"  Test W-F1     : {test_res['weighted_f1']:.4f}")
    print(f"  Total Time    : {elapsed_time:.1f} seconds")

    # 9. Save Outputs
    # Save predictions as compressed npz
    predictions_path = os.path.join(args.results_dir, "transformer_predictions.npz")
    np.savez_compressed(
        predictions_path,
        val_labels=val_res["labels"],
        val_predictions=val_res["predictions"],
        val_probabilities=val_res["probabilities"],
        test_labels=test_res["labels"],
        test_predictions=test_res["predictions"],
        test_probabilities=test_res["probabilities"],
    )
    print(f"Predictions and probabilities saved to {predictions_path}")

    # Save Run Info
    run_info = {
        "model_name": args.model_name,
        "smoke_test": args.smoke_test,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "hardware": hardware_info,
        "elapsed_seconds": round(elapsed_time, 2),
        "validation_accuracy": val_res["accuracy"],
        "validation_macro_f1": val_res["macro_f1"],
        "validation_weighted_f1": val_res["weighted_f1"],
        "test_accuracy": test_res["accuracy"],
        "test_macro_f1": test_res["macro_f1"],
        "test_weighted_f1": test_res["weighted_f1"],
    }
    run_info_path = os.path.join(args.results_dir, "transformer_run_info.json")
    save_json(run_info, run_info_path)
    print(f"Run metadata saved to {run_info_path}")

    return run_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT on LexGLUE SCOTUS.")
    parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true", help="Run fast smoke test on tiny subset")
    parser.add_argument("--smoke_train_size", type=int, default=20)
    parser.add_argument("--smoke_eval_size", type=int, default=10)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/transformer")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()
    train_transformer(args)
