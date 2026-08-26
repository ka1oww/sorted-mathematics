"""Measure the fine-tuned transformer on the same sealed test questions.

Deliberately prints the same shape of report as src/evaluate.py, minus the
strongest-words section. That section has no equivalent here, and its absence is
the honest headline of the comparison: Approach 1 can show you the twelve words
that made it choose a chapter, and Approach 2 cannot.

Run: python3 src/evaluate_transformer.py
"""

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate import report_confusions
from features import read_split
from train_transformer import BATCH_SIZE, MODEL_DIR, encode, id2label, pick_device


def predict_chapters(chapter_transformer, dataset, device):
    """Run the model over every question and return the chapter it chose."""
    chapter_transformer.eval()
    predicted = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in DataLoader(dataset, batch_size=BATCH_SIZE):
            outputs = chapter_transformer(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            )
            predicted.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return [id2label[index] for index in predicted]


def main():
    device = pick_device()
    question_tokeniser = AutoTokenizer.from_pretrained(MODEL_DIR)
    chapter_transformer = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    test_questions, y_test = read_split("test")
    test_dataset = encode(question_tokeniser, test_questions, y_test)
    predicted_chapters = predict_chapters(chapter_transformer, test_dataset, device)

    correct = sum(t == p for t, p in zip(y_test, predicted_chapters))
    print(f"{'=' * 72}")
    print(f"TEST ACCURACY  {correct / len(y_test):.1%}   ({correct} of {len(y_test)} questions)")
    print(f"{'=' * 72}")

    print(f"\n{'-' * 72}\nPER CHAPTER\n{'-' * 72}")
    print(classification_report(y_test, predicted_chapters, zero_division=0))

    report_confusions(y_test, predicted_chapters)


if __name__ == "__main__":
    main()
