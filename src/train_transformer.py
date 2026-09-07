"""Approach 2: fine-tune DistilBERT to pick the chapter.

66 million parameters. Bigger encoders score slightly better on public
benchmarks, but the constraint here is data rather than model size, and
DistilBERT fine-tunes on a laptop in minutes instead of hours.

The procedure deliberately mirrors src/train.py so the comparison is fair:
train on the training split, use validation to choose how many epochs, then
retrain on training plus validation for that many epochs, with the same
balanced class weights in the loss. The test questions stay sealed until
src/evaluate_transformer.py.

Run: python3 src/train_transformer.py
"""

import argparse

import numpy
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from chapters import CHAPTER_SLUGS
from features import read_split
from paths import MODELS_DIR, assert_inside_project

MODEL_DIR = MODELS_DIR / "chapter_transformer"

BASE_MODEL = "distilbert-base-uncased"

# Questions are short, but not as short as the old 256-token limit assumed. Counted through
# this tokeniser, 84 of the corpus's questions run past 256 and the longest is
# 544, losing more than half of itself; at 512 exactly one question is cut, by 32
# tokens. Approach 1 reads every word, so truncating here was quietly asking the
# two approaches different questions. 512 costs roughly twice the time per epoch
# and it buys a fair comparison. tools/measure_token_lengths.py is the count.
MAX_TOKENS = 512
BATCH_SIZE = 16

# 2e-5 is the standard fine-tuning rate for BERT-family models. Much higher and
# the pre-trained weights are destroyed faster than the new task is learned,
# which is the failure people mean by "catastrophic forgetting".
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
SEED = 42

# The chapter's position in CHAPTER_SLUGS is its integer class id. That order is
# fixed, and rearranging it after training would silently relabel every
# prediction the saved model makes.
label2id = {slug: index for index, slug in enumerate(CHAPTER_SLUGS)}
id2label = {index: slug for slug, index in label2id.items()}


def pick_device():
    """Use Apple's GPU when present, otherwise fall back to the processor."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def balanced_class_weights(chapters, device):
    """One weight per chapter, rarer chapters weighted up.

    Approach 1 passes class_weight balanced to its logistic regression for a
    reason that applies here word for word: Complex Numbers has 152 questions
    and Correlation and Regression has 32, so a model can raise its overall
    score by quietly never predicting the small chapters. Leaving this side of
    the comparison unweighted was measuring two different objectives and
    calling the difference an argument about architectures.

    The formula is sklearn's, taken from sklearn rather than rewritten, so the
    two approaches are handed the same definition of balanced. A chapter with
    no rows at all keeps weight 1: it contributes nothing to the loss either
    way, and dividing by its count would not survive the attempt.
    """
    present = numpy.unique(chapters)
    weight_by_chapter = dict(
        zip(
            present,
            compute_class_weight("balanced", classes=present, y=chapters),
        )
    )
    weights = [weight_by_chapter.get(slug, 1.0) for slug in CHAPTER_SLUGS]
    return torch.tensor(weights, dtype=torch.float, device=device)


def encode(question_tokeniser, questions, chapters, max_tokens=MAX_TOKENS):
    """Turn questions into the token ids the model reads, paired with answers.

    Unlike Approach 1, the vocabulary here is fixed before this project began.
    The tokeniser cannot learn anything from these questions, so there is no
    train-only rule to enforce at this step. That is a genuine difference
    between the two approaches, not an oversight.
    """
    encoded_questions = question_tokeniser(
        questions,
        truncation=True,
        padding="max_length",
        max_length=max_tokens,
        return_tensors="pt",
    )
    labels = torch.tensor([label2id[chapter] for chapter in chapters])
    return TensorDataset(
        encoded_questions["input_ids"], encoded_questions["attention_mask"], labels
    )


def run_one_epoch(
    chapter_transformer, batches, device, optimiser=None, class_weights=None
):
    """Train for one pass if given an optimiser, otherwise just measure.

    The loss is computed here rather than read off outputs.loss, because the
    model's own loss is unweighted cross entropy and class_weights is the whole
    point of the exercise. Passing labels in as well would have the model
    compute a second, unweighted loss and throw it away.
    """
    training = optimiser is not None
    chapter_transformer.train(training)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    correct = total = 0
    with torch.set_grad_enabled(training):
        for input_ids, attention_mask, labels in batches:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = chapter_transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )
            if training:
                loss_function(outputs.logits, labels).backward()
                optimiser.step()
                optimiser.zero_grad()
            correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def fine_tune(
    train_dataset,
    epochs,
    device,
    validation_dataset=None,
    class_weights=None,
    seed=SEED,
):
    """Fine-tune a fresh copy of the base model for a fixed number of epochs.

    Returns the model and, when a validation set is supplied, that set's
    accuracy after each epoch so the caller can decide where to stop.
    """
    torch.manual_seed(seed)
    chapter_transformer = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(CHAPTER_SLUGS), label2id=label2id, id2label=id2label
    ).to(device)
    optimiser = torch.optim.AdamW(chapter_transformer.parameters(), lr=LEARNING_RATE)

    train_batches = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_scores = []
    for epoch in range(1, epochs + 1):
        train_accuracy = run_one_epoch(
            chapter_transformer, train_batches, device, optimiser, class_weights
        )
        line = f"   epoch {epoch}   training accuracy {train_accuracy:.3f}"
        if validation_dataset is not None:
            batches = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
            score = run_one_epoch(chapter_transformer, batches, device)
            validation_scores.append(score)
            line += f"   validation accuracy {score:.3f}"
        print(line)
    return chapter_transformer, validation_scores


def fit_on_the_full_dataset(
    device,
    use_class_weights=True,
    max_tokens=MAX_TOKENS,
    seed=SEED,
    train_data=None,
    val_data=None,
):
    """Tune the epoch count on validation, then refit on training plus validation.

    Returns the tokeniser and the fitted model. main saves them. The seed and
    truncation runs in tools/ read the accuracy and throw the model away.

    train_data and val_data are (questions, chapters) pairs, and default to the
    whole of each split. The learning-curve runs in tools/learning_curve.py pass
    a group-aware fraction of them instead; nothing below this line can tell the
    difference, which is the point. The test split is not read here at all.
    """
    train_questions, train_chapters = train_data or read_split("train")
    val_questions, val_chapters = val_data or read_split("val")

    question_tokeniser = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = encode(
        question_tokeniser, train_questions, train_chapters, max_tokens
    )
    val_dataset = encode(question_tokeniser, val_questions, val_chapters, max_tokens)

    # Weights come from the rows being trained on, so the tuning run is weighted
    # by the training split and the refit by training plus validation. Deriving
    # them once from the whole corpus would let the validation rows' chapter
    # counts inform the run that is choosing the epoch count.
    tuning_weights = (
        balanced_class_weights(train_chapters, device) if use_class_weights else None
    )

    print(f"choosing epoch count on {len(val_questions)} validation questions:")
    _, validation_scores = fine_tune(
        train_dataset, MAX_EPOCHS, device, val_dataset, tuning_weights, seed
    )
    best_epochs = validation_scores.index(max(validation_scores)) + 1
    print(
        f"\nbest at {best_epochs} epochs, {max(validation_scores):.3f} validation accuracy"
    )

    # Retrain from scratch on training plus validation, exactly as Approach 1
    # refits after tuning. Continuing the existing run instead would mean the
    # final model had trained on the validation rows more than the training rows.
    print(
        f"\nretraining on all {len(train_questions) + len(val_questions)} for {best_epochs} epochs:"
    )
    final_questions = train_questions + val_questions
    final_chapters = train_chapters + val_chapters
    final_dataset = encode(
        question_tokeniser, final_questions, final_chapters, max_tokens
    )
    final_weights = (
        balanced_class_weights(final_chapters, device) if use_class_weights else None
    )
    chapter_transformer, _ = fine_tune(
        final_dataset, best_epochs, device, class_weights=final_weights, seed=seed
    )
    return question_tokeniser, chapter_transformer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT to pick the chapter, and save it "
        "beside Approach 1's model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="seed for the weight initialisation and the batch "
        "order (default %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="tokens of each question the model reads before "
        "the rest is truncated (default %(default)s)",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="drop the balanced class weights from the loss, "
        "which is the comparison they were added against",
    )
    arguments = parser.parse_args()

    device = pick_device()
    print(f"fine-tuning {BASE_MODEL} on {device}\n")
    question_tokeniser, chapter_transformer = fit_on_the_full_dataset(
        device,
        use_class_weights=not arguments.unweighted,
        max_tokens=arguments.max_tokens,
        seed=arguments.seed,
    )

    assert_inside_project(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    chapter_transformer.save_pretrained(MODEL_DIR)
    question_tokeniser.save_pretrained(MODEL_DIR)
    print(f"\nsaved to models/{MODEL_DIR.name}/")


if __name__ == "__main__":
    main()
