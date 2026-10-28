"""
Masked Language Modeling Comparison
Student Starter Template

Requirements:
- Create a file named corpus.txt in the same folder
- Install transformers and torch if needed:
pip install transformers torch

Instructions:
- Complete all TODO sections
- Keep the overall program structure
- Test with your own corpus.txt content
"""

import string
from collections import defaultdict, Counter
from transformers import pipeline


# -----------------------------
# LOAD AND PREPROCESS CORPUS
# -----------------------------

def load_corpus(filename):
    """
    Load the corpus text from a file.
    """

    #  Open the file and return the text
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def preprocess_text(text):
    """
    Convert text to lowercase, remove punctuation, and split into tokens.
    """

    #  Convert text to lowercase
    text = text.lower()

    #  Remove punctuation
    for p in string.punctuation:
        text = text.replace(p, "")

    #  Split into tokens
    tokens = text.split()

    return tokens


# -----------------------------
# BUILD CONTEXT MODEL
# -----------------------------

def build_context_model(tokens, window_size=2):
    """
    Build a simple context model:
    (left_context, right_context) -> Counter(possible words)

    Example idea:
    if the corpus contains:
    "the cat sat on the mat"

    then for the word "sat" with window_size=2:
    left_context = ("the", "cat")
    right_context = ("on", "the")
    target = "sat"
    """

    model = defaultdict(Counter)

    #  Loop through tokens and build the model
    for i in range(window_size, len(tokens) - window_size):
        left_context = tuple(tokens[i - window_size:i])

        right_context = tuple(
            tokens[i + 1:i + window_size + 1]
        )

        target_word = tokens[i]

        model[(left_context, right_context)][target_word] += 1

    return model


# -----------------------------
# MANUAL MASK PREDICTION
# -----------------------------

def predict_mask_manual(sentence, model, window_size=2, top_n=3):
    """
    Predict masked words using the manual context model.
    """

    words = sentence.lower().split()

    #  Check that [mask] is in the sentence
    if "[mask]" not in words:
        return []

    #  Find the index of [mask]
    mask_index = words.index("[mask]")

    #  Extract left context
    left = tuple(
        words[max(0, mask_index - window_size):mask_index]
    )

    #  Extract right context
    right = tuple(
        words[mask_index + 1: mask_index + 1 + window_size]
    )

    context = (left, right)

    #  If context not found, return empty list
    if context not in model:
        return []

    #  Return top_n predictions
    return model[context].most_common(top_n)


# -----------------------------
# BERT PREDICTION
# -----------------------------

def predict_mask_bert(sentence, fill_mask):
    """
    Predict masked words using a pretrained BERT model.
    """

    #  Replace [mask] with [MASK]
    sentence = sentence.replace("[mask]", "[MASK]")

    #  Run the fill-mask pipeline
    results = fill_mask(sentence)

    #  Return only the top 3 predictions
    return results[:3]


# -----------------------------
# DISPLAY FUNCTIONS
# -----------------------------

def print_manual_predictions(predictions):
    print("\nManual Model Predictions:")

    if not predictions:
        print(" No predictions found (context not in corpus).")
    else:
        for word, count in predictions:
            print(f" {word} (count: {count})")


def print_bert_predictions(predictions):
    print("\nBERT Predictions:")

    if not predictions:
        print(" No BERT predictions returned.")
    else:
        for pred in predictions:
            print(
                f" {pred['token_str']} "
                f"(score: {pred['score']:.4f})"
            )


def compare_predictions(sentence, manual_preds, bert_preds):
    print("\nComparison Notes:")

    print("TODO: Write or print a short comparison here.")

    if manual_preds:
        print("- Manual model found context in the corpus.")
    else:
        print("- Manual model could not find the context.")

    if bert_preds:
        print("- BERT used broader language knowledge.")

    print("- Discuss these results in your write-up.")


# -----------------------------
# MAIN PROGRAM
# -----------------------------

def main():
    # Step 1: Load corpus
    try:
        text = load_corpus("corpus.txt")

    except FileNotFoundError:
        print(
            "ERROR: corpus.txt not found. "
            "Please create the file."
        )
        return

    # Step 2: Preprocess corpus
    tokens = preprocess_text(text)

    print("Corpus loaded successfully.")
    print(f"Number of tokens: {len(tokens)}")

    # Step 3: Build manual context model
    model = build_context_model(tokens, window_size=2)

    # Step 4: Load BERT
    print(
        "\nLoading BERT model "
        "(this may take a moment)..."
    )

    fill_mask = pipeline(
        "fill-mask",
        model="bert-base-uncased"
    )

    # Step 5: Test sentences

    sentences = [
        "The cat sat on the [mask]",
        "I went to the [mask] to buy food",
        "She deposited money in the [mask]",
        "The trophy does not fit into the suitcase because it is too [mask]"
    ]

    print(
        "\n=== MASKED PREDICTION COMPARISON ===\n"
    )

    for sentence in sentences:
        print(f"Sentence: {sentence}")

        #  Get manual predictions
        manual_preds = predict_mask_manual(
            sentence,
            model,
            window_size=2,
            top_n=3
        )

        #  Get BERT predictions
        bert_preds = predict_mask_bert(
            sentence,
            fill_mask
        )

        # Print results
        print_manual_predictions(manual_preds)
        print_bert_predictions(bert_preds)

        # Compare results
        compare_predictions(
            sentence,
            manual_preds,
            bert_preds
        )

        print("\n" + "-" * 50)

        # Step 6: Required reflection/failure case reminder

    print("\nAssignment Reminder:")
    print(
        "- You must modify corpus.txt "
        "and observe how predictions change."
    )
    print(
        "- You must include at least one failure "
        "case for your manual model."
    )
    print(
        "- You must include at least one case "
        "where BERT is also imperfect."
    )


if __name__ == "__main__":
    main()
