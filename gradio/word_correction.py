import gradio as gr
import numpy as np


def load_vocab(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def word_correction(word):
    vocabs = load_vocab('../assets/vocab.txt')

    leven_distances = {vocab: levenshtein_distance(
        word, vocab) for vocab in vocabs}

    sorted_distances = dict(
        sorted(leven_distances.items(), key=lambda item: item[1]))
    correct_word = list(sorted_distances.keys())[0]

    return (
        f"Correct word: {correct_word}",
        f"Vocabulary: {', '.join(vocabs[:10])}...",
        f"Distances: {str(dict(list(sorted_distances.items())[:10]))}"
    )


iface = gr.Interface(
    fn=word_correction,
    inputs=gr.Textbox(label="Word"),
    outputs=[
        gr.Textbox(label="Correct Word"),
        gr.Textbox(label="Vocabulary (first 10 words)"),
        gr.Textbox(label="Distances (first 10 words)")
    ],
    title="Word Correction using Levenshtein Distance",
    description="Enter a word to find the closest match in the vocabulary."
)

if __name__ == "__main__":
    iface.launch()
