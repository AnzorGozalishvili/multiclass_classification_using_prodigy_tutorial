import json
import spacy
import os


def terms_and_classes(directory):
    for item in os.listdir(directory):
        if item.endswith('.txt'):
            yield item, item[:-4].upper()


def read_terms(path):
    with (open(path)) as file:
        return file.readlines()


def patterns_to_jsonl(data, path):
    with open(path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def generate_lower_pattern(text, nlp):
    doc = nlp(text)
    tokens = [item.text for item in doc]

    pattern = []
    for token in tokens:
        pattern.append({'LOWER': token})

    return pattern


def generate_lemma_pattern(text, nlp):
    doc = nlp(text)
    tokens = [item.lemma_ for item in doc]

    pattern = []
    for token in tokens:
        pattern.append({'LEMMA': token})

    return pattern


def generate_pos_pattern(text, nlp):
    doc = nlp(text)
    tokens = [item.pos_ for item in doc]

    pattern = []
    for token in tokens:
        pattern.append({'POS': token})

    return pattern


def generate_all_patterns(text, nlp):
    all_patterns = [
        generate_lower_pattern(text, nlp),
        generate_lemma_pattern(text, nlp),
        generate_pos_pattern(text, nlp)
    ]
    return all_patterns


if __name__ == '__main__':
    spacy_model = spacy.load('en_core_web_sm')

    terms_dir = 'prodigy/terms'
    terms_out_dir = 'prodigy/terms'

    for terms_file, class_name in terms_and_classes(terms_dir):
        generated_patterns = []
        terms = read_terms(os.path.join(terms_dir, terms_file))
        for term in terms:
            term_patterns = generate_all_patterns(term, spacy_model)

            for term_pattern in term_patterns:
                generated_patterns.append({"label": class_name, "pattern": term_pattern})

        patterns_to_jsonl(generated_patterns, os.path.join(terms_out_dir, f'{class_name.lower()}.jsonl'))
