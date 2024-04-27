import pandas as pd
import spacy
import unidecode
import contractions as contract
import re
import collections
import pkg_resources
from symspellpy import SymSpell
import swifter

data = pd.read_csv('SuicideAndDepression_Detection.csv')

nlp = spacy.load("en_core_web_sm")
vocab = collections.Counter()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


# Spell Check using Symspell
def fix_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    correctedtext = suggestions[0].term
    return correctedtext


deselect_stop_words = ['no', 'not']

for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


# Remove URL
def remove_url(text):
    return re.sub(r'http\S+', '', text)


def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)


def remove_special(text):
    return text.replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '')


def fix_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def text_preprocessing(text, accented_chars=True, contractions=True, convert_num=True,
                       extra_whitespace=True, lemmatization=True, lowercase=True,
                       url=True, symbols_digits=True, special_chars=True,
                       stop_words=True, lengthening=True, spelling=True):

    if accented_chars:
        text = remove_accented_chars(text)
    if contractions:
        text = contract.fix(text)
    if lowercase:
        text = text.lower()
    if url:
        text = remove_url(text)
    if symbols_digits:
        text = remove_symbols_digits(text)
    if special_chars:
        text = remove_special(text)
    if extra_whitespace:
        text = remove_whitespace(text)
    if lengthening:
        text = fix_lengthening(text)
    if spelling:
        text = fix_spelling(text)

    doc = nlp(text)
    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        if stop_words == True and token.is_stop and token.pos_ != 'NUM':
            flag = False
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            flag = False
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        if edit != "" and flag == True:
            clean_text.append(edit)
    return " ".join(clean_text)


data = data.dropna()
print(len(data))

data['cleaned_text'] = data['text'].swifter.apply(lambda row: text_preprocessing(row))
data.to_csv('Suicide_Detection_cleaned.csv', index=False)
