import re
import string


def remove_hyperlink(word):
    return re.sub(r"http\S+", "", word)


def to_lower(word):
    result = word.lower()
    return result


def remove_number(word):
    result = re.sub(r"\d+", "", word)
    return result


def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result


def remove_whitespace(word):
    result = word.strip()
    return result


def replace_newline(word):
    return word.replace("\n", " ")


def preprocess_pipeline(sentence):
    cleaning_utils = [
        remove_hyperlink,
        replace_newline,
        to_lower,
        remove_number,
        remove_punctuation,
        remove_whitespace,
    ]
    for util in cleaning_utils:
        sentence = util(sentence)

    return sentence
