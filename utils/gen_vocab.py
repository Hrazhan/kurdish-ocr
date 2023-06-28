import os
import random
import pandas as pd
from datasets import load_dataset
import re
import string
import argparse
from klpt.preprocess import Preprocess
from klpt.tokenize import Tokenize

preprocessor_ckb = Preprocess("Sorani", "Arabic", numeral="Arabic")
tokenizer_ckb = Tokenize("Sorani", "Arabic")



parser = argparse.ArgumentParser(description='Generate corpus for the OCR dataset.')
parser.add_argument('--lang', type=str, default='ckb', help='Language')
parser.add_argument('--wiki_date', type=str, default='20230201', help='Wiki date')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
parser.add_argument('--chars', type=str, default='٠١٢٣٤٥٦٧٨٩ئبپتجچحخدرڕزژسشعغفڤقکگلڵمنهاەوۆوویێء', help='Characters to be kept for the model to learn to recognize. This could be any combination of letters, numbers,and  punctuations.')
parser.add_argument('--oscar', action='store_true', help='Load oscar dataset')
parser.add_argument('--wiki', action='store_true', help='Load wikipedia dataset')
parser.add_argument('--min_length', type=int, default=10, help='Minimum length of a sentence')
parser.add_argument('--max_length', type=int, default=15, help='Maximum length of a sentence')

args = parser.parse_args()


LANG = args.lang
WIKIPEDIA_DATE = args.wiki_date
NUM_SAMPLES = args.num_samples
CHARS = args.chars
MIN_LENGTH = args.min_length
MAX_LENGTH = args.max_length
NO_REPEAT_NGRAM_SIZE = 2

def clean_text(text):
    text = preprocessor_ckb.preprocess(text)

    chars_to_keep = fr'[^{CHARS}\s]+'
    text = re.sub(chars_to_keep, '', text)
    text = re.sub(r"http\S+", "", text)  # remove urls
    punctuations_pattern = f"{string.punctuation}\u060c\u060d\u061b\u061f\u00bb\u00ab\u06D6-\u06ED\u005c«»"
    punctuations_to_remove = ''.join(set(punctuations_pattern) - set(chars_to_keep))
    text = text.translate(str.maketrans('', '', punctuations_to_remove))
    text = text.replace('\u200c', '')    
    return text


def contains_repeated_ngram(window, n):
    ngrams = generate_ngrams(window, n)
    ngram_set = set(ngrams)
    return len(ngrams) != len(ngram_set)


def generate_ngrams(text, n):
     words = text.split()
     output = []  
     for i in range(len(words)- n+1):
         output.append(tuple(words[i:i+n]))
     return output

def main():

    if args.oscar and args.wiki:
        oscar_dataset = load_dataset("oscar-corpus/OSCAR-2301", language=LANG, split='train', use_auth_token=True)
        wiki_dataset = load_dataset("wikipedia", language=LANG, date=WIKIPEDIA_DATE, split='train', beam_runner='DirectRunner')
        df_oscar = oscar_dataset.to_pandas()
        df_wiki = wiki_dataset.to_pandas()
        df = pd.concat([df_oscar, df_wiki], ignore_index=True)

    elif args.oscar:
        oscar_dataset = load_dataset("oscar-corpus/OSCAR-2301", language=LANG, split='train', use_auth_token=True)
        df = oscar_dataset.to_pandas()
    elif args.wiki:
        wiki_dataset = load_dataset("wikipedia", language=LANG, date=WIKIPEDIA_DATE, split='train', beam_runner='DirectRunner')
        df = wiki_dataset.to_pandas()
    else:
        raise ValueError('At least one of --oscar or --wiki must be specified.')


    # We do this so we don't have to clean the entire corpus and only pick the NUM_SAMPLES of sentences.
    # All the rows contains more than one sentence. so by default we are selecting more rows than our chosen NUM_SAMPLES
    if len(df) > NUM_SAMPLES:
        df = df.sample(NUM_SAMPLES)
    df["text"] = df["text"].apply(clean_text)

    # combine all the rows of column text into one giant string
    text = df["text"].str.cat(sep=" ")
    text = text.split()
    # sentences = tokenizer_ckb.sent_tokenize(text)
    tokenized_sentences = []
    
    print("Generating {} sentences for {} dataset".format(NUM_SAMPLES, LANG))
    
    
    start = 0
    count = 0
    while start + len(text) and count < NUM_SAMPLES:
        length = random.randint(MIN_LENGTH, MAX_LENGTH)
        end = start + length
        if end > len(text):
            # end = len(text)
            break

        sentence = " ".join(text[start:end])
        if NO_REPEAT_NGRAM_SIZE > 0 and contains_repeated_ngram(sentence, NO_REPEAT_NGRAM_SIZE):
            start += length
            continue

        tokenized_sentences.append(sentence)

        
        start += length
        count += 1



    df = pd.DataFrame(tokenized_sentences, columns=["sentences"])
    df.dropna(inplace=True)
    df = df[~df.sentences.str.contains("nan")]
    os.makedirs(f'data', exist_ok=True)
    df.to_csv(f'./data/{LANG}_corpus.txt', index=False, header=False)






if __name__ == '__main__':
    main()