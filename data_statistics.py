from corpus_processor import Loader
from corpus_processor import Type
import re
import tabulate
corpus_loader = Loader()


def corpus_statistics(corpus_name):
    whole_raw = corpus_loader.load_corpus(corpus_name=corpus_name, corpus_type=Type.whole_raw)
    whole_tok = corpus_loader.load_corpus(corpus_name=corpus_name, corpus_type=Type.whole_tok)
    sents_raw = corpus_loader.load_corpus(corpus_name=corpus_name, corpus_type=Type.sents_raw)
    space_pattern = r'[^\S\r\n\v\f]'
    zwnj_pattern = r'\u200c'
    print(f"Statistics for {corpus_name} corpus")
    print(f"total number of characters:-------------{len(whole_raw)}")
    print(f"total number of tokens:-----------------{len(whole_tok)}")
    print(f"total number of sentences:--------------{len(sents_raw)}")
    print()
    print(f"characters in tokens average:-----------{round(len(whole_raw) / len(whole_tok), 3)}")
    print(f"characters in sentences average:--------{round(len(whole_raw) / len(sents_raw), 3)}")
    print(f"tokens in sentences average:------------{round(len(whole_tok) / len(sents_raw), 3)}")
    print()
    print(
        f"spaces in sentences average:-------------{round(len(re.findall(space_pattern, whole_raw)) / len(sents_raw), 3)}")
    print(
        f"zwnj in sentences average:---------------{round(len(re.findall(zwnj_pattern, whole_raw)) / len(sents_raw), 3)}")
    print()
    print(f"space/character ratio:-----------{round(len(re.findall(space_pattern, whole_raw)) / len(whole_raw), 3)}")
    print(f"zwnj/character ratio:------------{round(len(re.findall(zwnj_pattern, whole_raw)) / len(whole_raw), 3)}")
    return [corpus_name, len(whole_raw), len(whole_tok), len(sents_raw), round(len(whole_raw) / len(whole_tok), 3),
            round(len(whole_raw) / len(sents_raw), 3), round(len(whole_tok) / len(sents_raw), 3),
            round(len(re.findall(space_pattern, whole_raw)) / len(sents_raw), 3),
            round(len(re.findall(zwnj_pattern, whole_raw)) / len(sents_raw), 3),
            round(len(re.findall(space_pattern, whole_raw)) / len(whole_raw), 3),
            round(len(re.findall(zwnj_pattern, whole_raw)) / len(whole_raw), 3)]


table = []
table.append(["Corpus Name", "Total Characters", "Total Tokens", "Total Sentences", "Characters per Token Average",
              "Characters per Sentence Average", "Tokens per Sentence Average", "Spaces per Sentence Average",
              "Zwnj per Sentence Average", "Spaces per Character Ratio", "Zwnj per Character Ratio"])
table.append(corpus_statistics("bijankhan"))
table.append(corpus_statistics("peykareh"))
table.append(corpus_statistics("all"))
table = list(map(list, zip(*table)))
print(tabulate.tabulate(table, headers="firstrow",tablefmt="fancy_grid"))
