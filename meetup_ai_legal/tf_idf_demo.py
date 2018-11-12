# Baseado em https://building.lang.ai/wtf-is-tf-idf-5c5b86ee7331

from collections import Counter
import math
import os
import spacy
import nltk
from nltk.corpus import stopwords

nlp = spacy.load('pt')

# Adicionando stopwords do NLTK e algumas customizadas (exemplos)
nltk.download('stopwords')
stopwords_direito = {'autor', 'réu', 'lei', 'artigo', 'art', 'arts', 'direito', 'tribunal', 'juiz', 'advogado'}

for stop_word in stopwords.words('portuguese'):
    nlp.vocab[stop_word].is_stop = True

for stop_word in stopwords_direito:
    nlp.vocab[stop_word].is_stop = True


# Cria um dicionário com o conteúdo de todas as peças
pecas = dict()
path = os.path.join(os.path.dirname(__file__), 'sentencas')

for filename in os.listdir(path):
    print(filename)
    f = open(os.path.join(os.path.dirname(__file__), 'sentencas', filename), 'r')
    pecas[filename] = f.read()


# Considera como válidos apenas os tokens alfanuméricos, que não são pessoas nem lugares - acurácia ruim do NER
# Remove stopwords -
def token_valido(token):
    is_valid = token.is_alpha and token.ent_type_ != 'PERSON' and token.ent_type_ != 'LOC'
    return is_valid and not token.is_stop


# Retorna um generator com todos os lemas dos tokens em minúsculos de cada peça considerados válidos
def tokenize_peca(peca):
    return [token.lemma_.lower() for token in nlp(peca) if token_valido(token)]


# Montando o vocabulário
vocabulario = set()
# Um counter é uma subclasse de dict que conta as ocorrências de cada palavra
idf_counter = Counter()

for key, conteudo in pecas.items():
    print("Processando peça {}...".format(key))
    peca_palavras = set(tokenize_peca(conteudo))
    vocabulario = vocabulario | peca_palavras
    idf_counter.update(peca_palavras)

print("Todas as peças processadas")
print("Tamanho do vocabulario {}".format(len(vocabulario)))
print("Palavras mais comuns no vocabulário {}".format(str(idf_counter.most_common(10))))


idf = {palavra: math.log(len(pecas) / df, 2) for palavra, df in idf_counter.items()}
print(idf)


def analisar_peca(key, conteudo):
    palavras_alvo = tokenize_peca(conteudo)
    tfidf = {
        palavra: (1 + math.log(_tf, 2)) * idf[palavra]
        for palavra, _tf in Counter(palavras_alvo).items()
    }
    numero_palavras = 20
    mais_frequente = [
        w for (w, _) in Counter(palavras_alvo).most_common(numero_palavras)
    ]

    tfidf_ordenado = [
        w for (w, _) in sorted(tfidf.items(), key=lambda kv: kv[1], reverse=True)
    ]
    print(key)
    print("Palavras mais frequentes: {}".format(mais_frequente))
    print("Palavras com maior TF-IDF: {}".format(tfidf_ordenado[:numero_palavras]))


if __name__ == '__main__':
    for key, conteudo in pecas.items():
        analisar_peca(key, conteudo)

