import spacy
from spacy import displacy

# Fonte http://www.cnj.jus.br/noticias/cnj/87993-conciliacao-evita-expulsao-de-9-mil-familias-de-terreno-em-bh
texto_demo = '''Um acordo homologado pela Justiça mineira nesta sexta-feira (9/11) evitou que 9 mil famílias fossem obrigadas a deixar um terreno que ocupam desde 2013 na Região Norte de Belo Horizonte. A conciliação viabilizou que uma solução fosse negociada entre proprietários da área de 1,8 milhão de hectares e os moradores da Ocupação Izidora, com a intervenção do Tribunal de Justiça de Minas Gerais (TJ-MG). O acordo foi um dos mais emblemáticos da Semana Nacional da Conciliação, mobilização anual em que Conselho Nacional de Justiça (CNJ) e tribunais de todo o país buscam dissuadir as partes em litígio a levar suas disputas à Justiça e a optar por uma solução de consenso.'''


nlp = spacy.load('pt')
doc = nlp(texto_demo)

# Imprime todos os tokens, inclusive espaços.
# Colocamos abaixo as aspas para melhor destacar cada token.
for token in doc:
    print('"' + token.text + '"')


# https://spacy.io/api/token
for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        token.text, # texto puro
        token.idx, # índice do caracter inicial
        token.lemma_, # forma básica do token em Unicode
        token.is_punct,
        token.is_space,
        token.shape_, # trasformação da string do token para demonstrar características ortográficas
        # https://github.com/explosion/spacy/blob/master/spacy/lang/pt/tag_map.py
        token.pos_, # Part of speech - classificação da categoria gramatical alto nível
        token.tag_ # Part of speech - classificação da categoria gramtial detalhada
    ))

print('---------------------------------------------------------------------------------------------------')

for ent in doc.ents:
    print(ent.text, ent.label_)


# displacy.serve(doc, style='ent')
displacy.serve(doc, style='dep')
