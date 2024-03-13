import amrlib
import spacy

amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test sentence.')

graphs = doc._.to_amr()
for graph in graphs:
    print(graph)
