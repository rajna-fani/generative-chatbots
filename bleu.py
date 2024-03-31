# Code for Table 1 calculations
'''
An individual N-gram score is the evaluation of just matching grams of a specific order, 
such as single words (1-gram) or word pairs (2-gram or bigram)
'''
from nltk.translate.bleu_score import sentence_bleu

reference = [['This', 'is', 'a', 'good', 'example', 'for', 'BLEU', 'metric']]
candidate = ['This', 'is', 'not', 'a', 'good', 'example', 'to', 'test', 'ROUGE', 'metric']
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
