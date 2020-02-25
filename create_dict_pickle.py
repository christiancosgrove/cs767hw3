import pickle
import os
from parlai.core.dict import DictionaryAgent

# path = os.path.join(os.path.expanduser("~"), 'Downloads/dat/MovieTriples_Dataset')
path = 'dat/MovieTriples_Dataset'

with open(os.path.join(path, 'Training.dict.pkl'), 'rb') as data_file:
    dictionary = pickle.load(data_file)


parlai_dict = DictionaryAgent({'vocab_size': 10004})#, 'dict_nulltoken':None, 'dict_starttoken':None, 'dict_endtoken':None,'dict_unktoken':None})

parlai_dict.default_tok = 'space'

dictionary = sorted(dictionary, key=lambda x: x[1])
print(dictionary[:10])

for word in dictionary:
    # print(word[0])
    parlai_dict.add_to_dict([word[0]])
    parlai_dict.freq[word[0]] = word[2]
    # print(word)

# print(parlai_dict)
# parlai_dict.add_to_dict(['hello'])

parlai_dict.save('test_hred.dict', sort=True)
