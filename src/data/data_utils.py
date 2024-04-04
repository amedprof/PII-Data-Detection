import re
from difflib import SequenceMatcher

import codecs
import os
from collections import Counter
from typing import Dict, List, Tuple

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from text_unidecode import unidecode
import joblib
import torch

# def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
#     return error.object[error.start : error.end].encode("utf-8"), error.end


# def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
#     return error.object[error.start : error.end].decode("cp1252"), error.end


# # Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
# codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
# codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


# def remove_double_spaces(text):
#     # Use a regular expression to replace consecutive spaces with a single space
#     # cleaned_text = re.sub(r'\s{2,}', ' | ', text)
#     cleaned_text = re.sub(r'\s+', ' ', text)
#     return cleaned_text

# def resolve_encodings_and_normalize(text: str) -> str:
#     """Resolve the encoding problems and normalize the abnormal characters."""
#     text = (
#         text.encode("raw_unicode_escape")
#         .decode("utf-8", errors="replace_decoding_with_cp1252")
#         .encode("cp1252", errors="replace_encoding_with_utf8")
#         .decode("utf-8", errors="replace_decoding_with_cp1252")
#     )
#     text = unidecode(text)
#     return text


# def clean_text(text):
#     text = text.replace(u'\x9d', u' ')
#     # text = resolve_encodings_and_normalize(text)
#     # text = text.replace(u'\xa0', u' ')
#     # text = text.replace(u'\x85', u'\n')
#     text = remove_double_spaces(text)
#     text = text.strip()
#     return text

# def add_text_to_df(test_df,data_folder):
#     mapper = {}
#     for idx in tqdm(test_df.essay_id.unique()):
#         with open(data_folder/f'{idx}.txt','r') as f:
#             texte = clean_text(f.read())
#             # texte = resolve_encodings_and_normalize(f.read())
#             # texte = texte.strip() 
#         mapper[idx] = texte

#     test_df['discourse_ids'] = np.arange(len(test_df))
#     test_df['essay_text'] = test_df['essay_id'].map(mapper)
#     test_df['discourse_text'] = test_df['discourse_text'].transform(clean_text)
#     test_df['discourse_text'] = test_df['discourse_text'].str.strip()

#     test_df['previous_discourse_end'] = 0
#     test_df['st_ed'] = test_df.apply(get_start_end('discourse_text'),axis=1)
#     test_df['discourse_start'] = test_df['st_ed'].transform(lambda x:x[0])
#     test_df['discourse_end'] = test_df['st_ed'].transform(lambda x:x[1])
#     test_df['previous_discourse_end'] = test_df.groupby("essay_id")['discourse_end'].transform(lambda x:x.shift(1).fillna(0)).astype(int)
#     test_df['st_ed'] = test_df.apply(get_start_end('discourse_text'),axis=1)
#     test_df['discourse_start'] = test_df['st_ed'].transform(lambda x:x[0]) #+ test_df['previous_discourse_end']
#     test_df['discourse_end'] = test_df['st_ed'].transform(lambda x:x[1]) #+ test_df['previous_discourse_end']

#     if 'target' in test_df.columns:
#         classe_mapper = {'Ineffective':0,"Adequate":1,"Effective":2}
#         test_df['target'] = test_df['discourse_effectiveness'].map(classe_mapper)
        
#     else:
#         test_df['target'] = 1 

#     return test_df

# def get_essays(df,n_cpu=4):
    
#     pool = joblib.Parallel(n_cpu)
#     mapper = joblib.delayed(_get_essay)
#     tasks = [mapper(df) for idx,df in df.groupby('id')]
#     ids = [idx for idx,_ in df.groupby('id')]
    
#     return pd.DataFrame({"id":ids,"essay":pool(tqdm(tasks))})

# def _get_essay(df):
#     text_recons = ''
#     for i, (id_,row) in enumerate(df.iterrows()):
#     #     print(row)
#         activity = row["activity"]
#         curs_pos = row["cursor_position"] # cursor position AFTER activity!
#         text_change = row["text_change"]


#         if activity == 'Input' or activity == 'Paste':
#             text_recons = text_recons[:curs_pos - len(text_change)] + text_change + text_recons[curs_pos - len(text_change):]   
#         if activity == 'Remove/Cut':
#             text_recons = text_recons[:curs_pos] + text_recons[curs_pos + len(text_change):]
#         if activity == 'Replace': # Combined remove and input operation
#             cut, add = text_change.split(' => ')
#             text_recons = text_recons[:curs_pos - len(add)] + add + text_recons[curs_pos - len(add) + len(cut):]

#         if "Move" in activity:
#             a, b, c, d = map(
#                         int,
#                         re.match(
#                             r"Move From \[(\d+), (\d+)\] To \[(\d+), (\d+)\]",
#                             activity,
#                         ).groups(),
#                     )

#             if a != c:
#                 if a < c:
#                     text_recons = text_recons[:a] + text_recons[b:d] + text_recons[a:b] + text_recons[d:]
#                 else:
#                     text_recons = text_recons[:c] + text_recons[a:b] + text_recons[c:a] + text_recons[b:]
                    
#     return text_recons


# def get_text_start_end(txt, s, search_from=0):
#     txt = txt[int(search_from):]
#     try:
#         idx = txt.find(s)
#         if idx >= 0:
#             st = idx
#             ed = st + len(s)
#         else:
#             raise ValueError('Error')
#     except:
#         res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
#         if len(res):
#             st, ed = res[0][0], res[0][1]
#         else:
#             m = SequenceMatcher(None, s, txt).get_opcodes()
#             for tag, i1, i2, j1, j2 in m:
#                 if tag == 'replace':
#                     s = s[:i1] + txt[j1:j2] + s[i2:]
#                 if tag == "delete":
#                     s = s[:i1] + s[i2:]

#             res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
#             if len(res):
#                 st, ed = res[0][0], res[0][1]
#             else:
#                 idx = txt.find(s)
#                 if idx >= 0:
#                     st = idx
#                     ed = st + len(s)
#                 else:
#                     st, ed = 0, 0
#     return st + search_from, ed + search_from


# def get_offset_mapping(full_text, tokens):
#     offset_mapping = []

#     current_offset = 0
#     for token in tokens:
#         start, end = get_text_start_end(full_text, token, search_from=current_offset)
#         offset_mapping.append((start, end))
#         current_offset = end

#     return offset_mapping

# def get_start_end(col):
#     def search_start_end(row):
#         txt = row.essay_text
#         search_from = row.previous_discourse_end
#         s = row[col]
#         # print(search_from)
#         return get_text_start_end(txt,s,search_from)
#     return search_start_end

# def batch_to_device(batch, device):
#     batch_dict = {key: batch[key].to(device) for key in batch}
#     return batch_dict


# def text_to_words(text):
#     word = text.split()
#     word_offset = []

#     start = 0
#     for w in word:
#         r = text[start:].find(w)

#         if r==-1:
#             raise NotImplementedError
#         else:
#             start = start+r
#             end   = start+len(w)
#             word_offset.append((start,end))
#         start = end

#     return word, word_offset

# def text_to_sentence(text):
#     sentences = re.split(r' *[\.\?!\n][\'"\)\]]* *', text)
#     sentences = [x for x in sentences if x!=""]
    
#     sentence_offset = []
#     start = 0
#     for w in sentences:
#         r = text[start:].find(w)

#         if r==-1:
#             raise NotImplementedError
#         else:
#             start = start+r
#             end   = start+len(w)
#             sentence_offset.append((start,end))
#         start = end

#     return sentences,sentence_offset

# def text_to_paragraph(text):
#     sentences = re.split(r' *[\n][\'"\)\]]* *', text)
#     sentences = [x for x in sentences if x!=""]
    
#     sentence_offset = []
#     start = 0
#     for w in sentences:
#         r = text[start:].find(w)

#         if r==-1:
#             raise NotImplementedError
#         else:
#             start = start+r
#             end   = start+len(w)
#             sentence_offset.append((start,end))
#         start = end

#     return sentences,sentence_offset


# def get_span_from_text(text,span_type="words"):
    
#     if span_type=="words":
#         spans,spans_offset = text_to_words(text)
#     elif span_type=="sentences":
#         spans,spans_offset = text_to_sentence(text)
#     else:
#         spans,spans_offset = text_to_paragraph(text)
    
#     return spans,spans_offset


# def get_span_len_from_text(text,span_type="words"):
    
#     if span_type=="words":
#         spans_len = len(text.split())
#     elif span_type=="sentences":
#         sentences = re.split(r' *[\.\?!\n][\'"\)\]]* *', text)
#         spans_len = len([x for x in sentences if x!=""])
#     else:
#         sentences = re.split(r' *[\n][\'"\)\]]* *', text)
#         spans_len = len([x for x in sentences if x!=""])
    
#     return spans_len


def to_gpu(data,device):
    if isinstance(data, dict):
        return {k: to_gpu(v,device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v,device) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    
def to_np(data):
    if isinstance(data, dict):
        return {k: to_np(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_np(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return data

# def get_start_end_offset(col):
#     def search_start_end(row):
#         txt = row.full_text
#         toks = row[col]
#         # print(search_from)
#         return get_offset_mapping(txt,toks)
#     return search_start_end





# # ======================================================================================== #
# import numpy as np

# def find_successive_numbers(input_array):
#     result = []
#     current_sublist = []

#     for num in input_array:
#         if not current_sublist or num == current_sublist[-1] + 1:
#             current_sublist.append(num)
#         else:
#             result.append(current_sublist)
#             current_sublist = [num]

#     if current_sublist:
#         result.append(current_sublist)

#     return result
# # ======================================================================================== #

# import re
# # from mimesis import Generic

# def generate_random_data_with_probabilities():
#     generic = Generic()

#     # Probabilities for each country
#     country_probabilities = {'fr': 0.5, 'en': 0.1, 'it': 0.1, 'de': 0.1, 'es': 0.2}

#     # Function to randomly choose a country based on probabilities
#     def choose_country():
#         return generic.random.choices(list(country_probabilities.keys()), weights=country_probabilities.values())

#     # Generate random data
#     country = choose_country()[0]
#     generic = Generic(locale=country)
#     name = generic.person.full_name()
# #     phone_number = generic.person.telephone()
# #     username = generic.person.username()
# #     email = generic.person.email()
# #     address = generic.address.address()
# #     surname = generic.person.surname()
#     ret = dict(
#               NAME_STUDENT=name
#               )
#     return ret
# # ======================================================================================== #

# def generate_ent(labels,tokens):
    
#     idx_lab = np.argwhere(np.array(labels)!="O").reshape(-1)
#     pos = sorted(find_successive_numbers(idx_lab),reverse=True,key=len)

#     lab = np.array(labels)
#     toks = np.array(tokens)

#     ent = {}
#     for i,p in enumerate(pos):
#         l = np.unique([x.split('-')[-1] for x in lab[p]]).tolist()
#         t = toks[p].tolist()
#         if 'NAME_STUDENT' in l:
#             full_name = " ".join(t)
#             ent[full_name] = l[-1]

#         else:
#             full_name = " ".join(t)
#             ent[full_name] = l[-1]
#     return ent
# # ======================================================================================== #
# from spacy.lang.en import English
# en_tokenizer = English().tokenizer

# def tokenize_with_spacy(text, tokenizer=en_tokenizer):
#     tokenized_text = tokenizer(text)
#     tokens = [token.text for token in tokenized_text]
#     offset_mapping = [(token.idx,token.idx+len(token)) for token in tokenized_text]
#     return {'tokens': tokens, 'offset_mapping': offset_mapping}
# # ======================================================================================== #

# def create_mapper_n_clean(full_text,labels,tokens,attribut=["NAME_STUDENT"]):
#     ent = generate_ent(labels,tokens)
#     mapper = {}
#     label_mapper = {}
#     for k,v in ent.items():
#         if v in attribut:      
#             dc_ent = generate_random_data_with_probabilities()
#             if 'NAME_STUDENT' in v:
#                 names = k.split()
#                 if k not in mapper.keys():
#                     mapper[k] = dc_ent[v]
#                     label_mapper[dc_ent[v]] = v
#                 if len(k.split())>1:
#                     map_ = dc_ent[v].split()
#                     if names[0] not in mapper.keys():
#                         mapper[names[0]] = map_[0]
#                         label_mapper[map_[0]] = v
#                     if " ".join(names[1:]) not in mapper.keys():
#                         mapper[" ".join(names[1:])] = " ".join(map_[1:]) 
#                         label_mapper[" ".join(names[1:])] = v
#                 else:
#                     map_ = dc_ent[v].split()
#                     if names[0] not in mapper.keys():
#                         mapper[names[0]] = map_[0]
#                         label_mapper[map_[0]] = v      

#             else:
#                 mapper[k] = dc_ent[v]
#                 label_mapper[dc_ent[v]] = v

#             if k in mapper.keys():
#                 full_text = re.sub(k,mapper[k],full_text)
#             else:
#                 full_text = re.sub(k,dc_ent[v],full_text)
#         else:
#             label_mapper[k] = v
    
#     full_text = clean_text(full_text)
    
#     # print(label_mapper)
#     # print(mapper)

#     tokenized_text = tokenize_with_spacy(full_text, tokenizer=en_tokenizer)
#     tokens = tokenized_text['tokens']
#     # tg = get_offset_mapping(full_text, list(label_mapper.keys()))
    
#     offs = {}
#     for s in list(label_mapper.keys()):
#         res = [(m.start(0), m.end(0)) for m in re.finditer(s,full_text)]
#         if len(res):
#             offs[s] = res
            
#     labels = []
#     for tok, off in zip(tokens, tokenized_text['offset_mapping']):
#         found_label = False
#         for k, ofs in offs.items():
#             for o in ofs:
#                 if o[0] <= off[0] < o[1] or o[0] < off[1] < o[1]:
#                     labels.append(label_mapper[k])
#                     found_label = True
#                     break
#             if found_label:
#                 break
#         else:
#             labels.append('O')
        
#     return full_text,labels,tokens
