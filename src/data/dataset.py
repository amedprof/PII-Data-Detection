import re
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tqdm.auto import tqdm
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


try:
    from faker import Faker
    fake = Faker()
    Faker.seed(0)
except:
    print('No faker installed')
    
try:
    from spacy.lang.en import English
    EN_TOK = English().tokenizer
except:
    print("No spacy")

    
    
def process_regex(pattern, reverse=False):
    replacements = {
        '(': r'\(',
        ')': r'\)',
        '[': r'\[',
        ']': r'\]',
        '|': r'\|',
        '?': r'\?',
        '*': r'\*',
        '+': r'\+'
    }
    
    if reverse:
        replacements = {v: k for k, v in replacements.items()}
    
    for old, new in replacements.items():
        pattern = pattern.replace(old, new)
    
    return pattern



LABEL2TYPE = ('NAME_STUDENT','EMAIL','USERNAME','ID_NUM', 'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS','O')
TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}
ID_TYPE = {"0-0":0,"0-1":1,
           "1-0":2,"1-1":3,
           "2-0":4,"2-1":5,
           "3-0":6,"3-1":7,
           "4-0":8,"4-1":9,
           "5-0":10,"5-1":11,
           "6-0":12,"6-1":13
          }

ID_NAME = {"0-0":"B-NAME_STUDENT","0-1":"I-NAME_STUDENT",
           "1-0":"B-EMAIL","1-1":"I-EMAIL",
           "2-0":"B-USERNAME","2-1":"I-USERNAME",
           "3-0":"B-ID_NUM","3-1":"I-ID_NUM",
           "4-0":"B-PHONE_NUM","4-1":"I-PHONE_NUM",
           "5-0":"B-URL_PERSONAL","5-1":"I-URL_PERSONAL",
           "6-0":"B-STREET_ADDRESS","6-1":"I-STREET_ADDRESS",
           "7-0":"O","7-1":"O"
          }

RE_ID_PHONE = r"""(\(?\+\s*\d{1,4}\s*\)?\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\s{0,2}\d{0,5}|\(?\+\s*\d{1,4}\s*\)?\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s{0,2}\d{0,5}|\(?\s*\d{1,4}\s*\)?\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s{0,2}\d{0,5}\s*[\.\-x]?\d{1,5}\s{0,2}\d{0,5}|\(?\s*\d{1,4}\s*\)?\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s*[\.\-x]?\d{1,5}\s{0,2}\d{0,5}|\(\s*\d{3}\s*\)\s*\d{3}\s*\-\s*\d{4}\s*\w{0,3}(\s*\d{1,8}\s*)?|\b\d{2,}-\d{2,}\.\d{2,}\.\d{2,}\.\d{2,}\b|\b\d{2,}-\d{2,}\-\d{2,}\-\d{2,}\-\d{2,}\b|\b\d{2,}\-\d{2,}\-\d{2,}\-\d{2,}\b|\b\d{2,}\.\d{2,}\.\d{2,}\.\d{2,}\b|\d{3}\s*\.\s*\d{3}\s*\.\s*\d{1,5}|\d{3}\s*\-\s*\d{3}\s*\-\s*\d{1,5}|\d{3}\s*x\s*\d{3}\s*x\s*\d{1,5}|\d{1,3}\s{0,2}\d{1,}\s{0,2}\d{1,}|\b\d{1,}\s*\d{1,}\s*\d{1,}|\b\d{2,}\-\d{2,}\-\d{2,}\b|\b\d{2,}\.\d{2,}\.\d{2,}\b|\b\d{1,}-\d{1,}|[\w\.\:\-\_\|]*\d{6,})"""
REGEX_COMPILE = re.compile(RE_ID_PHONE)


## =============================================================================== ##
class FeedbackDataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 use_re  = False,
                 add_text_prob=1.,
                 replace_text_prob=1.,
                 attrib_to_replace = ['NAME_STUDENT','EMAIL','USERNAME','ID_NUM',
                                      'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS']
                 ):
        
        self.tokenizer = tokenizer
        self.df = df
        self.use_re = use_re
        self.attrib_to_replace = attrib_to_replace

        if "labels" in self.df.columns:
            for name in LABEL2TYPE:
                self.df[name] = self.df['labels'].transform(lambda x:len([i for i in x if i.split('-')[-1]==name ]))
            self.df['has_label'] = (self.df['labels'].transform(lambda x:len([i for i in x if i!="O" ]))>0)*1
        print(f'Loaded {len(self)} samples.')

        assert 0 <= add_text_prob <= 1
        assert 0 <= replace_text_prob <= 1
        self.add_text_prob = add_text_prob
        self.replace_text_prob = replace_text_prob

        if self.use_re:
            print('using re aggregation')

        if self.add_text_prob:
            print(f'using add_text_prob {add_text_prob}')
        if self.replace_text_prob:
            print(f'using replace_text_prob {replace_text_prob}')
    # ======================================================================================== #
    def __len__(self):
        return len(self.df)
    # ======================================================================================== #
    def __getitem__(self, index):
        
        df = self.df.iloc[index]
        text_id = df['document']
        
        ## Adding space tokens
        if len(self.tokenizer.encode("\n\n"))==2:
            text = self.clean_text(df['full_text'].replace("\n\n"," | ").replace("\n"," [BR] "))
            txt_tokens = [self.clean_text(x.replace("\n\n"," | ").replace("\n"," [BR] ")) for x in df['tokens']]
        else:
            text = self.clean_text(df['full_text'])
            txt_tokens = [self.clean_text(x) for x in df['tokens']]
            
        labels = df['labels']
        has_label = df['has_label']
        offset_mapping_init = self.get_offset_mapping(text,txt_tokens)
        
        ##############   Augmentation   #############
        # Replace PII
#         prob = 
        if (np.random.random()< self.replace_text_prob) and (has_label>0):
            # print("rep")
            try:
                text,labels,txt_tokens = self.create_mapper_n_clean(text,labels,offset_mapping_init,
                                                                        attribut=self.attrib_to_replace)
                offset_mapping_init = self.get_offset_mapping(text, txt_tokens)
            except:
                pass

    
        # Add PII
        if np.random.random() < self.add_text_prob:
            # print("add")
            try:

                new_text,new_tokens,new_labels,new_offset_mapping = self.generate_fake_data()
                
            
                
                text,txt_tokens,labels,offset_mapping_init  = self.add_text(text,txt_tokens,labels,offset_mapping_init,
                                                                   new_text,new_tokens,new_labels,new_offset_mapping)
            except:
                pass

        ### Convert to regex space
        if self.use_re:
            re_offset_mapping,spacy_to_re,re_tokens,spacy_to_re_unique,re_labels = self.spacy_to_re_off(text,txt_tokens,offset_mapping_init,labels)
        else:
            spacy_to_re = np.arange(len(offset_mapping_init)).tolist()
            re_offset_mapping = offset_mapping_init
            re_tokens = txt_tokens
            spacy_to_re_unique = np.arange(len(offset_mapping_init)).tolist()
            re_labels = labels

        hf_tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(hf_tokens['input_ids'])
        attention_mask = torch.LongTensor(hf_tokens['attention_mask'])
        hf_offset_mapping = np.array(hf_tokens['offset_mapping'])

        num_tokens = len(input_ids)

        # token slices of words
        woff = np.array(re_offset_mapping)
        toff = np.array(hf_offset_mapping)
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1)
#         assert (ious > 0).any(-1).all()

        word_boxes = []
#         err = []
        for i,row in enumerate(ious):
            inds = row.nonzero()[0]
            try:
                word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
            except:
                word_boxes.append([-100, 0, -99, 1])
#                 err.append(i)
                
        word_boxes = torch.FloatTensor(word_boxes)


        gt_spans = []        
        for i,label in enumerate(re_labels):
            gt_spans.append([i,TYPE2LABEL[label.split('-')[-1] if label!="O" else "O"],0 if label.split('-')[0]=="B" else 1])
            
        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
#         if np.random.random() < self.mask_prob:
#             all_inds = np.arange(1, len(input_ids) - 1)
#             n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
#             np.random.shuffle(all_inds)
#             mask_inds = all_inds[:n_mask]
#             input_ids[mask_inds] = self.tokenizer.mask_token_id

        return dict(
                    text_id=text_id,
                    text=text,
                    labels = re_labels,
                    re_tokens = re_tokens,
                    spacy_to_re = spacy_to_re,
                    spacy_to_re_unique = spacy_to_re_unique,
                    tokens = df['tokens'],
                    tokens_clean = txt_tokens,
                    input_ids=input_ids,
                    offset_mapping_init = offset_mapping_init,
                    re_offset_mapping = re_offset_mapping,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)
    # ======================================================================================== #
    def name_student(self,v):
        # Add Name for Name/ and Mobile/Tel for phone Email for mail
        text = random.choices([f"Reflection – Visualization {v}",f'Person {v}',
                       f"STORYTELLER {v}",f"STORY TELLING {v}",f"{v}"],k=1,weights = [0.125,0.125,0.125,0.125,0.5])[0]

        return text
    # ======================================================================================== #
    def generate_fake_data(self):

        data = self.generate_random_data_with_probabilities()

        NB_PII_MAX = random.choice([1,2,3])
        piis_ent = random.sample(list(data.keys()),k=NB_PII_MAX)


        full_text = ""
        tokens = []
        labels = []
        offset_mapping = []
        off = 0
        for num, ent in enumerate(piis_ent):

            if ent=="NAME_STUDENT":
                text = self.name_student(data[ent])
                if num ==0:
                    full_text = text
                else:
                    off = off + 1 
                    full_text = full_text + " " + text

                toks = self.tokenize_with_spacy(text)
                tokns = toks['tokens']
                offset = toks['offset_mapping']
                labs = np.array(["O"]*len(tokns),dtype='<U50')
                idx = [i for i,x in enumerate(tokns) if x in data[ent]][0]
                labs[idx:] = "NAME_STUDENT"

                tokens = tokens + tokns
                labels = labels + labs.tolist()
                new_offset = [(x[0]+off,x[1]+off) for x in offset]
                offset_mapping = offset_mapping + new_offset
                off = offset_mapping[-1][1]

            else:
                text = data[ent]

                if num ==0:
                    full_text = text
                else:
                    off = off+1
                    full_text = full_text + " " + text

                toks = self.tokenize_with_spacy(text)
                tokns = toks['tokens']
                offset = toks['offset_mapping']
                labs = [ent]*len(tokns)

                tokens = tokens + tokns
                labels = labels + labs

                new_offset = [(x[0]+off,x[1]+off) for x in offset]
                offset_mapping = offset_mapping + new_offset
                off = offset_mapping[-1][1]

        return full_text,tokens,labels,offset_mapping

    def remove_double_spaces(self,text):
        # Use a regular expression to replace consecutive spaces with a single space
        # cleaned_text = re.sub(r'\s{2,}', ' | ', text)
        cleaned_text = re.sub(r'\s+', ' ', text)
        return cleaned_text

    def clean_text(self,text):
        text = text.replace(u'\x9d', u' ')
        # text = resolve_encodings_and_normalize(text)
        # text = text.replace(u'\xa0', u' ')
        # text = text.replace(u'\x85', u'\n')
        text = self.remove_double_spaces(text)
        text = text.strip()
        return text

    # ======================================================================================== #
    def find_successive_numbers(self,input_array):
        result = []
        current_sublist = []

        for num in input_array:
            if not current_sublist or num == current_sublist[-1] + 1:
                current_sublist.append(num)
            else:
                result.append(current_sublist)
                current_sublist = [num]

        if current_sublist:
            result.append(current_sublist)

        return result
    # ======================================================================================== #

    def generate_random_number(self,length):
        return ''.join(random.choice('0123456789') for _ in range(length))
    # ======================================================================================== #
    def generate_fake_social_media_urls(self,num_urls=1):
        social_media_platforms = {
            'LinkedIn': 'linkedin.com/in/',
            'YouTube': 'youtube.com/c/',
            'Instagram': 'instagram.com/',
            'GitHub': 'github.com/',
            'Facebook': 'facebook.com/',
            'Twitter': 'twitter.com/'
        }

        fake_social_media_urls = []

        for _ in range(num_urls):
            fake_user_name = fake.user_name()
            platform, domain = random.choice(list(social_media_platforms.items()))
            fake_url = f'https://{domain}{fake_user_name}'
            fake_social_media_urls.append(fake_url)

        return fake_social_media_urls[0]
    # ======================================================================================== #
    def generate_random_data_with_probabilities(self):

        name = random.choices([fake.name(),fake.first_name(), fake.last_name()],
                              weights = [0.7,0.15,0.15], k = 1)[0]  #generic.person.full_name()
        phone_number =  fake.phone_number()
        username = fake.user_name()
        email = fake.ascii_free_email()
        address = fake.address()
        id_num = random.choices([fake.passport_number(),fake.bban(),
                                 fake.iban(),self.generate_random_number(12)],k=1,weights = [0.1,0.10,0.15,0.65])[0]
        url_pers = self.generate_fake_social_media_urls()

        ret = dict(
                  NAME_STUDENT=name,
                  EMAIL=email,
                  USERNAME=username,
                  ID_NUM=id_num,
                  URL_PERSONAL=url_pers,
                  PHONE_NUM=phone_number,
                  STREET_ADDRESS=address
                  )

        for k,v in ret.items():
            ret[k] = self.clean_text(v.replace("\n\n"," | ").replace("\n"," [BR] "))
        return ret
    # ======================================================================================== #
    def generate_ent(self,text,labels,offset_mapping):

        idx_lab = np.argwhere(np.array(labels)!="O").reshape(-1)
        pos = self.find_successive_numbers(idx_lab)
        lab = np.array(labels)


        ent = {}
        ent_order = []
        ent_offset_in_order = []
        for i,p in enumerate(pos):
            l = [x.split('-')[-1] for x in lab[p]]

            if len(np.unique(l))>1:
                px = self.successive_positions(l)
                for pp in px:
                    full_name = text[offset_mapping[p[pp[0]]][0]:offset_mapping[p[pp[-1]]][1]].strip()
                    ent[full_name] = l[pp[-1]]
                    ent_order.append(full_name)
                    ent_offset_in_order.append((offset_mapping[p[pp[0]]][0],
                                                offset_mapping[p[pp[-1]]][1]))

            else:
                full_name = text[offset_mapping[p[0]][0]:offset_mapping[p[-1]][1]].strip()
                ent[full_name] = l[-1]
                ent_order.append(full_name)
                ent_offset_in_order.append((offset_mapping[p[0]][0],offset_mapping[p[-1]][1]))

        return ent,ent_order,ent_offset_in_order
    # ======================================================================================== #
    def tokenize_with_spacy(self,text,tok=EN_TOK):
        tokenized_text = tok(text)
        tokens = [token.text for token in tokenized_text]
        offset_mapping = [(token.idx,token.idx+len(token)) for token in tokenized_text]
        return {'tokens': tokens, 'offset_mapping': offset_mapping}
    
    def successive_positions(self,input_list):
        result = []
        current_group = []
        prev_element = None

        for i, element in enumerate(input_list):
            if element == prev_element:
                current_group.append(i)
            else:
                if current_group:
                    result.append(current_group)
                current_group = [i]
            prev_element = element

        if current_group:
            result.append(current_group)

        return result

    # ======================================================================================== #
    def get_offset_mapping(self,full_text, tokens):
        offset_mapping = []

        current_offset = 0
        for token in tokens:
            start, end = self.get_text_start_end(full_text, token, search_from=current_offset)
            offset_mapping.append((start, end))
            current_offset = end

        return offset_mapping
    # ======================================================================================== #
    def create_mapper_n_clean(self,full_text,labels,offset_mapping,attribut=["NAME_STUDENT"]):
        ent,ent_order,ent_offset_in_order = self.generate_ent(full_text,labels,offset_mapping)

        mapper = {}
        label_mapper = {}
        new_tokens = []
        txt_added = 0
        for num,k in enumerate(ent_order):
            v = ent[k]

            if v in attribut:      
                dc_ent = self.generate_random_data_with_probabilities()
                mapper[k] = dc_ent[v]
                label_mapper[dc_ent[v]] = v

                old_len = len(full_text)
                if k in mapper.keys():
                    full_text = full_text[:ent_offset_in_order[num][0]+txt_added] +" " +mapper[k] + " "+full_text[ent_offset_in_order[num][-1]+txt_added:]
                    txt_added+= len(full_text)-old_len

                    new_tokens.append(mapper[k])
                else:
                    full_text = full_text[:ent_offset_in_order[num][0]+txt_added] + " "+dc_ent[v] +" "+ full_text[ent_offset_in_order[num][-1]+txt_added:]

                    new_tokens.append(dc_ent[v])
                    txt_added+= len(full_text)-old_len
            else:
                label_mapper[k] = v
                new_tokens.append(k)

        full_text = self.clean_text(full_text)

        tokenized_text = self.tokenize_with_spacy(full_text)
        tokens = tokenized_text['tokens']
        tg = self.get_offset_mapping(full_text, new_tokens)


        woff = np.array(tokenized_text['offset_mapping'])
        labels = np.array(["O"]*len(woff),dtype='<U50')

        toff = np.array(tg)
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1)


        for i,row in enumerate(ious):
            inds = row.nonzero()[0]
            if len(inds):
                labels[i] = label_mapper[new_tokens[inds[0]]]

        labels = labels.tolist()

        return full_text,labels,tokens

    # ======================================================================================== #
    def get_text_start_end(self,txt, s, search_from=0):
        txt = txt[int(search_from):]
        try:
            idx = txt.find(s)
            if idx >= 0:
                st = idx
                ed = st + len(s)
            else:
                raise ValueError('Error')
        except:
            res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
            if len(res):
                st, ed = res[0][0], res[0][1]
            else:
                m = SequenceMatcher(None, s, txt).get_opcodes()
                for tag, i1, i2, j1, j2 in m:
                    if tag == 'replace':
                        s = s[:i1] + txt[j1:j2] + s[i2:]
                    if tag == "delete":
                        s = s[:i1] + s[i2:]

                res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
                if len(res):
                    st, ed = res[0][0], res[0][1]
                else:
                    idx = txt.find(s)
                    if idx >= 0:
                        st = idx
                        ed = st + len(s)
                    else:
                        st, ed = 0, 0
        return st + search_from, ed + search_from
    
    # ======================================================================================== #
    def find_patterns(self,text,regex):
        matches = [(match.group(0), match.start(), match.end()) for match in regex.finditer(text)]
        offsets = self.strip_offset_mapping(text,[(m[1],m[2]) for m in matches])
        return [m[0].strip() for m in matches],offsets
    # ======================================================================================== #
    def spacy_to_re_off(self,text,tokens,offset_mapping_init,labels):
        

        lab,off_matc = self.find_patterns(text,REGEX_COMPILE)


        if len(lab):

            spacy_to_re = []
            re_oof = []
            tokens_re = []

            # token slices of words
            woff = np.array(offset_mapping_init)
            toff = off_matc
            wx1, wx2 = woff.T
            tx1, tx2 = toff.T
            ix1 = np.maximum(wx1[..., None], tx1[None, ...])
            ix2 = np.minimum(wx2[..., None], tx2[None, ...])
            ux1 = np.minimum(wx1[..., None], tx1[None, ...])
            ux2 = np.maximum(wx2[..., None], tx2[None, ...])
            ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1+1e-12)

            for i,(spcy_of_set,tok,row) in enumerate(zip(offset_mapping_init,tokens,ious)):
                inds = row.nonzero()[0]
                if len(inds):
                    spacy_to_re.append(inds[0])
                    re_oof.append((off_matc[inds[0]].tolist()[0],off_matc[inds[0]].tolist()[1]))
                    tokens_re.append(lab[inds[0]] if len(lab[inds[0]])>len(tok) else tok)
                else:
                    spacy_to_re.append(len(lab)+i)
                    re_oof.append(spcy_of_set)
                    tokens_re.append(tok)

            re_oof = [(x,i) for i, x in enumerate(re_oof) if re_oof.index(x) == i]
            spacy_to_re_unique = [spacy_to_re[x[1]] for x in re_oof]
            labels_re = [labels[x[1]] for x in re_oof]
            re_oof = [x[0] for x in re_oof]
            
        else:
            spacy_to_re = np.arange(len(offset_mapping_init)).tolist()
            re_oof = offset_mapping_init
            tokens_re = tokens
            spacy_to_re_unique = np.arange(len(offset_mapping_init)).tolist()
            labels_re = labels

        return re_oof,spacy_to_re,tokens_re,spacy_to_re_unique,labels_re
    # ======================================================================================== #
    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)
    # ======================================================================================== #
    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)
    # ======================================================================================== #
    def custom_distribution(self,n):
        distribution = [0] * n
        middle_index = n // 2
        for i in range(middle_index):
            distribution[i] = (middle_index - i) / middle_index
            distribution[n - 1 - i] = (middle_index - i) / middle_index
        return distribution
    # ======================================================================================== #
    def add_text(self,full_text,tokens,labels,offset_mapping,
                 new_text,new_tokens,new_labels,new_offset_mapping):
        try:
            s = full_text.split('|')
            prob_dist = self.custom_distribution(len(s))
            id_ = random.choices(np.arange(len(s)),k=1,weights = prob_dist)[0]

            idx = [len(s[i]) for i in range(id_+1)]
            idx = sum(idx)
            full_text = full_text[:idx+id_+1] +" "+ new_text + full_text[idx+id_+1:]


            t_idx = [i for i,x in enumerate(offset_mapping) if x[1]==idx+id_+1][-1]+1


            tokens = tokens[:t_idx]+new_tokens+tokens[t_idx:]
            labels = labels[:t_idx]+new_labels+labels[t_idx:]


            v = offset_mapping[:t_idx][-1][1]
            new_offset_mappings = [(x[0]+v+1,x[1]+v+1) for x in new_offset_mapping]
            v1 = new_offset_mappings[-1][1]
            vx = v1-v
            
            old_offset_mapping =  [(x[0]+vx,x[1]+vx) for x in offset_mapping[t_idx:]]
            offset_mapping = offset_mapping[:t_idx]+new_offset_mappings+old_offset_mapping

        except:
            pass
            # print("Text not added")
        return full_text,tokens,labels,offset_mapping

## =============================================================================== ##
##                                                                                 ##
## =============================================================================== ##

class CustomCollator(object):
    def __init__(self, tokenizer, model):
        self.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, 'attention_window'):
            # For longformer
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/longformer/modeling_longformer.py#L1548
            self.attention_window = (model.config.attention_window
                                     if isinstance(
                                         model.config.attention_window, int)
                                     else max(model.config.attention_window))
        else:
            self.attention_window = None

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        if self.attention_window is not None:
            attention_window = self.attention_window
            padded_length = (attention_window -
                             max_seq_length % attention_window
                             ) % attention_window + max_seq_length
        else:
            padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        tokens = sample['tokens']
        tokens_clean = sample['tokens_clean']
        re_tokens = sample['re_tokens']
        spacy_to_re = sample['spacy_to_re']
        spacy_to_re_unique = sample['spacy_to_re_unique']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']

        return dict(text_id=text_id,
                    tokens_clean=tokens_clean,
                    tokens = tokens,
                    re_tokens = re_tokens,
                    spacy_to_re = spacy_to_re,
                    spacy_to_re_unique = spacy_to_re_unique,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)