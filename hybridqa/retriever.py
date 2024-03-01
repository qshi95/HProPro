import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from thefuzz import fuzz

# class Question_Image_Retriever(object):
#     # def __init__(self, )

#     def retriever_mmqa(self, data_entry):
#         print(data_entry['images'])
#         exit()

class Question_Passage_Retriever(object):
    def __init__(self, threshold=0.95, best_threshold=0.80):
        self.resource_path = './data/WikiTables-WithLinks'
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        stopWords = list(set(stopwords.words('english')))
        self.tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(2, 3), stop_words=stopWords)
        self.best_threshold = best_threshold
        self.threshold = threshold

    def url2text(self, url):
        if url.startswith('https://en.wikipedia.org'):
            url = url.replace('https://en.wikipedia.org', '')
        return url.replace('/wiki/', '').replace('_', ' ')

    # Finding the longest substring
    def longestSubstringFinder(self, S, T):
        S = S.lower()
        T = T.lower()
        m = len(S)
        n = len(T)
        counter = [[0]*(n+1) for x in range(m+1)]
        longest = 0
        lcs_set = set()
        for i in range(m):
            for j in range(n):
                if S[i] == T[j]:
                    c = counter[i][j] + 1
                    counter[i+1][j+1] = c
                    if c > longest:
                        lcs_set = set()
                        longest = c
                        lcs_set.add(S[i-c+1:i+1])
                    elif c == longest:
                        lcs_set.add(S[i-c+1:i+1])
        
        return longest, lcs_set

    # Measure the longest string overlap distance
    def longest_match_distance(self, str1s, str2s):
        longest_string, longest_pattern = [], []
        for str1 in str1s:
            longest_string.append([])
            longest_pattern.append([])
            for str2 in str2s:
                length, pattern = self.longestSubstringFinder(str1, str2)
                best_pattern = max(pattern, key=len) if pattern else ''
                longest_string[-1].append(1 - length / len(str1))
                longest_pattern[-1].append(best_pattern)
        return longest_string, longest_pattern


    def retriever_mmqa(self, data_entry):

        table = data_entry['table']

        # tokenize text into sentences, and retrieve based on sentences.
        ids, titles, paras, urls = [], [], [], []
        for _id, title, text, url in zip(data_entry['passages']['id'],
                                                data_entry['passages']['title'],
                                                data_entry['passages']['text'],
                                                data_entry['passages']['url']):
            
            for _ in self.tokenizer.tokenize(text):
                ids.append(_id)
                titles.append(title)
                paras.append(_)
                urls.append(url)
        
        qs = [data_entry['question']]
        try:
            para_feature = self.tfidf.fit_transform(paras)
            q_feature = self.tfidf.transform(qs)
        except Exception:
            print("failed on mmqa_table {}".format(data_entry['table']['table_id']))
            return []

        dist_match, dist_pattern = self.longest_match_distance(qs, paras)
        dist_match, dist_pattern = dist_match[0], dist_pattern[0]
        dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]


        # Find out the best matched passages based on distance
        tfidf_nodes = []
        min_dist = {}
        tfidf_best_match = ('N/A', None, 1.)
        for k, para, d in zip(urls, paras, dist):
            if d < min_dist.get(k, self.threshold):
                min_dist[k] = d
                if d < tfidf_best_match[-1]:
                    tfidf_best_match = (k, para, d)
                if d <= self.best_threshold:
                    tfidf_nodes.append((self.url2text(k), k, para, d))
        
        if tfidf_best_match[0] != 'N/A':
            if tfidf_best_match[-1] > self.best_threshold:
                k = tfidf_best_match[0]
                tfidf_nodes.append((self.url2text(k), k, tfidf_best_match[1], tfidf_best_match[2]))

        # Find the best matched paragraph string
        string_nodes = []
        min_dist = {}
        string_best_match = ('N/A', None, 1.)

        for k, para, d, pat in zip(urls, paras, dist_match, dist_pattern):
            if d < min_dist.get(k, self.threshold):
                min_dist[k] = d
                if d < string_best_match[-1]:
                    string_best_match = (k, para, pat, d)
                if d <= self.best_threshold:
                    string_nodes.append((self.url2text(k), k, para, pat, d))

        if string_best_match[0] != 'N/A':
            if string_best_match[-1] > self.best_threshold:
                k = string_best_match[0]
                string_nodes.append((self.url2text(k), k, string_best_match[1], string_best_match[2], string_best_match[3]))
    
        # consider both tf-idf and longest substring
        union_nodes = [node[:-1] for node in string_nodes if node[:-2] in [item[:-1] for item in tfidf_nodes]]

        table_contents = table['header'][0]
        for row in table['rows'][0]:
            table_contents.extend(row)
        table_contents.append(table['title'][0])
        table_contents = [item.lower() for item in list(set(table_contents))]

        ret_list = []
        for node in union_nodes:
            if all([fuzz.ratio(node[-1].strip().lower(), item)<60 for item in table_contents]):
                # if node[-1].strip()[0].isalpha():               # ensure the overlap is a complete part, not begin with punctuation（todo, need to filter）
                additional_knowledge = 'Knowledge regarding {}: {}'.format(node[0], node[-2])
                if additional_knowledge not in ret_list:
                    ret_list.append(additional_knowledge)

        return ret_list



    def retriever_hybridqa(self, data_entry):

        table_id = data_entry['table_id']
        with open('{}/request_tok/{}.json'.format(self.resource_path, table_id)) as f:
            requested_documents = json.load(f)
        with open('{}/tables_tok/{}.json'.format(self.resource_path, table_id)) as f:
            table = json.load(f)

        
        # Mapping entity link to cell, entity link to surface word
        mapping_entity = {}
        for row_idx, row in enumerate(table['data']):
            for col_idx, cell in enumerate(row):
                for i, ent in enumerate(cell[1]):
                    mapping_entity[ent] = mapping_entity.get(ent, []) + [(row_idx, col_idx)]
        
        # Convert the paragraph and question into TF-IDF features
        keys = []
        paras = []
        for k in mapping_entity:
            v = requested_documents[k]
            for _ in self.tokenizer.tokenize(v):
                keys.append(k)
                paras.append(_)
        
        qs = [data_entry['question']]
        # try:
        para_feature = self.tfidf.fit_transform(paras)
        q_feature = self.tfidf.transform(qs)
        # except Exception:
        #     print("failed on hybridqa_table {}".format(table_id))
        #     return []
        
        dist_match, dist_pattern = self.longest_match_distance(qs, paras)
        dist_match, dist_pattern = dist_match[0], dist_pattern[0]
        dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]

        # Find out the best matched passages based on distance
        tfidf_nodes = []
        min_dist = {}
        tfidf_best_match = ('N/A', None, 1.)
        for k, para, d in zip(keys, paras, dist):
            if d < min_dist.get(k, self.threshold):
                min_dist[k] = d
                if d < tfidf_best_match[-1]:
                    tfidf_best_match = (k, para, d)
                if d <= self.best_threshold:
                    for loc in mapping_entity[k]:
                        tfidf_nodes.append((self.url2text(k), loc, k, para, d))
        
        if tfidf_best_match[0] != 'N/A':
            if tfidf_best_match[-1] > self.best_threshold:
                k = tfidf_best_match[0]
                for loc in mapping_entity[k]:
                    tfidf_nodes.append((self.url2text(k), loc, k, tfidf_best_match[1], tfidf_best_match[2]))

        # Find the best matched paragraph string
        string_nodes = []
        min_dist = {}
        string_best_match = ('N/A', None, 1.)

        for k, para, d, pat in zip(keys, paras, dist_match, dist_pattern):
            if d < min_dist.get(k, self.threshold):
                min_dist[k] = d
                if d < string_best_match[-1]:
                    string_best_match = (k, para, pat, d)
                if d <= self.best_threshold:
                    for loc in mapping_entity[k]:
                        string_nodes.append((self.url2text(k), loc, k, para, pat, d))

        if string_best_match[0] != 'N/A':
            if string_best_match[-1] > self.best_threshold:
                k = string_best_match[0]
                for loc in mapping_entity[k]:
                    string_nodes.append((self.url2text(k), loc, k, string_best_match[1], string_best_match[2], string_best_match[3]))
    
        # consider both tf-idf and longest substring
        union_nodes = [node[:-1] for node in string_nodes if node[:-2] in [item[:-1] for item in tfidf_nodes]]

        table_contents = [cell[0] for cell in table['header']]
        for i, row in enumerate(table['data']):
            content = [cell[0] for cell in row]
            table_contents.extend(content)
        table_contents.append(table['title'])
        table_contents = [item.lower() for item in list(set(table_contents))]

        ret_list = []
        for node in union_nodes:
            if all([fuzz.ratio(node[-1].strip().lower(), item)<60 for item in table_contents]):
                # use cell content instead of wikipedia passage title.
                row_index, column_index = node[1]
                assert row_index < len(table['data']) and column_index < len(table['data'][0])
                cell_content = table['data'][row_index][column_index][0]

                additional_knowledge = '{}:\t{}'.format(cell_content, node[-2]) # Contain the entire sentence.
                # additional_knowledge = '{}:\t{}'.format(cell_content, node[-1]) # Only contain the same part.
                
                if additional_knowledge not in ret_list:
                    ret_list.append(additional_knowledge)

        return ret_list
