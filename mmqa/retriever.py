import os
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from thefuzz import fuzz
from PIL import Image
from typing import Any
from itertools import product
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import semantic_search
import torch

# class Question_Image_Retriever(object):
#     # def __init__(self, )

#     def retriever_mmqa(self, data_entry):
#         print(data_entry['images'])
#         exit()

with open('url_map.json') as f:
    url_map = json.load(f)

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

        table_contents = table['header'][0].copy()
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

class Retriever():
    def __init__(self, device='cuda') -> None:
        self.device = 'mps'
        # self.bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)
        # self.bi_encoder.max_seq_length = 256 
        # self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    def get_url_cell_map(self, example:dict[str, Any]) -> dict[str, tuple[int, int]]:
        url_cell_map = {}
        header = example['table']['header'][0]
        rows_with_links = example['table']['rows_with_links'][0]
        for i, j in product(range(len(rows_with_links)), range(len(header))):
            url = rows_with_links[i][j][2]
            if len(url) > 0:
                url_cell_map[url[0]] = (i, j)
        return url_cell_map

    def linearize_table_row(self, example: dict[str, Any], row_idx:int, mode='passage'):
        header = example['table']['header'][0]
        cell_info = example['table']['rows'][0][row_idx]
        prompt = "\nThe table row about this {0} is:\n{1}\n{2}".format(mode, '|'.join(header), '|'.join(cell_info))
        return prompt

    def retrieve_oracle(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        url_cell_map = self.get_url_cell_map(example)
        # passages
        support_id = set([doc_id for doc_id, doc_part in zip(example['supporting_context']['doc_id'], example['supporting_context']['doc_part']) 
                      if doc_part == 'text'])
        passages_info = []
        for id, url, title, text in zip(example['passages']['id'], 
                                        example['passages']['url'], 
                                        example['passages']['title'], 
                                        example['passages']['text']):
            if id not in support_id:
                continue
            passage_info = {
                "url": url, 
                "title": title, 
                "text": text, 
                "cell": url_cell_map[url] if url in url_cell_map else None
            }
            passages_info.append(passage_info)
        
        # images
        support_id = set([doc_id for doc_id, doc_part in zip(example['supporting_context']['doc_id'], example['supporting_context']['doc_part']) 
                      if doc_part == 'image'])
        images_info = []
        for id, url, title, path in zip(example['images']['id'], 
                                        example['images']['url'], 
                                        example['images']['title'], 
                                        example['images']['pic']):
            if id not in support_id:
                continue
            image_info = {
                "url": url, 
                "title": title, 
                "path": path, 
                "caption": url_map[url]['image']['new_caption'], 
                "cell": url_cell_map[url] if url in url_cell_map else None
            }
            images_info.append(image_info)
        
        return passages_info, images_info

    def retrieve_rerank(self, query:str, passages:list[str], retrieve_top_k:int=10):
        # retrieve
        passages_embeddings = self.bi_encoder.encode(passages, convert_to_tensor=True)
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        hits = semantic_search(question_embedding, passages_embeddings, top_k=retrieve_top_k)[0]

        # rerank
        cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
        cross_score = self.cross_encoder.predict(cross_inp)
        for idx in range(len(hits)):
            hits[idx]['cross_score'] = cross_score[idx]
        #hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)

        return hits

    def retrieve(self, 
                 example: dict[str, Any], 
                 use_caption:bool=True, 
                 retrieve_top_k:int=10, 
                 passage_rerank_top_k:int=3, 
                 image_rerank_top_k:int=5) -> list[dict[str, Any]]:
        url_cell_map = self.get_url_cell_map(example)
        id_score_map = {}
        if os.path.exists('id_score_map.json'):
            with open('id_score_map.json') as f:
                id_score_map = json.load(f)
        question_id = example['id']
        question = 'Question: {0}'.format(example['question'])

        # retrieve passages
        retrieved_passages = []
        passages_hits = []
        if len(example['passages']['url']) > 0:
            passages = []
            for url, title, text in zip(example['passages']['url'], 
                                        example['passages']['title'], 
                                        example['passages']['text']):
                passage = "This passage is about {0}. Its content is: {1}".format(title, text)
                if url in url_cell_map:
                    passage += self.linearize_table_row(example, url_cell_map[url][0])
                passages.append(passage)
            if question_id in id_score_map:
                passages_hits = sorted(id_score_map[question_id]['passage'], key=lambda x: x['score'], reverse=True)
                if len(passages_hits) > retrieve_top_k:
                    passages_hits = passages_hits[:retrieve_top_k]
            else:
                passages_hits = self.retrieve_rerank(question, passages, retrieve_top_k=retrieve_top_k)
            passages_hits = sorted(passages_hits, key=lambda x: x['cross_score'], reverse=True)
            if len(passages_hits) > passage_rerank_top_k:
                passages_hits = passages_hits[:passage_rerank_top_k]
            for hit in passages_hits:
                idx = hit['corpus_id']
                url = example['passages']['url'][idx]
                retrieved_passages.append({
                    "url": url, 
                    "title": example['passages']['title'][idx], 
                    "text": example['passages']['text'][idx], 
                    "cell": url_cell_map[url] if url in url_cell_map else None
                })

        # retrieve images
        retrieved_images = []
        images_hits = []
        if len(example['images']['url']) > 0:
            if use_caption:
                images = []
                for url, title, path in zip(example['images']['url'], 
                                            example['images']['title'], 
                                            example['images']['pic']):
                    image = "This image is about {0}. Here are the caption describe this image: {1}".format(title, url_map[url]['image']['new_caption'])
                    if url in url_cell_map:
                        passage += self.linearize_table_row(example, url_cell_map[url][0])
                    images.append(image)
                if question_id in id_score_map:
                    images_hits = sorted(id_score_map[question_id]['image'], key=lambda x: x['score'], reverse=True)
                    if len(images_hits) > retrieve_top_k:
                        images_hits = images_hits[:retrieve_top_k]
                else:
                    images_hits = self.retrieve_rerank(question, images, retrieve_top_k=retrieve_top_k)
                images_hits = sorted(images_hits, key=lambda x: x['cross_score'], reverse=True)
                if len(images_hits) > image_rerank_top_k:
                    images_hits = images_hits[:image_rerank_top_k]
                for hit in images_hits:
                    idx = hit['corpus_id']
                    url = example['images']['url'][idx]
                    retrieved_images.append({
                        "url": url, 
                        "title": example['images']['title'][idx], 
                        "path": example['images']['pic'][idx], 
                        "caption": url_map[url]['image']['new_caption'], 
                        "cell": url_cell_map[url] if url in url_cell_map else None
                    })
            else:
                raise NotImplementedError
        
        return retrieved_passages, retrieved_images

if __name__ == '__main__':
    retriever = Retriever()
    from datasets import load_dataset

    mmqa_dev = load_dataset(path='../data/mmqa.py', cache_dir='../data/mmqa_cache')['validation']

    new_dataset_split_loaded = []
    for data_item in mmqa_dev:
        data_item['table']['page_title'] = data_item['table']['title']
        new_dataset_split_loaded.append(data_item)
    mmqa_dev = new_dataset_split_loaded
    mmqa_dev = mmqa_dev[:-1]

    from tqdm import tqdm

    #threshold = [0.1 * (i + 1)]
    
    scores = {}
    for example in tqdm(mmqa_dev):
        question_id = example['id']
        passage_golden = [int(id in example['supporting_context']['doc_id']) for id in example['passages']['id']]
        image_golden = [int(id in example['supporting_context']['doc_id']) for id in example['images']['id']]
        passages_hits, images_hits = retriever.retrieve(example, retrieve_top_k=max(len(passage_golden), len(image_golden)))
        for idx in range(len(passages_hits)):
            passages_hits[idx]['score'] = float(passages_hits[idx]['score'])
            passages_hits[idx]['cross_score'] = float(passages_hits[idx]['cross_score'])
        for idx in range(len(images_hits)):
            images_hits[idx]['score'] = float(images_hits[idx]['score'])
            images_hits[idx]['cross_score'] = float(images_hits[idx]['cross_score'])
        scores[question_id] = {
            "passage": passages_hits, 
            "image": images_hits
        }
    with open('id_score_map.json', 'w') as f:
        json.dump(scores, f)

    # retrieve_top_ks = [3, 5, 7, 10, 15, 20, 100]
    # rerank_top_ks = [1, 2, 3, 4, 5, 7, 9]

    # import evaluate
    
    # with open('scores.json') as f:
    #     scores = json.load(f)

    # from itertools import product

    # for retrieve_k, rerank_k in product(retrieve_top_ks, rerank_top_ks):
    #     if retrieve_k < rerank_k:
    #         continue
    #     pevaluater = evaluate.combine(["f1", "precision", "recall"])
    #     ievaluater = evaluate.combine(["f1", "precision", "recall"])
    #     evaluater = {'passage': pevaluater, 'image': ievaluater}
    #     for score in tqdm(scores):
    #         for key in ['passage', 'image']:
    #             golden = score[key]['golden']
    #             prediction = [0 for _ in range(len(golden))]
    #             hits = score[key]['hits']
    #             hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    #             if retrieve_k < len(hits):
    #                 hits = hits[:retrieve_k]
    #             hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)
    #             if rerank_k < len(hits):
    #                 hits = hits[:rerank_k]
    #             for hit in hits:
    #                 prediction[hit['corpus_id']] = 1
    #             evaluater[key].add_batch(prediction, golden)

    #     print(f'retrieve k: {retrieve_k} rerank k: {rerank_k}')
    #     print(pevaluater.compute())
    #     print(ievaluater.compute())