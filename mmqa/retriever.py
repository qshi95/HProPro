import os
import json
from typing import Any
from itertools import product
from sentence_transformers.util import semantic_search

with open('url_map.json') as f:
    url_map = json.load(f)

class Retriever():
    def __init__(self) -> None:
        self.device = 'mps'

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

    def oracle(self, example: dict[str, Any]) -> list[dict[str, Any]]:
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
                "caption": url_map[url]['image']['caption'], 
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

        return hits

    def retrieve(self, 
                 example: dict[str, Any], 
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
            images = []
            for url, title in zip(example['images']['url'], 
                                  example['images']['title']):
                image = "This image is about {0}. Here are the caption describe this image: {1}".format(title, url_map[url]['image']['caption'])
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
                    "caption": url_map[url]['image']['caption'], 
                    "cell": url_cell_map[url] if url in url_cell_map else None
                })
        
        return retrieved_passages, retrieved_images