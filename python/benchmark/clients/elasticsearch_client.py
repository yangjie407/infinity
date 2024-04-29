import argparse
from typing import Any
from elasticsearch import Elasticsearch, helpers
import json
import time
from typing import List, Optional
import os
import h5py
import uuid
import numpy as np
import multiprocessing

from .base_client import BaseClient

def work(queries, collection_name, mode, topK):
    client = Elasticsearch("http://127.0.0.1:9200")
    for query in queries:
        if mode == 'fulltext':
            client.search(index=collection_name, source=["_id", "_score"], body=query, size=topK)
        else:
            client.search(index=collection_name, source=["_id", "_score"], knn=query, size=topK)
    client.close()

class ElasticsearchClient(BaseClient):
    def __init__(self,
                 mode: str,
                 options: argparse.Namespace,
                 drop_old: bool = True) -> None:
        """
        The mode configuration file is parsed to extract the needed parameters, which are then all stored for use by other functions.
        """
        with open(mode, 'r') as f:
            self.data = json.load(f)
        self.client = Elasticsearch(self.data['connection_url'])
        self.collection_name = self.data['name']
        self.path_prefix = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.threads = 16
        self.rounds = 1

    def upload_batch(self, actions: List):
        helpers.bulk(self.client, actions)

    def upload(self):
        """
        Upload data and build indexes (parameters are parsed by __init__).
        """
        if self.client.indices.exists(index=self.collection_name):
            self.client.indices.delete(index=self.collection_name)
        self.client.indices.create(index=self.collection_name, body=self.data['index'])
        batch_size = self.data["insert_batch_size"]
        dataset_path = os.path.join(self.path_prefix, self.data["data_path"])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")
        _, ext = os.path.splitext(dataset_path)
        if ext == '.json' or ext == '.jsonl':
            with open(dataset_path, 'r') as f:
                actions = []
                for i, line in enumerate(f):
                    if i % batch_size == 0 and i != 0:
                        self.upload_batch(actions)
                        actions = []
                    record = json.loads(line)
                    if '_id' in record:
                        record['id'] = record['_id']
                        del record['_id']
                    actions.append({"_index": self.collection_name, "_id": uuid.UUID(int=i).hex, "_source": record})
                if actions:
                    self.upload_batch(actions)
        elif ext == '.csv':
            with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                actions = []
                for i, line in enumerate(f):
                    if i % batch_size == 0 and i != 0:
                        self.upload_batch(actions)
                        actions = []
                    line = line.strip().split('\t')
                    record = {}
                    if len(line) == len(self.data['fields']):
                        for idx, field in enumerate(self.data['fields']):
                            record[field] = line[idx]
                    else:
                        continue
                    actions.append({"_index": self.collection_name, "_id": uuid.UUID(int=i).hex, "_source": record})
                if actions:
                    self.upload_batch(actions)          
        elif ext == '.hdf5' and self.data['mode'] == 'vector':
            with h5py.File(dataset_path, 'r') as f:
                actions = []
                for i, line in enumerate(f['train']):
                    if i % batch_size == 0 and i != 0:
                        self.upload_batch(actions)
                        actions = []
                    record = {self.data['vector_name']: line}
                    actions.append({"_index": self.collection_name, "_id": uuid.UUID(int=i).hex, "_source": record})
                if actions:
                    self.upload_batch(actions)
        else:
            raise TypeError("Unsupported file type")
        
        self.client.indices.forcemerge(index=self.collection_name, wait_for_completion=True)

    def parse_fulltext_query(self, query: dict) -> Any:
        key, value = list(query.items())[0]
        ret = {}
        if key == 'and':
            ret = {
                "query": {
                    "bool": {
                        "must": [{"match": item} for item in value]
                    }
                }
            }
        elif key == 'or':
            ret = {
                "query": {
                    "bool": {
                        "should": [{"match": item} for item in value]
                    }
                }
            }
        return ret

    def search(self) -> list[list[Any]]:
        """
        Execute the corresponding query tasks (vector search, full-text search, hybrid search) based on the parsed parameters.
        The function returns id list.
        """
        query_path = os.path.join(self.path_prefix, self.data["query_path"])
        results = []
        _, ext = os.path.splitext(query_path)
        if ext == '.json' or ext == '.jsonl':
            with open(query_path, 'r') as f:
                if self.data['mode'] == 'fulltext':
                    for line in f:
                        query = json.loads(line)
                        start = time.time()
                        body = self.parse_fulltext_query(query)
                        result = self.client.search(index=self.collection_name,
                                                    source=["_id", "_score"],
                                                    body=body,
                                                    size=self.data['topK'])
                        end = time.time()
                        latency = (end - start) * 1000
                        result = [(uuid.UUID(hex=hit['_id']).int, hit['_score']) for hit in result['hits']['hits']]
                        result.append(latency)
                        results.append(result)
        elif ext == '.hdf5' and self.data['mode'] == 'vector':
            with h5py.File(query_path, 'r') as f:
                for query in f['test']:
                    knn = {
                        "field": self.data["vector_name"],
                        "query_vector": query,
                        "k": self.data["topK"],
                        "num_candidates": 200
                    }
                    start = time.time()
                    result = self.client.search(index=self.collection_name,
                                                source=["_id", "_score"],
                                                knn=knn,
                                                size=self.data["topK"])
                    end = time.time()
                    latency = (end - start) * 1000
                    result = [(uuid.UUID(hex=hit['_id']).int, hit['_score']) for hit in result['hits']['hits']]
                    result.append(latency)
                    results.append(result)
        else:
            raise TypeError("Unsupported file type")
        return results
    
    def search_parallel(self) -> list[list[Any]]:
        query_path = os.path.join(self.path_prefix, self.data["query_path"])
        _, ext = os.path.splitext(query_path)
        results = []
        queries = [[] for _ in range(self.threads)]
        total_queries_count = 0
        if ext == '.hdf5':
            with h5py.File(query_path, 'r') as f:
                for i, line in enumerate(f['test']):
                    total_queries_count += 1
                    knn = {
                        "field": self.data["vector_name"],
                        "query_vector": line,
                        "k": self.data["topK"],
                        "num_candidates": 200
                    }
                    queries[i % self.threads].append(knn)
        elif ext == '.json' or ext == '.jsonl':
            with open(query_path, 'r') as f:
                for i, line in enumerate(f):
                    total_queries_count += 1
                    query = json.loads(line)
                    if self.data['mode'] == 'fulltext':
                        match_condition = self.parse_fulltext_query(query)
                        queries[i % self.threads].append(match_condition)
        else:
            raise TypeError("Unsupported file type")
        for i in range(self.rounds):
            p = multiprocessing.Pool(self.threads)
            start = time.time()
            for idx in range(self.threads):
                if self.data['mode'] == 'fulltext':
                    p.apply_async(work, args=(queries[idx], self.collection_name, self.data['mode'], self.data['topK']))
                else:
                    p.apply_async(work, args=(queries[idx], self.collection_name, "vector", self.data['topK'],))
            p.close()
            p.join()
            end = time.time()
            dur = end - start
            results.append(f"Round {i + 1}:")
            results.append(f"Total Dur: {dur:.2f} s")
            results.append(f"Query Count: {total_queries_count}")
            results.append(f"QPS: {(total_queries_count / dur):.2f}")

        for result in results:
            print(result)
    
    def check_and_save_results(self, results: List[List[Any]]):
        if 'ground_truth_path' in self.data:
            ground_truth_path = self.data['ground_truth_path']
            _, ext = os.path.splitext(ground_truth_path)
            precisions = []
            latencies = []
            if ext == '.hdf5':
                with h5py.File(ground_truth_path, 'r') as f:
                    expected_result = f['neighbors']
                    for i, result in enumerate(results):
                        ids = set(x[0] for x in result[:-1])
                        precision = len(ids.intersection(expected_result[i][:self.data['topK']])) / self.data['topK']
                        precisions.append(precision)
                        latencies.append(result[-1])
            elif ext == '.json' or ext == '.jsonl':
                with open(ground_truth_path, 'r') as f:
                    for i, line in enumerate(f):
                        expected_result = json.loads(line)
                        result = results[i]
                        ids = set(x[0] for x in result[:-1])
                        precision = len(ids.intersection(expected_result['expected_results'][:self.data['topK']])) / self.data['topK']
                        precisions.append(precision)
                        latencies.append(result[-1])
            
            print(f"mean_time: {np.mean(latencies)}, mean_precisions: {np.mean(precisions)}, \
                  std_time: {np.std(latencies)}, min_time: {np.min(latencies)}, \
                  max_time: {np.max(latencies)}, p95_time: {np.percentile(latencies, 95)}, \
                  p99_time: {np.percentile(latencies, 99)}")
        else:
            latencies = []
            for result in results:
                latencies.append(result[-1])
            print(f"mean_time: {np.mean(latencies)}, std_time: {np.std(latencies)}, \
                    min_time: {np.min(latencies)}, max_time: {np.max(latencies)}, \
                    p95_time: {np.percentile(latencies, 95)}, p99_time: {np.percentile(latencies, 99)}")