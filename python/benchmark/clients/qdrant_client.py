import argparse
from qdrant_client import QdrantClient as QC
from qdrant_client import models
from qdrant_client.models import VectorParams, Distance
import os
import time
import json
import h5py
from typing import Any, List, Optional
import numpy as np
import multiprocessing

from .base_client import BaseClient

def work(queries, collection_name, topK):
    client = QC('http://localhost:6333')
    for query in queries:
        client.search(
            collection_name=collection_name,
            query_vector=query,
            limit=topK,
        )
    client.close()
class QdrantClient(BaseClient):
    def __init__(self,
                 mode: str,
                 options: argparse.Namespace,
                 drop_old: bool = True) -> None:
        with open(mode, 'r') as f:
            self.data = json.load(f)
        self.client = QC(self.data['connection_url'], timeout=60)
        self.collection_name = self.data['name']
        if self.data['distance'] == 'cosine':
            self.distance = Distance.COSINE
        elif self.data['distance'] == 'L2':
            self.distance = Distance.EUCLID
        self.path_prefix = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.threads = 16
        self.rounds = 1
    
    def upload_bach(self, ids: list[int], vectors, payloads = None):
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            ),
            wait=True
        )

    def upload(self):
        # get the dataset (downloading is completed in run.py)
        if 'vector_index' in self.data:
            index_config = self.data['vector_index']
            if index_config['type'] == "HNSW":
                hnsw_params = index_config['index_params']
                hnsw_config = models.HnswConfigDiff(
                    m=hnsw_params.get("M", None),
                    ef_construct=hnsw_params.get("ef_construct", None),
                    full_scan_threshold=hnsw_params.get("full_scan_threshold", None),
                    max_indexing_threads=hnsw_params.get("max_indexing_threads", None),
                    on_disk=hnsw_params.get("on_disk", None),
                    payload_m=hnsw_params.get("payload_m", None),
                )
        else:
            hnsw_config = None
        
        self.client.recreate_collection(collection_name=self.collection_name,
                                        vectors_config=VectorParams(
                                            size=self.data['vector_size'],
                                            distance=self.distance),
                                            hnsw_config=hnsw_config)
        # set payload index
        if 'payload_index_schema' in self.data:
            for field_name, field_schema in self.data['payload_index_schema'].items():
                self.client.create_payload_index(collection_name=self.collection_name,
                                                field_name=field_name,
                                                field_schema=field_schema)
        
        # set full text index
        if 'full_text_index_schema' in self.data:
            for field_name, schema in self.data['full_text_index_schema'].items():
                field_schema = models.TextIndexParams(
                                        type="text",
                                        tokenizer=schema.get("tokenizer", None),
                                        min_token_len=schema.get("min_token_len", None),
                                        max_token_len=schema.get("max_token_len", None),
                                        lowercase=schema.get("lowercase", None),
                                    )
                self.client.create_payload_index(collection_name=self.collection_name,
                                                field_name=field_name,
                                                field_schema=field_schema)
        dataset_path = os.path.join(self.path_prefix, self.data['data_path'])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")
        vector_name = self.data['vector_name']
        batch_size=self.data['insert_batch_size']
        total_time = 0
        _, ext = os.path.splitext(dataset_path)
        if ext == '.json' or ext == '.jsonl':
            with open(dataset_path, 'r') as f:
                    vectors = []
                    payloads = []
                    for i, line in enumerate(f):
                        if i % batch_size == 0 and i != 0:
                            start_time = time.time()
                            self.upload_bach(list(range(i-batch_size, i)), vectors, payloads)
                            end_time = time.time()
                            total_time += end_time - start_time
                            # reset vectors and payloads for the next batch
                            vectors = []
                            payloads = []
                        record = json.loads(line)
                        vectors.append(record[vector_name])
                        del record[vector_name]
                        payloads.append(record)
                    if vectors:
                        start_time = time.time()
                        self.upload_bach(list(range(i-len(vectors)+1, i+1)), vectors, payloads)
                        end_time = time.time()
                        total_time += end_time - start_time
        elif ext == '.hdf5':
            with h5py.File(dataset_path) as f:
                vectors = []
                for i, line in enumerate(f['train']):
                    if i % batch_size == 0 and i != 0:
                        self.upload_bach(list(range(i-batch_size, i)), vectors)
                        vectors= []
                    vectors.append(line.tolist())
                if vectors:
                    self.upload_bach(list(range(i-len(vectors)+1, i+1)), vectors)
        else:
            raise TypeError("Unsupported file type")

    def search(self) -> list[list[Any]]:
        # get the queries path
        query_path = os.path.join(self.path_prefix, self.data['query_path'])
        results = []
        _, ext = os.path.splitext(query_path)
        if (ext == '.json' or ext == '.jsonl') and self.data['mode'] == 'vector':
            with open(query_path, 'r') as f:
                for line in f.readlines():
                    query = json.loads(line)
                    start = time.time()
                    res = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query['vector'],
                        limit=self.data.get('topK', 10),
                        with_payload=False
                    )
                    end = time.time()
                    latency = (end - start) * 1000
                    result = [(hit.id, hit.score) for hit in res]
                    result.append(latency)
                    results.append(result)
        elif ext == '.hdf5' and self.data['mode'] == 'vector':
            with h5py.File(query_path, 'r') as f:
                for line in f['test']:
                    start = time.time()
                    search_params=models.SearchParams(hnsw_ef=64, exact=False)
                    res = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=line,
                        limit=self.data.get('topK', 10),
                        search_params=search_params
                    )
                    end = time.time()
                    latency = (end-start) * 1000
                    result = [(hit.id, hit.score) for hit in res]
                    result.append(latency)
                    results.append(result)
                end = time.time()
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
                    queries[i % self.threads].append(line)
        else:
            raise TypeError("Unsupported file type")
        for i in range(self.rounds):
            p = multiprocessing.Pool(self.threads)
            start = time.time()
            for idx in range(self.threads):
                p.apply_async(work, args=(queries[idx], self.collection_name, self.data['topK']))
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

    def check_and_save_results(self, results: list[list[Any]]):
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