import numpy as np
from collections import defaultdict
from typing import List, Set, Dict, Tuple
import random


class GraphPartitioner3:
    def __init__(self, file_path: str, num_partitions: int):

        self.num_partitions = num_partitions

        self.edges = []
        self.edge_labels = {}
        self.adj = defaultdict(list)
        self.num_vertices = 0
        self.max_source_vertex = 0  # 保存第一列的最大值

        with open(file_path, 'r') as f:
            for line in f:
                src, dst, label = map(int, line.strip().split())
                self.edges.append((src, dst))
                self.edge_labels[(src, dst)] = label
                self.adj[src].append(dst)
                self.adj[dst].append(src)
                self.num_vertices = max(self.num_vertices, src + 1, dst + 1)
                self.max_source_vertex = max(self.max_source_vertex, src)  # 更新第一列的最大值

        self.num_edges = len(self.edges)
        self.partition_capacity = self.num_edges // num_partitions

        # 核心集和次级集
        self.core_sets = [set() for _ in range(num_partitions)]
        self.secondary_sets = [set() for _ in range(num_partitions)]

        # 外部度数
        self.external_degrees = defaultdict(int)

        # 边的分配
        self.edge_assignments = defaultdict(list)  # partition_id -> [edges]
        self.assigned_edges = set()

    def get_external_degree(self, vertex: int, partition_id: int) -> int:
        """计算顶点相对于某个分区的外部度数"""
        core = self.core_sets[partition_id]
        secondary = self.secondary_sets[partition_id]
        count = 0
        for neighbor in self.adj[vertex]:
            if neighbor not in core and neighbor not in secondary:
                count += 1
        return count

    def move_to_core(self, vertex: int, partition_id: int):
        """将顶点移动到核心集"""
        self.core_sets[partition_id].add(vertex)

        # 将邻居加入次级集
        for neighbor in self.adj[vertex]:
            if (neighbor not in self.core_sets[partition_id] and
                    neighbor not in self.secondary_sets[partition_id]):
                self.move_to_secondary(neighbor, partition_id)

    def move_to_secondary(self, vertex: int, partition_id: int):
        """将顶点移动到次级集"""
        self.secondary_sets[partition_id].add(vertex)

        # 更新外部度数
        self.external_degrees[vertex] = self.get_external_degree(vertex, partition_id)

        # 更新邻居的外部度数
        for neighbor in self.adj[vertex]:
            if neighbor in self.secondary_sets[partition_id]:
                self.external_degrees[neighbor] -= 1
                self.external_degrees[vertex] -= 1

        # 分配边
        for neighbor in self.adj[vertex]:
            if neighbor in self.core_sets[partition_id] or neighbor in self.secondary_sets[partition_id]:
                edge = tuple(sorted([vertex, neighbor]))
                self.assign_edge(edge)  # 使用统一管理函数分配边

    def assign_edge(self, edge):

        if edge in self.assigned_edges:
            return  # 如果边已经被分配过，则直接返回

        for partition_id in range(self.num_partitions):
            if len(self.edge_assignments[partition_id]) < self.partition_capacity:
                self.edge_assignments[partition_id].append(edge)
                self.assigned_edges.add(edge)
                return  # 分配成功后退出

        # 如果所有分区都满了，可以考虑其他策略，例如扩展分区容量或记录未分配的边
        print(f"Warning: Edge {edge} cannot be assigned as all partitions are full.")

    def partition(self) -> Dict[int, List[Tuple[int, int]]]:
        """执行NE分区算法"""
        vertices = set(range(self.num_vertices))

        for partition_id in range(self.num_partitions):
            # 选择种子顶点
            available_vertices = vertices - set().union(*self.core_sets)
            if not available_vertices:
                break
            seed = random.choice(list(available_vertices))
            self.move_to_core(seed, partition_id)

            # 扩展阶段
            while (len(self.secondary_sets[partition_id]) > 0 and
                   len(self.edge_assignments[partition_id]) < self.partition_capacity):

                # 选择外部度数最小的顶点
                min_vertex = min(
                    (v for v in self.secondary_sets[partition_id] if v not in self.core_sets[partition_id]),
                    key=lambda x: self.external_degrees[x],
                    default=None
                )

                if min_vertex is None:
                    break

                self.move_to_core(min_vertex, partition_id)

        return self.edge_assignments

    def save_to_C(self) -> List[Dict[int, List[int]]]:

        C = [{} for _ in range(self.num_partitions)]
        for partition_id, edges in self.edge_assignments.items():
            for edge in edges:
                v, u = edge
                if v > u:
                    v, u = u, v  # 确保v是较小的节点
                u = u - self.max_source_vertex - 1  # 调整较大节点的值
                if v not in C[partition_id]:
                    C[partition_id][v] = []
                C[partition_id][v].append(u)
        return C

    def save_to_C_itr(self) -> List[List[List[int]]]:

        C_itr = [[] for _ in range(self.num_partitions)]
        for partition_id, edges in self.edge_assignments.items():
            for edge in edges:
                v, u = edge
                if v > u:
                    v, u = u, v  # 确保v是较小的节点
                u = u - self.max_source_vertex - 1  # 调整较大节点的值
                label = self.edge_labels[edge]
                C_itr[partition_id].append([v, u, label])
        return C_itr


def process_partitions(input_file: str, num_partitions: int) -> Tuple[List[Dict[int, List[int]]], List[List[List[int]]]]:

    partitioner = GraphPartitioner3(input_file, num_partitions)
    partitioner.partition()
    C = partitioner.save_to_C()
    C_itr = partitioner.save_to_C_itr()
    return C, C_itr