from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import heapq
from collections import deque

State = Tuple[int,int]  # for grid demos

@dataclass
class SearchResult:
    found: bool
    path: List[State]
    explored: int
    cost: float

def neighbors_grid(pos: State, grid: List[List[int]]) -> List[State]:
    x,y = pos
    cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    out = []
    n = len(grid); m = len(grid[0])
    for nx,ny in cand:
        if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 0:
            out.append((nx,ny))
    return out

def bfs(start: State, goal: State, grid: List[List[int]]) -> SearchResult:
    q = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    explored = 0
    while q:
        s = q.popleft()
        explored += 1
        if s == goal:
            break
        for nb in neighbors_grid(s, grid):
            if nb not in parent:
                parent[nb] = s
                q.append(nb)
    if goal not in parent:
        return SearchResult(False, [], explored, float("inf"))
    # reconstruct
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return SearchResult(True, path, explored, len(path)-1)

def dfs(start: State, goal: State, grid: List[List[int]], depth_limit: int = 10_000) -> SearchResult:
    stack = [start]
    parent: Dict[State, Optional[State]] = {start: None}
    explored = 0
    while stack and explored < depth_limit:
        s = stack.pop()
        explored += 1
        if s == goal:
            break
        for nb in neighbors_grid(s, grid):
            if nb not in parent:
                parent[nb] = s
                stack.append(nb)
    if goal not in parent:
        return SearchResult(False, [], explored, float("inf"))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return SearchResult(True, path, explored, len(path)-1)

def ucs(start: State, goal: State, grid: List[List[int]]) -> SearchResult:
    pq = [(0.0, start)]
    parent: Dict[State, Optional[State]] = {start: None}
    dist: Dict[State, float] = {start: 0.0}
    explored = 0
    while pq:
        d, s = heapq.heappop(pq)
        explored += 1
        if s == goal:
            break
        if d != dist.get(s, float("inf")):
            continue
        for nb in neighbors_grid(s, grid):
            nd = d + 1.0
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                parent[nb] = s
                heapq.heappush(pq, (nd, nb))
    if goal not in parent:
        return SearchResult(False, [], explored, float("inf"))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return SearchResult(True, path, explored, dist[goal])

def manhattan(a: State, b: State) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(start: State, goal: State, grid: List[List[int]], h: Callable[[State,State], float] = manhattan) -> SearchResult:
    pq = [(h(start,goal), 0.0, start)]
    parent: Dict[State, Optional[State]] = {start: None}
    gscore: Dict[State, float] = {start: 0.0}
    explored = 0
    while pq:
        f, g, s = heapq.heappop(pq)
        explored += 1
        if s == goal:
            break
        if g != gscore.get(s, float("inf")):
            continue
        for nb in neighbors_grid(s, grid):
            ng = g + 1.0
            if ng < gscore.get(nb, float("inf")):
                gscore[nb] = ng
                parent[nb] = s
                heapq.heappush(pq, (ng + h(nb,goal), ng, nb))
    if goal not in parent:
        return SearchResult(False, [], explored, float("inf"))
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return SearchResult(True, path, explored, gscore[goal])
