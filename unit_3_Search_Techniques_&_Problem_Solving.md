# Unit 3: Search Techniques & Problem Solving

> **Goal:** Master classical AI search algorithms — the backbone of problem-solving in AI, still used in games, navigation, planning, and robotics today.

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [State Space Search](#1-state-space-search)
4. [Problem Formulation](#2-problem-formulation)
5. [Search Trees](#3-search-trees)
6. [Breadth-First Search (BFS)](#4-breadth-first-search-bfs)
7. [Depth-First Search (DFS)](#5-depth-first-search-dfs)
8. [Heuristic Search](#6-heuristic-search)
9. [Best-First Search](#7-best-first-search)
10. [A* Search](#8-a-search)
11. [Game Playing](#9-game-playing)
12. [Minimax Algorithm](#10-minimax-algorithm)
13. [Alpha-Beta Pruning](#11-alpha-beta-pruning)
14. [Iterative Deepening](#12-iterative-deepening)
15. [Algorithm Comparison](#algorithm-comparison)
16. [Code Examples](#code-examples)
17. [Real-World Applications](#real-world-applications)
18. [Interview Notes](#interview-notes)
19. [Common Mistakes](#common-mistakes--misconceptions)
20. [Key Takeaways](#key-takeaways)
21. [Revision Questions](#revision-questions)
22. [Summary](#summary)

---

## Overview

Before machine learning dominated AI, **search algorithms** were the primary tools for building intelligent systems. They are still used everywhere:

- Google Maps finding the shortest route (A*)
- Chess engines deciding the best move (Minimax + Alpha-Beta)
- Robot navigation through unknown terrain (BFS/DFS/A*)
- Game AI opponents (Minimax)
- Puzzle solvers like Rubik's cube and 8-puzzle

The core idea is elegant:
```
AI Problem Solving = Finding a PATH through a SEARCH SPACE
                     from START to GOAL
```

Every search algorithm asks: **"Given where I am now, which way should I go to reach my goal?"**

The difference between algorithms is HOW they decide which direction to explore next.

---

## Learning Objectives

By the end of this unit, you should be able to:

- [ ] Define state space and formulate any problem as a search problem
- [ ] Construct and traverse a search tree
- [ ] Implement BFS and DFS and compare their properties
- [ ] Explain what a heuristic is and why it is needed
- [ ] Implement Best-First Search and explain its limitation
- [ ] Implement A* Search and prove why it is optimal
- [ ] Model a two-player game as a game tree
- [ ] Implement the Minimax algorithm
- [ ] Apply Alpha-Beta pruning to reduce Minimax computation
- [ ] Implement Iterative Deepening DFS
- [ ] Choose the right search algorithm for a given problem

---

## 1. State Space Search

### What is a State?

A **state** is a complete description of the world (or the problem) at a particular moment in time.

```
Example: 8-Puzzle
Initial State:       Goal State:
+---+---+---+        +---+---+---+
| 7 | 2 | 4 |        | 1 | 2 | 3 |
+---+---+---+        +---+---+---+
| 5 |   | 6 |        | 4 | 5 | 6 |
+---+---+---+        +---+---+---+
| 8 | 3 | 1 |        | 7 | 8 |   |
+---+---+---+        +---+---+---+
State = the exact arrangement of all tiles
```

### What is a State Space?

The **state space** is the set of ALL possible states the problem can be in, connected by possible actions (transitions).

```
State Space = (S, A, T, I, G)

S = set of all possible states
A = set of possible actions
T = transition function: T(state, action) -> new_state
I = initial state
G = goal state(s)
```

### Visualising the State Space

```
                [Initial State]
                      |
           +----------+----------+
           |          |          |
         [S1]       [S2]       [S3]      <- states after 1 action
           |          |
        [S1a] [S1b] [S2a] [S2b]          <- states after 2 actions
           |
        [GOAL]                            <- found!
```

**Analogy:** The state space is a map of all possible worlds. Search is about navigating that map to find the destination.

### Key Properties

| Property | Meaning |
|---|---|
| **State** | A complete snapshot of the problem |
| **Initial state** | Where we start |
| **Goal state** | Where we want to be |
| **Action** | A move from one state to another |
| **Path** | A sequence of states from start to goal |
| **Solution** | A path from initial state to goal state |
| **Cost** | The total cost of all actions in the path |

---

## 2. Problem Formulation

Before writing any search algorithm, you must **formulate the problem** clearly. Every AI search problem requires exactly 5 components:

```
+---------------------------------------------------+
|           PROBLEM FORMULATION TEMPLATE            |
+---------------------------------------------------+
| 1. INITIAL STATE   - Where do we start?          |
| 2. ACTIONS         - What moves are possible?    |
| 3. TRANSITION MODEL- Result of each action?      |
| 4. GOAL TEST       - Have we reached the goal?   |
| 5. PATH COST       - How expensive is this path? |
+---------------------------------------------------+
```

### Example 1: Romania Road Problem

**Problem:** Find a route from city Arad to city Bucharest in Romania.

```
1. Initial State:   Currently in Arad
2. Actions:         Drive to any directly connected city
3. Transition:      Drive(Arad, Sibiu) -> Currently in Sibiu
4. Goal Test:       Are we in Bucharest?
5. Path Cost:       Sum of road distances (km)
```

### Example 2: 8-Puzzle

```
1. Initial State:   [7,2,4,5,_,6,8,3,1] (arrangement of tiles)
2. Actions:         Slide tile UP, DOWN, LEFT, or RIGHT into blank
3. Transition:      Slide(UP) -> new tile arrangement
4. Goal Test:       Is arrangement = [1,2,3,4,5,6,7,8,_]?
5. Path Cost:       Number of moves (each move costs 1)
```

### Example 3: N-Queens Problem

```
1. Initial State:   Empty N x N board
2. Actions:         Place a queen in any empty column row
3. Transition:      Board with one more queen placed
4. Goal Test:       N queens placed with no two attacking each other
5. Path Cost:       Number of placements (minimise conflicts)
```

### Why Problem Formulation Matters

Good formulation = efficient search. Bad formulation = exponential explosion.

**Key insight:** The same real-world problem can have many formulations. The right formulation dramatically reduces the search space.

---

## 3. Search Trees

A **search tree** is the tree of states explored by a search algorithm.

```
          [Arad]                      <- ROOT (initial state)
         /   |   \
    [Zerind][Sibiu][Timisoara]        <- Level 1 (after 1 action)
      /           \
[Oradea]      [Rimnicu Vilcea]        <- Level 2 (after 2 actions)
                   \
               [Pitesti]              <- Level 3
                   \
               [Bucharest]            <- GOAL (solution found!)
```

### Important Terminology

| Term | Meaning |
|---|---|
| **Root** | The initial state |
| **Node** | A state in the search tree |
| **Edge** | An action connecting two nodes |
| **Leaf** | A node with no children explored yet |
| **Frontier** | The set of leaf nodes awaiting expansion |
| **Expansion** | Generating all children of a node |
| **Depth** | Distance from root to a node |
| **Branching factor (b)** | Average number of children per node |
| **Explored set** | States already visited (avoids re-visiting) |

### Tree vs Graph Search

```
Tree Search:              Graph Search:
- Can revisit states      - Tracks visited states
- Simpler to implement    - More memory but avoids loops
- May run forever         - Always terminates (finite space)
- No explored set         - Maintains explored set
```

**Always use Graph Search in practice** to avoid infinite loops.

### Node Representation

Each node in the search tree stores:

```python
class Node:
    state         # the current state
    parent        # which node generated this one
    action        # what action produced this node
    path_cost     # total cost from root to this node
    depth         # depth in the tree
```

---

## 4. Breadth-First Search (BFS)

### Definition

**BFS** explores nodes **level by level** — all nodes at depth 1 before depth 2, all at depth 2 before depth 3, and so on.

**Data structure used:** **Queue** (First-In-First-Out — FIFO)

### How BFS Works

```
Step 1: Add initial state to queue. Mark as visited.
Step 2: Dequeue the front node.
Step 3: If it is the goal, return the path.
Step 4: Otherwise, expand it — add all unvisited neighbors to the back of queue.
Step 5: Repeat from Step 2 until queue is empty (no solution) or goal found.
```

### BFS Visualisation

```
Graph:
A -- B -- D -- G(goal)
|         |
C ------- E

BFS from A to G:
Queue: [A]
  Dequeue A -> Enqueue B, C         Queue: [B, C]
  Dequeue B -> Enqueue D            Queue: [C, D]
  Dequeue C -> Enqueue E            Queue: [D, E]
  Dequeue D -> Enqueue G            Queue: [E, G]
  Dequeue E -> (already visited)    Queue: [G]
  Dequeue G -> GOAL FOUND!

Order visited: A -> B -> C -> D -> E -> G
Path: A -> B -> D -> G
```

### BFS Properties

| Property | Value | Explanation |
|---|---|---|
| **Complete** | YES | Will always find a solution if one exists |
| **Optimal** | YES (equal costs) | Finds the shallowest (fewest steps) solution first |
| **Time Complexity** | O(b^d) | b=branching factor, d=depth of solution |
| **Space Complexity** | O(b^d) | Stores ALL nodes at the frontier |

**The space complexity is the critical weakness of BFS.**

```
Branching factor b=10, solution at depth d=6:
Nodes explored = 10^1 + 10^2 + ... + 10^6 = ~1,111,111
Memory needed = 10^6 nodes = potentially gigabytes!
```

### When to Use BFS

- When the solution is **shallow** (close to the start)
- When all actions have **equal cost**
- When **completeness** is required
- When memory is not a constraint

```python
from collections import deque

def bfs(graph, start, goal):
    """
    Breadth-First Search — finds shortest path in unweighted graph.

    graph: dict of {node: [neighbors]}
    start: starting node
    goal:  target node
    Returns: path from start to goal, or None
    """

    # Queue stores (current_node, path_taken_so_far)
    queue = deque([(start, [start])])

    # Track visited nodes to avoid revisiting (graph search)
    visited = set([start])

    while queue:
        # Dequeue from the FRONT (FIFO)
        current, path = queue.popleft()

        # Goal test
        if current == goal:
            return path

        # Expand: explore all unvisited neighbors
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                # Enqueue neighbor with updated path
                queue.append((neighbor, path + [neighbor]))

    return None   # No path found


# Test
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

path = bfs(graph, 'A', 'G')
print(f"BFS Path: {' -> '.join(path)}")
# Output: BFS Path: A -> C -> F -> G
```

**Line-by-line explanation:**
- `deque([(start, [start])])` — queue initialised with start node and path [start]
- `queue.popleft()` — FIFO dequeue from the front (BFS key: always dequeue from front)
- `if current == goal` — goal test: stop if we reached destination
- `for neighbor in graph.get(current, [])` — expand current node
- `path + [neighbor]` — track the path as we go (creates new list each time)
- `return None` — exhausted all possibilities without finding goal

---

## 5. Depth-First Search (DFS)

### Definition

**DFS** explores nodes by going **as deep as possible** before backtracking.

**Data structure used:** **Stack** (Last-In-First-Out — LIFO), or recursion.

### How DFS Works

```
Step 1: Add initial state to stack.
Step 2: Pop the top node.
Step 3: If it is the goal, return the path.
Step 4: Otherwise, expand it — push all unvisited neighbors onto TOP of stack.
Step 5: Repeat from Step 2.
```

### DFS Visualisation

```
Graph (same as above):
A -- B -- D
|         |
C -- F -- E
     |
     G (goal)

DFS from A to G (explores A->B first):
Stack: [A]
  Pop A  -> Push C, B             Stack: [C, B]  (B on top — explored first)
  Pop B  -> Push E, D             Stack: [C, E, D]
  Pop D  -> no unvisited children  Stack: [C, E]
  Pop E  -> Push F                Stack: [C, F]
  Pop F  -> Push G                Stack: [C, G]
  Pop G  -> GOAL FOUND!

Order: A -> B -> D -> E -> F -> G
```

### DFS Properties

| Property | Value | Explanation |
|---|---|---|
| **Complete** | NO (with cycles) / YES (finite acyclic) | Can loop forever without visited tracking |
| **Optimal** | NO | May find a long path before a short one |
| **Time Complexity** | O(b^m) | m = maximum depth of search space |
| **Space Complexity** | O(b*m) | Only stores nodes on the current path + siblings |

**Space efficiency is DFS's key strength** — it uses linear space, unlike BFS's exponential space.

```
DFS memory = O(b * m) = 10 * 100 = 1000 nodes (manageable!)
BFS memory = O(b^d) = 10^6 nodes (huge!)
```

### DFS vs BFS

| Feature | BFS | DFS |
|---|---|---|
| **Data structure** | Queue (FIFO) | Stack (LIFO) |
| **Explores** | Level by level | As deep as possible |
| **Finds** | Shortest path (equal costs) | Any path (possibly long) |
| **Memory** | O(b^d) — high | O(b*m) — low |
| **Complete** | Yes | Only with visited tracking |
| **Optimal** | Yes (equal costs) | No |
| **Best for** | Shallow solutions, optimality | Deep solutions, memory limits |

### When to Use DFS

- When memory is very limited
- When the solution is expected to be **deep** in the tree
- When any solution is acceptable (not necessarily optimal)
- When searching large or infinite state spaces

```python
def dfs(graph, start, goal):
    """
    Depth-First Search — explores deepest nodes first.

    Uses a stack (iterative version — avoids Python recursion limit).
    """

    # Stack stores (current_node, path_taken)
    stack = [(start, [start])]

    # Track visited to avoid cycles
    visited = set([start])

    while stack:
        # Pop from the TOP (LIFO)
        current, path = stack.pop()

        # Goal test
        if current == goal:
            return path

        # Expand: push all unvisited neighbors to TOP of stack
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))

    return None


# Recursive DFS (cleaner but limited by Python recursion depth)
def dfs_recursive(graph, start, goal, path=None, visited=None):
    """Recursive implementation of DFS."""
    if path is None:
        path = [start]
    if visited is None:
        visited = set([start])

    if start == goal:
        return path

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            visited.add(neighbor)
            result = dfs_recursive(graph, neighbor, goal,
                                   path + [neighbor], visited)
            if result is not None:
                return result

    return None   # Backtrack


graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

path_dfs = dfs(graph, 'A', 'G')
path_bfs = bfs(graph, 'A', 'G')

print(f"DFS Path: {' -> '.join(path_dfs)}")
print(f"BFS Path: {' -> '.join(path_bfs)}")
# DFS may find a longer path than BFS
```

---

## 6. Heuristic Search

### What is a Heuristic?

A **heuristic** (from Greek: "to find/discover") is a smart **estimate** of how far the current state is from the goal.

```
h(n) = heuristic estimate of cost from node n to the goal

h(n) = 0        means: we are at the goal
h(n) = 100      means: we estimate 100 cost units to reach goal
```

**Analogy:** When navigating a city without a map, you might use "go towards the tall building" as a heuristic. It is not guaranteed to work, but it is usually better than random walking.

### Why Do We Need Heuristics?

Without a heuristic, search algorithms treat all directions equally. With a heuristic, we can **guide the search toward the goal**, avoiding exploring dead ends.

```
Without heuristic:              With heuristic:
Explores blindly in all         Focuses effort toward the goal
directions:

     S                               S
   / | \                           / | \
  A  B  C                         A  B  C
 /|  |  |\                       /       \
D E  F  G H    <-- explores all D(far)   H(close) <-- explores closer
              directions         ^^^                   ^^^
                                 skip!              explore this!
```

### Properties of a Good Heuristic

**Admissibility (MUST for A* to be optimal):**
```
h(n) <= h*(n)   for all nodes n

h*(n) = true cost from n to goal
h(n)  = our estimate

An admissible heuristic NEVER overestimates.
It is optimistic — it thinks the goal is closer than it really is.
```

**Consistency (also called Monotonicity):**
```
h(n) <= c(n, a, n') + h(n')

c(n, a, n') = cost of action a from n to n'

The estimated cost from n is at most the actual cost
to n' plus the estimated cost from n'.
(Triangle inequality for heuristics)
```

If a heuristic is consistent, it is also admissible.

### Example Heuristics

**For navigation (finding route between cities):**
```
h(n) = Straight-line distance from n to goal

This is admissible because road distance >= straight-line distance.
You can never get there by travelling LESS than the straight-line distance.
```

**For 8-puzzle:**
```
Heuristic 1: Number of misplaced tiles
  Count tiles not in their goal position.
  Admissible: each misplaced tile needs at least 1 move.

Heuristic 2: Manhattan distance
  Sum of (horizontal + vertical) distances each tile must travel.
  Admissible: tiles can't pass through each other.

Example:
  Current:    Goal:
  7 2 4       1 2 3
  5 _ 6       4 5 6
  8 3 1       7 8 _

  Manhattan distance for tile 7:
    Current position: (0,0), Goal position: (2,0) -> distance = 2
  Manhattan distance for tile 1:
    Current position: (2,2), Goal position: (0,0) -> distance = 4
  Total Manhattan distance (sum for all tiles) = heuristic value
```

**Dominance:** A heuristic h2 dominates h1 if h2(n) >= h1(n) for all n.
A dominant heuristic explores fewer nodes and is more efficient.

```
h_manhattan >= h_misplaced for all states
Therefore, Manhattan distance DOMINATES misplaced tiles.
Use Manhattan distance for 8-puzzle — it is more informed.
```

---

## 7. Best-First Search

### Definition

**Best-First Search** always expands the node that looks **most promising according to the heuristic** — the node with the lowest h(n).

**Data structure:** Priority Queue (min-heap on h(n))

```
At each step: expand the node with the lowest h(n) value
```

Also called **Greedy Best-First Search** because it greedily picks what looks best right now.

### How It Works

```
Evaluation function: f(n) = h(n)
(only heuristic, no path cost considered)

Romania example: goal = Bucharest, h = straight-line distance to Bucharest

Start: Arad (h=366)
  Expand Arad -> {Zerind(h=374), Sibiu(h=253), Timisoara(h=329)}
  Best: Sibiu (h=253)

  Expand Sibiu -> {Fagaras(h=176), Rimnicu Vilcea(h=193), ...}
  Best: Fagaras (h=176)

  Expand Fagaras -> {Bucharest(h=0)}
  Best: Bucharest -> GOAL!

Path found: Arad -> Sibiu -> Fagaras -> Bucharest (450 km)
(But the OPTIMAL path is 418 km via a different route!)
```

### Properties of Best-First Search

| Property | Value | Explanation |
|---|---|---|
| **Complete** | NO | Can get stuck in a loop (infinite path if not tracking visited) |
| **Optimal** | NO | May find non-optimal path |
| **Time** | O(b^m) | Heuristic helps in practice, but worst case is still exponential |
| **Space** | O(b^m) | Keeps all nodes in memory |

### Why Is Best-First Search NOT Optimal?

It only looks at **h(n)** — the estimated cost to the goal — and completely **ignores the cost already paid** to reach the current node. It might take an expensive path to a node that "looks close" to the goal.

```
Example:
       A(h=5)
      / \
   B(h=2) C(h=4)
     |        |
   D(h=1)  GOAL(cost=1)

Cost A->C->GOAL = 1+1 = 2 (actual optimal)
Cost A->B->D = 1+1  = 2  (same, but D is not the goal)

Greedy picks B first (h=2 < h=4), then D, then gets stuck.
It ignores that C leads directly to the goal cheaply!
```

**Fix:** Also consider the cost already spent to reach the node. That gives us A*.

---

## 8. A* Search

### Definition

**A* (A-star)** combines Best-First Search (heuristic) with the actual cost paid so far. It is the most widely used search algorithm in AI.

```
Evaluation function: f(n) = g(n) + h(n)

g(n) = actual cost from start to node n (path cost so far)
h(n) = estimated cost from n to goal (heuristic)
f(n) = estimated TOTAL cost of cheapest path through n

A* always expands the node with the LOWEST f(n)
```

**Intuition:**
- `g(n)` = "How much have I already paid to get here?"
- `h(n)` = "How much more will I probably pay?"
- `f(n)` = "What is my best guess for the total trip cost?"

### Why A* is Better

```
Best-First:  f(n) = h(n)         (ignores past cost — greedy)
Dijkstra:    f(n) = g(n)         (ignores future — like BFS with costs)
A*:          f(n) = g(n) + h(n)  (BOTH past and future — optimal!)
```

### A* Example — Romania Map

```
Cities and connections (km):        Straight-line to Bucharest:
Arad -- Zerind: 75                  Arad: 366
Arad -- Sibiu: 140                  Sibiu: 253
Arad -- Timisoara: 118              Fagaras: 176
Sibiu -- Fagaras: 99                Bucharest: 0
Fagaras -- Bucharest: 211           Pitesti: 100
Sibiu -- Rimnicu: 80
Rimnicu -- Pitesti: 97
Pitesti -- Bucharest: 101

A* step by step (start=Arad, goal=Bucharest):

Open list (priority on f = g + h):
  [Arad: g=0, h=366, f=366]

Step 1: Expand Arad (f=366)
  Zerind:    g=75,  h=374, f=449
  Sibiu:     g=140, h=253, f=393
  Timisoara: g=118, h=329, f=447
  Open: [Sibiu:393, Timisoara:447, Zerind:449]

Step 2: Expand Sibiu (f=393)
  Fagaras:       g=239, h=176, f=415
  Rimnicu Vilcea: g=220, h=193, f=413
  Open: [Rimnicu:413, Fagaras:415, Timisoara:447, Zerind:449]

Step 3: Expand Rimnicu (f=413)
  Pitesti: g=317, h=100, f=417
  Open: [Fagaras:415, Pitesti:417, Timisoara:447, Zerind:449]

Step 4: Expand Fagaras (f=415)
  Bucharest: g=450, h=0, f=450
  Open: [Pitesti:417, Bucharest:450, Timisoara:447, Zerind:449]

Step 5: Expand Pitesti (f=417)
  Bucharest: g=418, h=0, f=418  <- better path to Bucharest!
  Open: [Bucharest:418, Timisoara:447, Zerind:449] (old path 450 discarded)

Step 6: Expand Bucharest (f=418) -> GOAL!

Optimal path: Arad->Sibiu->Rimnicu->Pitesti->Bucharest, cost=418 km
```

### A* Properties

| Property | Value | Condition |
|---|---|---|
| **Complete** | YES | If h(n) is admissible and b is finite |
| **Optimal** | YES | If h(n) is admissible |
| **Time** | O(b^d) | Exponential in worst case |
| **Space** | O(b^d) | Keeps all nodes in memory — biggest weakness |

**A* is optimally efficient:** No other optimal algorithm expands fewer nodes than A* (given the same heuristic).

### Proof of Optimality (intuition)

```
Claim: If h(n) is admissible, A* finds the optimal solution.

Proof sketch:
  Let G* = optimal goal node, cost C* = g(G*)
  Let G' = suboptimal goal node, cost C' > C*

  Suppose A* expands G' before G*:
    When G' is selected: f(G') = g(G') = C' (since h(G')=0)
    There must be a node n on the optimal path still in frontier.
    f(n) = g(n) + h(n) <= g(n) + h*(n) = C*  (admissibility)
    So f(n) <= C* < C' = f(G')
    But A* would have expanded n before G' (lower f value)!
    CONTRADICTION -> A* never expands G' before G*.
```

### Limitations of A*

- **Memory:** Keeps all generated nodes in memory. For large problems, this is infeasible.
- **Heuristic quality matters:** A bad heuristic makes A* degrade to BFS (h=0 is Dijkstra/UCS).
- **For game trees:** A* is not used directly — Minimax is used instead.

```python
import heapq

def astar(graph, heuristic, start, goal):
    """
    A* Search algorithm.

    graph:     dict of {node: [(neighbor, cost), ...]}
    heuristic: dict of {node: h_value}
    start:     starting node
    goal:      target node
    Returns:   (path, total_cost)
    """

    # Priority queue: (f_value, g_value, node, path)
    # f = g + h
    h_start = heuristic.get(start, 0)
    open_list = [(h_start, 0, start, [start])]

    # Closed set: best known g-value for each node
    closed = {}

    while open_list:
        # Pop node with lowest f value
        f, g, current, path = heapq.heappop(open_list)

        # Goal test
        if current == goal:
            return path, g

        # Skip if we already found a better path to current
        if current in closed and closed[current] <= g:
            continue
        closed[current] = g

        # Expand: explore all neighbors
        for neighbor, edge_cost in graph.get(current, []):
            new_g = g + edge_cost
            h = heuristic.get(neighbor, 0)
            new_f = new_g + h

            if neighbor not in closed or closed[neighbor] > new_g:
                heapq.heappush(open_list,
                               (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')   # No path found


# Romania road problem
romania_graph = {
    'Arad':     [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
    'Zerind':   [('Arad', 75), ('Oradea', 71)],
    'Oradea':   [('Zerind', 71), ('Sibiu', 151)],
    'Timisoara':[('Arad', 118), ('Lugoj', 111)],
    'Sibiu':    [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu', 80)],
    'Rimnicu':  [('Sibiu', 80), ('Pitesti', 97), ('Craiova', 146)],
    'Fagaras':  [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti':  [('Rimnicu', 97), ('Bucharest', 101), ('Craiova', 138)],
    'Bucharest':[('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90)],
}

# Straight-line distances to Bucharest (the heuristic)
h_to_bucharest = {
    'Arad': 366, 'Zerind': 374, 'Oradea': 380, 'Timisoara': 329,
    'Lugoj': 244, 'Sibiu': 253, 'Rimnicu': 193, 'Fagaras': 176,
    'Pitesti': 100, 'Craiova': 160, 'Bucharest': 0, 'Giurgiu': 77,
}

path, cost = astar(romania_graph, h_to_bucharest, 'Arad', 'Bucharest')
print(f"A* Path: {' -> '.join(path)}")
print(f"Total cost: {cost} km")
# A* Path: Arad -> Sibiu -> Rimnicu -> Pitesti -> Bucharest
# Total cost: 418 km
```

**Line-by-line explanation:**
- `heapq.heappop(open_list)` — extracts the node with lowest f value (priority queue magic)
- `if current in closed and closed[current] <= g` — skip if we already found a cheaper path to this node
- `new_g = g + edge_cost` — accumulate actual path cost
- `h = heuristic.get(neighbor, 0)` — look up heuristic estimate for neighbor
- `new_f = new_g + h` — the A* evaluation function f = g + h
- `heapq.heappush(...)` — add to priority queue with f as priority key

---

## 9. Game Playing

### Why Games?

Games are the perfect AI testbed because they have:
- Well-defined states and rules
- Clear win/lose conditions
- An **adversary** (the opponent) who tries to defeat you

Game-playing AI must not just find **any** solution — it must find the **best move** against a rational opponent.

### Two-Player Zero-Sum Games

**Zero-sum:** One player's gain is the other's loss. The sum of gains = 0.

```
Examples:
  Chess, Checkers, Tic-Tac-Toe, Go, Othello

Properties:
  Two players: MAX (we want to maximise) and MIN (they want to minimise)
  Perfect information: both players see the full board
  Deterministic: no randomness
  Zero-sum: what one gains, the other loses
```

### Game Tree

A **game tree** shows all possible moves and counter-moves from the current position.

```
Tic-Tac-Toe (simplified after several moves):

                [Current Board]            <- MAX's turn (X)
               /      |       \
          [X plays   [X plays  [X plays
           top-left]  center]  top-right]  <- After MAX moves
              /           |
         [O plays...]  [O plays...]        <- MIN's turn (O)
              |              |
         [X plays...]  [X plays...]        <- MAX's turn again
              ...           ...
         [X wins!]     [Draw]              <- Terminal states
         score=+1       score=0
```

### Terminal States and Utility

**Terminal states** are end states (win/draw/loss).

**Utility function** assigns a score to terminal states:
```
From MAX's perspective:
  MAX wins: utility = +1  (or +infinity for chess)
  Draw:     utility =  0
  MIN wins: utility = -1  (or -infinity)
```

---

## 10. Minimax Algorithm

### Definition

**Minimax** is the algorithm that finds the optimal move for the MAX player, assuming the MIN player also plays optimally.

```
Rule:
  MAX player: choose the move that MAXIMISES the outcome
  MIN player: choose the move that MINIMISES the outcome

Recursively evaluate all terminal states, then back-propagate
values up the tree.
```

### Minimax Example — Tic-Tac-Toe Endgame

```
Game tree (X is MAX, O is MIN):

                    A
                 [X turn]
                /    |    \
               B     C     D
            [O turn]  [O turn]  [O turn]
             / \      / \         |
            E   F    G   H        I
          [X] [X]  [X] [X]      [X]

Terminal values (from X's perspective):
  E = +1 (X wins)   F = -1 (O wins)
  G = +1 (X wins)   H = 0  (Draw)
  I = +1 (X wins)

Step 1: Compute MIN values (O's choices at B, C, D):
  B = min(E, F) = min(+1, -1) = -1   (O picks F)
  C = min(G, H) = min(+1,  0) =  0   (O picks H)
  D = min(I)   =  +1                  (D has only one child)

Step 2: Compute MAX value (X's choice at A):
  A = max(B, C, D) = max(-1, 0, +1) = +1  (X picks move D!)

X should choose move D (guaranteed win even with O playing perfectly).
```

### Minimax Diagram

```
                    MAX: choose highest
                   [A: max(-1,0,+1) = +1]
                  /          |          \
        MIN: choose lowest
     [B: min(+1,-1)=-1] [C: min(+1,0)=0] [D: min(+1)=+1]
        /     \             /    \              |
      [E=+1] [F=-1]     [G=+1] [H=0]        [I=+1]
     (X wins)(O wins) (X wins)(Draw)        (X wins)
```

### Minimax Properties

| Property | Value |
|---|---|
| **Complete** | YES (finite game tree) |
| **Optimal** | YES (against optimal opponent) |
| **Time** | O(b^m) where m = max depth |
| **Space** | O(b*m) — depth-first expansion |

**The exponential time is the problem.** Chess has branching factor b~35 and games last ~80 moves:
```
35^80 = impossibly large number of nodes to evaluate!
```

Solutions: Alpha-Beta Pruning, depth-limited search with evaluation function.

### Minimax Implementation

```python
def minimax(board, depth, is_maximising, game):
    """
    Minimax algorithm for two-player zero-sum games.

    board:           current game state
    depth:           how deep to search (0 = terminal check only)
    is_maximising:   True if it is MAX's turn
    game:            game object with methods: is_terminal, utility, get_moves, make_move
    Returns:         best score from this position
    """

    # Base case: terminal state or depth limit reached
    if game.is_terminal(board) or depth == 0:
        return game.utility(board)

    if is_maximising:
        # MAX player: find the move with highest value
        best_score = float('-inf')
        for move in game.get_moves(board):
            new_board = game.make_move(board, move)
            score = minimax(new_board, depth - 1, False, game)  # MIN's turn next
            best_score = max(best_score, score)
        return best_score

    else:
        # MIN player: find the move with lowest value
        best_score = float('inf')
        for move in game.get_moves(board):
            new_board = game.make_move(board, move)
            score = minimax(new_board, depth - 1, True, game)   # MAX's turn next
            best_score = min(best_score, score)
        return best_score


def best_move_minimax(board, game, depth=9):
    """
    Find the best move for MAX player using Minimax.
    Returns: the best move
    """
    best_score = float('-inf')
    best_mv = None

    for move in game.get_moves(board):
        new_board = game.make_move(board, move)
        # Evaluate this move (MIN plays next)
        score = minimax(new_board, depth - 1, False, game)
        if score > best_score:
            best_score = score
            best_mv = move

    return best_mv, best_score


# Tic-Tac-Toe Implementation
class TicTacToe:
    def is_terminal(self, board):
        """Check if game is over"""
        return self.get_winner(board) is not None or '_' not in board

    def utility(self, board):
        """Return score: +1 X wins, -1 O wins, 0 draw"""
        winner = self.get_winner(board)
        if winner == 'X': return +1
        if winner == 'O': return -1
        return 0

    def get_moves(self, board):
        """Return list of empty positions"""
        return [i for i, cell in enumerate(board) if cell == '_']

    def make_move(self, board, pos):
        """Make a move — alternates X and O based on board state"""
        board = list(board)
        x_count = board.count('X')
        o_count = board.count('O')
        board[pos] = 'X' if x_count == o_count else 'O'
        return ''.join(board)

    def get_winner(self, board):
        """Check win conditions"""
        lines = [(0,1,2),(3,4,5),(6,7,8),  # rows
                 (0,3,6),(1,4,7),(2,5,8),  # columns
                 (0,4,8),(2,4,6)]           # diagonals
        for a,b,c in lines:
            if board[a] == board[b] == board[c] != '_':
                return board[a]
        return None

    def print_board(self, board):
        for i in range(0, 9, 3):
            print(board[i:i+3])
        print()


# Test Minimax
game = TicTacToe()

# Position where X can win in one move
# X | O | X
# O | X | _
# _ | _ | O
board = 'XOXOX__O'

# Note: 8 chars, let's use standard 9-char board
board = 'XOX_X_O__'   # X's turn, can win at position 5

print("Board:")
game.print_board(board)

move, score = best_move_minimax(board, game)
print(f"Best move for X: position {move}, score: {score}")
```

---

## 11. Alpha-Beta Pruning

### The Problem with Pure Minimax

Minimax evaluates **every** node in the game tree, even branches that can never influence the final decision.

**Alpha-Beta Pruning** eliminates these branches, dramatically reducing computation.

### Core Idea

```
While searching, maintain two values:
  alpha (a) = best score MAX can guarantee so far (starts at -inf)
  beta  (B) = best score MIN can guarantee so far (starts at +inf)

PRUNE when: alpha >= beta
  -> The current branch cannot improve upon what both players already have.
  -> Stop exploring this branch immediately.
```

### Alpha-Beta Example

```
Game tree:
                MAX
               [A]
              /     \
           MIN[B]   MIN[C]
           / \       / \
         [3] [5]   [2] [9]

Without Alpha-Beta:
  Evaluate ALL leaf nodes: 3, 5, 2, 9 -> 4 evaluations

With Alpha-Beta:
  Step 1: Evaluate B: left child = 3
          alpha = -inf, beta = +inf
          B is MIN: beta = min(inf, 3) = 3 (best MIN can do at B so far)
          Evaluate right child of B = 5
          B = min(3, 5) = 3
          Pass B's value (3) up: A's alpha = max(-inf, 3) = 3

  Step 2: Evaluate C (MIN node): left child = 2
          At C: alpha = 3 (inherited from A), beta = +inf
          C is MIN: beta = min(inf, 2) = 2
          Now: alpha (3) >= beta (2) -> PRUNE!
          Do NOT evaluate right child of C (9).
          (Even if it's +inf, MIN would pick 2 over 9.
           MAX already has 3 from branch B, so C can never be better.)
          C = 2 (best MIN can guarantee)

  Step 3: MAX at A = max(3, 2) = 3

  Total evaluations: 3 (saved 1)
  For large trees the savings are exponential!
```

### Alpha-Beta Pruning Diagram

```
              MAX [A]
             alpha=-inf, beta=+inf
            /                \
       MIN [B]              MIN [C]     <- PRUNED branch
      a=-inf,b=+inf       a=3, b=+inf
       /      \               /    X
    [3]       [5]           [2]   [9]  <- never evaluated!
    B=3       B=3           beta=2
                            alpha(3) >= beta(2) -> PRUNE!

    B returns 3
    A: alpha = max(-inf, 3) = 3
```

### Best-Case Performance

| Scenario | Effective Branching Factor | Nodes Explored |
|---|---|---|
| **No pruning (Minimax)** | b | O(b^m) |
| **Perfect ordering** | sqrt(b) | O(b^(m/2)) |
| **Random ordering** | b^(3/4) approximately | O(b^(3m/4)) |

**With perfect move ordering:** Alpha-Beta can reduce the number of nodes explored by up to b^(m/2), meaning it can effectively search **twice as deep** in the same time as pure Minimax!

```
Chess example (b=35, m=4):
  Minimax:    35^4 = 1,500,625 nodes
  Alpha-Beta (perfect order): 35^2 = 1,225 nodes  <- 1000x fewer!
```

### Alpha-Beta Implementation

```python
def minimax_alphabeta(board, depth, alpha, beta, is_maximising, game):
    """
    Minimax with Alpha-Beta Pruning.

    alpha: best score MAX can guarantee (-inf initially)
    beta:  best score MIN can guarantee (+inf initially)
    Prune when alpha >= beta.
    """

    # Base case
    if game.is_terminal(board) or depth == 0:
        return game.utility(board)

    if is_maximising:
        best_score = float('-inf')
        for move in game.get_moves(board):
            new_board = game.make_move(board, move)
            score = minimax_alphabeta(new_board, depth-1, alpha, beta, False, game)
            best_score = max(best_score, score)

            # Update alpha: best MAX can guarantee
            alpha = max(alpha, best_score)

            # PRUNE: MIN will never choose this branch
            # (MIN already has a better option elsewhere)
            if beta <= alpha:
                break   # Beta cutoff — prune remaining branches

        return best_score

    else:  # Minimising
        best_score = float('inf')
        for move in game.get_moves(board):
            new_board = game.make_move(board, move)
            score = minimax_alphabeta(new_board, depth-1, alpha, beta, True, game)
            best_score = min(best_score, score)

            # Update beta: best MIN can guarantee
            beta = min(beta, best_score)

            # PRUNE: MAX will never choose this branch
            if beta <= alpha:
                break   # Alpha cutoff — prune remaining branches

        return best_score


def best_move_alphabeta(board, game, depth=9):
    """Find best move using Alpha-Beta pruned Minimax."""
    best_score = float('-inf')
    best_mv = None

    alpha = float('-inf')
    beta  = float('inf')

    for move in game.get_moves(board):
        new_board = game.make_move(board, move)
        # Initial call: alpha=-inf, beta=+inf
        score = minimax_alphabeta(new_board, depth-1, alpha, beta, False, game)
        if score > best_score:
            best_score = score
            best_mv = move
        alpha = max(alpha, best_score)   # Update alpha at root

    return best_mv, best_score


# Compare: Minimax vs Alpha-Beta
import time

game = TicTacToe()
board = '_________'   # Empty board, X goes first

# Time Minimax
start = time.perf_counter()
move_mm, score_mm = best_move_minimax(board, game)
time_mm = time.perf_counter() - start

# Time Alpha-Beta
start = time.perf_counter()
move_ab, score_ab = best_move_alphabeta(board, game)
time_ab = time.perf_counter() - start

print(f"Minimax:    move={move_mm}, score={score_mm}, time={time_mm:.4f}s")
print(f"Alpha-Beta: move={move_ab}, score={score_ab}, time={time_ab:.4f}s")
print(f"Speedup: {time_mm/time_ab:.1f}x faster")
# Both give the same answer; Alpha-Beta is significantly faster
```

**Key lines explained:**
- `alpha = max(alpha, best_score)` — MAX updates its lower bound on what it can guarantee
- `beta = min(beta, best_score)` — MIN updates its upper bound on what it can guarantee
- `if beta <= alpha: break` — the pruning condition: no point exploring further

---

## 12. Iterative Deepening

### The Problem

- **BFS:** Complete and optimal, but uses O(b^d) memory — too much for deep searches
- **DFS:** Uses O(b*m) memory, but not optimal and may go infinitely deep
- **Can we get the best of both?**

**Yes! — Iterative Deepening Depth-First Search (IDDFS)**

### Definition

**Iterative Deepening** performs a series of depth-limited DFS searches, increasing the depth limit by 1 each time.

```
Iteration 1: DFS with depth limit 0   (only root)
Iteration 2: DFS with depth limit 1   (root + level 1)
Iteration 3: DFS with depth limit 2   (root + levels 1-2)
...
Iteration d+1: DFS with depth limit d (finds solution at depth d!)
```

### Visualisation

```
Depth limit = 1:         Depth limit = 2:        Depth limit = 3:
     A                        A                       A
     |                       / \                     / \
   [B,C]                   B   C                   B   C
   (explored)             / \   \                 /|   |\
                         D  E    F              D  E   F  G
                       (explored)            (explored, G is GOAL!)
```

### Why Doesn't Re-Exploring Waste Time?

At first glance, revisiting all levels in each iteration seems wasteful. But mathematically, it is efficient:

```
Nodes explored in each iteration (b=10):

Depth 1: 10
Depth 2: 10 + 100 = 110
Depth 3: 10 + 100 + 1,000 = 1,110
Depth d: 10 + 100 + ... + 10^d

The last level dominates:
  At depth d, the last level has 10^d nodes
  All previous levels together = 10^d * (1 + 1/10 + 1/100 + ...) <= 10^d * 1.11

Overhead = at most 11% for b=10. Negligible!
```

The repeated exploration of upper levels is a small fraction of the total work because the tree grows exponentially.

### IDDFS Properties

| Property | Value | Explanation |
|---|---|---|
| **Complete** | YES | Will find solution if one exists |
| **Optimal** | YES (equal costs) | Always finds shallowest solution |
| **Time** | O(b^d) | Same asymptotic as BFS |
| **Space** | O(b*d) | Same as DFS — linear! |

**IDDFS is the preferred uninformed search when the depth of the solution is unknown.**

```
It combines:
  BFS property: completeness + optimality (for uniform costs)
  DFS property: O(b*d) linear memory
```

### IDDFS vs BFS vs DFS

| Algorithm | Time | Space | Complete | Optimal |
|---|---|---|---|---|
| BFS | O(b^d) | O(b^d) | Yes | Yes (equal cost) |
| DFS | O(b^m) | O(b*m) | No | No |
| **IDDFS** | **O(b^d)** | **O(b*d)** | **Yes** | **Yes (equal cost)** |
| A* | O(b^d) | O(b^d) | Yes | Yes (admissible h) |

IDDFS is the winner for uninformed search (no heuristic available).

```python
def iddfs(graph, start, goal):
    """
    Iterative Deepening Depth-First Search.
    Combines DFS space efficiency with BFS optimality.
    """

    def dls(node, goal, depth_limit, path, visited):
        """Depth-Limited Search — DFS up to depth_limit."""
        if node == goal:
            return path
        if depth_limit == 0:
            return None   # Reached depth limit, did not find goal

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                result = dls(neighbor, goal,
                             depth_limit - 1,
                             path + [neighbor],
                             visited)
                if result is not None:
                    return result
                visited.discard(neighbor)  # Backtrack: unmark for other branches
        return None

    # Iteratively increase depth limit
    depth = 0
    while True:
        print(f"  Searching at depth limit: {depth}")
        visited = set([start])
        result = dls(start, goal, depth, [start], visited)
        if result is not None:
            return result, depth
        depth += 1
        if depth > 100:   # Safety limit to avoid infinite loop
            return None, -1


graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

print("IDDFS from A to G:")
path, depth_found = iddfs(graph, 'A', 'G')
print(f"Path found at depth {depth_found}: {' -> '.join(path)}")

# Output:
# Searching at depth limit: 0  (only A, no goal)
# Searching at depth limit: 1  (A,B,C -- no G)
# Searching at depth limit: 2  (A,B,C,D,E,F -- no G)
# Searching at depth limit: 3  (finds G!)
# Path found at depth 3: A -> C -> F -> G
```

---

## Algorithm Comparison

### Uninformed Search (No Heuristic)

| Algorithm | Complete | Optimal | Time | Space | Best For |
|---|---|---|---|---|---|
| **BFS** | Yes | Yes* | O(b^d) | O(b^d) | Shallow goals, optimality |
| **DFS** | No | No | O(b^m) | O(b*m) | Deep goals, memory limits |
| **IDDFS** | Yes | Yes* | O(b^d) | O(b*d) | Unknown depth, memory limits |
| **UCS** | Yes | Yes | O(b^(C*/e)) | O(b^(C*/e)) | Varying path costs |

*For uniform step costs. UCS = Uniform Cost Search (Dijkstra).

### Informed Search (With Heuristic)

| Algorithm | Complete | Optimal | Time | Space | Best For |
|---|---|---|---|---|---|
| **Best-First** | No | No | O(b^m) | O(b^m) | Quick approximate solutions |
| **A*** | Yes | Yes | O(b^d) | O(b^d) | Optimal path, good heuristic |

### Game Search

| Algorithm | Complete | Optimal | Against Optimal? | Best For |
|---|---|---|---|---|
| **Minimax** | Yes | Yes | Yes | Two-player zero-sum games |
| **Alpha-Beta** | Yes | Yes | Yes | Same as Minimax, much faster |
| **IDDFS + A-B** | Yes | Depth-limited | Best effort | Deep game trees |

---

## Code Examples

### Complete Search Comparison

```python
from collections import deque
import heapq
import time

# =====================================================
# COMPLETE COMPARISON OF ALL SEARCH ALGORITHMS
# Problem: Find path in a graph
# =====================================================

# Graph: node -> [(neighbor, cost)]
weighted_graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('S', 1), ('C', 2), ('D', 5)],
    'B': [('S', 4), ('D', 1), ('G', 10)],
    'C': [('A', 2), ('G', 8)],
    'D': [('A', 5), ('B', 1), ('G', 3)],
    'G': [('C', 8), ('D', 3), ('B', 10)]
}

# Unweighted version for BFS/DFS
unweighted_graph = {k: [n for n,_ in v] for k,v in weighted_graph.items()}

# Heuristic (estimated cost to G)
heuristic = {'S': 10, 'A': 7, 'B': 6, 'C': 3, 'D': 2, 'G': 0}

start, goal = 'S', 'G'

# --- BFS ---
def bfs_demo(graph, start, goal):
    queue = deque([(start, [start])])
    visited = {start}
    nodes_expanded = 0
    while queue:
        node, path = queue.popleft()
        nodes_expanded += 1
        if node == goal:
            return path, nodes_expanded
        for nbr in graph.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, path + [nbr]))
    return None, nodes_expanded

# --- DFS ---
def dfs_demo(graph, start, goal):
    stack = [(start, [start])]
    visited = {start}
    nodes_expanded = 0
    while stack:
        node, path = stack.pop()
        nodes_expanded += 1
        if node == goal:
            return path, nodes_expanded
        for nbr in reversed(graph.get(node, [])):
            if nbr not in visited:
                visited.add(nbr)
                stack.append((nbr, path + [nbr]))
    return None, nodes_expanded

# --- A* ---
def astar_demo(graph, heuristic, start, goal):
    pq = [(heuristic[start], 0, start, [start])]
    closed = {}
    nodes_expanded = 0
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node in closed:
            continue
        closed[node] = g
        nodes_expanded += 1
        if node == goal:
            return path, g, nodes_expanded
        for nbr, cost in graph.get(node, []):
            if nbr not in closed:
                new_g = g + cost
                new_f = new_g + heuristic.get(nbr, 0)
                heapq.heappush(pq, (new_f, new_g, nbr, path + [nbr]))
    return None, float('inf'), nodes_expanded

# --- Run and Compare ---
print("=" * 55)
print("SEARCH ALGORITHM COMPARISON")
print("Graph: S->G (with weights)")
print("=" * 55)

bfs_path, bfs_exp = bfs_demo(unweighted_graph, start, goal)
dfs_path, dfs_exp = dfs_demo(unweighted_graph, start, goal)
astar_path, astar_cost, astar_exp = astar_demo(weighted_graph, heuristic, start, goal)

print(f"\nBFS:")
print(f"  Path:  {' -> '.join(bfs_path)}")
print(f"  Nodes expanded: {bfs_exp}")

print(f"\nDFS:")
print(f"  Path:  {' -> '.join(dfs_path)}")
print(f"  Nodes expanded: {dfs_exp}")

print(f"\nA*:")
print(f"  Path:  {' -> '.join(astar_path)}")
print(f"  Cost:  {astar_cost}")
print(f"  Nodes expanded: {astar_exp}")
```

### 8-Puzzle Solver Using A*

```python
import heapq

def solve_8puzzle(initial_state, goal_state):
    """
    Solve the 8-puzzle using A* with Manhattan distance heuristic.

    State: tuple of 9 numbers, 0 represents blank
    Example: (1,2,3,4,5,6,7,8,0) = solved state
    """

    def manhattan_distance(state, goal):
        """Heuristic: sum of Manhattan distances of each tile from its goal."""
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0:   # Don't count the blank
                goal_pos = goal.index(tile)
                curr_row, curr_col = i // 3, i % 3
                goal_row, goal_col = goal_pos // 3, goal_pos % 3
                distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return distance

    def get_neighbors(state):
        """Generate all states reachable by sliding one tile."""
        state = list(state)
        zero_pos = state.index(0)
        row, col = zero_pos // 3, zero_pos % 3
        neighbors = []

        # Try moving blank: UP, DOWN, LEFT, RIGHT
        moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
        for dr, dc, direction in moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = state.copy()
                swap_pos = new_row * 3 + new_col
                new_state[zero_pos], new_state[swap_pos] = \
                    new_state[swap_pos], new_state[zero_pos]
                neighbors.append((tuple(new_state), direction))
        return neighbors

    # A* search
    start = tuple(initial_state)
    goal  = tuple(goal_state)

    h = manhattan_distance(start, goal)
    # Priority queue: (f, g, state, path_of_moves)
    pq = [(h, 0, start, [])]
    visited = {}

    while pq:
        f, g, state, moves = heapq.heappop(pq)

        if state == goal:
            return moves, g   # Solution found!

        if state in visited:
            continue
        visited[state] = g

        for next_state, move in get_neighbors(state):
            if next_state not in visited:
                new_g = g + 1
                new_h = manhattan_distance(next_state, goal)
                heapq.heappush(pq, (new_g + new_h, new_g, next_state, moves + [move]))

    return None, -1   # No solution

# Test
initial = [1, 2, 3,
           4, 0, 5,
           7, 8, 6]

goal    = [1, 2, 3,
           4, 5, 6,
           7, 8, 0]

solution, num_moves = solve_8puzzle(initial, goal)
print(f"8-Puzzle solved in {num_moves} moves: {solution}")
```

---

## Real-World Applications

| Algorithm | Real-World Use | Example |
|---|---|---|
| **BFS** | Social network distance | "How many connections to reach this person?" |
| **BFS** | Web crawling | Explore web pages level by level |
| **DFS** | Maze generation | Recursive backtracking maze algorithms |
| **DFS** | Topological sort | Task scheduling with dependencies |
| **A*** | Navigation / GPS | Google Maps, car GPS routing |
| **A*** | Video game pathfinding | NPC movement in open-world games |
| **A*** | Robot navigation | Autonomous robots navigating obstacles |
| **Minimax** | Chess engines | Stockfish, Komodo (combined with evaluation) |
| **Alpha-Beta** | All modern game AI | Every commercial chess/checkers/Go engine |
| **Minimax + IDDFS** | Time-limited game search | Real-time game AI with time controls |
| **IDDFS** | Compiler optimization | Register allocation, code optimisation |

---

## Interview Notes

**Q1: What are the 5 components of problem formulation?**

> Initial state, Actions, Transition model (result of actions), Goal test, and Path cost. Every search problem must be expressed in these terms before selecting an algorithm.

---

**Q2: BFS vs DFS — when do you use each?**

> BFS is used when you need the shortest path (fewest hops), when the solution is shallow, or when you need a complete search. DFS is used when memory is limited, when solutions may be deep, or when any valid solution is acceptable. IDDFS combines the best of both.

---

**Q3: What is a heuristic? What makes one admissible?**

> A heuristic h(n) is an estimate of the cost from state n to the goal. It is admissible if it never overestimates the true cost — h(n) <= h*(n) for all n. An admissible heuristic ensures A* finds the optimal solution. Example: Manhattan distance for 8-puzzle, straight-line distance for navigation.

---

**Q4: Why is A* optimal? Explain intuitively.**

> A* evaluates f(n) = g(n) + h(n) — actual cost so far plus estimated remaining cost. If h is admissible (never overestimates), A* will never bypass the optimal path. If a cheaper path existed, it would have a lower f-value and would have been expanded first. A* essentially "can't miss" the optimal solution because it is always looking at the most promising unexplored frontier node.

---

**Q5: What is Minimax? What is its time complexity?**

> Minimax finds the optimal move in a two-player zero-sum game by assuming both players play perfectly. MAX maximises the score, MIN minimises it. The algorithm recursively evaluates the entire game tree and backs up values. Time complexity is O(b^m) where b is the branching factor and m is the maximum depth.

---

**Q6: How does Alpha-Beta pruning improve Minimax?**

> Alpha-Beta maintains alpha (best MAX can guarantee) and beta (best MIN can guarantee). If during search beta <= alpha, the current branch cannot influence the final decision — it is pruned. With perfect move ordering, Alpha-Beta reduces complexity from O(b^m) to O(b^(m/2)), effectively doubling the search depth in the same time.

---

**Q7: What is IDDFS and why is it preferred over DFS or BFS?**

> IDDFS performs DFS with increasing depth limits (0, 1, 2, ...) until the goal is found. It has BFS's completeness and optimality (for uniform costs) combined with DFS's linear memory usage O(b*d). The repeated exploration overhead is small (~11% for b=10) because lower levels have exponentially fewer nodes.

---

**Q8: What is the difference between Best-First Search and A*?**

> Best-First Search uses f(n) = h(n) — only the estimated cost to the goal. It is fast but not optimal because it ignores the cost already paid. A* uses f(n) = g(n) + h(n) — both actual cost and estimated cost. A* is optimal (given admissible h) because it considers the complete trip cost, not just the remaining estimated cost.

---

**Q9: What happens if h(n) = 0 in A*?**

> If h(n) = 0 for all nodes, A* reduces to Uniform Cost Search (Dijkstra's algorithm), which expands nodes purely based on g(n) — actual cost. It is still complete and optimal, but explores many more nodes than A* with a good heuristic.

---

**Q10: Can Alpha-Beta pruning miss the optimal move?**

> No. Alpha-Beta always produces the exact same result as full Minimax — it just skips branches that cannot affect the outcome. The pruned branches would never have been chosen by either player if they played optimally. Alpha-Beta is a safe optimisation, not an approximation.

---

## Common Mistakes & Misconceptions

**Mistake 1: "DFS is always faster than BFS"**

Correction: DFS explores one path deeply before backtracking. If the goal is shallow and near the start, BFS finds it much faster. DFS may go to maximum depth before finding a shallow solution.

---

**Mistake 2: "A higher heuristic value is always better in A*"**

Correction: An admissible heuristic must never overestimate. Inflating h(n) beyond h*(n) makes A* inadmissible and non-optimal. The heuristic should be as tight as possible without exceeding the true cost.

---

**Mistake 3: "Alpha-Beta gives a different answer than Minimax"**

Correction: Alpha-Beta always produces exactly the same result as full Minimax. It is a lossless optimisation that skips branches which cannot influence the final decision.

---

**Mistake 4: "BFS always finds the shortest path"**

Correction: BFS finds the path with the fewest edges (steps). If edges have different weights/costs, BFS does NOT find the cheapest path. For weighted graphs, use Dijkstra (UCS) or A*.

---

**Mistake 5: "IDDFS wastes time by re-exploring nodes"**

Correction: The overhead is mathematically small — at most O(1/(1-1/b)) times more nodes than BFS. For b=10, this is only ~11% extra work. The memory savings (O(b*d) vs O(b^d)) far outweigh this overhead.

---

**Mistake 6: "Minimax works for games with chance (dice, cards)"**

Correction: Pure Minimax assumes deterministic, perfect-information games. For games with randomness (backgammon, poker), you need Expectiminimax which adds chance nodes computing expected values. For imperfect information games (poker), you need completely different frameworks.

---

**Mistake 7: "Best-First Search is optimal because it always picks the best-looking node"**

Correction: Best-First is greedy — it picks what looks best NOW, ignoring how expensive the current path already is. A node close to the goal via an expensive route looks attractive but leads to a suboptimal solution. A* fixes this by also accounting for g(n).

---

## Key Takeaways

```
=====================================================================
                    UNIT 3 KEY TAKEAWAYS
               Search Techniques & Problem Solving
=====================================================================

PROBLEM FORMULATION (always do this first!)
  5 components: Initial State, Actions, Transition, Goal Test, Cost
  Good formulation = exponentially smaller search space

UNINFORMED SEARCH
  BFS:    Queue (FIFO), level-by-level, optimal (equal costs), high memory
  DFS:    Stack (LIFO), depth-first, not optimal, low memory
  IDDFS:  Best of both — BFS quality + DFS memory. USE THIS by default.

INFORMED SEARCH (heuristic)
  Heuristic h(n) = estimated cost to goal
  Admissible: h(n) <= true cost (never overestimates)
  Best-First: f=h(n) only — fast but NOT optimal
  A*:         f=g(n)+h(n) — optimal + complete (with admissible h)
  Key insight: A* is the gold standard for single-agent search

GAME SEARCH
  Game tree: all possible moves and counter-moves
  Minimax: MAX maximises, MIN minimises — assumes perfect play
  Time: O(b^m) — exponential, impractical for deep games
  Alpha-Beta: prunes branches that cannot affect outcome
    alpha = best MAX guarantee, beta = best MIN guarantee
    Prune when beta <= alpha
    Best case: O(b^(m/2)) — effectively doubles search depth!
  IDDFS + Alpha-Beta = practical real-world game AI

THE HIERARCHY
  A* > Best-First (optimality)
  IDDFS > DFS (completeness, optimality)
  IDDFS = BFS quality + DFS memory
  Alpha-Beta = Minimax result + much faster
=====================================================================
```

---

## Revision Questions

### Conceptual (Short Answer)
1. What are the 5 components of problem formulation? Give an example for each using the N-Queens problem.
2. What data structure does BFS use? What about DFS? How does the choice affect the search order?
3. Define admissibility of a heuristic. Why does it matter for A*?
4. What is the evaluation function f(n) in A*? What does each component mean?
5. What is the difference between a search tree and a state space graph?

### Analytical (Medium)
6. Trace BFS and DFS on this graph from S to G: S-{A,B}, A-{C,D}, B-{D,E}, D-{G}. Show the frontier at each step.
7. Why is Best-First Search (f=h) NOT optimal? Construct a small counterexample.
8. For the 8-puzzle: which heuristic is better — misplaced tiles or Manhattan distance? Why? Which dominates?
9. Trace Minimax on this tree: MAX at root, children MIN nodes with values [[3,5],[2,9]]. Show alpha and beta.
10. Why does IDDFS not waste much time despite re-exploring nodes at each depth?

### Advanced (Long Answer)
11. Prove informally that A* is optimal when using an admissible heuristic.
12. Explain step by step how Alpha-Beta pruning works. What is the best-case improvement over Minimax?
13. Compare BFS, DFS, IDDFS, and A* on all four properties: completeness, optimality, time, space. When would you use each?
14. Design an admissible heuristic for a robot finding the shortest path in a room with obstacles.
15. Explain why Alpha-Beta gives the same result as Minimax but faster. Under what ordering is it fastest?

### Applied / Code
16. Implement BFS for a word ladder problem (change one letter at a time: cat -> dog).
17. Write the Minimax function for a simple number game: players alternate picking from [1..10], the player who reaches exactly 21 wins.
18. How would you adapt A* to find the K shortest paths, not just 1?

---

## Summary

```
=====================================================================
                        UNIT 3 SUMMARY
                Search Techniques & Problem Solving
=====================================================================

Search = finding a path from initial state to goal state
          through a state space

UNINFORMED (blind) SEARCH:
  BFS    - complete, optimal, high memory
  DFS    - incomplete, not optimal, low memory
  IDDFS  - complete, optimal, low memory (USE THIS)

INFORMED (heuristic) SEARCH:
  h(n) = estimated cost to goal
  Best-First: f=h(n) -- greedy, fast, NOT optimal
  A*:         f=g(n)+h(n) -- optimal with admissible h
  Admissible: h(n) <= true cost (never overestimates)

GAME SEARCH:
  Game tree = all possible moves + counter-moves
  Minimax: finds optimal play for MAX vs MIN
    MAX picks max child, MIN picks min child
    Time: O(b^m) -- too slow for real games
  Alpha-Beta Pruning:
    Same result as Minimax but prunes useless branches
    alpha = best MAX guarantee (lower bound)
    beta  = best MIN guarantee (upper bound)
    Prune when beta <= alpha
    Best case: O(b^(m/2)) -- double the search depth!
  Iterative Deepening + Alpha-Beta = practical game AI

KEY PRINCIPLE:
  All search is about smart exploration of the search space.
  The better your heuristic, the less you search.
  Pruning removes what cannot help -- always safe in Minimax.
=====================================================================
```

---

*Next Unit: Machine Learning Fundamentals — supervised, unsupervised, and reinforcement learning.*

---

*Last updated: 2025 | Format: GitHub README.md | Subject: AI/ML from Beginner to Advanced*
