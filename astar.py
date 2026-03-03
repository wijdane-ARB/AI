import heapq

def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic_zero(a, b):
    return 0

def astar(grid, start, goal, weight=1.0, pure_greedy=False, use_h_zero=False):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    h_func = heuristic_zero if use_h_zero else heuristic_manhattan
    
    closed = set()
    nodes_expanded = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal], nodes_expanded
            
        closed.add(current)
        
        for nx, ny, cost in grid.neighbors(*current):
            neighbor = (nx, ny)
            tentative_g = g_score[current] + cost
            
            if neighbor in closed and tentative_g >= g_score.get(neighbor, float('inf')):
                continue
                
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                if pure_greedy:
                    f = h_func(neighbor, goal)
                else:
                    f = tentative_g + weight * h_func(neighbor, goal)
                    
                heapq.heappush(open_set, (f, neighbor))
                
    return None, float('inf'), nodes_expanded