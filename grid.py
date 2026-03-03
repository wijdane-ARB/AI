class Grid:
    def __init__(self, width, height, obstacles, start, goal):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.start = start
        self.goal = goal

    def is_free(self, x, y):
        return (0 <= x < self.width and 0 <= y < self.height and 
                (x, y) not in self.obstacles)

    def neighbors(self, x, y):
        # 4-voisins, coût uniforme = 1
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return [(x+dx, y+dy, 1) for dx, dy in dirs if self.is_free(x+dx, y+dy)]

    def all_free_cells(self):
        return [(x, y) for x in range(self.width) 
                for y in range(self.height) if self.is_free(x, y)]