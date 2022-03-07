import cv2
import numpy as np
import random as rnd
import time


def make_nodes(width, height):
    n = []
    for i in range(width):
        tmp = []
        for j in range(height):
            tmp.append(Node((255, 255, 255)))
        n.append(tmp)
    return n


def make_edges(m):
    # Determine the start node.
    x = 0
    y = rnd.randint(0, m.height)

    # Make it red for checking
    # m.set_node_colour(x, y, (0, 0, 255))

    # Generate edges with Prim's algorithm
    edges = prim(m, x, y)
    return edges


def prim(m, x, y):
    edges = []

    stack = [(x, y)]
    m.set_node_visited(x, y, True)

    while len(stack) > 0:

        neighbours_positions = m.get_neighbours_positions(stack[-1][0], stack[-1][1])

        unvisited_neighbours_positions = []

        for i in neighbours_positions:
            if not m.get_node(i[0], i[1]).visited and i not in stack:
                unvisited_neighbours_positions.append(i)

        if len(unvisited_neighbours_positions) == 0:
            stack.pop()
        else:
            # select a random neighbour
            rand_neigh = rnd.choice(unvisited_neighbours_positions)
            m.set_node_visited(rand_neigh[0], rand_neigh[1], True)

            # Add forward and backwards edges
            edges.append((rand_neigh, stack[-1]))
            edges.append((stack[-1], rand_neigh))

            stack.append(rand_neigh)

    return edges


def display(full, zoom):
    img = cv2.cvtColor(full.rendering, cv2.COLOR_BGR2RGB)
    scaled = cv2.resize(img, (img.shape[0] * zoom, img.shape[1] * zoom), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("maze", scaled)
    cv2.waitKey(0)


def write_to_file(full):
    cv2.imwrite("maze.png", full.rendering)


class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nodes = make_nodes(width, height)
        self.edges = make_edges(self)
        self.rendering = self.generate_image()

    def generate_image(self):
        full = np.full(((self.height * 2 + 1), (self.width * 2 + 1), 3), 0, dtype=np.uint8)

        # Paint Nodes
        for x in range(self.width):
            for y in range(self.height):
                full[2 * y + 1, 2 * x + 1, 0] = self.nodes[x][y].colour[0]
                full[2 * y + 1, 2 * x + 1, 1] = self.nodes[x][y].colour[1]
                full[2 * y + 1, 2 * x + 1, 2] = self.nodes[x][y].colour[2]

        # Paint edges
        for i in self.edges:
            if i[0][0] == i[1][0]:
                # Vertical edge
                if i[0][1] > i[1][1]:
                    # Upward edge
                    full[2 * i[0][1], 2 * i[0][0] + 1, 0] = 255
                    full[2 * i[0][1], 2 * i[0][0] + 1, 1] = 255
                    full[2 * i[0][1], 2 * i[0][0] + 1, 2] = 255

                else:
                    # Downward edge
                    full[2 * i[0][1] + 2, 2 * i[0][0] + 1, 0] = 255
                    full[2 * i[0][1] + 2, 2 * i[0][0] + 1, 1] = 255
                    full[2 * i[0][1] + 2, 2 * i[0][0] + 1, 2] = 255

            else:
                # Horizontal edge
                if i[0][0] > i[1][0]:
                    # Left edge
                    full[2 * i[0][1] + 1, 2 * i[0][0], 0] = 255
                    full[2 * i[0][1] + 1, 2 * i[0][0], 1] = 255
                    full[2 * i[0][1] + 1, 2 * i[0][0], 2] = 255

                else:
                    # Right edge
                    full[2 * i[0][1] + 1, 2 * i[0][0] + 2, 0] = 255
                    full[2 * i[0][1] + 1, 2 * i[0][0] + 2, 1] = 255
                    full[2 * i[0][1] + 1, 2 * i[0][0] + 2, 2] = 255


        return full

    def set_node_colour(self, x, y, colour):
        self.nodes[x][y].colour = colour

    def set_node_visited(self, x, y, visited):
        self.nodes[x][y].visited = visited

    def get_node(self, x, y):
        return self.nodes[x][y]

    def get_neighbours_positions(self, x, y):
        ret = []

        # Left
        if x > 0:
            ret.append((x-1, y))

        if x < self.width - 1:
            ret.append((x+1, y))

        if y > 0:
            ret.append((x, y-1))

        if y < self.height - 1:
            ret.append((x, y+1))

        return ret




class Node:
    def __init__(self, colour):
        self.colour = colour
        self.visited = False
        self.distance = float('inf')
        

# Main script
if __name__ == '__main__':

    start = time.time()
    my_maze = Maze(100, 100)
    write_to_file(my_maze)
    print("Generated a maze with:", str(my_maze.width * my_maze.height), "nodes in:", str(time.time() - start), "seconds.")
    display(my_maze, 6)



