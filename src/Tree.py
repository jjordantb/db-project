from Node import Node


class Tree:

    def __init__(self, num_classes, img_dim, num_clusters):
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.root = Node(num_clusters)

    def build_tree(self):
        pass
