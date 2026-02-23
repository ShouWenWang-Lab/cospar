class Tree:
    """
    A lightweight, vendored alternative to ete3.Tree.
    Parses basic Newick format strings and provides standard node traversal methods.
    """

    def __init__(self, newick=None, name="", dist=0.0):
        self.up = None
        self.children = []
        self.name = name
        self.dist = dist

        if newick is not None:
            self._parse_newick(newick)

    def add_child(self, child=None, name="", dist=0.0):
        """Adds a child to the current node."""
        if child is None:
            child = Tree(name=name, dist=dist)
        child.up = self
        self.children.append(child)
        return child

    def is_leaf(self):
        """Returns True if the node has no children."""
        return len(self.children) == 0

    def is_root(self):
        """Returns True if the node has no parent."""
        return self.up is None

    def get_leaves(self):
        """Returns a list of all leaf nodes descending from this node."""
        return [node for node in self.traverse("postorder") if node.is_leaf()]

    def search_nodes(self, **kwargs):
        """Finds descendant nodes matching specific attributes (e.g., name='Cell_A')."""
        results = []
        for node in self.traverse("preorder"):
            match = True
            for k, v in kwargs.items():
                if getattr(node, k, None) != v:
                    match = False
                    break
            if match:
                results.append(node)
        return results

    def traverse(self, strategy="levelorder"):
        """Yields nodes in the specified topological order."""
        if strategy == "preorder":
            yield self
            for child in self.children:
                yield from child.traverse("preorder")

        elif strategy == "postorder":
            for child in self.children:
                yield from child.traverse("postorder")
            yield self

        elif strategy == "levelorder":
            queue = [self]
            while queue:
                curr = queue.pop(0)
                yield curr
                queue.extend(curr.children)

    def __repr__(self):
        return self.write()

    def write(self):
        """Returns the tree in basic Newick format."""
        dist_str = f":{self.dist}" if self.dist is not None and self.dist != 0.0 else ""
        name_str = str(self.name) if self.name else ""

        if self.is_leaf():
            return f"{name_str}{dist_str}"
        else:
            children_str = ",".join([c.write() for c in self.children])
            newick = f"({children_str}){name_str}{dist_str}"
            if self.is_root():
                newick += ";"
            return newick

    def _parse_newick(self, newick_str):
        """Internal recursive parser for Newick strings."""
        s = newick_str.strip().rstrip(";")
        self._parse_node(s)

    def _parse_node(self, s):
        if s.startswith("("):
            depth = 0
            close_idx = -1
            for i, char in enumerate(s):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        close_idx = i
                        break

            children_str = s[1:close_idx]
            info_str = s[close_idx + 1 :]

            child_strs = []
            curr_child = []
            depth = 0
            for char in children_str:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1

                if char == "," and depth == 0:
                    child_strs.append("".join(curr_child))
                    curr_child = []
                else:
                    curr_child.append(char)
            if curr_child:
                child_strs.append("".join(curr_child))

            for child_str in child_strs:
                child = Tree()
                child._parse_node(child_str)
                self.add_child(child)

            self._set_info(info_str)
        else:
            self._set_info(s)

    def _set_info(self, s):
        """Extracts name and distance from a Newick node string."""
        if ":" in s:
            name, dist = s.rsplit(":", 1)
            self.name = name.strip()
            self.dist = float(dist.strip())
        else:
            self.name = s.strip()
            self.dist = 0.0
