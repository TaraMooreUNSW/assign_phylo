"""Helper functions for HW3"""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
from matplotlib.axes import Axes
from Bio import Align
from Bio.Align import substitution_matrices

GAP_PENALTY = 8

CORONAVIRUS_MAJOR_GROUPS = [
    "Alphacoronavirus",
    "Betacoronavirus",
    "Gammacoronavirus",
    "Deltacoronavirus"
]

C_GROUP_TO_COLOUR_MAP = {
    "Alphacoronavirus": "red",
    "Betacoronavirus": "blue",
    "Gammacoronavirus": "purple",
    "Deltacoronavirus": "green",
    None: "#000",
}

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        taxonomy: str = None,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        taxonomy: str
            The matched major group of coronavirus
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.taxonomy = taxonomy
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # create nodes for all labels
    nodes = []
    for n in labels:
        nodes.append(Node(n, None, 0, None, 0))

    dist_copy = distances.copy()
    labels_copy = labels.copy()

    # repeat with new node and distances until convergance
    while len(labels_copy) > 2:
        n = len(labels_copy)
        # initialise q_array
        q_array = np.zeros_like(dist_copy)
        rows, cols = np.triu_indices(dist_copy.shape[0], k=0)

        # based on the current dist matrix, calculate Q
        for i, j in zip(rows, cols):
            if i == j:
                q_array[i,j] = 0
                continue

            # sum(D_ik)
            d_ik = sum(dist_copy[i, range(0, n)])

            # sum(D_jk)
            d_jk = sum(dist_copy[j, range(0, n)])

            # Q = (n-2)D_ij - sum(D_ik) - sum(D_jk)
            q = (n - 2) * dist_copy[i,j] - d_ik - d_jk
            q_array[i,j] = q

        q_array = q_array + q_array.T

        # find a pair of distinct nodes i and j, (i != j)
        # for which Q is smallest
        q_copy = q_array.copy()
        np.fill_diagonal(q_copy, np.inf)
        min_row, min_col = np.unravel_index(np.nanargmin(q_copy), q_copy.shape)
        # Make a new node that joins the nodes i and j
        # connect the new node to the central node
        new_node_name = f"{labels_copy[min_row]}{labels_copy[min_col]}"

        # calculate the dist from each of the node in the pair to this new node
        # delta_iu = 1/2*D_ij + 1/2(n-2)*(sum(D_ik) - sum(D_jk))
        # sum(D_ik)
        new_d_ik = sum(dist_copy[min_row, range(0, n)])

        # sum(D_jk)
        new_d_jk = sum(dist_copy[min_col, range(0, n)])
        delta_iu = 1/2*dist_copy[min_row,min_col] + 1/(2*(n-2))*(new_d_ik - new_d_jk)
        delta_ju = dist_copy[min_row,min_col] - delta_iu

        new_node_left_dist = delta_iu
        new_node_right_dist = delta_ju

        # create new node & update list of nodes
        new_node = Node(new_node_name, nodes[min_row], new_node_left_dist, nodes[min_col], new_node_right_dist)

        # calculate new distance matrix
        new_dist_matrix = np.zeros((n - 1, n - 1))

        label_1_remove = labels_copy[min_row]
        label_2_remove = labels_copy[min_col]
        labels_copy.remove(label_1_remove)
        labels_copy.remove(label_2_remove)
        labels_copy.append(new_node_name)

        # calculate the dist from each of the nodes outside of this pair
        # to the new node
        # d_uk = 1/2 * (D_ik + D_jk - D_ij)
        remaining_labels = [k for k in range(n) if k != min_row and k != min_col]

        for ind_i, og_i in enumerate(remaining_labels):
            for ind_j, og_j in enumerate(remaining_labels):
                new_dist_matrix[ind_i, ind_j] = dist_copy[og_i, og_j]

        for ind_k, og_k in enumerate(remaining_labels):
            d_uk = 1/2 * (dist_copy[min_row, og_k] + dist_copy[min_col, og_k] - dist_copy[min_row, min_col])
            new_dist_matrix[ind_k, -1] = d_uk
            new_dist_matrix[-1, ind_k] = d_uk

        new_dist_matrix[0,0] = 0.0
        nodes = [nodes[k] for k in remaining_labels] + [new_node]

        dist_copy = new_dist_matrix

    if len(nodes) == 2:
        # connect remaining nodes to root node
        node_dist = dist_copy[0,1] * 1/2
        return Node("Root", nodes[0], node_dist, nodes[1], node_dist)



def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    node_coords = {}
    edges = []

    def get_leaf_nodes(node: Node):
        if node is None:
            return []

        if node.left is None and node.right is None:
            return [node.name]

        return get_leaf_nodes(node.left) + get_leaf_nodes(node.right)

    def count_leaf_nodes(node: Node):
        if node is None:
            return 0
        if not node.left and not node.right:
            return 1
        return count_leaf_nodes(node.left) + count_leaf_nodes(node.right)

    leaf_nodes = get_leaf_nodes(tree)
    no_leaves = len(leaf_nodes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))

    def recursive_plot_nj_tree(node: Node, x_left, x_right):
        if node is None:
            return

        if node == tree:
            left_leaves = count_leaf_nodes(node.left)
            right_leaves = count_leaf_nodes(node.right)
            total_leaves = left_leaves + right_leaves

            if total_leaves > 0:
                curr_x = x_left + (x_right - x_left) * (left_leaves/total_leaves)
            else:
                curr_x = (x_left + x_right) / 2

            node_coords[node.name] = (curr_x, 0)

        curr_x, curr_y = node_coords[node.name]

        # left tree
        if node.left:
            left_sub_leaves = count_leaf_nodes(node.left.left)
            right_sub_leaves = count_leaf_nodes(node.left.right)
            total_sub_leaves = left_sub_leaves + right_sub_leaves

            if total_sub_leaves > 0:
                child_x = x_left + (curr_x - x_left) * (left_sub_leaves / total_sub_leaves)
            else:
                child_x = (x_left + curr_x) / 2

            child_y = curr_y - node.left_distance
            node_coords[node.left.name] = (child_x, child_y)
            edges.append((node.name, node.left.name, node.left_distance))

            recursive_plot_nj_tree(node.left, x_left, curr_x)

        # right tree
        if node.right:
            left_sub_leaves = count_leaf_nodes(node.right.left)
            right_sub_leaves = count_leaf_nodes(node.right.right)
            total_sub_leaves = left_sub_leaves + right_sub_leaves

            if total_sub_leaves > 0:
                child_x = curr_x + (x_right - curr_x) * (left_sub_leaves / total_sub_leaves)
            else:
                child_x = (curr_x + x_right) / 2

            child_y = curr_y - node.right_distance
            node_coords[node.right.name] = (child_x, child_y)
            edges.append((node.name, node.right.name, node.right_distance))

            recursive_plot_nj_tree(node.right, curr_x, x_right)

    recursive_plot_nj_tree(tree, 0, no_leaves)

    name_to_node = {}
    def map_name_to_node(node: Node):
        if node:
            name_to_node[node.name] = node
            map_name_to_node(node.left)
            map_name_to_node(node.right)

    map_name_to_node(tree)

    for parent, child, _ in edges:
        px, py = node_coords[parent]
        cx, cy = node_coords[child]

        ax.plot([py,py,cy], [px,cx,cx], color="#000")


    for name, (x,y) in node_coords.items():
        if name in leaf_nodes:
            curr_node = name_to_node.get(name)
            node_taxa = curr_node.taxonomy
            label_colour = C_GROUP_TO_COLOUR_MAP.get(node_taxa, "#000")

            ax.text(
                y, x, f"  {name}",
                va="center", fontsize=12,
                color=label_colour
            )

    all_x_coords = [coords[0] for coords in node_coords.values()]
    min_x, max_x = min(all_x_coords), max(all_x_coords)
    canvas_width = max_x - min_x

    stem_length = 0.05 * canvas_width
    ry, rx = node_coords[tree.name]
    ax.plot([rx + stem_length, rx], [ry,ry], color="#000")

    # add colour legend
    legend_handles = [mpatches.Patch(color=colour, label=label if label else "Not a coronvirus") for label, colour in C_GROUP_TO_COLOUR_MAP.items()]
    ax.legend(handles=legend_handles, title="Coronavirus Taxonomies", loc="center right", bbox_to_anchor=(1.70, 0.5), borderaxespad=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(abs(x))}"))

    fig.subplots_adjust(right=0.75)
    plt.tight_layout(pad=0.2)
    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree is None:
        return 0

    if tree.left is None and tree.right is None:
        return 1

    left_leaves = sort_children_by_leaves(tree.left)
    right_leaves = sort_children_by_leaves(tree.right)

    if left_leaves > right_leaves:
        tree.left, tree.right = tree.right, tree.left

    return left_leaves + right_leaves


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

# Global alignment function from ass1 pt 1

def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    rows = len(seq2) + 1
    cols = len(seq1) + 1

    # cache scoring function to improve efficiency
    uniq_seq1 = set(seq1)
    uniq_seq2 = set(seq2)
    store_cache = {
        (aa_i, aa_j): scoring_function(aa_i, aa_j)
        for aa_i in uniq_seq1
        for aa_j in uniq_seq2
    }

    # initialise matrix
    alignment_matrix = [[{} for _ in range(cols)] for _ in range(rows)]
    alignment_matrix[0][0] = 0

    # set matrix initial row
    for i in range(rows):
        alignment_matrix[i][0] = {"score": -i*GAP_PENALTY,
                                  "prev_cell_row": i - 1,
                                  "prev_cell_col": 0}

    # set matrix initial col
    for j in range(cols):
        alignment_matrix[0][j] = {"score": -j*GAP_PENALTY,
                                  "prev_cell_row": 0,
                                  "prev_cell_col": j - 1}

    # recurrence
    for i in range(1, rows):
        for j in range(1, cols):
            recurr_one = alignment_matrix[i-1][j-1]["score"] + store_cache[(seq1[j-1], seq2[i-1])]
            recurr_two = alignment_matrix[i-1][j]["score"] - GAP_PENALTY
            recurr_three = alignment_matrix[i][j-1]["score"] - GAP_PENALTY

            scores = [recurr_one, recurr_two, recurr_three]
            max_score = max(scores)
            max_score_index = scores.index(max_score)

            alignment_matrix[i][j]["score"] = max_score

            # traceback recording
            if max_score_index == 0:
                alignment_matrix[i][j]["prev_cell_row"] = i - 1
                alignment_matrix[i][j]["prev_cell_col"] = j - 1
            elif max_score_index == 1:
                alignment_matrix[i][j]["prev_cell_row"] = i - 1
                alignment_matrix[i][j]["prev_cell_col"] = j
            elif max_score_index == 2:
                alignment_matrix[i][j]["prev_cell_row"] = i
                alignment_matrix[i][j]["prev_cell_col"] = j - 1

    # Traceback from bottom right
    seq1_aligned = list(seq1)
    seq2_aligned = list(seq2)
    score = alignment_matrix[rows - 1][cols - 1]["score"]

    t_i = len(seq2)
    t_j = len(seq1)
    while t_i > 0 or t_j > 0:
        i_backpoint = alignment_matrix[t_i][t_j]["prev_cell_row"]
        j_backpoint = alignment_matrix[t_i][t_j]["prev_cell_col"]

        if i_backpoint == t_i:
            t_j = j_backpoint           # gap in seq2
            seq2_aligned.insert(t_i, '-')
        elif j_backpoint == t_j:
            t_i = i_backpoint           # gap in seq1
            seq1_aligned.insert(t_j, '-')
        else:
            t_i = i_backpoint
            t_j = j_backpoint

    seq1_final = "".join(seq1_aligned)
    seq2_final = "".join(seq2_aligned)

    return(seq1_final, seq2_final, f"{score:.1f}")


def scoring_blosum62(aa_i, aa_j):
    """
    Scoring function using the BLOSUM62 substitution matrix

    Parameters
    ----------
    aa_i: str
        First amino acid.
    aa_j: str
        Second amino acid.

    Returns
    -------
    float
        Score of match between amino acids.

    Examples
    --------
    >>> scoring_BLOSUM('W', 'W')
    (11.0)
    """
    aligner = Align.PairwiseAligner()
    blosum_matrix = substitution_matrices.load("BLOSUM62")
    aligner.substitution_matrix = blosum_matrix

    score = aligner.score(aa_i, aa_j)
    return score
