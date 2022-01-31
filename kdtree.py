import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


def sort_key_by_vale(key, value):
    """
        将key和value按照value的大小顺序进行排列
    """
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    """
        输出下一次分割应该依据的维度
    """
    if axis == dim-1:
        return 0
    else:
        return axis + 1


def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    """

    :param root:
    :param db: NxD
    :param db_sorted_idx_inv: NxD
    :param point_idx: M
    :param axis: scalar
    :param leaf_size: scalar
    :return:
    """
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    # 当当前结点中的点数量大于一定值时才进行分割
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # 排序
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        # 找到中点进行排序
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5
        # === get the split position ===
        # 因为是要构建kd树，所以左右两侧要分别构建之后的root才是该点的left和right
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right,
                                           db,
                                           point_indices_sorted[middle_right_idx:],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)
    return root


def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        # 如果是叶结点，则获得结点，计算距离，尝试添加到result_set中
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    # 如果不是叶结点，则需要找左侧或者右侧进行进一步迭代
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        # 在遍历一侧结束之后，如果查询点与当前点的距离之差小于result_set的最大距离
        # 则说明当前点的另一侧也有可能有符合要求的点存在。
        # 
        # 这个地方就是和八叉树的差别，因为八叉树节点存储的是cube的尺寸，
        # 所以在访问当前节点时就可以知道树的另一侧分布的范围；而kd树则不同，
        # 以这里为例，根节点的右子节点有可能存在时必须遍历所有右子节点的左结点
        # 才能知道结果
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)

    return False



def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    db_np = np.random.rand(db_size, dim)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()