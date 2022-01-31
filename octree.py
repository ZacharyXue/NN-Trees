import random
import math
import numpy as np
import time
# KNNResultSet类和RadiusNNResultSet类利用相应方法获得满足条件的
# 结点列表，但是在这两个类中不涉及 dist 的计算
from result_set import KNNResultSet, RadiusNNResultSet


class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        # 这里的这些属性很重要啊
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    # 设置 print 时候显示的内容
    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output


def traverse_octree(root: Octant, depth, max_depth):
    # 访问当前结点
    # 如果是叶结点，则答应当前结点
    # 如果不是，则访问该结点子结点
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1


def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        # 假如该root为空，则创建心得八叉树结点，开始时初始化为叶结点
        # 如果后续操作发现不是叶结点，则会修改该属性
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    # 注意这里的终止条件
    if len(point_indices) <= leaf_size or extent <= min_extent:
        # 不需要分割，将 is_leaf 标记为 True
        root.is_leaf = True
    else:
        # 需要分割
        root.is_leaf = False

        # 初始化8个子结点列表
        children_point_indices = [[] for i in range(8)]
        # 遍历所有结点
        for point_idx in point_indices:
            point_db = db[point_idx]
            # 这里使用或运算编码点云数据和center之间的关系
            # 很巧妙，假如使用numpy应该可以用矩阵运算同时完成
            # 所有计算-->发现效率并没有明显提升
            # 2^3 = 8
            morton_code = 0
            if point_db[0] > center[0]:
                # 1 == 0b001
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                # 2 == 0b010
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                # 4 == 0b100
                morton_code = morton_code | 4
            # 整个过程很巧妙啊
            # 这里存储过程类似于哈希表，存储的是点的index
            children_point_indices[morton_code].append(point_idx)
        # create children
        factor = [-0.5, 0.5]
        for i in range(8):
            # 计算中心结点
            # factor[*] *号部分决定子cube的中心点在root中心点的位置
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            # 缩小子cube的尺寸
            child_extent = 0.5 * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            # 最后递归得到子cube内的结点分割结果
            root.children[i] = octree_recursive_build(root.children[i],
                                                      db,
                                                      child_center,
                                                      child_extent,
                                                      children_point_indices[i],
                                                      leaf_size,
                                                      min_extent)
    return root


def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    # 这种情况下搜索范围的球在cube内，所以看的是
    # 中心点距球心距离加上球半径和cube尺寸的关系
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)


def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    # 注意这里的前提是球心距离和cube中心距离xyz都小于半径＋cube半边长
    # 这时候如果有两边长度小于cube半边长，则说明该球心在cube六个面中某个面
    # 的正对面，结合上一种情况，便得出球与cube相交的结论
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    # 在上一种情况考虑完球心对应cube六个面后，接下来考虑球心与cube边角之间的距离关系
    # 这种情况就是求距离进行判断
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


def contains(query: np.ndarray, radius: float, octant:Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    # np.linalg.norm() 求矩阵的二阶范数，这里就是向量的模
    # 如果下面这种情况存在，则cube被球完全包含
    return np.linalg.norm(query_offset_to_farthest_corner) < radius


def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 如果球包含该cube，则返回False（不在遍历其子cube），
    # 该cube内所有点均为所求点
    if contains(query, result_set.worstDist(), root):
        # compare the contents of the octant
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # don't need to check any child
        return False

    # 在root是叶结点的情况下，如果球没有完全包含cube，也还需要遍历
    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 对于不是叶结点的root，进行遍历与球相交的子结点
    # no need to go to most relevant child first, because anyway we will go through all children
    for c, child in enumerate(root.children):
        if child is None:
            continue
        # 如果没有相交
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        # 如果相交，则迭代该子cube
        if octree_radius_search_fast(child, db, result_set, query):
            return True

    return inside(query, result_set.worstDist(), root)


def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # go to the relevant child first
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4

    if octree_radius_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_radius_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    # 该函数所有的返回值都是为了方便遍历过程中是否需要停止遍历
    # 和外界无关
    if root is None:
        # 实际上这一条件只有外界输入时用得上，
        # 内部迭代时会跳过为空的cube
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # 如果该结点是叶结点，则在遍历完该cube内结点后则停止继续迭代
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            # 遍历所有点，尝试添加到KNN队列中，add_point()函数会检测是否加入
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        # 如果该cube是叶结点，并且搜索点的KNN队列中最远距离在该cube内。
        # 说明其他相邻cube内不可能有更近的点，则停止搜索
        return inside(query, result_set.worstDist(), root)

    # 先根据查询点找到最相近的cube
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    # 查询当前结点和查询点最近的子cube
    if octree_knn_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children
    for c, child in enumerate(root.children):
        # 跳过已经遍历的结点和空结点
        if c == morton_code or child is None:
            continue
        # 如果查询节点和当前子cube完全没有交集，则跳过该cube
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        # 在有重叠情况下迭代该子cube
        if octree_knn_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    # 这里所有利用 inside() 函数进行判断是否停止遍历，其
    # 原理都是：随着遍历KNN中存储的K个点的最大距离会不断
    # 缩小，如果当前cube已经完全包住了以query为球心以
    # result_set.worstDist()为半径的球，则说明没有访问其他
    # 邻近cube的必要了
    return inside(query, result_set.worstDist(), root)


def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    # 貌似使用np.min也可以
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    # 确定该cube的半边长
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    # 确定该cube的中心点
    db_center = db_np_min + db_extent

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root

def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    # depth = [0]
    # max_depth = [0]
    # traverse_octree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # octree_knn_search(root, db_np, result_set, query)
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])

    begin_t = time.time()
    print("Radius search normal:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.5)
        octree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    begin_t = time.time()
    print("Radius search fast:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius = 0.5)
        octree_radius_search_fast(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t)*1000))



if __name__ == '__main__':
    main()