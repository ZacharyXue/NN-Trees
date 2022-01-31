import copy


class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index
    
    # 定义了该类比较大小的方式
    def __lt__(self, other):
        return self.distance < other.distance


class KNNResultSet:
    """
        创建一个容纳KNN排列结果的类，KNN的排列结果中的每一个元素
        使用DistIndex表示，DistIndex中存储点的index和距离查询位置
        的距离
    """
    def __init__(self, capacity):
        self.capacity = capacity  # 定义KNN中的K值
        self.count = 0
        self.worst_dist = 1e10  # 这里使用 float('inf')可能更好
        # 定义KNN中存储K个结点的列表，为了方便对结点进行比较，
        # 这里定义了DistIndex类包含结点距离以及相应的index
        self.dist_index_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))

        self.comparison_counter = 0

    def size(self):
        return self.count

    def full(self):
        return self.count == self.capacity

    def worstDist(self):
        return self.worst_dist

    def add_point(self, dist, index):
        self.comparison_counter += 1
        # 添加新的结点时候如果距离大于K个结点中的最远距离，
        # 则不影响结果
        if dist > self.worst_dist:
            return

        if self.count < self.capacity:
            self.count += 1

        # 这里要完成的任务是将新的结点加入self.dist_index_list列表中，
        # 首先知道最后一个元素是要被pop出去的，这里使用深拷贝进行对象移动，
        # 从而给新的结点腾出位置来，感觉这里其实不用深拷贝，直接移动对象就可以。
        # 移动完成之后将新结点的信息赋给相应位置上的DistIndex对象
        i = self.count - 1
        while i > 0:
            if self.dist_index_list[i-1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i-1])
                i -= 1
            else:
                break

        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = index
        self.worst_dist = self.dist_index_list[self.capacity-1].distance
        
    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d comparison operations.' % self.comparison_counter
        return output


class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []

        self.comparison_counter = 0

    def size(self):
        return self.count

    def worstDist(self):
        return self.radius

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.radius:
            return

        # 使用KNN方法的时候只获取最近的K个结点，而在RadiusNN中会
        # 取 radius 内的所有结点，实际上在类似VoxelNet中仍有上限
        self.count += 1
        self.dist_index_list.append(DistIndex(dist, index))

    def __str__(self):
        self.dist_index_list.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
                  % (self.count, self.radius, self.comparison_counter)
        return output


