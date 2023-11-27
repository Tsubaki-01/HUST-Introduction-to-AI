import numpy as np

############################################
#              八方向A_star算法              #
############################################


MOVE = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)]


class Point:  # 节点类

    def __init__(self, row, col, parent=None, move=None):
        self.row = row
        self.col = col
        self.f = 0
        self.parent = parent
        self.move = move

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def get_h(self, end_point):  # 曼哈顿距离
        return 10 * abs(self.row - end_point.row) + 10 * abs(self.col - end_point.col)

    def get_g(self):  # g代价
        if not self.parent:
            return 0
        else:
            return self.parent.get_g() + 10 if abs(self.move[0]) + abs(self.move[1]) == 1 \
                else self.parent.get_g() + 14

    def get_f(self):
        return self.f

    def get_row_and_col(self):
        return self.row, self.col

    def set_f(self, state, end_point):  # f = g + h
        self.f = self.get_h(end_point) + self.get_g()


class State:  # map状态类

    def __init__(self, state, current_point=Point(-1, -1), end_point=Point(-1, -1)):
        self.state = state
        self.current_point = current_point
        self.end_point = end_point

    def __eq__(self, other):
        return self.current_point == other.current_point

    def get_current_point(self):  # 获取当前位置
        return self.current_point


def search_next_step(map: State, open_list, close_list, wrong_list):  # 寻找下个落脚点
    walkable_points = []
    # 获取当前位置信息
    current_point = map.get_current_point()
    row, col = current_point.get_row_and_col()
    # 边界条件
    row_border = map.state.shape[0]
    col_border = map.state.shape[1]
    # 遍历八个方向
    for move in MOVE:
        next_step = (row + move[0], col + move[1])

        if row_border > next_step[0] > 0 and col_border > next_step[1] > 0 \
                and map.state[next_step[0], next_step[1]] == 0:  # walkable
            next_point = Point(next_step[0], next_step[1], parent=current_point, move=move)

            if next_point not in wrong_list:  # next_point不是死路
                next_point.set_f(state, map.end_point)

                if next_point not in close_list:  # next_point还没被走过
                    #   f更小就更新，大就不动
                    for compare_point in (temp for temp in open_list if temp == next_point):
                        if compare_point.get_f() > next_point.get_f():
                            open_list.remove(compare_point)
                        else:
                            next_point = compare_point
                    # 将点加入open_list
                    if next_point not in open_list:
                        open_list.append(next_point)
                    # 将点加入walkable_list
                    walkable_points.append(next_point)

    # 排序walkable_list，没有walkable point就是死路，有的话选取f最小的作为下一个落脚点
    walkable_points.sort(key=compare)
    if len(walkable_points) < 1:
        wrong_list.append(current_point)
        close_list.remove(current_point)

        map.current_point = close_list[-1]
    else:
        map.current_point = walkable_points[0]
        open_list.remove(walkable_points[0])
        close_list.append(walkable_points[0])


def compare(point):
    return point.get_f()


# A_STAR
def a_star(map, open_list, close_list, wrong_list):
    while not map.current_point == map.end_point:
        search_next_step(map, open_list, close_list, wrong_list)


def showInfo(map, path):
    walked_points = []
    end_state = map.state

    # 从后往前遍历取路径点
    temp_point = path[-1]
    start_point = path[0]
    while not temp_point == start_point:
        walked_points.append(temp_point)
        temp_point = temp_point.parent
    walked_points.append(start_point)
    walked_points.reverse()

    # 打印路径
    t = 0
    for point in walked_points:
        row, col = point.get_row_and_col()
        print(f'({row}, {col}) ', end='  ')
        end_state[row, col] = -1
        t += 1
        if t == 4:
            print('\n')
            t = 0
    print('\n')

    # 打印地图
    for i in range((map.state.shape[0])):
        for j in range((map.state.shape[1])):
            if end_state[i][j] < 0:
                print('*', end='  ')
            else:
                print(end_state[i][j], end='  ')
        print('\n')


if __name__ == '__main__':
    open_list = []
    close_list = []
    wrong_list = []

    state = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    )
    # state = np.array([
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]
    # )

    # 起点终点
    start_point = Point(0, 0)
    end_point = Point(9, 9)
    # 寻路
    Map = State(state, Point(0, 0), end_point)
    a_star(Map, open_list, close_list, wrong_list)
    # 打印路径
    path = [start_point] + close_list
    print('Best Way:')
    showInfo(Map, path)
