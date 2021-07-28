from icecream import ic


def row_column_index_creator(index_row_size, index_col_size):
    row_col_index_list = []
    index_row_size -= 1
    index_col_size += 1
    for row_index in range(index_col_size):
        row_index += 1
        for col_index in range(index_row_size):
            col_index += 1
            row_col_index_list.append([row_index, col_index])
    return row_col_index_list


def row_column_index_creator_v2(index_row_size, index_col_size):
    row_col_index_list = []
    index_row_size += 1
    index_col_size += 1
    [row_col_index_list.append([row_index, col_index]) for row_index in range(1, index_row_size) for col_index in
     range(1, index_col_size)]
    return row_col_index_list


ic(row_column_index_creator(4, 3))
ic(row_column_index_creator_v2(4, 3))
