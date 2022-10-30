# Piotr Rogula, 249801
def from_list(list_of_cords: list[list[int, int]]):
    return [[it[0] for it in list_of_cords], [it[1] for it in list_of_cords]]


def to_list(cords: list[list[int], list[int]]):
    return [it for it in zip(cords[0], cords[1])]