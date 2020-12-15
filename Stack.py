from collections import deque


class Stack:
    def __init__(self):
        self.__list = deque()

    def push(self, key):
        self.__list.append(key)

    def pop(self):
        return self.__list.pop()

    def peek(self):
        key = self.__list.pop()
        self.__list.append(key)
        return key

    def is_empty(self):
        return len(self.__list) == 0

    def __str__(self):
        return "[" + ", ".join(self.__list) + "]"

    def __len__(self):
        return len(self.__list)