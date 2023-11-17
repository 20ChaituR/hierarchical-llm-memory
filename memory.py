from __future__ import annotations

import os
import tiktoken

from abc import ABC, abstractmethod
from typing import List

TOKENIZER_MODEL = "gpt-4"


class Counter(object):
    """
    A simple counter class.

    Attributes:
        value (int): The current count.
    """

    def __init__(self) -> None:
        """
        Initializes the Counter instance with a count of 0.
        """
        self.value = 0

    def increase(self):
        """
        Increases the count by 1.
        """
        self.value += 1

    def equals(self, target):
        """
        Checks if the current count equals the target value.

        Args:
            target (int): The target value to compare against.

        Returns:
            bool: True if the current count equals the target, False otherwise.
        """
        return self.value == target


class HierarchicalEntity(ABC):
    """
    An abstract base class representing a hierarchical entity in a file system,
    such as a file or directory. This class provides functionality to manage the hierarchy,
    status (open or closed) of the entity, and interaction with its contents.

    Attributes:
        path (str): The file system path of the entity.
        opened (bool): Indicates whether the entity is open.
        opened_time (int): The time at which the entity was opened, or None if not opened.
        contents (List[HierarchicalEntity]): A list of contained hierarchical entities.
    """

    cur_time = 0  # Static variable to keep track of the current time across instances.

    def __init__(self, path: str) -> None:
        """
        Initializes a new instance of the HierarchicalEntity class.

        Args:
            path (str): The file system path of the entity.
        """
        self.path: str = path
        self.opened: bool = False
        self.opened_time = None  # None indicates the entity has not been opened.
        self.contents: List[HierarchicalEntity] = []

    @abstractmethod
    def _path_matches(self, path: str) -> bool:
        """
        Abstract method to check if the provided path matches the entity's path.

        Args:
            path (str): The path to be matched.

        Returns:
            bool: True if the path matches, False otherwise.
        """
        pass

    @abstractmethod
    def _can_open(self) -> bool:
        """
        Abstract method to determine if the entity can be opened.

        Returns:
            bool: True if the entity can be opened, False otherwise.
        """
        pass

    @abstractmethod
    def _fill_contents(self) -> None:
        """
        Abstract method to fill the contents of the entity.
        """
        pass

    @abstractmethod
    def _str_path(self) -> str:
        """
        Abstract method to get the string representation of the entity's path.

        Returns:
            str: The string representation of the path.
        """
        pass

    def _add(self, entity: HierarchicalEntity) -> None:
        """
        Adds a hierarchical entity to the contents.

        Args:
            entity (HierarchicalEntity): The entity to be added.
        """
        self.contents.append(entity)

    def _get_latest_time(self) -> int:
        """
        Retrieves the latest time any part of the hierarchy was opened.

        Returns:
            int: The latest time recorded in the hierarchy.
        """
        if not self.opened:
            return None

        latest_time = None
        for content in self.contents:
            time = content._get_latest_time()
            if time is not None and (latest_time is None or time > latest_time):
                latest_time = time

        if latest_time is not None:
            return latest_time
        return self.opened_time

    def _close(self) -> None:
        """
        Closes this entity and all contained entities recursively.
        """
        if self.opened:
            self.opened = False
            self.opened_time = -1
            for content in self.contents:
                content._close()

    def _open(self) -> None:
        """
        Opens this entity. If not already opened, updates the opened status,
        sets the opened time, and calls _fill_contents to populate the contents.
        """
        if not self.opened:
            self.opened = True
            self.opened_time = HierarchicalEntity.cur_time
            HierarchicalEntity.cur_time += 1
            self._fill_contents()

    def open(self, path: str) -> bool:
        """
        Opens the entity if the path matches or recursively opens a contained entity.

        Args:
            path (str): The path of the entity to be opened.

        Returns:
            bool: True if the entity or any contained entity was successfully opened, False otherwise.
        """
        if not self.opened:
            if self._path_matches(path) and self._can_open():
                self._open()
                return True
        else:
            for content in self.contents:
                has_opened = content.open(path)
                if has_opened:
                    return True

        return False

    def open_by_index(self, target_idx: int, cur_idx: Counter = None) -> bool:
        """
        Opens the entity by its index in a sequence.

        Args:
            target_idx (int): The target index to match for opening.
            cur_idx (Counter, optional): The current index counter, used for recursive calls.

        Returns:
            bool: True if the entity or any contained entity was successfully opened by index, False otherwise.
        """
        if cur_idx is None:
            cur_idx = Counter()

        if not self.opened and self._can_open():
            cur_idx.increase()

        if cur_idx.equals(target_idx):
            self._open()
            return True

        if self.opened:
            for content in self.contents:
                has_opened = content.open_by_index(target_idx, cur_idx)
                if has_opened:
                    return True

        return False

    def close(self, path: str) -> bool:
        """
        Closes the entity if the path matches or recursively closes a contained entity.

        Args:
            path (str): The path of the entity to be closed.

        Returns:
            bool: True if the entity or any contained entity was successfully closed, False otherwise.
        """
        if self.opened:
            for content in self.contents:
                has_closed = content.close(path)
                if has_closed:
                    return True

            if self._path_matches(path) and self._can_open():
                self._close()
                return True

        return False

    def close_oldest(self) -> None:
        """
        Closes the oldest opened entity in the hierarchy.
        """
        if self.opened:
            earliest_time = None
            earliest_content = None
            for content in self.contents:
                time = content._get_latest_time()
                if time is not None and (earliest_time is None or time < earliest_time):
                    earliest_time = time
                    earliest_content = content

            if earliest_time is None:
                self._close()
            else:
                earliest_content.close_oldest()

    def to_str(self, indent: str = "", idx: Counter = None) -> str:
        """
        Converts the hierarchical structure of the entity into a string representation.

        Args:
            indent (str): The indentation to be used for each hierarchical level.

        Returns:
            str: The string representation of the hierarchical structure.
        """
        if idx is None:
            idx = Counter()

        result = indent + self._str_path() + "\n"
        if not self.opened:
            if self._can_open():
                idx.increase()
                result += indent + f"  ({idx.value}) ...\n"
        else:
            for content in self.contents:
                result += content.to_str(indent + "  ", idx)
        return result

    def size(self) -> int:
        """
        Calculates the size of the string representation in terms of tokens.

        Returns:
            int: Number of tokens in the string representation.
        """
        encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
        string_representation = self.to_str()
        tokens = encoding.encode(string_representation)
        return len(tokens)


class Directory(HierarchicalEntity):
    """
    A class representing a directory in a file system. Inherits from HierarchicalEntity.

    This class overrides abstract methods from the HierarchicalEntity class to handle
    directory-specific operations like matching paths, filling contents with other
    directories and files, and getting a string representation of the directory path.
    """

    def _path_matches(self, path: str) -> bool:
        """
        Checks if the provided path matches the directory's path.

        Args:
            path (str): The path to be matched.

        Returns:
            bool: True if the path matches the directory's path, False otherwise.
        """
        return self.path.endswith(path)

    def _can_open(self) -> bool:
        """
        Checks if the directory can be opened.

        Returns:
            bool: Always true, as directories can always be opened.
        """
        return True

    def _fill_contents(self) -> None:
        """
        Fills the contents of the directory with its subdirectories and files.
        """
        self.contents = []
        for entry in os.listdir(self.path):
            filepath = os.path.join(self.path, entry)
            if os.path.isdir(filepath):
                self.contents.append(Directory(filepath))
            else:
                self.contents.append(File(filepath))

    def _str_path(self) -> str:
        """
        Gets the string representation of the directory's path.

        Returns:
            str: The basename of the directory path.
        """
        return os.path.basename(self.path)


class File(HierarchicalEntity):
    """
    A class representing a file in a file system. Inherits from HierarchicalEntity.

    This class is specifically designed to handle text files and their contents,
    distinguishing them from other file types and representing their content as a
    hierarchy of blocks.

    Attributes:
        is_text (bool): Indicates if the file is a text file.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes a new instance of the File class.

        Args:
            path (str): The file system path of the file.
        """
        super().__init__(path)
        self.is_text: bool = self._is_text_file(path)

    def _path_matches(self, path: str) -> bool:
        """
        Checks if the provided path matches the file's path and if the file is a text file.

        Args:
            path (str): The path to be matched.

        Returns:
            bool: True if the path matches and the file is a text file, False otherwise.
        """
        return self.path.endswith(path)

    def _can_open(self) -> bool:
        """
        Checks if the file is a text file, and therefore can be opened.

        Returns:
            bool: True if the file is a text file, False otherwise.
        """
        return self.is_text

    def _fill_contents(self) -> None:
        """
        Fills the contents of the file with blocks representing lines of text with their indentation.
        """
        self.contents = []
        with open(self.path) as file:
            stack: List[Block] = []
            for line in file:
                if len(line.strip()) == 0:
                    continue

                indent_level = len(line) - len(line.lstrip())
                block = Block(line.strip(), indent_level)

                while stack and stack[-1].indent_level >= indent_level:
                    stack.pop()
                if stack:
                    stack[-1]._add(block)
                else:
                    self.contents.append(block)

                stack.append(block)

    def _str_path(self) -> str:
        """
        Gets the string representation of the file's path.

        Returns:
            str: The basename of the file path.
        """
        return os.path.basename(self.path)

    def _is_text_file(self, path: str) -> bool:
        """
        Determines if the file at the given path is a text file.

        Args:
            path (str): The path of the file to be checked.

        Returns:
            bool: True if the file is a text file, False otherwise.
        """
        try:
            with open(path, "rb") as file:
                chunk = file.read(1024)
            return b"\x00" not in chunk
        except IOError:
            return False


class Block(HierarchicalEntity):
    """
    A class representing a block of text within a file. Inherits from HierarchicalEntity.

    A block is a line of text in a file, along with its indentation level. This class
    is used to represent the hierarchical structure of text within a file, especially
    in the context of programming languages or structured text.

    Attributes:
        indent_level (int): The indentation level of the block.
    """

    def __init__(self, path: str, indent_level: int) -> None:
        """
        Initializes a new instance of the Block class.

        Args:
            path (str): The text of the block.
            indent_level (int): The indentation level of the block.
        """
        super().__init__(path)
        self.indent_level: int = indent_level

    def _path_matches(self, path: str) -> bool:
        """
        Checks if the provided path (text) is part of the block's text.

        Args:
            path (str): The text to be matched.

        Returns:
            bool: True if the text is part of the block's text, False otherwise.
        """
        return path in self.path

    def _can_open(self) -> bool:
        """
        Checks if the block contains any subblocks within it, and therefore can be opened.

        Returns:
            bool: True if the contains any subblocks, False otherwise.
        """
        return len(self.contents) > 0

    def _fill_contents(self) -> None:
        """
        This method is not applicable for Block as blocks should be initialized with all of their contents.
        """
        pass

    def _str_path(self) -> str:
        """
        Gets the string representation of the block's text.

        Returns:
            str: The text of the block.
        """
        return self.path
