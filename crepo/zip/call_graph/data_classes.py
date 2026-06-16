from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich import print as rprint
from rich.tree import Tree


@dataclass(slots=True)
class FunctionNode:
    name: str
    file_name: str
    file_path: str | None = None
    is_external: bool = False
    is_static: bool = False
    macro_expansion: Optional["FunctionNode"] = None
    # ❌ callbacks REMOVED — it's per-call-site, not per-definition
    start_line: int = -1
    end_line: int = -1

    def label_with_line(
        self, line: int | None = None, callbacks: list["FunctionNode"] | None = None
    ) -> str:
        if not line and not callbacks:
            return self.label

        if line:
            base = (
                f"[{line}]{self.name}"
                if self.is_external
                else f"[{self.file_name}:{line}]{self.name}"
            )
        else:
            base = self.unique_id

        suffix = ""
        if self.start_line != -1:
            suffix = f"[{self.start_line}:{self.end_line}]"

        if self.macro_expansion:
            base += f" (macro expansion)-> {self.macro_expansion.name}"
        if callbacks:
            cb_names = ", ".join(cb.name for cb in callbacks)
            base += f" (accepts callback)-> {cb_names}"

        return base + suffix

    @property
    def unique_id(self) -> str:
        if self.is_external:
            return self.name
        return f"[{self.file_name}]{self.name}"

    @property
    def label(self) -> str:
        base = self.unique_id
        suffix = ""
        if self.start_line != -1:
            suffix = f"[{self.start_line}:{self.end_line}]"
        if self.macro_expansion:
            return f"{base} (macro expansion)-> {self.macro_expansion.name}"
        return base + suffix


@dataclass(slots=True)
class CallSite:
    """
    Represents a specific call event (an edge in the graph).
    Links a caller to a callee at a specific line number.
    """

    callee: FunctionNode
    line_number: int  # at which it was called to build the intial context..
    callbacks: list[FunctionNode] = field(default_factory=list)


@dataclass(slots=True)
class CallTreeNode:
    """
    Represents a node in the logical call tree structure.
    Contains the actual FunctionNode data and its children.
    """

    func_node: FunctionNode
    children: list[CallTreeNode] = field(default_factory=list)
    called_line_number: int | None = None
    callbacks: list[FunctionNode] = field(default_factory=list)  # ← from CallSite

    def add_child(self, child_node: CallTreeNode):
        self.children.append(child_node)

    @property
    def get_display_label(self) -> str:
        """
        Formats the node label as:
        [file_name:line_number]function_name OR
        [line_number]function_name (if file missing)
        """
        name = self.func_node.name
        file_name = self.func_node.file_name
        line = self.called_line_number
        # if name == 'pmf_addevent':
        #     print("Sending the line_number", line)
        label = self.func_node.label_with_line(
            line=line, callbacks=self.callbacks if self.callbacks else None
        )

        return label

    def to_rich_tree(self) -> Tree:
        """Converts this logical node and its children into a Rich Tree for visualization."""
        rich_tree = Tree(self.get_display_label)
        for child in self.children:
            rich_tree.add(child.to_rich_tree())
        return rich_tree


@dataclass(slots=True)
class custom_tree:
    name: str
    children: list[custom_tree] = field(default_factory=list)

    def add(self, child: custom_tree):
        self.children.append(child)

    @property
    def get_name(self) -> str:
        return re.sub(r"\[([^\[\]]*)\]", "", self.name)
