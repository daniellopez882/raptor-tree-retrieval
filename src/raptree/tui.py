"""A Textual browser for one RAPTOR tree, with per-question retrieval highlighting."""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, ClassVar

from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, OptionList, Static, Tree
from textual.widgets.option_list import Option

from raptree.retrieve import METHODS, Method, Retrieval, Retriever, query_text

if TYPE_CHECKING:
    from textual.widgets.tree import TreeNode

    from raptree.embed import Embedder
    from raptree.quality import Question
    from raptree.tree import RaptorTree


def _preview(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _ranges(ids: list[int]) -> str:
    """Compact leaf ranges such as ``12-19, 23``."""
    if not ids:
        return "-"
    parts: list[str] = []
    start = prev = ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = i
    parts.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(parts)


class TreeExplorer(App[None]):
    """Left: the tree from the top layer down. Right: node details and the article's questions."""

    TITLE = "raptree"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    #body { height: 1fr; }
    #tree { width: 3fr; border: round $primary; overflow-x: hidden; }
    #side { width: 2fr; }
    #detail-box { height: 2fr; border: round $secondary; padding: 0 1; }
    #questions { height: 1fr; border: round $secondary; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("m", "next_method", "Method"),
        Binding("e", "expand_level", "Expand a level"),
        Binding("c", "collapse", "Collapse"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        tree: RaptorTree,
        questions: list[Question],
        embedder: Embedder | None,
        *,
        budget: int = 2000,
        beam: int = 3,
        preview_chars: int = 48,
    ) -> None:
        super().__init__()
        self.raptor = tree
        self.questions = questions
        self.embedder = embedder
        self.retriever = Retriever(tree, budget, beam)
        self.method: Method = "raptor-collapsed"
        self.preview_chars = preview_chars
        self.retrieval: Retrieval | None = None
        self.rank: dict[int, int] = {}
        self.question_index: int | None = None
        self._parents = tree.parents()
        self._expanded_levels = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield Tree(self._root_label(), id="tree")
            with Vertical(id="side"):
                with VerticalScroll(id="detail-box"):
                    yield Static(id="detail")
                yield OptionList(
                    *[
                        Option(
                            Text(f"Q{i + 1}{' (hard)' if q.hard else ''} {q.question}"), id=str(i)
                        )
                        for i, q in enumerate(self.questions)
                    ],
                    id="questions",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self.raptor.title} · {self.method}"
        widget = self.query_one("#tree", Tree)
        widget.show_root = True
        widget.root.expand()
        for node_id in self.raptor.layers[-1]:
            self._add(widget.root, node_id)
        self._show_overview()

    # ---- tree construction -------------------------------------------------------------

    def _root_label(self) -> Text:
        sizes = "/".join(str(len(layer)) for layer in reversed(self.raptor.layers))
        return Text(f"{self.raptor.title} · nodes per layer {sizes}", style="bold")

    def _label(self, node_id: int) -> Text:
        node = self.raptor.nodes[node_id]
        rank = self.rank.get(node_id)
        marker = f"●{rank:>2}" if rank is not None else "   "
        kind = f"{len(node.children)} ch" if node.children else "leaf"
        head = f"{marker} L{node.layer} #{node.id:<4} {node.tokens:>3} tok {kind:>5}  "
        label = Text(head, style="bold green" if rank is not None else "dim")
        label.append(_preview(node.text, self.preview_chars))
        return label

    def _add(self, parent: TreeNode[int], node_id: int) -> TreeNode[int]:
        has_children = bool(self.raptor.nodes[node_id].children)
        return parent.add(self._label(node_id), data=node_id, allow_expand=has_children)

    def _populate(self, item: TreeNode[int]) -> None:
        if item.data is None or item.children:
            return
        for child in self.raptor.nodes[item.data].children:
            self._add(item, child)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[int]) -> None:
        self._populate(event.node)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[int]) -> None:
        if event.node.data is None:
            self._show_overview()
        else:
            self._show_node(event.node.data)

    def _walk(self) -> list[TreeNode[int]]:
        widget = self.query_one("#tree", Tree)
        out: list[TreeNode[int]] = []
        stack = list(reversed(widget.root.children))
        while stack:
            item = stack.pop()
            out.append(item)
            stack.extend(reversed(item.children))
        return out

    def _relabel(self) -> None:
        for item in self._walk():
            if item.data is not None:
                item.set_label(self._label(item.data))

    # ---- actions -------------------------------------------------------------------------

    def action_expand_level(self) -> None:
        """Expand every visible node one more level down."""
        self._expanded_levels += 1
        self.expand_to_depth(self._expanded_levels)

    def expand_to_depth(self, depth: int) -> None:
        widget = self.query_one("#tree", Tree)
        frontier = list(widget.root.children)
        for _ in range(depth):
            nxt: list[TreeNode[int]] = []
            for item in frontier:
                if item.allow_expand:
                    self._populate(item)
                    item.expand()
                    nxt.extend(item.children)
            frontier = nxt
        self._expanded_levels = depth

    def action_collapse(self) -> None:
        widget = self.query_one("#tree", Tree)
        for item in widget.root.children:
            item.collapse_all()
        self._expanded_levels = 0

    def action_next_method(self) -> None:
        self.method = METHODS[(METHODS.index(self.method) + 1) % len(METHODS)]
        self.sub_title = f"{self.raptor.title} · {self.method}"
        if self.question_index is not None:
            self.ask(self.question_index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self.ask(int(event.option.id))

    def ask(self, index: int) -> None:
        """Retrieve for question ``index`` with the current method and highlight the nodes."""
        q = self.questions[index]
        query = query_text(q.question, q.options)
        qvec = self.embedder.embed([query])[0] if self.embedder is not None else None
        if qvec is None and self.method not in ("truncation", "bm25"):
            self.query_one("#detail", Static).update("No embedder loaded; only bm25/truncation.")
            return
        self.question_index = index
        self.retrieval = self.retriever.run(self.method, query, qvec)
        self.rank = {node_id: r + 1 for r, node_id in enumerate(self.retrieval.node_ids)}
        self._reveal(self.retrieval.node_ids)
        self._relabel()
        self._show_question(q, self.retrieval)

    def _reveal(self, node_ids: list[int]) -> None:
        """Expand one path from the top layer to each retrieved node."""
        widget = self.query_one("#tree", Tree)
        for node_id in node_ids:
            path = [node_id]
            while self._parents[path[-1]]:
                path.append(self._parents[path[-1]][0])
            item = widget.root
            for step in reversed(path):
                match = next((c for c in item.children if c.data == step), None)
                if match is None:
                    break
                if step != node_id:
                    self._populate(match)
                    match.expand()
                item = match

    def focus_node(self, node_id: int) -> None:
        """Expand down to ``node_id`` and put the cursor on it once the tree has re-rendered."""
        self._reveal([node_id])
        widget = self.query_one("#tree", Tree)

        def move() -> None:
            for item in self._walk():
                if item.data == node_id:
                    widget.move_cursor(item)
                    break

        self.call_after_refresh(move)

    # ---- right-hand panel -----------------------------------------------------------------

    def _show_overview(self) -> None:
        t = self.raptor
        lines = [
            f"{t.title}",
            f"{len(t.segments)} sentences, {len(t.nodes)} nodes, {t.depth} layers",
            f"built with {t.meta.get('embedder')}, {t.meta.get('reducer')},",
            f"{t.meta.get('summarizer')}",
            "",
            "e expands a level, m changes the retrieval method,",
            "Enter on a question highlights what that method retrieves.",
        ]
        self.query_one("#detail", Static).update("\n".join(lines))

    def _show_node(self, node_id: int) -> None:
        t = self.raptor
        node = t.nodes[node_id]
        n_leaves = len(t.layers[0])
        lines = [f"L{node.layer} #{node.id} · {node.tokens} tokens"]
        if node.children:
            lines.append(f"summary of {len(node.children)} children: {_ranges(node.children)}")
        lines.append(
            f"covers leaves {_ranges(node.leaves)} "
            f"({len(node.leaves)} of {n_leaves}, {100 * len(node.leaves) / n_leaves:.0f}%)"
        )
        parents = self._parents[node_id]
        if parents:
            lines.append(f"parents: {', '.join(f'#{p}' for p in parents)}")
        segs = [t.segments[s] for s in node.segments]
        if node.layer > 0 and all(s.origin == "article" for s in segs):
            lines.append(f"{len(segs)} sentences, all copied verbatim from the article")
        elif segs and segs[0].start is not None:
            lines.append(f"article characters {segs[0].start}-{segs[-1].end}")
        if node_id in self.rank:
            lines.append(f"retrieved at rank {self.rank[node_id]} for the current question")
        lines += ["", node.text]
        self.query_one("#detail", Static).update(Text("\n".join(lines)))

    def _show_question(self, q: Question, found: Retrieval) -> None:
        layers: dict[int, int] = {}
        for node_id in found.node_ids:
            layer = self.raptor.nodes[node_id].layer
            layers[layer] = layers.get(layer, 0) + 1
        mix = ", ".join(f"L{k}×{v}" for k, v in sorted(layers.items(), reverse=True))
        # The retrieval summary goes first so it stays visible when the options are long.
        lines = [
            f"{found.method}: {len(found.node_ids)} nodes, {found.tokens} of "
            f"{self.retriever.budget} tokens ({mix})",
            "",
            f"{q.question}",
            *[f"{'*' if i == q.gold else ' '} {'ABCD'[i]}. {o}" for i, o in enumerate(q.options)],
            "",
            "* marks the gold answer; ● and a rank mark retrieved nodes in the tree.",
        ]
        self.query_one("#detail", Static).update(Text("\n".join(lines)))


def screen_text(app: App[None]) -> str:
    """The current screen as plain text (the same render path Textual uses for screenshots)."""
    width, height = app.size
    console = Console(
        width=width,
        height=height,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        record=True,
        legacy_windows=False,
        safe_box=False,
    )
    console.print(
        app.screen._compositor.render_update(full=True, screen_stack=app._background_screens)
    )
    return "\n".join(line.rstrip() for line in console.export_text().splitlines()) + "\n"


def capture_screen(
    tree: RaptorTree,
    questions: list[Question],
    embedder: Embedder | None,
    *,
    method: Method = "raptor-collapsed",
    depth: int = 0,
    node: int | None = None,
    question: int | None = None,
    size: tuple[int, int] = (120, 32),
    budget: int = 2000,
) -> str:
    """Run the explorer headless, apply the scripted steps, and return the screen as text.

    ``question`` is a 0-based index into ``questions``; ``node`` puts the cursor on that node.
    """
    explorer = TreeExplorer(tree, questions, embedder, budget=budget)
    explorer.method = method

    async def script() -> str:
        async with explorer.run_test(size=size) as pilot:
            if depth:
                explorer.expand_to_depth(depth)
            if question is not None:
                explorer.ask(question)
            if node is not None:
                await pilot.pause()
                explorer.focus_node(node)
            await pilot.pause()
            await pilot.pause()
            return screen_text(explorer)

    return asyncio.run(script())
