"""Demo application for cjm-transcript-segmentation library.

Demonstrates text segmentation with card stack navigation, split/merge operations,
and keyboard navigation. Works standalone without the full transcript workflow.

Run with: python demo_app.py
"""

from typing import List, Dict, Any, Callable, Tuple
from functools import wraps
import asyncio

from fasthtml.common import (
    fast_app, Div, H1, H2, P, Span, Button, Input, Script,
    APIRouter, Details, Summary,
)

# DaisyUI components
from cjm_fasthtml_daisyui.core.resources import get_daisyui_headers
from cjm_fasthtml_daisyui.core.testing import create_theme_persistence_script
from cjm_fasthtml_daisyui.components.data_display.badge import badge, badge_styles, badge_sizes
from cjm_fasthtml_daisyui.components.data_display.collapse import (
    collapse, collapse_title, collapse_content, collapse_modifiers
)
from cjm_fasthtml_daisyui.utilities.semantic_colors import bg_dui, text_dui, border_dui
from cjm_fasthtml_daisyui.utilities.border_radius import border_radius

# Tailwind utilities
from cjm_fasthtml_tailwind.utilities.spacing import p, m
from cjm_fasthtml_tailwind.utilities.sizing import w, h, min_h, container, max_w
from cjm_fasthtml_tailwind.utilities.typography import font_size, font_weight, text_align, uppercase, tracking
from cjm_fasthtml_tailwind.utilities.layout import overflow, display_tw
from cjm_fasthtml_tailwind.utilities.borders import border
from cjm_fasthtml_tailwind.utilities.effects import ring
from cjm_fasthtml_tailwind.utilities.transitions_and_animation import transition, duration
from cjm_fasthtml_tailwind.utilities.flexbox_and_grid import (
    flex_display, flex_direction, justify, items, gap, grow
)
from cjm_fasthtml_tailwind.core.base import combine_classes

# Lucide icons
from cjm_fasthtml_lucide_icons.factory import lucide_icon

# App core
from cjm_fasthtml_app_core.core.routing import register_routes
from cjm_fasthtml_app_core.core.htmx import handle_htmx_request
from cjm_fasthtml_app_core.core.layout import wrap_with_layout

# Interactions library
from cjm_fasthtml_interactions.core.state_store import get_session_id

# State store
from cjm_workflow_state.state_store import SQLiteWorkflowStateStore

# Plugin system
from cjm_plugin_system.core.manager import PluginManager
from cjm_plugin_system.core.scheduling import SafetyScheduler

# Keyboard navigation
from cjm_fasthtml_keyboard_navigation.core.manager import ZoneManager
from cjm_fasthtml_keyboard_navigation.components.system import render_keyboard_system
from cjm_fasthtml_keyboard_navigation.components.hints import render_keyboard_hints

# Card stack library
from cjm_fasthtml_card_stack.keyboard.actions import build_card_stack_url_map
from cjm_fasthtml_card_stack.components.controls import render_width_slider
from cjm_fasthtml_card_stack.components.progress import render_progress_indicator
from cjm_fasthtml_card_stack.core.constants import DEFAULT_VISIBLE_COUNT, DEFAULT_CARD_WIDTH

# Segmentation library imports
from cjm_transcript_segmentation.models import TextSegment, SegmentationUrls
from cjm_transcript_segmentation.services.segmentation import SegmentationService
from cjm_transcript_segmentation.html_ids import SegmentationHtmlIds
from cjm_transcript_segmentation.components.card_stack_config import (
    SEG_CS_CONFIG, SEG_CS_IDS, SEG_CS_BTN_IDS, SEG_TS_IDS,
)
from cjm_transcript_segmentation.components.keyboard_config import (
    SD_SEG_ENTER_SPLIT_BTN, SD_SEG_EXIT_SPLIT_BTN,
    SD_SEG_SPLIT_BTN, SD_SEG_MERGE_BTN, SD_SEG_UNDO_BTN,
    create_seg_kb_parts,
)
from cjm_transcript_segmentation.components.step_renderer import (
    render_seg_column_body, render_toolbar, render_seg_stats,
    render_seg_footer_content, render_seg_mini_stats_text,
)
from cjm_transcript_segmentation.routes.init import init_segmentation_routers
from cjm_transcript_segmentation.routes.handlers import SegInitResult, _handle_seg_init
from cjm_transcript_segmentation.utils import calculate_segment_stats


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_TEXT = """November the 10th, Wednesday, 9 p.m. I'm standing in a dark alley. After waiting several hours, the time has come. A woman with long dark hair approaches. I have to act and fast before she realises what has happened. I must find out."""


# =============================================================================
# Demo HTML IDs
# =============================================================================

class DemoHtmlIds:
    """HTML IDs for demo app layout."""
    CONTAINER = "seg-demo-container"
    COLUMN = "seg-demo-column"
    COLUMN_HEADER = "seg-demo-column-header"
    COLUMN_CONTENT = "seg-demo-column-content"
    MINI_STATS = "seg-demo-mini-stats"
    KEYBOARD_SYSTEM = "seg-demo-kb-system"
    SHARED_HINTS = "seg-demo-hints"
    SHARED_TOOLBAR = "seg-demo-toolbar"
    SHARED_CONTROLS = "seg-demo-controls"
    SHARED_FOOTER = "seg-demo-footer"


# =============================================================================
# Single-Zone Keyboard System
# =============================================================================

def build_single_zone_kb_system(
    urls: SegmentationUrls,
) -> Tuple[ZoneManager, Any]:
    """Build single-zone keyboard system for segmentation only (no alignment)."""
    # Get segmentation-specific building blocks
    seg_zone, seg_actions, seg_modes = create_seg_kb_parts(
        ids=SEG_CS_IDS,
        button_ids=SEG_CS_BTN_IDS,
        config=SEG_CS_CONFIG,
    )

    # Assemble into ZoneManager (single zone, no zone switching)
    kb_manager = ZoneManager(
        zones=(seg_zone,),
        actions=seg_actions,
        modes=seg_modes,
        initial_zone_id=seg_zone.id,
        state_hidden_inputs=True,
    )

    # Build URL maps
    # Include: card stack focused index (segment_index) and token selector anchor (split position)
    include_selector = f"#{SEG_CS_IDS.focused_index_input}, #{SEG_TS_IDS.anchor_input}"

    # URL mappings
    url_map = {
        **build_card_stack_url_map(SEG_CS_BTN_IDS, urls.card_stack),
        SD_SEG_ENTER_SPLIT_BTN: urls.enter_split,
        SD_SEG_EXIT_SPLIT_BTN: urls.exit_split,
        SD_SEG_SPLIT_BTN: urls.split,
        SD_SEG_MERGE_BTN: urls.merge,
        SD_SEG_UNDO_BTN: urls.undo,
    }

    # Target maps
    target = f"#{SEG_CS_IDS.card_stack}"
    target_map = {btn_id: target for btn_id in url_map}

    # Include maps
    include_map = {btn_id: include_selector for btn_id in url_map}

    # Swap map (none for all - OOB swaps handle updates)
    swap_map = {btn_id: "none" for btn_id in url_map}

    kb_system = render_keyboard_system(
        kb_manager,
        url_map=url_map,
        target_map=target_map,
        include_map=include_map,
        swap_map=swap_map,
        show_hints=False,
        include_state_inputs=True,
    )

    return kb_manager, kb_system


def render_keyboard_hints_collapsible(
    manager: ZoneManager,
    container_id: str = "seg-demo-kb-hints",
) -> Any:
    """Render keyboard shortcut hints in a collapsible DaisyUI collapse."""
    hints = render_keyboard_hints(
        manager,
        include_navigation=True,
        include_zone_switch=False,
        badge_style="outline",
        container_id=container_id,
        use_icons=False
    )

    return Details(
        Summary(
            "Keyboard Shortcuts",
            cls=combine_classes(collapse_title, font_size.sm, font_weight.medium)
        ),
        Div(
            hints,
            cls=collapse_content
        ),
        cls=combine_classes(collapse, collapse_modifiers.arrow, bg_dui.base_200)
    )


# =============================================================================
# Mock Services
# =============================================================================

class MockSourceService:
    """Mock source service that provides static sample data from two audio sources."""

    def get_source_blocks(self, selected_sources: List[Dict]) -> List[Any]:
        """Return sample text as two source blocks (simulating chunked audio)."""
        from cjm_source_provider.models import SourceBlock
        return [
            SourceBlock(
                id="demo-source-1",
                provider_id="demo-provider",
                text=SAMPLE_TEXT,
            ),
            SourceBlock(
                id="demo-source-2",
                provider_id="demo-provider",
                text=SAMPLE_TEXT,
            ),
        ]


# =============================================================================
# Init Handler Wrapper (Simplified - No Alignment)
# =============================================================================

def create_demo_init_wrapper(
    urls: SegmentationUrls,
) -> Callable:
    """Create wrapper for seg init that builds KB system (no alignment column)."""

    async def wrapped_init(
        state_store: SQLiteWorkflowStateStore,
        workflow_id: str,
        source_service: MockSourceService,
        segmentation_service: SegmentationService,
        request,
        sess,
        urls: SegmentationUrls,
        visible_count: int = DEFAULT_VISIBLE_COUNT,
        card_width: int = DEFAULT_CARD_WIDTH,
    ):
        """Wrapped init that adds KB system and chrome."""
        # Call pure domain handler
        result: SegInitResult = await _handle_seg_init(
            state_store, workflow_id, source_service, segmentation_service,
            request, sess, urls, visible_count, card_width,
        )

        # Build single-zone KB system
        kb_manager, kb_system = build_single_zone_kb_system(urls)

        # OOB swap for keyboard system container
        kb_system_oob = Div(
            kb_system.script,
            kb_system.hidden_inputs,
            kb_system.action_buttons,
            id=DemoHtmlIds.KEYBOARD_SYSTEM,
            hx_swap_oob="innerHTML"
        )

        # Hints OOB
        hints_oob = Div(
            render_keyboard_hints_collapsible(kb_manager),
            id=DemoHtmlIds.SHARED_HINTS,
            hx_swap_oob="innerHTML"
        )

        # Toolbar OOB
        toolbar_oob = Div(
            render_toolbar(
                reset_url=urls.reset, ai_split_url=urls.ai_split, undo_url=urls.undo,
                can_undo=(result.history_depth > 0),
                visible_count=result.visible_count,
                is_auto_mode=result.is_auto_mode,
            ),
            id=DemoHtmlIds.SHARED_TOOLBAR,
            hx_swap_oob="innerHTML"
        )

        # Controls OOB (width slider)
        controls_oob = Div(
            render_width_slider(SEG_CS_CONFIG, SEG_CS_IDS, card_width=result.card_width),
            id=DemoHtmlIds.SHARED_CONTROLS,
            hx_swap_oob="innerHTML"
        )

        # Footer OOB
        footer_oob = Div(
            render_seg_footer_content(result.segments, result.focused_index),
            id=DemoHtmlIds.SHARED_FOOTER,
            hx_swap_oob="innerHTML"
        )

        # Mini-stats badge OOB
        mini_stats_oob = Span(
            render_seg_mini_stats_text(result.segments),
            id=DemoHtmlIds.MINI_STATS,
            cls=combine_classes(badge, badge_styles.ghost, badge_sizes.sm),
            hx_swap_oob="true",
        )

        return (
            result.column_body, kb_system_oob, hints_oob,
            toolbar_oob, controls_oob, footer_oob, mini_stats_oob,
        )

    return wrapped_init


# =============================================================================
# Mutation Handler Wrappers (Pass-through - No Alignment Status)
# =============================================================================

def wrap_mutation_handler(handler: Callable) -> Callable:
    """Wrap mutation handler (pass-through for demo - no alignment status)."""
    @wraps(handler)
    async def wrapped(
        state_store: SQLiteWorkflowStateStore,
        workflow_id: str,
        *args,
        **kwargs
    ):
        if asyncio.iscoroutinefunction(handler):
            result = await handler(state_store, workflow_id, *args, **kwargs)
        else:
            result = handler(state_store, workflow_id, *args, **kwargs)
        return result
    return wrapped


# =============================================================================
# Demo Page Renderer
# =============================================================================

def render_demo_page(
    urls: SegmentationUrls,
) -> Callable:
    """Create the demo page content factory."""

    def page_content():
        """Render the demo page with card stack column."""

        # Column header
        header = Div(
            Span(
                "Text Segmentation",
                cls=combine_classes(
                    font_size.sm, font_weight.bold,
                    uppercase, tracking.wide,
                    text_dui.base_content.opacity(50)
                )
            ),
            Span(
                "--",
                id=DemoHtmlIds.MINI_STATS,
                cls=combine_classes(badge, badge_styles.ghost, badge_sizes.sm)
            ),
            id=DemoHtmlIds.COLUMN_HEADER,
            cls=combine_classes(
                flex_display, justify.between, items.center,
                p(3), bg_dui.base_200,
                border_dui.base_300, border.b()
            )
        )

        # Column content (loading state with auto-trigger)
        from cjm_fasthtml_card_stack.components.states import render_loading_state

        content = Div(
            render_loading_state(SEG_CS_IDS, message="Initializing segments..."),
            Div(
                hx_post=urls.init,
                hx_trigger="load",
                hx_target=f"#{SegmentationHtmlIds.COLUMN_CONTENT}",
                hx_swap="outerHTML"
            ),
            id=SegmentationHtmlIds.COLUMN_CONTENT,
            cls=combine_classes(grow(), overflow.hidden, flex_display, flex_direction.col, p(4))
        )

        # Column
        column_cls = combine_classes(
            w.full, max_w._4xl, m.x.auto,
            min_h(0),
            flex_display, flex_direction.col,
            bg_dui.base_100, border_dui.base_300, border(1),
            border_radius.box,
            overflow.hidden,
            transition.all, duration._200,
            ring(1), "ring-primary",
        )

        column = Div(
            header,
            content,
            id=DemoHtmlIds.COLUMN,
            cls=column_cls
        )

        # Placeholder chrome
        hints = Div(
            P("Keyboard hints will appear here after initialization.",
              cls=combine_classes(font_size.sm, text_dui.base_content.opacity(50))),
            id=DemoHtmlIds.SHARED_HINTS,
            cls=str(p(2))
        )

        toolbar = Div(
            P("Toolbar actions will appear here after initialization.",
              cls=combine_classes(font_size.sm, text_dui.base_content.opacity(50))),
            id=DemoHtmlIds.SHARED_TOOLBAR,
            cls=str(p(2))
        )

        controls = Div(
            P("Width controls will appear here after initialization.",
              cls=combine_classes(font_size.sm, text_dui.base_content.opacity(50))),
            id=DemoHtmlIds.SHARED_CONTROLS,
            cls=str(p(2))
        )

        footer = Div(
            P("Footer with progress will appear here after initialization.",
              cls=combine_classes(font_size.sm, text_dui.base_content.opacity(50))),
            id=DemoHtmlIds.SHARED_FOOTER,
            cls=combine_classes(
                p(1), bg_dui.base_100,
                border_dui.base_300, border.t(),
                flex_display, justify.center, items.center
            )
        )

        # Keyboard system container (empty initially, populated by init handler)
        kb_container = Div(id=DemoHtmlIds.KEYBOARD_SYSTEM)

        return Div(
            # Header
            Div(
                H1("Text Segmentation Demo",
                   cls=combine_classes(font_size._3xl, font_weight.bold)),
                P(
                    "Split and merge text segments using keyboard navigation and the token selector.",
                    cls=combine_classes(text_dui.base_content.opacity(70), m.b(2))
                ),
            ),

            # Shared chrome
            hints,
            toolbar,
            controls,

            # Content area
            Div(
                column,
                cls=combine_classes(
                    grow(),
                    min_h(0),
                    flex_display,
                    flex_direction.col,
                    overflow.hidden,
                    p(1),
                )
            ),

            # Footer
            footer,

            # Keyboard system container
            kb_container,

            id=DemoHtmlIds.CONTAINER,
            cls=combine_classes(
                container, max_w._5xl, m.x.auto,
                h.full,
                flex_display, flex_direction.col,
                p(4), p.x(2), p.b(0)
            )
        )

    return page_content


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Initialize the segmentation demo and start the server."""
    print("\n" + "=" * 70)
    print("Initializing cjm-transcript-segmentation Demo")
    print("=" * 70)

    # Initialize FastHTML app
    app, rt = fast_app(
        pico=False,
        hdrs=[*get_daisyui_headers(), create_theme_persistence_script()],
        title="Segmentation Demo",
        htmlkw={'data-theme': 'light'},
        secret_key="demo-secret-key"
    )

    router = APIRouter(prefix="")

    # -------------------------------------------------------------------------
    # Set up state store and services
    # -------------------------------------------------------------------------
    import tempfile
    from pathlib import Path

    temp_db = Path(tempfile.gettempdir()) / "cjm_transcript_segmentation_demo_state.db"
    state_store = SQLiteWorkflowStateStore(temp_db)
    workflow_id = "seg-demo"

    print(f"  State store: {temp_db}")

    # -------------------------------------------------------------------------
    # Set up plugin manager and load NLTK plugin
    # -------------------------------------------------------------------------
    print("\n[Plugin System]")
    plugin_manager = PluginManager(scheduler=SafetyScheduler())

    # Discover plugins from JSON manifests
    plugin_manager.discover_manifests()

    # Load the NLTK plugin
    nltk_plugin_name = "cjm-text-plugin-nltk"
    nltk_meta = plugin_manager.get_discovered_meta(nltk_plugin_name)
    if nltk_meta:
        try:
            success = plugin_manager.load_plugin(nltk_meta, {"language": "english"})
            status = "loaded" if success else "failed"
            print(f"  {nltk_plugin_name}: {status}")
        except Exception as e:
            print(f"  {nltk_plugin_name}: error - {e}")
    else:
        print(f"  {nltk_plugin_name}: not found (will use fallback)")

    # Create services
    source_service = MockSourceService()
    segmentation_service = SegmentationService(plugin_manager, nltk_plugin_name)

    # Initialize selection state with demo source
    # (Required because init handler reads selected_sources from selection state)
    def init_demo_state(sess):
        """Ensure demo state is initialized for session."""
        session_id = get_session_id(sess)
        workflow_state = state_store.get_state(workflow_id, session_id)
        if "step_states" not in workflow_state:
            workflow_state["step_states"] = {}
        if "selection" not in workflow_state["step_states"]:
            workflow_state["step_states"]["selection"] = {
                "selected_sources": [
                    {"record_id": "demo-source-1", "provider_id": "demo-provider"},
                    {"record_id": "demo-source-2", "provider_id": "demo-provider"},
                ]
            }
            state_store.update_state(workflow_id, session_id, workflow_state)

    # -------------------------------------------------------------------------
    # Set up segmentation routes
    # -------------------------------------------------------------------------
    # Create wrapped handlers
    from cjm_transcript_segmentation.routes.handlers import (
        _handle_seg_split, _handle_seg_merge, _handle_seg_undo,
        _handle_seg_reset, _handle_seg_ai_split,
    )

    # We'll create the init wrapper after we have URLs
    # For now, set up routes with pass-through wrappers for mutations
    wrapped_handlers = {
        "split": wrap_mutation_handler(_handle_seg_split),
        "merge": wrap_mutation_handler(_handle_seg_merge),
        "undo": wrap_mutation_handler(_handle_seg_undo),
        "reset": wrap_mutation_handler(_handle_seg_reset),
        "ai_split": wrap_mutation_handler(_handle_seg_ai_split),
    }

    # Initialize routers
    seg_routers, seg_urls, seg_routes = init_segmentation_routers(
        state_store=state_store,
        workflow_id=workflow_id,
        source_service=source_service,
        segmentation_service=segmentation_service,
        prefix="/seg",
        wrapped_handlers=wrapped_handlers,
    )

    # Now create and register the wrapped init handler
    wrapped_init = create_demo_init_wrapper(seg_urls)

    # Override the init route with our wrapped version
    init_router = APIRouter(prefix="/seg/workflow")

    @init_router
    async def init(request, sess):
        """Initialize segments with KB system."""
        init_demo_state(sess)
        return await wrapped_init(
            state_store, workflow_id, source_service, segmentation_service,
            request, sess, urls=seg_urls,
        )

    # -------------------------------------------------------------------------
    # Page routes
    # -------------------------------------------------------------------------
    page_content = render_demo_page(seg_urls)

    @router
    def index(request, sess):
        """Demo homepage."""
        init_demo_state(sess)
        return handle_htmx_request(request, page_content)

    # -------------------------------------------------------------------------
    # Register routes
    # -------------------------------------------------------------------------
    register_routes(app, router, init_router, *seg_routers)

    # Debug output
    print("\n" + "=" * 70)
    print("Registered Routes:")
    print("=" * 70)
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"  {route.path}")
    print("=" * 70)
    print("Demo App Ready!")
    print("=" * 70 + "\n")

    return app


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    app = main()

    port = 5034
    host = "0.0.0.0"
    display_host = 'localhost' if host in ['0.0.0.0', '127.0.0.1'] else host

    print(f"Server: http://{display_host}:{port}")
    print()
    print("Controls:")
    print("  Arrow Up/Down     - Navigate segments")
    print("  Enter/Space       - Enter split mode")
    print("  Arrow Left/Right  - Move caret (in split mode)")
    print("  Enter/Space       - Execute split (in split mode)")
    print("  Escape            - Exit split mode")
    print("  Backspace         - Merge with previous segment")
    print("  Ctrl+Z            - Undo")
    print("  [ / ]             - Adjust viewport width")
    print()

    timer = threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}"))
    timer.daemon = True
    timer.start()

    uvicorn.run(app, host=host, port=port)
