"""Small ipywidgets display helpers."""


def display_widgets(widget_list) -> None:
    """Show widgets by setting CSS visibility to visible."""
    for widget in widget_list:
        widget.layout.visibility = "visible"


def hide_widgets(widget_list) -> None:
    """Hide widgets by setting CSS visibility to hidden."""
    for widget in widget_list:
        widget.layout.visibility = "hidden"


def enable_widget(widget) -> None:
    """Restore the default widget display."""
    widget.layout.display = ""


def disable_widget(widget) -> None:
    """Hide a widget from layout flow."""
    widget.layout.display = "none"
