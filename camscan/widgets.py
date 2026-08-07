"""
Extra widgets and interface elements used by the application.
"""

import tkinter as tk
import typing as t

import customtkinter as ctk


class Tooltip:
    """
    A tooltip that is shown when the mouse cursors hovers over a widget for some
    time, and then disappears when the mouse cursors leaves.
    :param widget: Widget that should show the tooltip when hovered over
    :param text: Text which is shown in the tooltip
    :param x_offset: Horizontal distance from the cursor and the top left corner of the tooltip
    :param y_offset: Horizontal distance from the cursor and the top left corner of the tooltip
    :param font_size: Font size of the text in the tooltip
    :param padx: Horizontal padding of the text inside the tooltip
    :param pady: Vertical padding of the text inside the tooltip
    :param wraplength: Horizontal pixel width of the text to wrap around
    :param display_delay: Time to wait before showing the tooltip when hovering
    """

    def __init__(
        self,
        widget: ctk.CTkBaseClass,
        text: str,
        x_offset: int = 16,
        y_offset: int = 16,
        font_size: int = 14,
        padx: int = 10,
        pady: int = 5,
        wraplength: int = 200,
        display_delay: float = 0.5,
    ):
        self.widget = widget
        self.text = text
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.font_size = font_size
        self.padx = padx
        self.pady = pady
        self.wraplength = wraplength

        # We require a minimum delay for the code to work properly
        self.display_delay_ms = round(1000 * max(display_delay, 0.05))

        # Bind events to the widget when the cursor enters or leaves
        self.widget.bind("<Enter>", self._enter)
        self.widget.bind("<Leave>", self._leave)

        # If the triggering widget has children, each of them might be able to
        # trigger the <Enter> and <Leave> events. This has to be handled later.
        self.widget_children = self.widget.winfo_children()

        # Some widgets, like the CTk.CheckBox, has child elements like a canvas
        # that are not bound to the leave event. For this code to work, this has
        # to be added.
        for child in self.widget_children:
            child.bind("<Leave>", self._leave, add="+")

        # Variable keeping track of if the tooltip is currently showing
        self.showing: bool = False
        # Reference to the created tooltip window
        self.window: ctk.CTkToplevel | None = None

    def _enter(self, _: t.Any) -> None:
        """
        Callback function for when the cursor enters the widget, or potentially
        one of its children.
        NOTE: If the cursor moves quickly, it seems that the <Leave> event can
        trigger before the <Enter> event. This will lead to problems where we
        might try to first destroy the old tooltip window, and then create a new
        one. This produces a tooltip that will not disappear until the user
        hovers over the widget again. This is handled here though, since we
        introduce a delay before showing the window and double-check that the
        cursor is still hovering above the widget.
        NOTE: This function might be called at the same time by parallel event
        triggers, which can cause multiple Tooltip windows to spawn. For that
        reason, we need to set the 'showing' variable to prevent additional
        events from triggering until the window has been shown.
        """

        def _enter_delayed() -> None:
            # We only want to show the tooltip if it is not already showing
            if not self.showing:
                # After the delay, we can no longer be sure that the cursor is
                # hovering over the widget or one of its children. Check this.
                x, y = self.widget.winfo_pointerxy()
                widget_under_mouse = self.widget.winfo_containing(x, y)
                if widget_under_mouse in self.widget_children:
                    self.show()

        self.widget.after(ms=self.display_delay_ms, func=_enter_delayed)

    def _leave(self, _: t.Any) -> None:
        """
        Callback function for when the cursor leaves the widget, or potentially
        one of its children.
        """
        # If there is no current tooltip window open, simply return
        if not self.window:
            return

        # Get the cursor position and check what widget is under the mouse
        x, y = self.widget.winfo_pointerxy()
        widget_under_mouse = self.widget.winfo_containing(x, y)

        # If the widget under the mouse is one of the original widget's children
        # we can consider that we are still hovering over the widget. If so,
        # keep the tooltip window open and return. An example of this is a
        # CTkButton that is made up of a canvas with a central label. When the
        # cursor moves from the label onto the canvas, a <Leave> event will be
        # generated, even though we want to keep the tooltip open.
        if widget_under_mouse in self.widget_children:
            return

        # Otherwise, the cursor has left the widget and we hide the tooltip
        self.hide()

    def show(self) -> None:
        """
        Show the tooltip by creating a new TopLevel window and place it at an
        offset from the current cursor position.
        """
        # Set the 'showing' variable to True to prevent further show attempts
        self.showing = True

        # This is only a Tooltip window, so instruct window manager to ignore it
        window = ctk.CTkToplevel(self.widget)
        window.wm_overrideredirect(1)

        # Set the window position to the cursor with the offset
        # Geometry (position) can be set using "<width>x<height>+<x>+<y>"
        px, py = self.widget.winfo_pointerxy()
        x = px + self.x_offset
        y = py + self.y_offset
        window.wm_geometry(f"+{x}+{y}")

        # Create the label element and pack it
        label = ctk.CTkLabel(
            window,
            text=self.text,
            justify=tk.LEFT,
            font=ctk.CTkFont(size=self.font_size),
            wraplength=self.wraplength,
        )
        label.pack(padx=self.padx, pady=self.pady)

        # Update Tooltip window to ensure it has correct position and size
        window.update()

        # Get the tooltip window position and size
        tt_x = window.winfo_rootx()
        tt_y = window.winfo_rooty()
        # It is not reliable to use winfo_width or winfo_height on the window
        # so we manually calculate the size of the label with its padding
        tt_w = label.winfo_width() + self.padx * 2
        tt_h = label.winfo_height() + self.pady * 2

        # Get position and size of the toplevel window containing the widget
        # that triggered the tooltip
        tl_window = self.widget.winfo_toplevel()
        tl_x = tl_window.winfo_rootx()
        tl_y = tl_window.winfo_rooty()
        tl_w = tl_window.winfo_width()
        tl_h = tl_window.winfo_height()

        # If the tooltip is too far to the right, snap it back to the left
        if tt_x + tt_w > tl_x + tl_w:
            tt_x = tl_x + tl_w - tt_w

        # If the tooltip is too far to down, snap it back to the bottom
        if tt_y + tt_h > tl_y + tl_h:
            tt_y = tl_y + tl_h - tt_h

        # If the tooltip is too far to the left, snap it back to the right
        tt_x = max(tt_x, tl_x)

        # If the tooltip is too far up, snap it back down
        tt_y = max(tt_y, tl_y)

        # Final check: The tooltip window is not allowed to be on top of the
        # cursor, so move it down a bit if that happens
        if (tt_x <= px <= tt_x + tt_w) and (tt_y <= py <= tt_y + tt_h):
            tt_y = py + 16

        # Set the new position of the tooltip window
        window.wm_geometry(f"+{tt_x}+{tt_y}")

        # Update the reference variable that keeps track of the tooltip window
        self.window = window

    def hide(self) -> None:
        """
        Hide the tooltip window.
        """
        # Destroy the actual CTkToplevel object to remove the window
        if self.window is not None:
            self.window.destroy()
        self.widget.update()
        # Update references in this class to allow a new tooltip to be created
        self.window = None
        self.showing = False


def _set_value_label(widget: ctk.CTkLabel, name: str, value: t.Any) -> None:
    widget.configure(text=f"{name}: {value:g}")


class InputNumber(ctk.CTkFrame):
    def __init__(
        self,
        master: t.Any,
        integer: bool,
        label: str,
        value: float,
        min_value: float | None,
        max_value: float | None,
        number_of_steps: int | None,
        step_size: float | None,
        default_value: float | None,
        on_value: t.Callable | None,
    ) -> None:
        super().__init__(master=master)

        min_value = min_value if min_value is not None else 0.0
        max_value = max_value if max_value is not None else 1000.0

        if min_value > max_value:
            raise ValueError(f"{min_value=} must be less than {max_value=}")

        # Set up widgets
        variable = tk.IntVar(value=int(value)) if integer else tk.DoubleVar(value=value)
        label_widget = ctk.CTkLabel(master=self)

        _set_value_label(label_widget, label, value)

        if number_of_steps is not None and step_size is not None:
            raise ValueError("Only one of number_of_steps and step_size can be used")
        elif step_size is not None:
            number_of_steps = int((max_value - min_value) / step_size)

        def _reset_default_value() -> None:
            if default_value is not None:
                new_value = int(default_value) if integer else default_value
                _set_value_label(label_widget, label, new_value)
                slider_widget.set(new_value)
                if on_value is not None:
                    on_value(new_value)

        def _on_value(value: float) -> None:
            new_value = int(value) if integer else value
            _set_value_label(label_widget, label, new_value)
            if on_value is not None:
                on_value(new_value)

        slider_widget = ctk.CTkSlider(
            master=self,
            from_=min_value,
            to=max_value,
            variable=variable,
            number_of_steps=number_of_steps,
            command=_on_value,
        )

        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        label_widget.grid(row=0, column=0, columnspan=2)
        slider_widget.grid(row=1, column=0)

        if default_value is not None:
            default_button = ctk.CTkButton(
                master=self,
                text="⟲",
                height=24,
                width=24,
                fg_color="transparent",
                command=_reset_default_value,
            )
            default_button.grid(row=1, column=1)

        self.pack()


class InputFloat(InputNumber):
    def __init__(
        self,
        master: t.Any,
        label: str = "Value",
        value: float = 0,
        min_value: float | None = None,
        max_value: float | None = None,
        number_of_steps: int | None = None,
        step_size: float | None = 0.01,
        default_value: float | None = None,
        on_value: t.Callable | None = None,
    ) -> None:
        super().__init__(
            master=master,
            integer=False,
            label=label,
            value=value,
            min_value=min_value,
            max_value=max_value,
            number_of_steps=number_of_steps,
            step_size=step_size,
            default_value=default_value,
            on_value=on_value,
        )


class InputInt(InputNumber):
    def __init__(
        self,
        master: t.Any,
        label: str = "Value",
        value: int = 0,
        min_value: int | None = None,
        max_value: int | None = None,
        number_of_steps: int | None = None,
        step_size: int | None = 1,
        default_value: int | None = None,
        on_value: t.Callable | None = None,
    ) -> None:
        super().__init__(
            master=master,
            integer=True,
            label=label,
            value=value,
            min_value=min_value,
            max_value=max_value,
            number_of_steps=number_of_steps,
            step_size=step_size,
            default_value=default_value,
            on_value=on_value,
        )
