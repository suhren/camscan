"""
Tooltip widget with a text box shown when hovering over another widget.
"""

import tkinter as tk
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
        font_size: int = 12,
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
        self.showing = False
        # Reference to the created tooltip window
        self.window = None

    def _enter(self, _):
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

        def _enter_delayed():
            # We only want to show the tooltip if it is not already showing
            if not self.showing:
                # After the delay, we can no longer be sure that the cursor is
                # hovering over the widget or one of its children. Check this.
                x, y = self.widget.winfo_pointerxy()
                widget_under_mouse = self.widget.winfo_containing(x, y)
                if widget_under_mouse in self.widget_children:
                    self.show()

        self.widget.after(ms=self.display_delay_ms, func=_enter_delayed)

    def _leave(self, _):
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

    def show(self):
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
        if tt_x < tl_x:
            tt_x = tl_x

        # If the tooltip is too far up, snap it back down
        if tt_y < tl_y:
            tt_y = tl_y

        # Final check: The tooltip window is not allowed to be on top of the
        # cursor, so move it down a bit if that happens
        if (tt_x <= px <= tt_x + tt_w) and (tt_y <= py <= tt_y + tt_h):
            tt_y = py + 16

        # Set the new position of the tooltip window
        window.wm_geometry(f"+{tt_x}+{tt_y}")

        # Update the reference variable that keeps track of the tooltip window
        self.window = window

    def hide(self):
        """
        Hide the tooltip window.
        """
        # Destroy the actual CTkToplevel object to remove the window
        self.window.destroy()
        self.widget.update()
        # Update references in this class to allow a new tooltip to be created
        self.window = None
        self.showing = False
