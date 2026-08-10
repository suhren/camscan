"""
Extra widgets and interface elements used by the application.
"""

import tkinter as tk
import typing as t

import customtkinter as ctk

from camscan.logging import logger


def clamp(value: float, min_value: float | None, max_value: float | None) -> float:
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


class NumberEntry(ctk.CTkFrame):
    def __init__(
        self,
        master: t.Any,
        integer: bool,
        value: float,
        label: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        step_size: float | None = None,
        default_value: float | None = None,
        on_value: t.Callable | None = None,
    ) -> None:
        super().__init__(master=master)

        self.initial_value = value
        self.integer = integer
        self.min_value = min_value
        self.max_value = max_value
        self.on_value = on_value
        self.step_size = step_size or 1.0
        self.default_value = default_value
        self.label = label

        self.entry_text_variable = tk.StringVar()
        self.entry_value_variable = (
            tk.IntVar(value=int(self.initial_value))
            if self.integer
            else tk.DoubleVar(value=self.initial_value)
        )

        self.set_value(self.initial_value)

        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_columnconfigure(3, weight=0)

        def _on_entry_event(_event: t.Any) -> None:
            try:
                new_text = self.entry_text_variable.get()
                self.set_value(new_text)
                if self.on_value is not None:
                    self.on_value(self.get_value())
            except ValueError as e:
                logger.error(f"Unable to parse entry text {new_text} to a number: {e}")
                self.set_value(self._previous_value)

        def _reset_default_value() -> None:
            self.set_value(self.default_value)
            if on_value is not None:
                on_value(self.get_value())

        def _increment() -> None:
            self.set_value(self.get_value() + self.step_size)
            if self.on_value is not None:
                self.on_value(self.get_value())

        def _decrement() -> None:
            self.set_value(self.get_value() - self.step_size)
            if self.on_value is not None:
                self.on_value(self.get_value())

        self.entry_widget = ctk.CTkEntry(
            self,
            textvariable=self.entry_text_variable,
            font=ctk.CTkFont(size=12),
            width=64,
        )

        # https://stackoverflow.com/questions/73786863/hi-how-can-i-change-a-string-number-from-entry-of-tkinter-into-int-number
        # https://stackoverflow.com/questions/76411353/about-tkinter-entry-and-the-focusout-focusin
        self.entry_widget.bind(sequence="<Return>", command=_on_entry_event)
        self.entry_widget.bind(sequence="<FocusOut>", command=_on_entry_event)

        self.entry_up_button = ctk.CTkButton(
            master=self,
            text="🔼",
            height=12,
            width=12,
            font=ctk.CTkFont(size=8),
            fg_color="transparent",
            command=_increment,
        )
        self.entry_down_button = ctk.CTkButton(
            master=self,
            text="🔽",
            height=12,
            width=12,
            font=ctk.CTkFont(size=8),
            fg_color="transparent",
            command=_decrement,
        )

        if self.default_value is not None:
            reset_button = ctk.CTkButton(
                master=self,
                text="⟲",
                height=24,
                width=24,
                fg_color="transparent",
                command=_reset_default_value,
            )
            reset_button.grid(row=0, column=3, rowspan=2)

        if self.label is not None:
            self.label_widget = ctk.CTkLabel(master=self, text=self.label)
            self.label_widget.grid(row=0, column=0, rowspan=2, sticky="nswe")

        self.entry_widget.grid(row=0, column=1, rowspan=2, sticky="nswe")
        self.entry_up_button.grid(row=0, column=2, sticky="nswe")
        self.entry_down_button.grid(row=1, column=2, sticky="nswe")

    def _cast(self, value: t.Any) -> float:
        return int(value) if self.integer else float(value)

    def _update_entry_text(self) -> None:
        self.entry_text_variable.set(f"{self.get_value():g}")

    def get_value(self) -> float:
        return self.entry_value_variable.get()

    def set_value(self, value: t.Any) -> None:
        self._previous_value = self.get_value()
        value = clamp(self._cast(value), self.min_value, self.max_value)
        self.entry_value_variable.set(value)
        self._update_entry_text()


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

        self.integer = integer
        self.label = label
        self.initial_value = value
        self.min_value = min_value if min_value is not None else 0.0
        self.max_value = max_value if max_value is not None else 1000.0
        self.number_of_steps = number_of_steps
        self.step_size = step_size
        self.default_value = default_value
        self.on_value = on_value
        self._previous_value = self.initial_value

        if self.min_value > self.max_value:
            raise ValueError(f"{self.min_value=} must be less than {self.max_value=}")

        if self.number_of_steps is not None and self.step_size is not None:
            raise ValueError("Only one of number_of_steps and step_size can be used")

        if self.number_of_steps is None and self.step_size is None:
            raise ValueError("One of number_of_steps and step_size must be used")

        if self.step_size is not None:
            self.number_of_steps = int(
                (self.max_value - self.min_value) / self.step_size
            )
        elif self.number_of_steps is not None:
            self.step_size = (self.max_value - self.min_value) / self.number_of_steps

        self.variable = (
            tk.IntVar(value=int(self.initial_value))
            if self.integer
            else tk.DoubleVar(value=self.initial_value)
        )

        def _on_slider_value(value: float) -> None:
            self.set_value(value)
            if on_value is not None:
                on_value(self.get_value())

        def _on_entry_value(value: float) -> None:
            if on_value is not None:
                on_value(value)
            self.set_value(value)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        if self.label is not None:
            self.label_widget = ctk.CTkLabel(master=self, text=self.label)
            self.label_widget.grid(row=0, column=0, columnspan=2, sticky="nws")

        self.number_entry = NumberEntry(
            master=self,
            value=self.initial_value,
            min_value=self.min_value,
            max_value=self.max_value,
            step_size=self.step_size,
            integer=self.integer,
            default_value=self.default_value,
            on_value=_on_entry_value,
        )

        self.slider_widget = ctk.CTkSlider(
            master=self,
            from_=self.min_value,
            to=self.max_value,
            variable=self.variable,
            number_of_steps=self.number_of_steps,
            command=_on_slider_value,
        )

        self.slider_widget.grid(row=1, column=0)
        self.number_entry.grid(row=1, column=1, sticky="e")

        self.pack()

    def _cast(self, value: t.Any) -> float:
        return int(value) if self.integer else float(value)

    def _update_slider(self) -> None:
        self.slider_widget.set(self.get_value())

    def _update_entry(self) -> None:
        self.number_entry.set_value(self.get_value())

    def get_value(self) -> float:
        return self.variable.get()

    def set_value(self, value: t.Any) -> None:
        self._previous_value = self.get_value()
        value = clamp(self._cast(value), self.min_value, self.max_value)
        self.variable.set(value)
        self._update_slider()
        self._update_entry()


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
