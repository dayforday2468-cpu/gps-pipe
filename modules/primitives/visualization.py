from collections.abc import Iterator
import matplotlib.pyplot as plt
import polars as pl

from matplotlib.ticker import LinearLocator, FormatStrFormatter


class GPSVisualizer:
    def __init__(
        self,
        title: str | None = None,
        show_legend: bool = True,
        tick_spacing: int = 200,
    ) -> None:
        self.title = title
        self.show_legend = show_legend
        self.tick_spacing = tick_spacing
        self._layers = []

    def add(
        self,
        data: pl.DataFrame,
        label: str | None = None,
        point_size: float = 5,
        point_color: str | None = None,
        show_line: bool = False,
        line_color: str | None = None,
        line_style: str = "-",
        line_width: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        self._layers.append(
            {
                "data": data,
                "label": label,
                "point_size": point_size,
                "point_color": point_color,
                "show_line": show_line,
                "line_color": line_color,
                "line_style": line_style,
                "line_width": line_width,
                "alpha": alpha,
            }
        )

    def add_batches(
        self,
        batches: Iterator[pl.DataFrame],
        label: str | None = None,
        point_size: float = 5,
        point_color: str | None = None,
        show_line: bool = False,
        line_color: str | None = None,
        line_style: str = "-",
        line_width: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        for df in batches:
            self.add(
                df,
                label=label,
                point_size=point_size,
                point_color=point_color,
                show_line=show_line,
                line_color=line_color,
                line_style=line_style,
                line_width=line_width,
                alpha=alpha,
            )

    def show(self) -> None:
        fig, ax = plt.subplots()

        for layer in self._layers:
            data = layer["data"]

            longitude = data["longitude"]
            latitude = data["latitude"]

            ax.scatter(
                longitude,
                latitude,
                s=layer["point_size"],
                c=layer["point_color"],
                label=layer["label"],
                alpha=layer["alpha"],
            )

            if layer["show_line"]:
                ax.plot(
                    longitude,
                    latitude,
                    color=layer["line_color"],
                    linestyle=layer["line_style"],
                    linewidth=layer["line_width"],
                    alpha=layer["alpha"],
                )

        if self.title:
            ax.set_title(self.title)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

        if self.show_legend:
            ax.legend()

        def update_ticks(event=None):
            bbox = ax.get_window_extent()

            x_ticks = max(2, int(bbox.width / self.tick_spacing))
            y_ticks = max(2, int(bbox.height / self.tick_spacing))

            ax.xaxis.set_major_locator(
                LinearLocator(numticks=x_ticks)
            )
            ax.yaxis.set_major_locator(
                LinearLocator(numticks=y_ticks)
            )

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", update_ticks)

        fig.canvas.draw()
        update_ticks()

        plt.show()