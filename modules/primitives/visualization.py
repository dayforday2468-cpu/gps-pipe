from collections.abc import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import polars as pl

from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, LinearLocator


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
        self._road_network = None

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

    def add_road_network(
        self,
        graph: nx.MultiDiGraph,
    ) -> None:
        self._road_network = graph

    def _draw_road_network(self, ax) -> None:
        if self._road_network is None:
            return

        ox.plot_graph(
            self._road_network,
            ax=ax,
            show=False,
            close=False,
            node_size=0,
            edge_linewidth=0.5,
            edge_alpha=0.5,
        )

    def _setup_axes(self, fig, ax) -> None:
        if self.title:
            ax.set_title(self.title)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

        def update_ticks(event=None):
            bbox = ax.get_window_extent()

            x_ticks = max(2, int(bbox.width / self.tick_spacing))
            y_ticks = max(2, int(bbox.height / self.tick_spacing))

            ax.xaxis.set_major_locator(LinearLocator(numticks=x_ticks))
            ax.yaxis.set_major_locator(LinearLocator(numticks=y_ticks))

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", update_ticks)

        fig.canvas.draw()
        update_ticks()

    def show(self) -> None:
        fig, ax = plt.subplots()

        self._draw_road_network(ax)

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
                zorder=3,
            )

            if layer["show_line"]:
                ax.plot(
                    longitude,
                    latitude,
                    color=layer["line_color"],
                    linestyle=layer["line_style"],
                    linewidth=layer["line_width"],
                    alpha=layer["alpha"],
                    zorder=2,
                )

        self._setup_axes(fig, ax)

        if self.show_legend:
            ax.legend()

        plt.show()

    def animate(
        self,
        interval: int = 100,
        repeat: bool = False,
        mode: str = "index",
    ) -> None:
        if mode not in {"index", "time"}:
            raise ValueError("mode must be 'index' or 'time'")

        fig, ax = plt.subplots()

        self._draw_road_network(ax)

        if self._road_network is None:
            all_longitudes = []
            all_latitudes = []

            for layer in self._layers:
                data = layer["data"]

                all_longitudes.extend(data["longitude"].to_list())
                all_latitudes.extend(data["latitude"].to_list())

            if all_longitudes and all_latitudes:
                ax.set_xlim(min(all_longitudes), max(all_longitudes))
                ax.set_ylim(min(all_latitudes), max(all_latitudes))

        artists = []

        for layer in self._layers:
            scatter = ax.scatter(
                [],
                [],
                s=layer["point_size"],
                c=layer["point_color"],
                label=layer["label"],
                alpha=layer["alpha"],
                zorder=3,
            )

            line = None

            if layer["show_line"]:
                (line,) = ax.plot(
                    [],
                    [],
                    color=layer["line_color"],
                    linestyle=layer["line_style"],
                    linewidth=layer["line_width"],
                    alpha=layer["alpha"],
                    zorder=2,
                )

            artists.append(
                {
                    "scatter": scatter,
                    "line": line,
                }
            )

        self._setup_axes(fig, ax)

        if self.show_legend:
            ax.legend()

        if mode == "index":
            frames = range(max(layer["data"].height for layer in self._layers))

        else:
            timestamps = []

            for layer in self._layers:
                data = layer["data"]

                if "timestamp" in data.columns:
                    timestamps.extend(data["timestamp"].to_list())

            frames = sorted(set(timestamps))

        def update(frame):
            updated_artists = []

            for layer, artist in zip(self._layers, artists):
                data = layer["data"]

                if mode == "index":
                    end = min(frame + 1, data.height)
                    visible = data[:end]

                else:
                    if "timestamp" not in data.columns:
                        visible = data
                    else:
                        visible = data.filter(pl.col("timestamp") <= frame)

                longitude = visible["longitude"]
                latitude = visible["latitude"]

                coordinates = visible.select(
                    "longitude",
                    "latitude",
                ).to_numpy()

                artist["scatter"].set_offsets(coordinates)
                updated_artists.append(artist["scatter"])

                if artist["line"] is not None:
                    artist["line"].set_data(
                        longitude.to_numpy(),
                        latitude.to_numpy(),
                    )
                    updated_artists.append(artist["line"])

            return updated_artists

        animation = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            repeat=repeat,
            blit=False,
        )

        self._animation = animation

        plt.show()
