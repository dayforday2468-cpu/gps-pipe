from datetime import datetime

from modules.parameter_tuning import (
    estimate_jump_threshold,
    estimate_same_place_threshold,
)
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer
from modules.sudden_position_jump import remove_sudden_position_jumps

if __name__ == "__main__":
    batches = initialize_pipeline()

    # 하루치 raw position 필터링
    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    jump_thres = estimate_jump_threshold(raw_filtered)

    same_place_thres = estimate_same_place_threshold(
        raw_filtered,
        jump_thres=jump_thres,
    )

    print(f"Jump Threshold: {jump_thres:.2f} m")
    print(f"Same Place Threshold: {same_place_thres:.2f} m")

    # sudden position jump 제거
    cleaned_positions = remove_sudden_position_jumps(
        raw_filtered,
        jump_thres=jump_thres,
        same_place_thres=same_place_thres,
    )

    # 제거된 point 추출
    removed_positions = raw_filtered.join(
        cleaned_positions.select("timestamp"),
        on="timestamp",
        how="anti",
    )

    print(f"before:  {raw_filtered.height}")
    print(f"after:   {cleaned_positions.height}")
    print(f"removed: {removed_positions.height}")

    # clean 결과 시각화
    visualizer = GPSVisualizer(
        title=f"Sudden Position Jump Removal - {time_range}",
        show_legend=True,
    )

    visualizer.add(
        raw_filtered,
        label="Raw GPS",
        point_size=2,
        point_color="gray",
        show_line=True,
        line_color="gray",
        line_width=0.5,
        alpha=0.15,
    )

    visualizer.add(
        cleaned_positions,
        label="Cleaned GPS",
        point_size=7,
        point_color="blue",
        show_line=True,
        line_color="blue",
        line_width=1.2,
        alpha=0.8,
    )

    visualizer.add(
        removed_positions,
        label="Removed Jump",
        point_size=50,
        point_color="red",
        show_line=False,
        alpha=1.0,
    )

    visualizer.show()
    