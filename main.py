from modules.dataload import load_raw_positions


if __name__=="__main__":
    raw_positions_df = load_raw_positions("data/timeline.json")

    for df in raw_positions_df:
        print(df)
