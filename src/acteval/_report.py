from pandas import DataFrame


def print_markdown(data: DataFrame):
    print(data.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))
