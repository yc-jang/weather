import os
import pandas as pd


def load_and_filter_data(file_path, encoding="cp949"):
    columns_to_use = [
        "일시",
        "평균기온(°C)",
        "최저기온(°C)",
        "최고기온(°C)",
        "일강수량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
    ]

    data = pd.read_csv(file_path, encoding=encoding, usecols=columns_to_use)

    data["일강수량(mm)"].fillna(0, inplace=True)

    return data


target_files = [
    file
    for file in os.listdir(".")
    if file.startswith("SURFACE_ASOS_266_DAY_") or file.startswith("OBS_ASOS_DD_")
]

all_data = pd.concat(
    [load_and_filter_data(file) for file in target_files],
    ignore_index=True,
)

all_data.sort_values(by="일시", inplace=True)
