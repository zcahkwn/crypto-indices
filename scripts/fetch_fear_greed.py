from fear_and_greed import FearAndGreedIndex
import pandas as pd
import numpy as np
from datetime import datetime
from nodiensenv.constants import DATA_DIR

fng = FearAndGreedIndex()

start_date = datetime(2015, 1, 1)

historical_data = fng.get_historical_data(start_date)

df = pd.DataFrame(historical_data)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df.rename(columns={"value": "fear_greed_index"}, inplace=True)
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.drop(columns=["time_until_update"])
df = df.sort_values("date", ascending=True)
df = df.reset_index(drop=True)
df["fear_greed_log_return"] = (
    df["fear_greed_index"].pct_change().apply(lambda x: np.log(1 + x))
)

df.to_csv(DATA_DIR / "fear_greed_index_new.csv", index=False)

# # Plotting the Fear and Greed Index
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.plot(df["date"], df["value"], label="Fear and Greed Index", color='blue')
# plt.title("Fear and Greed Index Over Time")
# plt.xlabel("Date")
# plt.ylabel("Index Value")
# plt.axhline(y=50, color='red', linestyle='--', label='Neutral Line (50)')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(FIGURE_DIR / "fear_greed_index_plot.png")
# plt.show()

# # plot the log returns
# plt.figure(figsize=(12, 6))
# plt.plot(df["date"], df["fear_greed_log_return"], label="Fear and Greed Log Return", color='orange')
# plt.title("Fear and Greed Log Return Over Time")
# plt.xlabel("Date")
# plt.ylabel("Fear and Greed Log Return")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(FIGURE_DIR / "fear_greed_log_return_plot.png")
