import pandas as pd

df = pd.read_csv("data/eyepacs/trainLabels.csv")

# take balanced small sample
subset = (
    df.groupby("level")
      .apply(lambda x: x.sample(200, random_state=42))
      .reset_index(drop=True)
)

subset.to_csv("data/eyepacs/subset_labels.csv", index=False)

print("Subset created:", len(subset))

