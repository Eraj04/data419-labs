import pandas as pd

test = pd.read_csv("test.csv")

submission = pd.DataFrame({
    "id": test["id"],
    "poisonous": 0.5,
})

submission.to_csv("submission.csv", index=False)
