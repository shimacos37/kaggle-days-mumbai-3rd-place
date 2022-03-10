import ast
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def clean_text(texts):
    res = []
    for text in tqdm(texts):
        if text[0] == "[":
            if "'" in text:
                text = text.replace("'", "")
            if text[1] != "'":
                text = "['" + text[1:-1] + "']"
            res.append(" ".join(ast.literal_eval(text)))
        else:
            res.append(text)
    return res


def main():
    kf = KFold(n_splits=4, shuffle=True, random_state=777)
    df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    for n_fold, (_, valid_idx) in enumerate(kf.split(df)):
        df.loc[valid_idx, "fold"] = n_fold
    for text_col in ["s1", "s2"]:
        texts = df[text_col].to_list()
        texts = clean_text(texts)
        df[text_col] = texts
        texts = test_df[text_col].to_list()
        texts = clean_text(texts)
        test_df[text_col] = texts
    df.to_csv("../input/train_fold.csv", index=False)
    test_df.to_csv("../input/test_clean.csv", index=False)


if __name__ == "__main__":
    main()
