import pandas as pd
import csv

def remove_null_bytes(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    # remove NULL bytes
    cleaned = data.replace(b"\x00", b"")

    temp_path = file_path.replace(".csv", "_no_nulls.csv")

    with open(temp_path, "wb") as f:
        f.write(cleaned)

    return temp_path


def build_clean_dataset(input_path, output_path):

    # STEP 1: remove NULL bytes first
    clean_file = remove_null_bytes(input_path)

    rows = []

    # STEP 2: now safely read
    with open(clean_file, encoding="latin-1") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) >= 2:

                label = row[0].strip().lower()
                text = " ".join(row[1:]).strip()

                if "spam" in label:
                    label = "spam"
                elif "ham" in label:
                    label = "ham"
                else:
                    continue

                rows.append([label, text])

    df = pd.DataFrame(rows, columns=["label", "text"])
    df.dropna(inplace=True)

    df.to_csv(output_path, index=False)

    print("Clean dataset created:", output_path)
    print("Rows:", len(df))


if __name__ == "__main__":
    build_clean_dataset(
        "dataset/spam.csv",
        "dataset/clean_spam.csv"
    )
