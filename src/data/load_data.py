import pandas as pd

def load_data(path):
    """
    Load dataset from csv file
    """

    data = pd.read_csv(path)

    return data


if __name__ == "__main__":

    data = load_data("data/raw/student_data.csv")

    print(data.head())
