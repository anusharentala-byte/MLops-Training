def build_features(data):

    """
    Split dataset into features and target
    """

    X = data[["hours_studied"]]

    y = data["score"]

    return X, y