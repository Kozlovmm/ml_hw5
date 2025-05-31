from sklearn.metrics import accuracy_score, precision_score, recall_score

def test_metric_calculations():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 1, 1, 2, 1]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')

    assert 0 <= acc <= 1, "Неверное значение accuracy"
    assert 0 <= prec <= 1, "Неверное значение precision"
    assert 0 <= rec <= 1, "Неверное значение recall"
