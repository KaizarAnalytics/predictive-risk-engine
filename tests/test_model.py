def test_model_outputs_probabilities():
    clf = joblib.load("src/models/model.pkl")
    df = pd.DataFrame({...})
    proba = clf.predict_proba(df)
    assert (proba >= 0).all() and (proba <= 1).all()
