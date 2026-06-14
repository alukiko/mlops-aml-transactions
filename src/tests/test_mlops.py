import pandas as pd

from aml_monitoring.data import normalize_transactions
from aml_monitoring.drift import categorical_psi, ks_statistic, psi, run_drift
from aml_monitoring.features import FEATURE_COLS, build_preprocessor, engineer_features, to_feature_matrix
from aml_monitoring.retraining import find_best_threshold, load_stratified_training_sample


def sample_frame():
    return pd.DataFrame(
        [
            {
                "Timestamp": "2022/09/01 00:20",
                "From Bank": "010",
                "From Account": "A1",
                "To Bank": "010",
                "To Account": "A2",
                "Amount Received": 100.0,
                "Receiving Currency": "US Dollar",
                "Amount Paid": 100.0,
                "Payment Currency": "US Dollar",
                "Payment Format": "Wire",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2022/09/01 23:20",
                "From Bank": "011",
                "From Account": "A1",
                "To Bank": "012",
                "To Account": "A3",
                "Amount Received": 250.0,
                "Receiving Currency": "Euro",
                "Amount Paid": 200.0,
                "Payment Currency": "US Dollar",
                "Payment Format": "ACH",
                "Is Laundering": 1,
            },
        ]
    )


def test_engineer_features_creates_model_columns():
    engineered = engineer_features(sample_frame())
    missing = [column for column in FEATURE_COLS if column not in engineered.columns]
    assert missing == []


def test_saved_preprocessor_keeps_category_encoding_stable():
    train = sample_frame()
    preprocessor = build_preprocessor(train)
    single = train.iloc[[0]]
    with_extra_category = pd.concat(
        [
            single,
            pd.DataFrame(
                [
                    {
                        **single.iloc[0].to_dict(),
                        "Payment Format": "Never Seen Format",
                        "From Bank": "999",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    one_row_features = to_feature_matrix(single, preprocessor=preprocessor).iloc[0]
    batch_features = to_feature_matrix(with_extra_category, preprocessor=preprocessor).iloc[0]
    assert one_row_features["Payment Format"] == batch_features["Payment Format"]
    assert one_row_features["From Bank"] == batch_features["From Bank"]


def test_unknown_categories_map_to_unk():
    preprocessor = build_preprocessor(sample_frame())
    unknown = sample_frame().iloc[[0]].copy()
    unknown["Payment Format"] = "Unknown Format"
    features = to_feature_matrix(unknown, preprocessor=preprocessor)
    assert features.iloc[0]["Payment Format"] == "Unknown Format"


def test_normalize_transactions_handles_duplicate_account_columns():
    raw = sample_frame().rename(columns={"From Account": "Account", "To Account": "Account.1"})
    normalized = normalize_transactions(raw)
    assert "From Account" in normalized.columns
    assert "To Account" in normalized.columns


def test_drift_scores_are_low_for_identical_samples():
    expected = pd.Series([1, 2, 3, 4, 5])
    actual = pd.Series([1, 2, 3, 4, 5])
    assert psi(expected, actual) < 0.01
    assert ks_statistic(expected, actual) == 0
    assert categorical_psi(pd.Series(["a", "b", "a"]), pd.Series(["a", "b", "a"])) < 0.01


def test_drift_scores_increase_for_shifted_samples():
    expected = pd.Series([1, 2, 3, 4, 5])
    actual = pd.Series([100, 110, 120, 130, 140])
    assert psi(expected, actual) > 0.1
    assert ks_statistic(expected, actual) > 0.5


def test_drift_returns_not_enough_data_for_small_recent_batch(monkeypatch, tmp_path):
    monkeypatch.setattr("aml_monitoring.drift.get_reference", lambda: sample_frame())
    monkeypatch.setattr("aml_monitoring.drift.write_report", lambda result: (tmp_path / "drift.json", tmp_path / "drift.html"))
    result = run_drift(batch=[sample_frame().iloc[0].to_dict()], min_rows=2, source="recent_predictions")
    assert result["status"] == "not_enough_data"
    assert result["source"] == "recent_predictions"
    assert result["rows"]["current"] == 1


def test_find_best_threshold_is_not_fixed_to_099():
    y_true = pd.Series([0, 0, 1, 1]).to_numpy()
    probabilities = pd.Series([0.05, 0.2, 0.45, 0.8]).to_numpy()
    threshold, metrics = find_best_threshold(y_true, probabilities, min_precision=0.1)
    assert threshold < 0.99
    assert metrics["recall"] > 0
    assert metrics["f2"] > 0


def test_stratified_training_sample_contains_positives(tmp_path):
    rows = sample_frame()
    data = pd.concat([rows, rows.assign(**{"Is Laundering": 0})], ignore_index=True)
    path = tmp_path / "sample.csv"
    data.to_csv(path, index=False)
    sampled = load_stratified_training_sample([path], max_rows=4, min_positives=1, chunksize=2)
    assert sampled["Is Laundering"].sum() >= 1

