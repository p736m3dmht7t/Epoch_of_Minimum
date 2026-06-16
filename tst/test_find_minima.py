import io
import sys
import types

import pandas as pd
import pytest

import find_minima as fm
from kwee_and_van_woerden import kwee_van_woerden


def _sequence_df(times, mags):
    return pd.DataFrame(
        {
            "BJD_TDB": times,
            "Source_AMag_T1": mags,
            "Source_AMag_Err_T1": [0.01] * len(times),
        }
    )


def test_trim_and_validate_sequence_returns_centered_odd_window():
    sequence_df = _sequence_df(
        [10, 11, 12, 13, 14, 15, 16],
        [1.0, 3.0, 5.0, 9.0, 5.0, 3.0, 1.0],
    )

    result = fm._trim_and_validate_sequence(
        sequence_df,
        sequence_length=5,
        monotonic_length=3,
    )

    assert result is not None
    assert result["BJD_TDB"].tolist() == [11, 12, 13, 14, 15]
    assert result["Source_AMag_T1"].tolist() == [3.0, 5.0, 9.0, 5.0, 3.0]


def test_trim_and_validate_sequence_rejects_edge_center_or_bad_monotonicity():
    edge_center_df = _sequence_df(
        [1, 2, 3, 4, 5],
        [9.0, 5.0, 3.0, 2.0, 1.0],
    )
    non_monotonic_df = _sequence_df(
        [10, 11, 12, 13, 14, 15, 16],
        [1.0, 5.0, 4.0, 9.0, 5.0, 3.0, 1.0],
    )

    edge_result = fm._trim_and_validate_sequence(
        edge_center_df,
        sequence_length=5,
        monotonic_length=3,
    )
    non_monotonic_result = fm._trim_and_validate_sequence(
        non_monotonic_df,
        sequence_length=5,
        monotonic_length=3,
    )

    assert edge_result is None
    assert non_monotonic_result is None


def test_trim_and_validate_sequence_validates_parameters():
    sequence_df = _sequence_df([1, 2, 3], [1.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="positive odd integer"):
        fm._trim_and_validate_sequence(
            sequence_df,
            sequence_length=4,
            monotonic_length=1,
        )

    with pytest.raises(ValueError, match="must be non-negative"):
        fm._trim_and_validate_sequence(
            sequence_df,
            sequence_length=3,
            monotonic_length=-1,
        )

    with pytest.raises(ValueError, match="must not exceed"):
        fm._trim_and_validate_sequence(
            sequence_df,
            sequence_length=3,
            monotonic_length=3,
        )


def test_analyze_sequence_builds_centered_output(monkeypatch):
    sequence_df = _sequence_df(
        [11, 12, 13, 14, 15],
        [3.0, 5.0, 9.0, 5.0, 3.0],
    )

    fake_module = types.SimpleNamespace(
        kwee_van_woerden=lambda times, mags: (123.456, 0.25)
    )
    monkeypatch.setitem(sys.modules, "kwee_and_van_woerden", fake_module)

    result = fm._analyze_sequence(
        sequence_df,
        sequence_length=5,
        monotonic_length=3,
        mag_error_column="Source_AMag_Err_T1",
    )

    assert result is not None
    output_df = result["output_df"]
    assert output_df.columns.tolist() == [
        "BJD_TDB",
        "Source_AMag_T1",
        "TOM",
        "TOM_uncertainty",
    ]
    assert output_df["TOM"].notna().sum() == 1
    assert output_df.loc[2, "TOM"] == pytest.approx(123.456)
    assert output_df.loc[2, "TOM_uncertainty"] == pytest.approx(0.25)
    assert result["integer_date"] == 123


def test_analyze_sequence_raises_when_solver_fails(monkeypatch):
    sequence_df = _sequence_df(
        [11, 12, 13, 14, 15],
        [3.0, 5.0, 9.0, 5.0, 3.0],
    )

    def _raise_value_error(times, mags):
        raise ValueError("synthetic failure")

    fake_module = types.SimpleNamespace(kwee_van_woerden=_raise_value_error)
    monkeypatch.setitem(sys.modules, "kwee_and_van_woerden", fake_module)

    with pytest.raises(ValueError, match="synthetic failure"):
        fm._analyze_sequence(
            sequence_df,
            sequence_length=5,
            monotonic_length=3,
            mag_error_column="Source_AMag_Err_T1",
        )


def test_write_minimum_result_writes_csv_and_pdf(monkeypatch, tmp_path):
    minimum_result = {
        "candidate_df": _sequence_df([11, 12, 13], [3.0, 9.0, 3.0]),
        "output_df": pd.DataFrame(
            {
                "BJD_TDB": [11.0, 12.0, 13.0],
                "Source_AMag_T1": [3.0, 9.0, 3.0],
                "TOM": [float("nan"), 12.25, float("nan")],
                "TOM_uncertainty": [float("nan"), 0.1, float("nan")],
            }
        ),
        "tom": 12.25,
        "tom_uncertainty": 0.1,
        "integer_date": 12,
        "mag_error_column": "Source_AMag_Err_T1",
        "star_name": "TZ Boo",
        "filter_name": "B",
    }
    render_calls = []

    def fake_render(result, pdf_output_file):
        render_calls.append((result["integer_date"], pdf_output_file))
        with open(pdf_output_file, "wb") as handle:
            handle.write(b"%PDF-1.4\n")

    monkeypatch.setattr(fm, "_render_minimum_visualization", fake_render)

    csv_buffer = io.StringIO()
    output_file = tmp_path / "result.csv"
    wrote = fm._write_minimum_result(
        minimum_result,
        is_first_valid_minimum=True,
        csvfile=csv_buffer,
        output_file=str(output_file),
    )

    assert wrote is True
    assert "TOM_uncertainty" in csv_buffer.getvalue()
    assert render_calls == [(12, str(tmp_path / "result_12.pdf"))]
    assert (tmp_path / "result_12.pdf").exists()


def test_find_variable_star_minima_uses_global_median_and_default_monotonic_length(
    monkeypatch,
    tmp_path,
    capsys,
):
    input_file = tmp_path / "sample.csv"
    output_file = tmp_path / "nested" / "result.csv"
    input_df = pd.DataFrame(
        {
            "BJD_TDB": [
                1.04,
                0.03,
                1.00,
                0.01,
                0.00,
                1.03,
                0.02,
                1.01,
                1.02,
                0.04,
            ],
            "Source_AMag_T1": [
                0.0,
                5.0,
                0.0,
                3.0,
                0.0,
                5.0,
                7.0,
                3.0,
                7.0,
                0.0,
            ],
            "Source_AMag_Err_T1": [0.02] * 10,
        }
    )
    input_df.to_csv(input_file, index=False)

    calls = []

    def fake_analyze_sequence(
        sequence_df,
        *,
        sequence_length,
        monotonic_length,
        mag_error_column,
    ):
        calls.append(
            {
                "times": sequence_df["BJD_TDB"].tolist(),
                "mags": sequence_df["Source_AMag_T1"].tolist(),
                "sequence_length": sequence_length,
                "monotonic_length": monotonic_length,
                "mag_error_column": mag_error_column,
            }
        )
        return {
            "candidate_df": sequence_df.copy(),
            "output_df": pd.DataFrame(
                {
                    "BJD_TDB": sequence_df["BJD_TDB"],
                    "Source_AMag_T1": sequence_df["Source_AMag_T1"],
                    "TOM": [float("nan")] * len(sequence_df),
                    "TOM_uncertainty": [float("nan")] * len(sequence_df),
                }
            ),
            "tom": 2460828.7,
            "tom_uncertainty": 0.1,
            "integer_date": 2460828,
            "mag_error_column": mag_error_column,
        }

    def fake_write_minimum_result(
        minimum_result,
        *,
        is_first_valid_minimum,
        csvfile,
        output_file,
    ):
        csvfile.write("processed\n")
        calls[-1]["is_first_valid_minimum"] = is_first_valid_minimum
        calls[-1]["output_file"] = output_file
        return True

    monkeypatch.setattr(fm, "_analyze_sequence", fake_analyze_sequence)
    monkeypatch.setattr(fm, "_write_minimum_result", fake_write_minimum_result)

    fm.find_variable_star_minima(
        str(input_file),
        str(output_file),
        sequence_length=5,
        monotonic_length=None,
        night_mode="land",
        night_gap_hours=8.0,
        mag_error_column="Source_AMag_Err_T1",
    )

    captured = capsys.readouterr()
    assert len(calls) == 2
    assert calls[0]["times"] == [0.02, 0.03]
    assert calls[0]["mags"] == [7.0, 5.0]
    assert calls[1]["times"] == [1.02, 1.03]
    assert calls[1]["mags"] == [7.0, 5.0]
    assert all(call["sequence_length"] == 5 for call in calls)
    assert all(call["monotonic_length"] == 3 for call in calls)
    assert all(call["mag_error_column"] == "Source_AMag_Err_T1" for call in calls)
    assert calls[0]["is_first_valid_minimum"] is True
    assert calls[1]["is_first_valid_minimum"] is False
    assert output_file.read_text() == "processed\nprocessed\n"
    assert "Successfully processed the file" in captured.out


def test_find_variable_star_minima_raises_for_invalid_configuration(tmp_path):
    input_file = tmp_path / "sample.csv"
    output_file = tmp_path / "result.csv"
    _sequence_df([1, 2, 3], [1.0, 2.0, 1.0]).to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="positive odd integer"):
        fm.find_variable_star_minima(
            str(input_file),
            str(output_file),
            sequence_length=4,
            night_mode="space",
            mag_error_column="Source_AMag_Err_T1",
        )


def test_find_variable_star_minima_raises_for_invalid_input_data(tmp_path):
    input_file = tmp_path / "sample.csv"
    output_file = tmp_path / "result.csv"
    pd.DataFrame(
        {
            "BJD_TDB": [1.0, "bad", 3.0],
            "Source_AMag_T1": [1.0, 2.0, 1.0],
            "Source_AMag_Err_T1": [0.01, 0.01, 0.01],
        }
    ).to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        fm.find_variable_star_minima(
            str(input_file),
            str(output_file),
            sequence_length=3,
            night_mode="space",
            mag_error_column="Source_AMag_Err_T1",
        )


def test_build_argument_parser_parses_main_options():
    parser = fm._build_argument_parser()

    args = parser.parse_args(
        [
            "TZ_Boo_Measurement_Table_B.csv",
            "land",
            "7",
            "--output",
            "custom.csv",
            "--monotonic-length",
            "2",
            "--night-gap-hours",
            "10",
            "--mag-error-column",
            "Custom_Err",
            "--star-name",
            "Custom Star",
            "--filter-name",
            "V",
        ]
    )

    assert args.input_file == "TZ_Boo_Measurement_Table_B.csv"
    assert args.night_mode == "land"
    assert args.sequence_length == 7
    assert args.output_file == "custom.csv"
    assert args.monotonic_length == 2
    assert args.night_gap_hours == 10.0
    assert args.mag_error_column == "Custom_Err"
    assert args.star_name == "Custom Star"
    assert args.filter_name == "V"


def test_path_helpers_use_input_and_output_directories():
    assert fm._resolve_input_path("TZ_Boo_Measurement_Table_B.csv") == (
        "input/TZ_Boo_Measurement_Table_B.csv"
    )
    assert fm._resolve_input_path("input/already.csv") == "input/already.csv"
    assert fm._default_output_path("TZ_Boo_Measurement_Table_B.csv") == (
        "output/TZ_Boo_Measurement_Table_B_minima.csv"
    )
    assert fm._pdf_output_path("output/TZ_Boo_Measurement_Table_B_minima.csv", 2460828) == (
        "output/TZ_Boo_Measurement_Table_B_minima_2460828.pdf"
    )


def test_plot_label_helpers_infer_defaults_and_allow_overrides():
    assert fm._infer_plot_labels("input/TZ_Boo_Measurement_Table_B.csv") == (
        "TZ Boo",
        "B",
    )
    assert fm._infer_plot_labels(
        "input/TZ_Boo_Measurement_Table_B.csv",
        star_name="Custom Star",
        filter_name="V",
    ) == ("Custom Star", "V")
    assert fm._x_axis_label(2460828, 2460828.69962974, 0.00026342) == (
        "BJD_TDB - 2460828 (T0 = 2460828.699630, SE = 0.000263)"
    )
    assert fm._y_axis_label("TZ Boo", "B") == "TZ Boo B Magnitude"


def test_kwee_van_woerden_rejects_even_length_sequences():
    with pytest.raises(ValueError, match="odd number of data points"):
        kwee_van_woerden(
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 2.0, 1.0],
        )


def test_render_minimum_visualization_creates_pdf(tmp_path):
    minimum_result = {
        "candidate_df": _sequence_df(
            [2460828.67, 2460828.68, 2460828.69, 2460828.70, 2460828.71],
            [11.42, 11.50, 11.60, 11.50, 11.42],
        ),
        "output_df": None,
        "tom": 2460828.6995,
        "tom_uncertainty": 0.0002,
        "integer_date": 2460828,
        "mag_error_column": "Source_AMag_Err_T1",
        "star_name": "TZ Boo",
        "filter_name": "B",
    }
    pdf_path = tmp_path / "minimum.pdf"

    fm._render_minimum_visualization(minimum_result, str(pdf_path))

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
