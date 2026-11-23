import pytest

from scripts.download_zenodo import parse_subjects, select_files

FILES = {name: {"size": 1, "md5": "x"} for name in [
    "subject_info.xlsx", "LICENSE.txt",
    "001_ECG.mat", "001_PPG.mat", "002_ECG.mat", "002_PPG.mat", "024_ECG.mat",
]}


def test_parse_subjects_ids_and_ranges():
    assert parse_subjects(["6", "006", "001-003"]) == ["001", "002", "003", "006"]


def test_parse_subjects_rejects_reversed_range():
    with pytest.raises(ValueError):
        parse_subjects(["010-001"])


def test_select_files_ecg_and_ppg_plus_shared_files():
    names, warnings = select_files(["001", "002"], FILES)
    assert names == ["subject_info.xlsx", "LICENSE.txt",
                     "001_ECG.mat", "001_PPG.mat", "002_ECG.mat", "002_PPG.mat"]
    assert warnings == []


def test_select_files_ecg_only_subject_warns():
    names, warnings = select_files(["024"], FILES)
    assert "024_ECG.mat" in names and "024_PPG.mat" not in names
    assert len(warnings) == 1


def test_select_files_unknown_subject():
    with pytest.raises(ValueError):
        select_files(["099"], FILES)
