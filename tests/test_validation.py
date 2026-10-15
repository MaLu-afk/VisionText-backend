import pytest

from app.services.validation import content_validator


@pytest.mark.parametrize("filename", [
    "badger_badger-0213.jpg",
    "armadillo_armadillo-0061.jpg",
    "african_elephant_african_elephant-0005.jpg",
])
def test_dataset_filenames_are_allowed(filename):
    is_valid, banned = content_validator.validate_image_metadata(filename)

    assert is_valid, banned


@pytest.mark.parametrize("filename, word", [
    ("dead_bird.jpg", "dead"),
    ("gun2.png", "gun"),
    ("my-weapon.jpeg", "weapon"),
])
def test_banned_words_in_filenames_are_detected(filename, word):
    is_valid, banned = content_validator.validate_image_metadata(filename)

    assert not is_valid
    assert word in banned


def test_description_matches_whole_words_only():
    assert content_validator.validate_image_metadata("perro.jpg", "un armadillo en el campo")[0]
    assert not content_validator.validate_image_metadata("perro.jpg", "un perro con un arma")[0]
