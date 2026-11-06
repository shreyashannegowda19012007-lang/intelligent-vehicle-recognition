"""
Automated unit tests for class mapping and hierarchical attribute decoding (Make, Model, Year).
"""

from __future__ import annotations

from src.classification.metrics import ClassMappingDecoder


def test_class_mapping_decoder_loading():
    decoder = ClassMappingDecoder()
    assert len(decoder.class_info) == 196
    assert len(decoder.class_names) == 196
    assert len(decoder.makes) == 196
    assert len(decoder.models) == 196
    assert len(decoder.years) == 196


def test_class_mapping_all_196_classes_valid():
    decoder = ClassMappingDecoder()
    unique_makes = set()

    for cid in range(196):
        info = decoder.get_info(cid)
        assert info["class_id"] == cid
        assert isinstance(info["class_name"], str) and len(info["class_name"]) > 3
        assert isinstance(info["make"], str) and len(info["make"]) > 0
        assert isinstance(info["model"], str) and len(info["model"]) > 0
        assert isinstance(info["year"], int) and 1990 <= info["year"] <= 2025

        make, model, year = decoder.decode_attributes(cid)
        assert make == info["make"]
        assert model == info["model"]
        assert year == info["year"]

        unique_makes.add(make)

    # Stanford Cars official dataset contains exactly 49 distinct makes
    assert len(unique_makes) == 49


def test_class_mapping_unknown_class():
    decoder = ClassMappingDecoder()
    unknown_info = decoder.get_info(999)
    assert unknown_info["class_id"] == 999
    assert unknown_info["make"] == "Unknown"
    assert unknown_info["model"] == "Unknown"
    assert unknown_info["year"] == 0
