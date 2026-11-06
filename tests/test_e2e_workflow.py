"""
Automated End-to-End Workflow Integration Test for Prompt 10.
Executes the full pipeline workflow and asserts that all 10 core verification
requirements pass without error.
"""

from __future__ import annotations

from scripts.verify_e2e import verify_e2e_workflow


def test_complete_end_to_end_workflow():
    """Run full 10-step end-to-end vehicle recognition workflow."""
    audit_summary = verify_e2e_workflow()
    assert audit_summary["pipeline_init_ok"] is True
    assert audit_summary["image_decoding_ok"] is True
    assert audit_summary["bounding_boxes_generated"] is True
    assert audit_summary["classification_attributes_ok"] is True
    assert audit_summary["top5_predictions_ok"] is True
    assert audit_summary["model_comparison_ok"] is True
    assert audit_summary["timing_profiling_ok"] is True
    assert audit_summary["invalid_bytes_handled"] is True
    assert audit_summary["blank_image_handled"] is True
    assert audit_summary["fastapi_endpoints_ok"] is True
