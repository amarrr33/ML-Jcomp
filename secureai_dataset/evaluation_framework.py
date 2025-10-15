"""Small evaluation helpers for dataset validation and metadata checks."""
import json


def validate_metadata(path):
    try:
        with open(path) as fh:
            j = json.load(fh)
        return True, None
    except Exception as e:
        return False, str(e)

