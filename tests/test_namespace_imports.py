from calculation.SROS import SRO as LegacySRO
from calculation.generate_random import create_original_structure as legacy_create_original_structure
from sros.calculation.SROS import SRO
from sros.calculation.generate_random import create_original_structure


def test_sros_namespace_reexports_legacy_calculation_modules():
    assert SRO is LegacySRO
    assert create_original_structure is legacy_create_original_structure
