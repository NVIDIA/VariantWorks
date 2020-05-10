# Custom neural types for VariantWorks

from nemo.core.neural_types import ElementType, LabelsType

class VariantPositionType(ElementType):
    """Element type to represent position of variant ing enome.
    """

class VariantAlleleType(LabelsType):
    """Element type to represent variant allele.
    """

class VariantType(LabelsType):
    """Element type to represent variant type (no variant, heterozygous or homozygous.
    """
