# Custom neural types for VariantWorks

from nemo.core.neural_types import ElementType, LabelsType, ChannelType

class VariantEncodingType(ChannelType):
    """Element type to represent a variant encoding.
    """

class VariantAlleleType(LabelsType):
    """Element type to represent variant allele.
    """

class VariantZygosityType(LabelsType):
    """Element type to represent variant type (no variant, heterozygous or homozygous.
    """
