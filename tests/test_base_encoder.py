import pytest

from claragenomics.variantworks.base_encoder import base_enum_encoder 

def test_base_encoder():
    seq = "ATCGNTCGA"
    encoded_seq = [str(base_enum_encoder[n]) for n in seq]
    encoded_seq = "".join(encoded_seq)
    assert encoded_seq == "123452341"
