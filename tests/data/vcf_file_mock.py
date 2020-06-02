from enum import Enum
import io
import vcf


from variantworks.io.vcfio import VCFReader


def mock_file_input():
    return io.StringIO("""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED
1	139098	.	CT	T	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50
1	139295	.	G	AC	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	139738	.	G	C,A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	139861	.	T	A	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50
1	139976	.	G	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	139988	.	T	A	50	.	DP=34;AF=0.0194118	GT:GQ	0/1:50
1	139994	.	G	C	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	140009	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	140013	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	140016	.	T	C	50	.	DP=34;AF=0.0194118	GT:GQ	1:50
1	240021	.	T	C	50	.	DP=34;AF=0.0294118	GT:GQ	1:50
1	240023	.	A	G	50	.	DP=35;AF=0.0285714	GT:GQ	1:50
1	240046	.	C	A	50	.	DP=34;AF=0.0294118	GT:GQ	1:50
1	240090	.	T	A	50	.	DP=22;AF=0.0454545	GT:GQ	1:50
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	1:50
1	240154	.	T	C	50	.	DP=13;AF=0.0769231	GT:GQ	1:50
""")


def mock_invalid_file_input():
    """Returns a string stream of a vcf file content which is supposed to raise a RuntimeError (more than one called
    sample)
    """
    return io.StringIO("""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED  CALLED2
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	1:50    1/1:50
1	240154	.	T	C	50	.	DP=13;AF=0.0769231	GT:GQ	1:50    0/1:50
""")


def mock_small_filtered_file_input():
    return io.StringIO("""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED
1	139861	.	T	A	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50
1	139976	.	G	A	50	.	DP=35;AF=0.0185714	GT:GQ	1/1:50
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	0/1:50
""")


class MockPyVCFReader:

    class ContentType(Enum):
        """VCF file content type for mocking
        """
        UNFILTERED = 0
        INVALID = 1
        SMALL_FILTERED = 2

    original_pyvcf_reader_init_function = vcf.Reader.__init__

    @staticmethod
    def _new_vcf_reader_init(self, *args, **kargs):
        MockPyVCFReader.original_pyvcf_reader_init_function(
            self, mock_file_input())

    @staticmethod
    def _new_invalid_vcf_reader_init(self, *args, **kargs):
        MockPyVCFReader.original_pyvcf_reader_init_function(
            self, mock_invalid_file_input())

    @staticmethod
    def _new_small_vcf_reader_init(self, *args, **kargs):
        MockPyVCFReader.original_pyvcf_reader_init_function(
            self, mock_small_filtered_file_input())

    _content_type_to_mocked_init_method = {
        ContentType.UNFILTERED:         _new_vcf_reader_init.__func__,
        ContentType.INVALID:            _new_invalid_vcf_reader_init.__func__,
        ContentType.SMALL_FILTERED:     _new_small_vcf_reader_init.__func__,
    }

    @staticmethod
    def get_reader(mp, vcf_bam_list, content_type):
        with mp.context() as m:
            # Mock vcf.Reader.__init__() return value
            m.setattr(vcf.Reader, "__init__", MockPyVCFReader._content_type_to_mocked_init_method[content_type])
            vcf_loader = VCFReader(vcf_bam_list)
        return vcf_loader

    @staticmethod
    def set_mocked_reader_content_and_call_function(mp, content_type, function_to_call):
        with mp.context() as m:
            # Mock vcf.Reader.__init__() return value
            m.setattr(vcf.Reader, "__init__", MockPyVCFReader._content_type_to_mocked_init_method[content_type])
            function_to_call()
