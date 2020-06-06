"""Contains mocked file object inputs for tests."""

import io


def mock_file_input():
    """Return a string stream of an unfiltered vcf file content."""
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
    """Returns a string stream of a vcf file content which is supposed to raise a RuntimeError.

    More than one called sample
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


def mock_vcf_file_reader_input(dummy_file_path):
    """Return string stream of small filtered vcf content."""
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
