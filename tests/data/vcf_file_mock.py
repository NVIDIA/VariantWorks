import io
import random
import string
import bgzip
import subprocess


def mock_file_input(*args, **kwargs):
    return io.StringIO("""
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED      CALLED2
1	139098	.	CT	T	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50  .
1	139295	.	G	AC	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .
1	139642	.	T	A	50	.	DP=34;AF=0.0194118	GT:GQ	0/1:50  1/1:55
1	139738	.	G	C,A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .
1	139861	.	T	A	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50  .
1	139976	.	G	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .
1	139988	.	T	A	50	.	DP=34;AF=0.0194118	GT:GQ	0/1:50  .
1	139994	.	G	C	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .
1	140009	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .
1	140013	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .
1	140016	.	T	C	50	.	DP=34;AF=0.0194118	GT:GQ	1:50    .
1	240021	.	T	C	50	.	DP=34;AF=0.0294118	GT:GQ	1:50    .
1	240023	.	A	G	50	.	DP=35;AF=0.0285714	GT:GQ	1:50    .
1	240046	.	C	A	50	.	DP=34;AF=0.0294118	GT:GQ	1:50    .
1	240090	.	T	A	50	.	DP=22;AF=0.0454545	GT:GQ	1:50    .
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	1:50    .
1	240154	.	T	C	50	.	DP=13;AF=0.0769231	GT:GQ	1:50    .
""")


def mock_vcf_file_reader_input(dummy_file_path, tmp_folder_path):
    if str(dummy_file_path) == '/dummy/path1.gz':
        file_content = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED      CALLED2
1	139642	.	T	A	50	.	DP=34;AF=0.0194118	GT:GQ	0/1:50  1/1:55
1	139738	.	G	C,A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .:.
1	139861	.	T	A	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50  .:.
1	139976	.	G	A	50	.	DP=35;AF=0.0185714	GT:GQ	1/1:50  .:.
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	0/1:50  .:.
"""
    elif str(dummy_file_path) == '/dummy/path2.gz':
        file_content = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED      CALLED2
1	139098	.	CT	T	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50  .:.
1	139295	.	G	AC	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .:.
1	140009	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50  .:.
1	140016	.	T	C	50	.	DP=34;AF=0.0194118	GT:GQ	1/1:50  .:.
1	240021	.	T	C	50	.	DP=34;AF=0.0294118	GT:GQ	1/1:50  .:.
"""
    else:
        raise RuntimeError("Wrong filepath for reader")
    input_file_name = \
        tmp_folder_path + "/test_file_" + ''.join(random.choices(string.ascii_lowercase, k=4)) + '.vcf.gz'
    with open(input_file_name, "wb") as raw:
        with bgzip.BGZipWriter(raw) as fh:
            fh.write(file_content)
    tabix_cmd_response = subprocess.run(['tabix', '-p', 'vcf', raw.name])
    tabix_cmd_response.check_returncode()
    return raw.name, raw.name + ".tbi"
