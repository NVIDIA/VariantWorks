from variantworks.io import vcfio
from variantworks import pon
import variantworks.merge_filter as mf

import cudf as cudf
import pandas as pd
import cyvcf2
from collections import defaultdict

import time
import sys
import argparse


def add_variant(chunk, variant, tag_columns=[]):
    chunk["chrom"].append(variant.CHROM)
    chunk["start_pos"].append(variant.POS)
    variant_len = abs(len(variant.REF) - len(variant.ALT))
    chunk["end_pos"].append(variant.POS + variant_len)
    chunk["ref"].append(variant.REF)
    chunk["alt"].append(variant.ALT[0])
    for i in tag_columns:
        chunk[i].append(variant.INFO.get(i))


def add_multiallelic(chunk, variant, tag_columns):
    num_alleles = len(variant.ALT)
    for i in range(0, num_alleles):
        chunk["chrom"].append(variant.CHROM)
        chunk["start_pos"].append(variant.POS)
        variant_len = abs(len(variant.REF) - len(variant.ALT[i]))
        chunk["end_pos"].append(variant.POS + variant_len)
        chunk["ref"].append(variant.REF)
        chunk["alt"].append(variant.ALT[i])
        for tag in tag_columns:
            val = variant.INFO.get(tag)
            if val is None:
                chunk[tag].append(None)
            else:
                chunk[tag].append(variant.INFO.get(tag)[i])


def append_chunk(df, chunk):
    chunk_df = pd.DataFrame.from_dict(chunk)
    df = df.append(chunk_df, ignore_index=True, sort=False)
    chunk.clear()
    return df


def parse_region(vcf, tag_columns=[], parse_infos=False, prefix=None, chunksize=10000, use_cudf=True):
    variants_read = 0
    chunk = defaultdict(list)
    vcf_df = pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "ref", "alt"])
    for variant in vcf:
        if len(variant.ALT) == 1:
            add_variant(chunk, variant, tag_columns)
        else:
            add_multiallelic(chunk, variant, tag_columns)
        if len(chunk["chrom"]) == chunksize:
            vcf_df = append_chunk(vcf_df, chunk)
    if len(chunk) > 0:
        vcf_df = append_chunk(vcf_df, chunk)
    return vcf_df


def vcf_load(fi, tag_columns=[], regions=[], labels={}, parse_infos=False, prefix=None, chunksize=10000, use_cudf=True):
    vcf_df = pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "ref", "alt"])
    vcf = cyvcf2.VCF(fi)
    if len(regions) > 0:
        for region in regions:
            vcf_df = vcf_df.append(parse_region(vcf(region), tag_columns, parse_infos,
                                                prefix, chunksize), ignore_index=True, sort=False)
    else:
        vcf_df = parse_region(vcf, tag_columns, parse_infos, prefix, chunksize)

    if use_cudf:
        return cudf.DataFrame(vcf_df)
    return vcf_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g",
                        "--gpu",
                        dest="gpu", help="Use GPU-backed datastructures. [no]",
                        action="store_true")
    # parser.add_argument("-l", "--list",
    #                     type=str,
    #                     help="A file containing a list of sample VCF files, one per line.", required=False)
    # parser.add_argument("-L", "--labels",
    #                     help="A file containing key:value labels, one per line, matched with the line of the file provided with --list.",
    #                     required=False, dest="label_list")
    # parser.add_argument("-T", "--tumor")
    # parser.add_argument("-N", "--normal")
    # parser.add_argument("-a", "--annotation",
    #                     help="An annotation VCF to used to annotate input VCF(s).", type=str,
    #                     required=False)
    # parser.add_argument("-p", "--pon",
    #                     help="A panel-of-normals VCF used to filter input VCF(s).", type=str,
    #                     required=False)

    return parser.parse_args()


class SimpleTimer:

    def __init__(self):
        self._start = None
        self._end = None
        self._time = None

    def start(self):
        self._start = time.time()

    def end(self):
        self._end = time.time()
        self._time = self._end - self._start

    def time(self):
        return _end - _start

    def print_time(self, msg=""):
        print("elapsed time (s) : " + msg + " :", round(self._time, 4), file=sys.stderr)


if __name__ == "__main__":
    # Read in example VCF

    args = parse_args()

    st = SimpleTimer()
    st.start()
    sample_vcf = vcf_load("/aztlan/data/2790b964-63e3-49aa-bf8c-9a00d3448c25.consensus.20160830.somatic.snv_mnv.vcf.gz",
                          tag_columns=["Callers", "t_alt_count"], use_cudf=args.gpu)
    st.end()
    st.print_time("Load sample VCF")

    st.start()
    second_sample_vcf = vcf_load(
        "/aztlan/data/ff870342-f0d6-4450-8f9c-344c046a0baf.consensus.20160830.somatic.snv_mnv.vcf.gz", tag_columns=["Callers", "t_alt_count"], use_cudf=args.gpu)
    st.end()
    st.print_time("Load second VCF")

    # Read in 1000 Genomes VCF
    st.start()
    one_kg = vcf_load(
        "/aztlan/data/1000Genomes/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.vcf.gz", tag_columns=["AF", "EAS_AF", "AFR_AF", "EUR_AF", "AMF_AF", "SAS_AF", "AC"], regions=["22"], chunksize=100000, use_cudf=args.gpu)
    st.end()
    st.print_time("Load 1kg VCF")
    # Create 1KG PON
    one_kg_pon = pon.create_pon(one_kg)

    # Filter example VCF
    st.start()
    filt = one_kg_pon.filter_by_allele_frequency(sample_vcf, 0.05, dropNA=False, dropAnnotations=False)
    st.end()
    st.print_time("PON filter, AF, first VCF")

    st.start()
    filt = one_kg_pon.filter_by_count(sample_vcf, 10, count_variable="AC")
    st.end()
    st.print_time("PON filter, count, first VCF")

    filt_second = one_kg_pon.filter_by_allele_frequency(second_sample_vcf, 0.05)
    filt_second = one_kg_pon.filter_by_count(second_sample_vcf, 10, count_variable="AC")

    #filt = mf.bind_rows(filt, filt_second)
