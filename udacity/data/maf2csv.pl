#!/usr/bin/perl

use strict;

my($debug) = 0;

# Mutataions by sample, by gene
my(%mutSample);

# All genes
my(%genes);

#-------------------------------------------------------------------------------
# Porcess each MAF file
#-------------------------------------------------------------------------------
sub mutations($) {
	my($f) = @_;

	my($sample, $lineNum, $l);
	if($f =~ /(.*).maf.txt/) { $sample = $1; }

	print("File: '$f', sample: '$sample'\n") if $debug;
	open FILE, $f;
	for($lineNum=0; $l=<FILE>; $lineNum++) {
		chomp $l;
		my($gene, $geneId, $center, $ncbiBuild, $chr, $start, $end, $strand, $var_class) = split /\t/, $l;
		if(($var_class eq 'Silent') or ($var_class eq 'RNA')) { next; }
		print("$f:$lineNum: ($gene, $geneId, $center, $ncbiBuild, $chr, $start, $end, $strand, $var_class)\n") if $debug;

		$genes{$gene} = 1;
		$mutSample{$sample}->{$gene} = 1;
	}
	close FILE;
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
my($f);
foreach $f (@ARGV) {
	mutations($f);
}

my($sample, $g);
# Show title
print("sample");
foreach $g (sort keys %genes) {
	print(",$g");
}
print("\n");

# Show data
foreach $sample (sort keys %mutSample) {
	print("$sample");
	foreach $g (sort keys %genes) {
		print("," . ($mutSample{$sample}->{$g} ? '1' : '0'))
	}
	print("\n");
}
