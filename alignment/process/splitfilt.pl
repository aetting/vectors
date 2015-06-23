#!/usr/bin/perl -w

my $file = shift @ARGV;

open $ff,$file;
open(my $filef, ">", shift @ARGV);
open(my $filee, ">", shift @ARGV);

while (<$ff>) {
 my ($sf, $se) = split /\s*\|\|\|\s*/;
 print $filef $sf;
 print $filef "\n";
 print $filee $se;
 print;
 }
close $ff;
close $filef;
close $filee;