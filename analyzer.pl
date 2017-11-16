#!/usr/bin/perl
use strict;
use warnings;

my $cmd;
#my @clients = (5, 473, 1298, 3987, 6000);
my @clients = (63);
my @programs = ("simple_recommender","optimized_recommender","fast_recommender");
my $prefix = "medidas200m";

#max users 6040
#max movies 3952

for (my $u = 250; $u <= 6000; $u=$u+500) {
  for (my $m = 200; $m <= 200; $m=$m+200) {
    foreach my $c (@clients) {
      foreach my $p (@programs) {
        my $px = $p;
        $px =~ s/_recommender// ;
        $cmd  = "( time ./$p $u $m $c 0 10 > /dev/null) 2>&1 | grep real > ".$prefix."_".$px."_".$u."_".$m."_".$c;
        print $cmd."\n";
      }
    }
  }
}
