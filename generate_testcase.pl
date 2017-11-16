#!/usr/bin/perl
use strict;
use warnings;

my $cmd;
my @clients = (63);
my @programs = ("simple_recommender","optimized_recommender","fast_recommender");
my $prefix = "demo";

#max users 6040
#max movies 3952
my $t = 0;
for (my $u = 250; $u <= 250; $u=$u+250) {
  for (my $m = 250; $m <= 1000; $m=$m+250) {
    foreach my $c (@clients) {
      foreach my $p (@programs) {
        my $px = $p;
        $px =~ s/_recommender// ;
        $cmd  = "( time ./$p $u $m $c 0 10 > ".$prefix."_".$px."_".$u."_".$m."_".$c.".out) 2>&1 | grep real > ".$prefix."_".$px."_".$u."_".$m."_".$c;
        print $cmd."\n";
        print "echo \"  \"\n";
        print "echo \"##TEST $t##\"\n";
        print "echo \"  \"\n";
        print "echo \"Running: $p $u $m $c 0 10 > ".$prefix."_".$px."_".$u."_".$m."_".$c.".out\"\n";
        print "echo \"  \"\n";
        print "echo \"Result: \"\n";
        print "cat ".$prefix."_".$px."_".$u."_".$m."_".$c.".out\n";
        print "echo \"  \"\n";
        print "echo \"Ellapsed Time: \"\n";
        print "cat ".$prefix."_".$px."_".$u."_".$m."_".$c."\n";
        print "echo \"  \"\n";
        print "echo \"  \"\n";
        $t++;
      }
    }
  }
}
