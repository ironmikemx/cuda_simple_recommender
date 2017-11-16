use warnings;
use strict;


my $cmd = "ls clientfast_recommender_*";
my @files = `$cmd`;

foreach my $f (@files) {
  my $time = `cat $f`;
  chomp($time);
  $time =~ s/real\s+//;
  my @tokens = split("_", $f);
  $time =~ /([\d\.]+)m([\d\.]+)s/;
  my $secs = ($1 * 60) + $2;
  chomp($tokens[5]);
  print "$tokens[0],$tokens[2],$tokens[3],$tokens[4],$tokens[5],$secs";
  print "\n";
}
