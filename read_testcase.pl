use warnings;
use strict;


my $cmd = "ls demo_* | grep -v out";
my @files = `$cmd`;

foreach my $f (@files) {
  my $time = `cat $f`;
  chomp($time);
  $time =~ s/real\s+//;
  my @tokens = split("_", $f);
  $time =~ /([\d\.]+)m([\d\.]+)s/;
  my $secs = ($1 * 60) + $2;
  chomp($tokens[4]);
  print "$tokens[0],$tokens[1],$tokens[2],$tokens[3],$tokens[4],$secs";
  print "\n";
}
