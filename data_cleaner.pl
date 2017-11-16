# data_cleaner.pl
# Miguel Velazquez 2017
# This script cleans u.data from MovieLense 1M data set and
# creates a data file readable for our recommender

use strict;
use warnings;

my $filename = 'u.data';
open(my $fh, '<:encoding(UTF-8)', $filename)
  or die "Could not open file '$filename' $!";

my $u = 943;
my $m = 1682;


 
my %users;
while (my $row = <$fh>) {
  chomp $row;
  my @items = split('\s', $row);
  #-1 to match an array indexing. IDs started in 1 on original data
  my $user_id = $items[0]-1;
  my $movie_id = $items[1]-1;
  my $rating = $items[2];
  
  $users{$user_id}->{$movie_id} = $rating;
}

close $fh;
my $against_user_id = 80;


for (my $user_index = 0; $user_index < $u; $user_index++) {
  for (my $movie_index = 0; $movie_index < $m; $movie_index++) {
    my $one_dimension_index = $movie_index + ($user_index * $m);
	
	#Put movie index and leave 2 spaces after. Eg. movie_id = 10 turns into 1000
    my $movie_id_plus_rating_encoded =  $movie_index  * 100; 
    
    if(defined $users{$user_index}->{$movie_index}) {
      $movie_id_plus_rating_encoded = $movie_id_plus_rating_encoded  + $users{$user_index}->{$movie_index};
    } 
	print "d_matrixA[$one_dimension_index]  =  $movie_id_plus_rating_encoded;";
  }
  print "\n";
}

for (my $user_index = 0; $user_index < $u; $user_index++) {
  for (my $movie_index = 0; $movie_index < $m; $movie_index++) {
    my $one_dimension_index = $movie_index + ($user_index * $m);
	
	#Put movie index and leave 2 spaces after. Eg. movie_id = 10 turns into 1000
    my $movie_id_plus_rating_encoded =  $movie_index  * 100; 
    
    if(defined $users{$against_user_id}->{$movie_index}) {
      $movie_id_plus_rating_encoded = $movie_id_plus_rating_encoded  + $users{$against_user_id}->{$movie_index};
    } 
	print "d_matrixB[$one_dimension_index]  =  $movie_id_plus_rating_encoded;";
  }
  print "\n";
}

my $count;
for (my $user_index = 0; $user_index < $u; $user_index++) {
  $count = 0;
  print "user $user_index :  :";
  for (my $movie_index = 0; $movie_index < $m; $movie_index++) {
    my $one_dimension_index = $movie_index + ($user_index * $m);
	
	#Put movie index and leave 2 spaces after. Eg. movie_id = 10 turns into 1000
    my $movie_id_plus_rating_encoded =  0; 
    
    if(defined $users{$user_index}->{$movie_index}) {
      $movie_id_plus_rating_encoded = $movie_id_plus_rating_encoded  + $users{$user_index}->{$movie_index};
	  $count += 1;
    } 
	print "$movie_id_plus_rating_encoded ";
  }
  print ": : $count\n";
}

foreach my $key (sort keys %users) {
  print "user $key :  :";
  my %u = %{$users{$key}};
 
  foreach my $m (sort keys %u) {
    print " $m,".$u{$m};
  }
  print "\n";
}


exit 1;
