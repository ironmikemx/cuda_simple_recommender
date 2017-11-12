# @author Miguel Angel Velázquez Ramos
# 2017
#
# This script is used to convert the 1m movie lense dataset into 
# a data.dat file we can use in our movie recommender
# It generates a matrix with user as rows and movie ratings
# as columns
#
use strict;
use warnings;

my $filename = 'ratings.dat';
open(my $fh, '<:encoding(UTF-8)', $filename)
    or die "Could not open file '$filename' $!";

#data set size in ratings.dat
my $u = 6040;
my $m = 3952;

my %users;

#Read teh data and sort it by user
while (my $row = <$fh>) {
    chomp $row;
    my @items = split('::', $row);
    #-1 to match an array indexing. IDs started in 1 on original data
    my $user_id = $items[0]-1;
    my $movie_id = $items[1]-1;
    my $rating = $items[2];
  
    $users{$user_id}->{$movie_id} = $rating;
}

close $fh;


#output the data per user. If the user has not rated a particular movie
#put a 0
for (my $user_index = 0; $user_index < $u; $user_index++) {
    for (my $movie_index = 0; $movie_index < $m; $movie_index++) {

        my $one_dimension_index = $movie_index + ($user_index * $m);
        my $movie_id_plus_rating_encoded = 0;
        if(defined $users{$user_index}->{$movie_index}) {
            $movie_id_plus_rating_encoded = $movie_id_plus_rating_encoded  + $users{$user_index}->{$movie_index};
        } 
	print "$movie_id_plus_rating_encoded ";
    }
     print "\n";
}


exit 1;
