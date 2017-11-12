#include <thrust/device_vector.h>
#include <stdio.h>

/*
 * Function:  load
 * --------------------
 * copies a subset u * m of data into output vector
 *
 *  output_vector: destination output vector
 *  u: number of users to copy
 *  m: number of movies to copy
 *
 *  returns: Nothing
 */
void load(thrust::device_vector<int> & output_vector, const int u, const int m) {

    int num_of_movies = 3952;

    FILE *file;
    int temp;

    file = fopen("data.dat", "r");

    //For each user in subset
    for(int k = 0; k < u; k++) {
       //Read an entire row of movies of size num_of_movies
       for(int i = 0; i < num_of_movies; i++) {
           fscanf(file, "%d", &temp);
           //As is a sequential file we need to read the rest of movie
           //ratings but we only care of the ones which id < m
           //that means the size of our movie subset
           if(i < m) {
              output_vector[i+(k*m)] = temp;
           }
        } 
    }

    fclose(file);

    return;
}

/*
 * Function:  load
 * --------------------
 * generates dataset of size u * m with the ratings of the client
 *
 *  output_vector: destination output vector
 *  u: number of users to copy
 *  m: number of movies to copy
 *  client_id: user_id of client
 *
 *  returns: Nothing
 */
void load(thrust::device_vector<int> & output_vector, const int u, const int m, const int client_id) {

    int num_of_movies = 3952;
    thrust::device_vector<int> client_ratings(m);


    FILE *file;
    int temp;

    file = fopen("data.dat", "r");

    //Read from the file until we reach the row of client_id
    for(int k = 0; k < client_id * num_of_movies; k++) {
       fscanf(file, "%d", &temp);
    } 
    //Read the client_id movie ratings
    for(int i = 0; i < m; i++) {
       fscanf(file, "%d", &temp);
       client_ratings[i] = temp;
    }
    fclose(file);

    //Copy the client_ratings to form a matrix of u size
    for(int i = 0; i < u; i++) {
        thrust::copy(client_ratings.begin(), client_ratings.end(), output_vector.begin()+(i*m)); 
    }

  return;

}


/*
 * Function:  load
 * --------------------
 * copies a subset u * m of data into output vector
 *
 *  output_vector: destination output vector
 *  u: number of users to copy
 *  m: number of movies to copy
 *
 *  returns: Nothing
 */
void load_char(thrust::device_vector<char> & output_vector, const int u, const int m) {

    int num_of_movies = 3952;

    FILE *file;
    int temp;

    file = fopen("data.dat", "r");

    //For each user in subset
    for(int k = 0; k < u; k++) {
       //Read an entire row of movies of size num_of_movies
       for(int i = 0; i < num_of_movies; i++) {
           fscanf(file, "%d", &temp);
           //As is a sequential file we need to read the rest of movie
           //ratings but we only care of the ones which id < m
           //that means the size of our movie subset
           if(i < m) {
              output_vector[i+(k*m)] = (char)temp;
           }
        } 
    }

    fclose(file);

    return;
}

/*
 * Function:  load
 * --------------------
 * generates dataset of size u * m with the ratings of the client
 *
 *  output_vector: destination output vector
 *  u: number of users to copy
 *  m: number of movies to copy
 *  client_id: user_id of client
 *
 *  returns: Nothing
 */
void load_char(thrust::device_vector<char> & output_vector, const int u, const int m, const int client_id) {

    int num_of_movies = 3952;
    thrust::device_vector<int> client_ratings(m);


    FILE *file;
    int temp;

    file = fopen("data.dat", "r");

    //Read from the file until we reach the row of client_id
    for(int k = 0; k < client_id * num_of_movies; k++) {
       fscanf(file, "%d", &temp);
    } 
    //Read the client_id movie ratings
    for(int i = 0; i < m; i++) {
       fscanf(file, "%d", &temp);
       client_ratings[i] = (char)temp;
    }
    fclose(file);

    //Copy the client_ratings to form a matrix of u size
    for(int i = 0; i < u; i++) {
        thrust::copy(client_ratings.begin(), client_ratings.end(), output_vector.begin()+(i*m)); 
    }

  return;

}



/*
 * Function:  load
 * --------------------
 * copies a subset u * m of data into output vector
 *
 *  output_vector: destination output vector
 *  u: number of users to copy
 *  m: number of movies to copy
 *
 *  returns: Nothing
 */
void load_char_from(thrust::device_vector<char> & output_vector, const int u, const int m, const int offset) {

    int num_of_movies = 3952;

    FILE *file;
    int temp;

    file = fopen("data.dat", "r");

    //Read from the file until we reach the row of offset
    for(int k = 0; k < offset * num_of_movies; k++) {
       fscanf(file, "%d", &temp);
    }


    //For each user in subset
    for(int k = 0; k < u; k++) {
       //Read an entire row of movies of size num_of_movies
       for(int i = 0; i < num_of_movies; i++) {
           fscanf(file, "%d", &temp);
           //As is a sequential file we need to read the rest of movie
           //ratings but we only care of the ones which id < m
           //that means the size of our movie subset
           if(i < m) {
              output_vector[i+(k*m)] = (char)temp;
           }
        }
    }

    fclose(file);

    return;
}

