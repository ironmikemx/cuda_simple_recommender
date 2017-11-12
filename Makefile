all:	simple_recommender optimized_recommender  fast_recommender

simple_recommender:	data.o simple_recommender.o
	nvcc data.o simple_recommender.o -o simple_recommender

simple_recommender.o:	simple_recommender.cu
	nvcc -c simple_recommender.cu -o simple_recommender.o

optimized_recommender:	data.o  optimized_recommender.o
	nvcc data.o optimized_recommender.o -o optimized_recommender

optimized_recommender.o:  optimized_recommender.cu
	nvcc -c optimized_recommender.cu -o optimized_recommender.o

fast_recommender:    data.o fast_recommender.o
	nvcc data.o fast_recommender.o -o fast_recommender

fast_recommender.o:  fast_recommender.cu
	nvcc -c fast_recommender.cu -o fast_recommender.o

data.o: data.cu
	nvcc -c data.cu -o data.o


clean:
	rm -f *.o simple_recommender optimized_recommender optimized_recommender 


