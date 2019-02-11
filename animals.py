import numpy as np

# 4.1 Animal Species

def import_animal_data():
    animal_data_array=np.empty((32,84))
    animal_data = [i.strip().split() for i in open("animals.dat").readlines()]
    animal_data= animal_data[0][0].split(',')
    j = -1
    for i in range(len(animal_data)):
        if i%84 !=0:
            k+=1
            animal_data_array[j][k] = (animal_data[i])
        else:
            j+=1
            k = 0
            animal_data_array[j] = np.empty(84)
            animal_data_array[j][k] = (animal_data[i])
    animal_names = [i.strip().split() for i in open("animalnames.txt").readlines()]
    return animal_data_array, animal_names

def distance_measure(pattern, weight):
    dist_vec = pattern-weight
    dist = np.empty(len(dist_vec))
    for ind,vec in enumerate(dist_vec):
        dist[ind] = np.linalg.norm(vec)
    return dist

def animal_species_SOM():
    eta = 0.2
    input_data,animal_names = import_animal_data()
    output_nodes = 100
    patterns = 84
    W = np.random.rand(output_nodes, patterns)
    epochs = 20
    for epoch in range(epochs):
        neighbourhood_size = (50 - round(2.5 * epoch))
        for sample in input_data:
            dist = distance_measure(sample, W)
            winner = np.argmin(dist)
            if winner<neighbourhood_size:
                idx = np.arange(0,neighbourhood_size)
            else:
                idx = np.arange(winner-neighbourhood_size,winner)
            W[idx,:] = W[idx,:] + eta * (sample - W[idx,:])

    order = []
    for ind,sample in enumerate(input_data):
        dist = distance_measure(sample, W)
        winner = np.argmin(dist)
        order.append(winner)
    order = np.argsort(order)

    print("Length", order)
    for i in order:
        print(animal_names[i][0])


animal_species_SOM()