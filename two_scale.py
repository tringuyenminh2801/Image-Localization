
from cv2 import destroyAllWindows, destroyWindow
import numpy as np
import matplotlib.pyplot as plt
import random
import math as m
import cv2 as cv

template = cv.imread("skku_0.png", 0)
np.set_printoptions(suppress=True)
#function for genetic algorithm
def population_init(size_of_population, shape = np.shape(template)):
    #create first resident
    #individual : [scale_percent_x, scale_percent_y]
    population = np.array([[random.uniform(2, 100), random.uniform(2, 100)]], dtype = float)
    #must use loop to create random residents
    for i in range(size_of_population - 1):
        scale_percent_x = random.uniform(2, 100)
        scale_percent_y = random.uniform(2, 100)
        population = np.append(population, np.array([[scale_percent_x, scale_percent_y]]), axis=0)
    return population

def compute_fitness(individual, population_size, template = cv.imread("skku_0.png", 0), img = cv.imread("skku_1.png", 0)):
    #input: individual: 1 x 1
    #dim: height x width 
    dim = (int(m.ceil(template.shape[1] * individual[0] / 100)), int(m.ceil(template.shape[0] * individual[1] / 100)))
    similarity_map = cv.matchTemplate(img, cv.resize(template, dim, interpolation=cv.INTER_AREA) , cv.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(similarity_map)
    return maxLoc, maxVal

def binary_selection(population, population_size):
    random_numbers = np.array([(random.randint(0,population_size-1), random.randint(0,population_size-1)) for _ in range(population_size)])
    population_new = np.zeros(np.shape(population), dtype=float)
    for i in range(population_size):
        a = population[random_numbers[i,0],:]
        b = population[random_numbers[i,1],:]
        if (a[-1] > b[-1]):
            population_new[i] = a
        elif (b[-1] > a[-1]):
            population_new[i] = b
        else:
            population_new[i] = a
    return population_new

def crossover(individual1, individual2, crossover_rate = 0.9, chromosome_length = 2):
    in1_new = individual1.copy()
    in2_new = individual2.copy()
    for i in range(chromosome_length):
        rand_num = random.uniform(0,1)
        if (rand_num < crossover_rate):
            in1_new[i] = individual2[i]
            in2_new[i] = individual1[i]
    return in1_new, in2_new

def mutation(population, size_of_population, mutation_rate = 0.05, chromosome_length = 2):
    for i in range(size_of_population):
        for j in range(chromosome_length):
            a = random.uniform(0,1)
            if (a < mutation_rate):
                population[i,j] = random.uniform(2,100)
    return population

def create_new_population(population, generation, size_of_population, chromosome_length = 2, template = cv.imread("skku_0.png", 0), img = cv.imread("skku_1.png", 0)):
    population_fitness_score = np.zeros((size_of_population, 3), dtype = float)
    for i in range(size_of_population):
        (location, score) = compute_fitness(population[i], size_of_population)
        population_fitness_score[i] = np.array([[location[0], location[1], score]])
    population = np.column_stack([population, population_fitness_score])
    #population: 2d array [scale_percent_x, scale_percent_y, x, y, score]
    selected_population = binary_selection(population, size_of_population)
    #selected_population : 2d array [scale_percent_x, scale_percent_y, x, y, score]
    #take the best location from selected population and its score
    selected_population = selected_population[selected_population[:,-1].argsort()] #sort according to the last column
    #best_location: x, y
    best_scale = selected_population[-1, :2]
    best_location = (selected_population[-1, 2], selected_population[-1, 3])
    best_score = selected_population[-1, -1]
    #crossover for new population
    for i in range(int(size_of_population/2)):
        selected_population[2*i, :2], selected_population[2*i+1, :2] = crossover(selected_population[2*i, :2], selected_population[2*i+1, :2])
    new_population = selected_population[:,:2]
    new_population = np.resize(new_population, (size_of_population, 2))
    new_population = mutation(new_population, size_of_population)
    return best_scale, best_location, best_score, new_population

def main():
    #read and show image
    #template & image format : numpy.ndarray
    template = cv.imread("skku_0.png", 0)
    img = cv.imread("skku_1.png", 0)

    #New part
    img = cv.equalizeHist(img)
    img1 = cv.imread("skku_1.png", 1)
    template1 = cv.imread("skku_0.png", 1)

    #Show original image
    img_show = np.squeeze(img1)
    cv.imshow("Original image", img_show)
    k1 = cv.waitKey()
    if k1 == ord('q'):
        cv.destroyWindow(winname="Original image")

    #Show template image
    template_show = np.squeeze(template1)
    cv.imshow("Template", template_show)
    k2 = cv.waitKey()
    if (k2 == ord('q')):
        cv.destroyWindow(winname="Template")

    size_of_population = 30
    chromosome_length = 2
    num_of_generations = 100
    #plotting loss function
    losses = []
    generations_x = [i+1 for i in range(num_of_generations)]
    population = population_init(size_of_population)

    for i in range(num_of_generations):
        (best_scale, best_location, best_score, population) = create_new_population(population, i+1, size_of_population)
        losses.append(best_score)
        print(f"{i+1} generation")

    plt.plot(generations_x, losses)
    plt.ylabel("Similar values")
    plt.xlabel("Generations")
    plt.title("Genetic Algorithm similarity function")
    plt.show()

    print("Precision: ", best_score)
    print("Best scale: ", best_scale)

    start_point = (int(m.ceil(best_location[1])), int(m.ceil(best_location[0])))
    x_end = int(m.ceil(best_location[0]+(img.shape[1]*best_scale[0])/100))
    y_end = int(m.ceil(best_location[1]+(img.shape[0]*best_scale[1])/100))
    end_point = (x_end, y_end)

    color = (255, 0, 0)
    image = cv.rectangle(img1, start_point, end_point, color, thickness=2)

    cv.imshow("Result", image)
    k = cv.waitKey()
    if k == ord('q'):
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
