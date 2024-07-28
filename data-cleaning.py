import numpy as np

with open('trip-advisor-reviews.dat') as f: # PrefLib Trip Advisor Data
    lines = f.readlines()

print(lines[0])

data = []

for review in lines[1:]:
    review = review.split(',')

    if int(review[8]) > 0:
        data.append([int(review[0]), review[3], int(review[8])]) # extract hotel ID, location, and cleanliness rating

print('Data length =', len(data)) # number of reviews

transposed_data = list(map(list, zip(*data)))
print('Transposed data length =', len(transposed_data))

hotel_IDs = []

for id in transposed_data[0]:
    if id not in hotel_IDs:
        hotel_IDs.append(id)

print('Number of distinct hotel IDs =', len(hotel_IDs))

hotel_ID_occurrences = [] # how many reviews are associated with each hotel ID

for id in hotel_IDs:
    hotel_ID_occurrences.append(len(np.where(np.array(transposed_data[0]) == id)[0]))

distinct_hotel_locations = []

for location in transposed_data[1]:
    if location not in distinct_hotel_locations:
        distinct_hotel_locations.append(location)

print('Number of distinct hotel locations =', len(distinct_hotel_locations))
print(distinct_hotel_locations[2]) # Punta Cana Dominican Republic

hotels_per_location = []

for location in distinct_hotel_locations:
    arr = []

    for i in range(len(transposed_data[0])): # for each review
        if (transposed_data[1][i] == location) & (transposed_data[0][i] not in arr): # if the review has the right location and its ID has not been saved
            arr.append(transposed_data[0][i]) # save its ID
    
    hotels_per_location.append(arr) # append list of hotel IDs for the selected location

print('Length of hotels per location =', len(hotels_per_location)) # same as number of distinct hotel locations

hotel_IDs_with_500_reviews = np.take(hotel_IDs, np.where(np.array(hotel_ID_occurrences) >= 500)[0])

hotels_with_500_reviews_per_location = []

for location in hotels_per_location:
    arr = []

    for hotel in location:
        if hotel in hotel_IDs_with_500_reviews:
            arr.append(hotel)
    
    hotels_with_500_reviews_per_location.append(arr) # append list of hotel IDs for the selected location with >= 500 reviews

print(hotels_with_500_reviews_per_location)

print(hotels_with_500_reviews_per_location[2])

# for i in range(len(data)):
#     if transposed_data[0][i] in hotels_with_500_reviews_per_location[2]:
#         print(transposed_data[1][i])

def get_cleanliness_ratings(id):
    return np.take(transposed_data[2], np.where(np.array(transposed_data[0]) == id)[0]) >= 4

hotel_149397_cleanliness_ratings = get_cleanliness_ratings(149397)
hotel_149397_mean = np.mean(hotel_149397_cleanliness_ratings)

hotel_149399_cleanliness_ratings = get_cleanliness_ratings(149399)
hotel_149399_mean = np.mean(hotel_149399_cleanliness_ratings)

hotel_150841_cleanliness_ratings = get_cleanliness_ratings(150841)
hotel_150841_mean = np.mean(hotel_150841_cleanliness_ratings)

hotel_218486_cleanliness_ratings = get_cleanliness_ratings(218486)
hotel_218486_mean = np.mean(hotel_218486_cleanliness_ratings)

np.save('hotel-149397-cleanliness-ratings', hotel_149397_cleanliness_ratings)
np.save('hotel-149399-cleanliness-ratings', hotel_149399_cleanliness_ratings)
np.save('hotel-150841-cleanliness-ratings', hotel_150841_cleanliness_ratings)
np.save('hotel-218486-cleanliness-ratings', hotel_218486_cleanliness_ratings)

np.save('hotel-means', np.array([hotel_149397_mean, hotel_149399_mean, hotel_150841_mean, hotel_218486_mean]))

print(np.array([hotel_149397_mean, hotel_149399_mean, hotel_150841_mean, hotel_218486_mean]))