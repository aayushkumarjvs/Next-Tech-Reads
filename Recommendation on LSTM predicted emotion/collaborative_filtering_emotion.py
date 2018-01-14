#!/usr/bin/env python
# Implementation of collaborative filtering recommendation engine



from recommendation_data_emotion import dataset_emotion

from math import sqrt

def similarity_score(person1,person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 

	both_viewed = {}		# To get both rated items by person1 and person2

	for item in dataset_emotion[person1]:
		if item in dataset_emotion[person2]:
			both_viewed[item] = 1

		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0

		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	

		for item in dataset_emotion[person1]:
			if item in dataset_emotion[person2]:
				sum_of_eclidean_distance.append(pow(dataset_emotion[person1][item] - dataset_emotion[person2][item],2)) #example:- sqrt(pow(5-4,2)+pow(4-1,2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

		return 1/(1+sqrt(sum_of_eclidean_distance))  #euclidean distance is only successful when the similarity score is one, so we divide it by zero and add 1 to the denominator.



def pearson_correlation(person1,person2):

	# To get both rated items
	both_rated = {}
	for item in dataset_emotion[person1]:
		if item in dataset_emotion[person2]:
			both_rated[item] = 1

	number_of_ratings = len(both_rated)		
	
	# Checking for number of ratings in common
	if number_of_ratings == 0:
		return 0

	# Add up all the preferences of each user
	person1_preferences_sum = sum([dataset_emotion[person1][item] for item in both_rated])
	person2_preferences_sum = sum([dataset_emotion[person2][item] for item in both_rated])

	# Sum up the squares of preferences of each user
	person1_square_preferences_sum = sum([pow(dataset_emotion[person1][item],2) for item in both_rated])
	person2_square_preferences_sum = sum([pow(dataset_emotion[person2][item],2) for item in both_rated])

	# Sum up the product value of both preferences for each item
	product_sum_of_both_users = sum([dataset_emotion[person1][item] * dataset_emotion[person2][item] for item in both_rated])

	# Calculate the pearson score
	numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
	#numerator => Sxy = sum(xy)-[(sum x)(sum y)]/n
	denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
        #denominator => sqrt(Sxx.Syy)
	#Sxx = sum(x,2) - [(sum x),2]/n
	#Syy = sum(y,2) - [(sum y),2]/n
	if denominator_value == 0:
	   	return 0
	else:
		r = numerator_value/denominator_value
		return r 

def most_similar_users(person,number_of_users):
	# returns the number_of_users (similar persons) for a given specific person.
	scores = [(pearson_correlation(person,other_person),other_person) for other_person in dataset_emotion if  other_person != person ]
	
	# Sort the similar persons so that highest scores person will appear at the first
	scores.sort()
	scores.reverse()
	return scores[0:number_of_users]

def user_reommendations(person):

	# Gets recommendations for a person by using a weighted average of every other user's rankings
	totals = {}
	simSums = {}
	rankings_list =[]
	for other in dataset_emotion:
		# don't compare me to myself
		if other == person:
			continue
		sim = pearson_correlation(person,other)
		#print ">>>>>>>",sim

		# ignore scores of zero or lower
		if sim <=0: 
			continue
		for item in dataset_emotion[other]:

			# only score books I haven't read yet
			if item not in dataset_emotion[person] or dataset_emotion[person][item] == 0:

			# Similrity * score
				totals.setdefault(item,0)
				totals[item] += dataset_emotion[other][item]* sim
				# sum of similarities
				simSums.setdefault(item,0)
				simSums[item]+= sim

		# Create the normalized list

	rankings = [(total/simSums[item],item) for item,total in totals.items()]
	rankings.sort()
	rankings.reverse()
	# returns the recommended items
	recommendataions_list = [recommend_item for score,recommend_item in rankings]
	return recommendataions_list
		

print (user_reommendations('sadness'))
